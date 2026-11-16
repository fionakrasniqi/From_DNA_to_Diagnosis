"""
s02_fetch_genomes.py — assembly metadata + contig FASTA download (proposal §2.1).

Fixes the defect that invalidated the first attempt. The BV-BRC data API returns
25 records by default, so `genome_sequence/?eq(genome_id,X)` returned only the
first 25 contigs of each assembly: median 1.16 Mb against a true E. coli genome
of ~5.07 Mb, i.e. roughly a fifth of each genome, chosen arbitrarily.

Two independent guards now make that failure impossible to repeat silently:
  1. every sequence request carries an explicit limit() clause;
  2. every download is checked against the genome_length BV-BRC itself reports
     for that assembly, and anything below MIN_COMPLETENESS is rejected.

The metadata pull is needed anyway: MLST supports the cluster validation in
stage 05, and isolation_country drives the geographic hold-out in stage 05.

FASTAs are stored gzipped (~11 GB rather than ~44 GB). Resumable — re-running
skips genomes already downloaded and validated.

Output:
    data/genomes/<genome_id>.fna.gz
    data/processed/genome_metadata.csv
    data/processed/download_manifest.csv
"""

from __future__ import annotations

import argparse
import gzip
import io
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import config as C

log = logging.getLogger("s02")

METADATA_FIELDS = [
    "genome_id", "genome_name", "mlst", "genome_quality", "contigs",
    "genome_length", "gc_content", "checkm_completeness", "checkm_contamination",
    "isolation_country", "collection_year", "host_name", "isolation_source",
]


def make_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(total=4, backoff_factor=1.5,
                  status_forcelist=(429, 500, 502, 503, 504),
                  allowed_methods=frozenset(["GET", "POST"]))
    session.mount("https://", HTTPAdapter(max_retries=retry,
                                          pool_connections=64, pool_maxsize=64))
    return session


def fetch_metadata(session: requests.Session, genome_ids: list[str]) -> pd.DataFrame:
    """Assembly QC and epidemiological metadata for every genome in the cohort."""
    frames = []
    for start in range(0, len(genome_ids), C.METADATA_BATCH):
        batch = genome_ids[start:start + C.METADATA_BATCH]
        query = (f"in(genome_id,({','.join(batch)}))"
                 f"&select({','.join(METADATA_FIELDS)})"
                 f"&limit({C.METADATA_BATCH * 2})")
        resp = session.post(f"{C.API}/genome/", data=query,
                            headers=C.RQL_HEADERS, timeout=C.REQUEST_TIMEOUT)
        resp.raise_for_status()
        if len(resp.content) > 50:
            frames.append(pd.read_csv(io.StringIO(resp.text), sep="\t",
                                      dtype={"genome_id": str}))
        log.info("metadata %d/%d", min(start + C.METADATA_BATCH, len(genome_ids)),
                 len(genome_ids))
        time.sleep(0.3)

    md = pd.concat(frames, ignore_index=True).drop_duplicates("genome_id")
    md["genome_id"] = md["genome_id"].astype(str).str.strip().str.strip('"')
    missing = set(genome_ids) - set(md["genome_id"])
    if missing:
        log.warning("no metadata for %d genome(s); they cannot be "
                    "length-validated and will be skipped", len(missing))
    return md


def measure_fasta(text: str) -> tuple[int, int]:
    """Return (n_contigs, total_bp) without materialising the sequence."""
    n_contigs = total_bp = 0
    for line in text.splitlines():
        if line.startswith(">"):
            n_contigs += 1
        else:
            total_bp += len(line.strip())
    return n_contigs, total_bp


def fetch_one(session: requests.Session, gid: str, expected_bp: float) -> dict:
    """Download one genome, validate length, store gzipped."""
    dest = C.GENOMES / f"{gid}.fna.gz"
    base = {"genome_id": gid, "expected_bp": expected_bp}

    if dest.exists():
        try:
            with gzip.open(dest, "rt") as fh:
                n_contigs, observed_bp = measure_fasta(fh.read())
            if observed_bp >= expected_bp * C.MIN_COMPLETENESS:
                return {**base, "status": "cached",
                        "n_contigs": n_contigs, "observed_bp": observed_bp}
        except (OSError, EOFError):
            pass
        dest.unlink(missing_ok=True)

    url = (f"{C.API}/genome_sequence/?eq(genome_id,{gid})"
           f"&limit({C.CONTIG_LIMIT})&http_accept=application/dna+fasta")
    try:
        resp = session.get(url, headers={"Accept": "application/dna+fasta"},
                           timeout=C.REQUEST_TIMEOUT)
    except requests.RequestException as exc:
        return {**base, "status": "error", "n_contigs": 0, "observed_bp": 0,
                "note": str(exc)[:120]}

    if resp.status_code != 200:
        return {**base, "status": f"http_{resp.status_code}",
                "n_contigs": 0, "observed_bp": 0}

    n_contigs, observed_bp = measure_fasta(resp.text)

    # the guard that would have caught the original bug
    if expected_bp and observed_bp < expected_bp * C.MIN_COMPLETENESS:
        return {**base, "status": "truncated",
                "n_contigs": n_contigs, "observed_bp": observed_bp}

    with gzip.open(dest, "wt", compresslevel=6) as fh:
        fh.write(resp.text)
    return {**base, "status": "ok", "n_contigs": n_contigs, "observed_bp": observed_bp}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16,
                    help="concurrent downloads; lower on a slow connection")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                        datefmt="%H:%M:%S")

    list_path = C.PROCESSED / "genome_list.txt"
    if not list_path.exists():
        log.error("missing %s — run s01_fetch_phenotypes.py first", list_path)
        return 1
    genome_ids = [g for g in list_path.read_text().split() if g]
    log.info("%d genomes in cohort", len(genome_ids))

    session = make_session()

    meta_path = C.PROCESSED / "genome_metadata.csv"
    if meta_path.exists():
        md = pd.read_csv(meta_path, dtype={"genome_id": str})
        log.info("reusing cached metadata (%d rows)", len(md))
    else:
        md = fetch_metadata(session, genome_ids)
        md.to_csv(meta_path, index=False)
        log.info("metadata -> %s", meta_path)

    md = md[md["genome_length"].notna() & (md["genome_length"] > 0)]
    log.info("expected ~%.1f GB uncompressed, ~%.1f GB gzipped",
             md["genome_length"].sum() / 1e9, md["genome_length"].sum() / 4e9)

    targets = list(zip(md["genome_id"], md["genome_length"].astype(float)))
    records, done = [], 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(fetch_one, session, gid, bp) for gid, bp in targets]
        for fut in as_completed(futures):
            records.append(fut.result())
            done += 1
            if done % 200 == 0:
                ok = sum(r["status"] in ("ok", "cached") for r in records)
                log.info("%d/%d complete (%d valid)", done, len(targets), ok)

    manifest = pd.DataFrame(records)
    manifest.to_csv(C.PROCESSED / "download_manifest.csv", index=False)

    print("\n" + "=" * 58)
    print("DOWNLOAD SUMMARY")
    print("=" * 58)
    print(manifest["status"].value_counts().to_string())

    valid = manifest[manifest["status"].isin(["ok", "cached"])]
    if not valid.empty:
        bp = valid["observed_bp"]
        print(f"\nvalidated genomes : {len(valid):,}")
        print(f"median length     : {bp.median():>12,.0f} bp")
        print(f"min / max         : {bp.min():,.0f} / {bp.max():,.0f} bp")
        print("\nSANITY CHECK: median must be near 5,000,000 bp.")
        print("Near 1,000,000 means the contig limit is not being applied — stop.")

    bad = manifest[~manifest["status"].isin(["ok", "cached"])]
    if not bad.empty:
        print(f"\n{len(bad)} genome(s) failed — re-run to retry; valid files are skipped.")

    print("\nnext: python src/s03_build_kmers.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
