"""
s01_fetch_phenotypes.py — build the labelled cohort (proposal §2.1).

Queries BV-BRC's genome_amr table for E. coli susceptibility results across all
four study antibiotics, keeps only lab-measured Resistant/Susceptible calls, and
writes one tidy long-format table plus the union genome list that stage 02 will
download.

Replaces the previous manual CSV export from the BV-BRC web interface, so the
cohort is reproducible from code (proposal §3.2).

Output:
    data/phenotypes/phenotypes_long.csv   one row per (genome, antibiotic)
    data/processed/genome_list.txt        unique genome IDs to download
    results/tables/cohort_summary.csv     per-antibiotic counts and balance
"""

from __future__ import annotations

import io
import logging
import sys
import time

import pandas as pd
import requests

import config as C

log = logging.getLogger("s01")

# BV-BRC returns display-style column names for genome_amr
RENAME = {
    "Genome ID": "genome_id",
    "Genome Name": "genome_name",
    "Antibiotic": "antibiotic",
    "Resistant Phenotype": "phenotype",
    "Evidence": "evidence",
    "Measurement Value": "mic_value",
    "Measurement Sign": "mic_sign",
    "Laboratory Typing Method": "lab_method",
    "Testing Standard": "testing_standard",
    "PubMed": "pubmed",
}


def total_records(session: requests.Session, antibiotic: str) -> int:
    """Read the true record count from the API's content-range header.

    Paging until a short page is returned is not reliable here, and paging to a
    hard-coded ceiling silently drops data — gentamicin alone has 133,597 rows.
    The server reports the total, so use it and verify against it.
    """
    query = f"eq(taxon_id,{C.TAXON_ID})&eq(antibiotic,{antibiotic})&limit(1)"
    resp = session.post(f"{C.API}/genome_amr/", data=query,
                        headers={**C.RQL_HEADERS, "Accept": "application/json"},
                        timeout=C.REQUEST_TIMEOUT)
    resp.raise_for_status()
    content_range = resp.headers.get("content-range", "")
    if "/" not in content_range:
        raise RuntimeError(f"no content-range header for {antibiotic}; cannot "
                           f"determine record count safely")
    return int(content_range.rsplit("/", 1)[1])


def fetch_antibiotic(session: requests.Session, antibiotic: str) -> pd.DataFrame:
    """Page through every genome_amr record for one antibiotic."""
    expected = total_records(session, antibiotic)
    log.info("  %s: %d records reported by API", antibiotic, expected)

    pages, fetched = [], 0
    for offset in range(0, expected, C.PAGE_SIZE):
        query = (f"eq(taxon_id,{C.TAXON_ID})&eq(antibiotic,{antibiotic})"
                 f"&limit({C.PAGE_SIZE},{offset})")
        resp = session.post(f"{C.API}/genome_amr/", data=query,
                            headers=C.RQL_HEADERS, timeout=C.REQUEST_TIMEOUT)
        resp.raise_for_status()
        if len(resp.content) < 200:
            break
        page = pd.read_csv(io.StringIO(resp.text), sep="\t", dtype={"Genome ID": str})
        pages.append(page)
        fetched += len(page)
        if offset % (C.PAGE_SIZE * 5) == 0:
            log.info("  %s: %d/%d", antibiotic, fetched, expected)
        time.sleep(0.2)

    if not pages:
        return pd.DataFrame()

    # fail loudly rather than quietly analysing a partial table
    if fetched < expected:
        raise RuntimeError(f"{antibiotic}: retrieved {fetched} of {expected} "
                           f"records — refusing to continue on partial data")

    log.info("  %s: %d/%d retrieved, complete", antibiotic, fetched, expected)
    return pd.concat(pages, ignore_index=True)


def clean(raw: pd.DataFrame, antibiotic: str) -> pd.DataFrame:
    """Apply the inclusion criteria, logging what each filter removes (§3.2)."""
    df = raw.rename(columns=RENAME)
    df = df[[c for c in RENAME.values() if c in df.columns]].copy()
    df["genome_id"] = df["genome_id"].astype(str).str.strip().str.strip('"')

    steps = [("fetched", len(df))]

    df = df[df["genome_id"].str.match(r"^\d+\.\d+$", na=False)]
    steps.append(("valid genome_id", len(df)))

    # lab-measured only — exclude BV-BRC's own computational predictions
    df = df[df["evidence"].astype(str).str.strip().str.lower() == C.REQUIRED_EVIDENCE]
    steps.append(("lab-measured", len(df)))

    df = df[df["phenotype"].isin(C.VALID_PHENOTYPES)]
    steps.append(("R/S only", len(df)))

    df = df.drop_duplicates(subset=["genome_id", "antibiotic"])
    steps.append(("deduplicated", len(df)))

    df["label"] = df["phenotype"].map(C.LABEL_MAP)
    df["antibiotic"] = antibiotic

    log.info("  %s: %s", antibiotic,
             " -> ".join(f"{name} {n}" for name, n in steps))
    return df


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                        datefmt="%H:%M:%S")
    session = requests.Session()

    frames = []
    for antibiotic in C.ANTIBIOTICS:
        log.info("querying %s", antibiotic)
        raw = fetch_antibiotic(session, antibiotic)
        if raw.empty:
            log.warning("  %s: no records returned", antibiotic)
            continue
        frames.append(clean(raw, antibiotic))

    if not frames:
        log.error("no phenotype data retrieved — check network access to BV-BRC")
        return 1

    long = pd.concat(frames, ignore_index=True)
    long.to_csv(C.PHENOTYPES / "phenotypes_long.csv", index=False)

    genome_ids = sorted(long["genome_id"].unique())
    (C.PROCESSED / "genome_list.txt").write_text("\n".join(genome_ids))

    # ── cohort summary ────────────────────────────────────────────────────────
    summary = (long.groupby("antibiotic")
                   .agg(n_genomes=("genome_id", "nunique"),
                        n_resistant=("label", "sum"))
                   .assign(n_susceptible=lambda d: d.n_genomes - d.n_resistant,
                           pct_resistant=lambda d: (d.n_resistant / d.n_genomes * 100).round(1))
                   .reset_index()
                   .sort_values("n_genomes", ascending=False))
    summary.to_csv(C.TABLES / "cohort_summary.csv", index=False)

    print("\n" + "=" * 62)
    print("COHORT SUMMARY (lab-confirmed phenotypes only)")
    print("=" * 62)
    print(summary.to_string(index=False))
    print(f"\nunique genomes to download : {len(genome_ids):,}")
    print(f"estimated download         : ~{len(genome_ids) * 5.07 / 1000:.1f} GB "
          f"uncompressed, ~{len(genome_ids) * 5.07 / 4000:.1f} GB gzipped")
    print(f"\nwrote {C.PHENOTYPES / 'phenotypes_long.csv'}")
    print(f"wrote {C.PROCESSED / 'genome_list.txt'}")
    print("\nnext: python src/s02_fetch_genomes.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
