"""
s11_external_validation.py — external validation on NCBI Pathogen Detection (proposal §2.1).

The models deployed in stage 10 were trained on BV-BRC. This stage tests them,
unchanged, on E. coli genomes from a different source: NCBI Pathogen Detection,
whose isolates carry submitter-reported ciprofloxacin susceptibility results.

The external cohort is shipped as data/external/ncbi_cohort_ciprofloxacin.csv.
It was built from the NCBI metadata release PDG000000004.6345 by keeping isolates
with a Resistant or Susceptible ciprofloxacin result and an assembly, then
removing every isolate that also appears in the BV-BRC laboratory cohort
(matched on BioSample, SRA run, assembly accession or strain name). Result:
7,554 genomes, 1,204 resistant. Rebuild it with --rebuild-cohort (downloads the
~550 MB NCBI metadata file and queries BV-BRC).

For each genome the assembly is streamed from the NCBI Datasets API, checked,
turned into the same features used in training (QRDR residues and hexamer
frequencies), and discarded — nothing is written to disk except the features.
The same size rule as stage 03 applies (4.4–6.0 Mb).

The QRDR model follows the application's decision rule: if gyrA 83 or parC 80
cannot be read, no prediction is made and the genome is reported as
"cannot determine". Metrics are computed on the remaining genomes, and the
k-mer model is scored on exactly the same genomes so the comparison is fair.

Resumable: progress is saved every 200 genomes; re-running continues.

Run from src/:
    python s11_external_validation.py            # ~30-45 min, needs internet

Output:
    data/external/ncbi_features.csv          QRDR residues and status per genome
    data/external/ncbi_kmer_freq.npy         hexamer frequencies (rows = ncbi_kmer_ids.npy)
    results/tables/external_validation.csv   AUROC (95% CI), sensitivity, specificity
    results/tables/external_predictions.csv  per-genome predictions
    results/figures/07_external_validation.png
"""

from __future__ import annotations

import argparse
import io
import logging
import re
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from sklearn.metrics import roc_auc_score, roc_curve

import config as C
from s03_build_kmers import count_kmers
from s04_build_mechanism_features import QRDR_FEATURES, QRDR_SITES, call_qrdr_site

log = logging.getLogger("s11")

EXTERNAL = C.DATA / "external"
COHORT = EXTERNAL / "ncbi_cohort_ciprofloxacin.csv"
FEATURES = EXTERNAL / "ncbi_features.csv"
KMER_FREQ = EXTERNAL / "ncbi_kmer_freq.npy"
KMER_IDS = EXTERNAL / "ncbi_kmer_ids.npy"

DATASETS_API = "https://api.ncbi.nlm.nih.gov/datasets/v2/genome/accession"
NCBI_METADATA = ("https://ftp.ncbi.nlm.nih.gov/pathogen/Results/Escherichia_coli_Shigella/"
                 "latest_snps/Metadata/")
WORKERS = 3                     # NCBI allows ~5 requests/s without an API key
CHECKPOINT_EVERY = 200
PRIMARY_SITES = ("gyrA_83", "parC_80")
N_BOOTSTRAP = 1000


# ── 1. optional: rebuild the external cohort ──────────────────────────────────
def rebuild_cohort() -> None:
    """Rebuild the deduplicated NCBI cohort from scratch (~550 MB download)."""
    listing = requests.get(NCBI_METADATA, timeout=120).text
    name = re.search(r'href="(PDG\d+\.\d+\.metadata\.tsv)"', listing).group(1)
    log.info("downloading NCBI metadata %s (~550 MB)", name)
    meta = pd.read_csv(NCBI_METADATA + name, sep="\t", dtype=str, low_memory=False,
                       usecols=["asm_acc", "biosample_acc", "Run", "strain",
                                "scientific_name", "geo_loc_name", "collection_date",
                                "asm_stats_length_bp", "AST_phenotypes"])
    meta["phenotype"] = meta["AST_phenotypes"].str.extract(r"ciprofloxacin=([A-Z]+)")[0]
    meta = meta[meta["phenotype"].isin(["R", "S"])
                & meta["scientific_name"].str.contains("Escherichia coli", na=False)
                & meta["asm_acc"].notna() & (meta["asm_acc"] != "NULL")]

    log.info("querying BV-BRC identifiers for deduplication")
    session = requests.Session()
    ids = set()
    for antibiotic in C.ANTIBIOTICS:
        total = int(session.post(
            f"{C.API}/genome_amr/", data=f"eq(taxon_id,{C.TAXON_ID})&eq(antibiotic,{antibiotic})&limit(1)",
            headers={**C.RQL_HEADERS, "Accept": "application/json"},
            timeout=C.REQUEST_TIMEOUT).headers["content-range"].rsplit("/", 1)[1])
        for offset in range(0, total, 25_000):
            query = (f"eq(taxon_id,{C.TAXON_ID})&eq(antibiotic,{antibiotic})"
                     f"&select(genome_id,evidence)&limit(25000,{offset})")
            page = pd.read_csv(io.StringIO(session.post(
                f"{C.API}/genome_amr/", data=query, headers=C.RQL_HEADERS,
                timeout=300).text), sep="\t", dtype=str)
            lab = page[page["evidence"].str.strip().str.lower() == C.REQUIRED_EVIDENCE]
            ids |= set(lab["genome_id"].str.strip().str.strip('"'))
    ids = sorted(ids)
    frames = []
    for start in range(0, len(ids), 200):
        query = (f"in(genome_id,({','.join(ids[start:start + 200])}))"
                 f"&select(genome_id,assembly_accession,biosample_accession,"
                 f"sra_accession,strain)&limit(400)")
        for attempt in range(5):
            resp = session.post(f"{C.API}/genome/", data=query, headers=C.RQL_HEADERS,
                                timeout=C.REQUEST_TIMEOUT)
            if resp.ok:
                break
            time.sleep(5 * (attempt + 1))
        resp.raise_for_status()
        frames.append(pd.read_csv(io.StringIO(resp.text), sep="\t", dtype=str))
    bv = pd.concat(frames)

    def tokens(column) -> set:
        out = set()
        for value in bv[column].dropna():
            for t in re.split(r"[,;\s]+", value):
                if t:
                    out.add(t.split(".")[0] if t.startswith(("GCA_", "GCF_")) else t)
        return out

    biosamples, runs, assemblies = (tokens("biosample_accession"),
                                    tokens("sra_accession"), tokens("assembly_accession"))
    strains = {s.strip() for s in bv["strain"].dropna() if len(s.strip()) >= 4}

    def overlaps(row) -> bool:
        row_runs = [r for r in re.split(r"[,;\s]+", str(row["Run"])) if r]
        return (row["biosample_acc"] in biosamples
                or any(r in runs for r in row_runs)
                or str(row["asm_acc"]).split(".")[0] in assemblies
                or str(row["strain"]).strip() in strains)

    new = meta[~meta.apply(overlaps, axis=1)]
    cohort = pd.DataFrame({
        "asm_acc": new["asm_acc"], "biosample_acc": new["biosample_acc"],
        "label": (new["phenotype"] == "R").astype(int), "phenotype": new["phenotype"],
        "country": new["geo_loc_name"].str.split(":").str[0].str.strip(),
        "collection_date": new["collection_date"],
        "reported_length_bp": new["asm_stats_length_bp"],
    }).drop_duplicates("asm_acc").sort_values("asm_acc")
    cohort.to_csv(COHORT, index=False)
    log.info("external cohort: %d genomes (%d resistant), %d removed as overlapping",
             len(cohort), int(cohort["label"].sum()), len(meta) - len(new))


# ── 2. download and featurise ─────────────────────────────────────────────────
def fetch_sequence(session: requests.Session, accession: str) -> str:
    """Download one assembly from NCBI Datasets and return its sequence."""
    for attempt in range(6):
        try:
            resp = session.get(f"{DATASETS_API}/{accession}/download",
                               params={"include_annotation_type": "GENOME_FASTA"},
                               timeout=180)
            if resp.status_code == 429 or resp.status_code >= 500:
                time.sleep(2 * (attempt + 1))
                continue
            resp.raise_for_status()
            archive = zipfile.ZipFile(io.BytesIO(resp.content))
            fasta = next(n for n in archive.namelist() if n.endswith(".fna"))
            text = archive.read(fasta).decode("ascii", errors="ignore")
            return "".join(line.strip().upper() for line in text.splitlines()
                           if not line.startswith(">"))
        except (requests.RequestException, zipfile.BadZipFile, StopIteration):
            time.sleep(2 * (attempt + 1))
    raise RuntimeError("download failed after retries")


def featurise(session: requests.Session, row: dict) -> tuple[dict, np.ndarray | None]:
    record = {"asm_acc": row["asm_acc"]}
    try:
        sequence = fetch_sequence(session, row["asm_acc"])
    except RuntimeError as exc:
        return {**record, "status": "download_failed", "note": str(exc)}, None

    record["length_bp"] = len(sequence)
    if not 4_400_000 <= len(sequence) <= 6_000_000:
        return {**record, "status": "qc_size"}, None

    for site, (flank, side, _, _) in QRDR_SITES.items():
        record[site] = call_qrdr_site(sequence, flank, side)
    counts = count_kmers(sequence)
    total = counts.sum()
    if total == 0:
        return {**record, "status": "no_kmers"}, None
    return {**record, "status": "ok"}, (counts / total).astype(np.float32)


def save_checkpoint(rows: list[dict], kmer_ids: list[str], kmers: list[np.ndarray]) -> None:
    pd.DataFrame(rows).to_csv(FEATURES, index=False)
    if kmers:
        np.save(KMER_FREQ, np.vstack(kmers))
        np.save(KMER_IDS, np.array(kmer_ids))


def build_features(cohort: pd.DataFrame) -> None:
    rows, kmer_ids, kmers = [], [], []
    if FEATURES.exists():
        rows = pd.read_csv(FEATURES, dtype={"asm_acc": str}).to_dict("records")
        if KMER_FREQ.exists() and KMER_IDS.exists():
            kmer_ids = [str(x) for x in np.load(KMER_IDS, allow_pickle=True)]
            kmers = list(np.load(KMER_FREQ))
        ok_with_kmers = set(kmer_ids)
        # keep only consistent rows: "ok" rows must have their k-mer vector saved
        rows = [r for r in rows if r["status"] != "ok" or r["asm_acc"] in ok_with_kmers]
        log.info("resuming: %d genomes already processed", len(rows))

    done = {r["asm_acc"] for r in rows}
    todo = cohort[~cohort["asm_acc"].isin(done)].to_dict("records")
    log.info("%d genomes to download and featurise", len(todo))

    session = requests.Session()
    started, completed = time.time(), 0
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = [pool.submit(featurise, session, row) for row in todo]
        for future in as_completed(futures):
            record, vector = future.result()
            rows.append(record)
            if vector is not None:
                kmer_ids.append(record["asm_acc"])
                kmers.append(vector)
            completed += 1
            if completed % CHECKPOINT_EVERY == 0:
                save_checkpoint(rows, kmer_ids, kmers)
                rate = completed / (time.time() - started)
                log.info("%d/%d  (%.1f genomes/s, ~%.0f min left)", completed, len(todo),
                         rate, (len(todo) - completed) / rate / 60)
    save_checkpoint(rows, kmer_ids, kmers)


# ── 3. evaluate the deployed models ───────────────────────────────────────────
def qrdr_vector(row: pd.Series) -> np.ndarray:
    vector = np.zeros(len(QRDR_FEATURES), dtype=np.float32)
    for col, (site, (_, _, wt, resistant)) in enumerate(QRDR_SITES.items()):
        residue = row.get(site)
        if residue in resistant:
            vector[col * 2] = 1
        elif residue == wt:
            vector[col * 2 + 1] = 1
    return vector


def site_read(row: pd.Series, site: str) -> bool:
    _, _, wt, resistant = QRDR_SITES[site]
    residue = row.get(site)
    return isinstance(residue, str) and (residue == wt or residue in resistant)


def metrics(y: np.ndarray, proba: np.ndarray, rng: np.random.Generator) -> dict:
    pred = proba >= 0.5
    boot = []
    for _ in range(N_BOOTSTRAP):
        idx = rng.integers(0, len(y), len(y))
        if 0 < y[idx].sum() < len(idx):
            boot.append(roc_auc_score(y[idx], proba[idx]))
    return {"n": int(len(y)), "n_resistant": int(y.sum()),
            "auroc": roc_auc_score(y, proba),
            "auroc_ci_low": float(np.percentile(boot, 2.5)),
            "auroc_ci_high": float(np.percentile(boot, 97.5)),
            "sensitivity": float(pred[y == 1].mean()),
            "specificity": float((~pred[y == 0]).mean())}


def evaluate(cohort: pd.DataFrame) -> pd.DataFrame:
    features = pd.read_csv(FEATURES, dtype={"asm_acc": str})
    data = cohort.merge(features, on="asm_acc", how="inner")
    status = data["status"].value_counts()
    ok = data[data["status"] == "ok"].copy()

    ok["qrdr_determinable"] = ok.apply(
        lambda r: all(site_read(r, s) for s in PRIMARY_SITES), axis=1)

    qrdr_model = joblib.load(C.MODELS / f"qrdr_{C.PRIMARY_ANTIBIOTIC}.joblib")
    kmer_model = joblib.load(C.MODELS / f"kmer_{C.PRIMARY_ANTIBIOTIC}.joblib")

    X_qrdr = np.vstack([qrdr_vector(r) for _, r in ok.iterrows()])
    ok["p_qrdr"] = qrdr_model.predict_proba(X_qrdr)[:, 1]

    kmer_ids = [str(x) for x in np.load(KMER_IDS, allow_pickle=True)]
    kmer_rows = {a: i for i, a in enumerate(kmer_ids)}
    X_kmer = np.load(KMER_FREQ)[[kmer_rows[a] for a in ok["asm_acc"]]]
    ok["p_kmer"] = kmer_model.predict_proba(X_kmer)[:, 1]

    ok.drop(columns=[c for c in ["note"] if c in ok], errors="ignore").to_csv(
        C.TABLES / "external_predictions.csv", index=False)

    rng = np.random.default_rng(C.SEED)
    scored = ok[ok["qrdr_determinable"]]
    y = scored["label"].to_numpy()
    rows = [
        {"model": "QRDR + XGBoost", "genomes": "determinable",
         **metrics(y, scored["p_qrdr"].to_numpy(), rng)},
        {"model": "k-mer + LogisticRegression", "genomes": "determinable",
         **metrics(y, scored["p_kmer"].to_numpy(), rng)},
        {"model": "k-mer + LogisticRegression", "genomes": "all passing QC",
         **metrics(ok["label"].to_numpy(), ok["p_kmer"].to_numpy(), rng)},
    ]
    table = pd.DataFrame(rows)
    table.round(4).to_csv(C.TABLES / "external_validation.csv", index=False)

    # ── figure (Albanian labels) ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 5.5))
    for column, label, colour in [("p_qrdr", "QRDR", "#4C72B0"),
                                  ("p_kmer", "K-mer", "#C44E52")]:
        fpr, tpr, _ = roc_curve(y, scored[column])
        auc = roc_auc_score(y, scored[column])
        ax.plot(fpr, tpr, color=colour, linewidth=2, label=f"{label} (AUROC {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k:", linewidth=1, label="Niveli i rastësishëm")
    ax.set_xlabel("1 − specificiteti")
    ax.set_ylabel("Ndjeshmëria")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.01)
    ax.legend(frameon=False, loc="lower right")
    plt.tight_layout()
    plt.savefig(C.FIGURES / "07_external_validation.png", dpi=150)
    plt.close()

    # ── report ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("EXTERNAL VALIDATION — NCBI Pathogen Detection, ciprofloxacin")
    print("=" * 72)
    print(f"external cohort         : {len(cohort):,} genomes "
          f"({int(cohort['label'].sum()):,} resistant)")
    print("processing status       : " + ", ".join(f"{k} {v}" for k, v in status.items()))
    print(f"passed QC               : {len(ok):,}")
    print(f"QRDR cannot determine   : {int((~ok['qrdr_determinable']).sum()):,} "
          f"({(~ok['qrdr_determinable']).mean():.1%})")
    print(f"scored (determinable)   : {len(scored):,} "
          f"({int(scored['label'].sum()):,} resistant)\n")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    card_path = C.MODELS / "model_card.json"
    if card_path.exists():
        import json
        card = json.loads(card_path.read_text()).get("models", {})
        q = card.get(f"qrdr_{C.PRIMARY_ANTIBIOTIC}", {})
        k = card.get(f"kmer_{C.PRIMARY_ANTIBIOTIC}", {})
        print("\nfor comparison, cross-validated on BV-BRC (MLST-grouped):")
        print(f"  QRDR  AUROC {q.get('cv_auroc_mlst_grouped')}  "
              f"sens {q.get('sensitivity_mlst_grouped')}  spec {q.get('specificity_mlst_grouped')}")
        print(f"  k-mer AUROC {k.get('cv_auroc_mlst_grouped')}  "
              f"sens {k.get('sensitivity_mlst_grouped')}  spec {k.get('specificity_mlst_grouped')}")

    print(f"\nwrote {C.TABLES / 'external_validation.csv'}")
    print(f"wrote {C.TABLES / 'external_predictions.csv'}")
    print(f"wrote {C.FIGURES / '07_external_validation.png'}")
    return table


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild-cohort", action="store_true",
                    help="rebuild the NCBI cohort from scratch (~550 MB download)")
    ap.add_argument("--limit", type=int, default=None,
                    help="process only the first N genomes (for a quick trial)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                        datefmt="%H:%M:%S")
    EXTERNAL.mkdir(parents=True, exist_ok=True)

    if args.rebuild_cohort or not COHORT.exists():
        rebuild_cohort()
    cohort = pd.read_csv(COHORT, dtype={"asm_acc": str, "biosample_acc": str})
    if args.limit:
        cohort = cohort.sample(n=min(args.limit, len(cohort)), random_state=C.SEED)
        log.info("trial run on %d genomes", len(cohort))

    build_features(cohort)
    evaluate(cohort)
    return 0


if __name__ == "__main__":
    sys.exit(main())
