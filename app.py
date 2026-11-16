"""
app.py, From DNA to Diagnosis

Interactive application for the dissertation. Three views:

  Predict   select or upload a genome, read the QRDR codons directly from the
            assembly, predict resistance, and contrast the mechanism model with
            the hexamer-composition model on the same genome
  Results   the figures and tables from stages 05 to 09
  Method    pipeline, deviations from the approved proposal, limitations

The Predict view leads with the sequence rather than with a probability. A
single codon substitution is what flips the phenotype, so showing the flanking
context with the deciding codon marked is both the most characteristic view of
this subject and the evidence behind the call.

STYLING NOTE. All custom styling here is inline `style="..."` on the elements
themselves, and the palette lives in .streamlit/config.toml. Streamlit
sanitises HTML with DOMPurify, which removes <style> and <link> elements but
retains their text content, so an injected stylesheet does not apply, it
prints its own CSS onto the page as visible text. Inline style attributes
survive sanitisation; theme colours set in config.toml cannot be stripped at
all. Do not reintroduce a <style> block here.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))

import config as C                                          # noqa: E402
from s03_build_kmers import count_kmers                     # noqa: E402
from s04_build_mechanism_features import (                  # noqa: E402
    CODON_TABLE, QRDR_FEATURES, QRDR_SITES, _revcomp, call_qrdr_site)

st.set_page_config(page_title="From DNA to Diagnosis",
                   page_icon="🧬", layout="wide")

# ── palette ───────────────────────────────────────────────────────────────────
# Navy, white and shades of blue for the interface. The verdict alone uses
# colour as a signal: red for resistant, green for susceptible, grey-blue when
# no call can be made.
INK = "#0B2545"
INK_SOFT = "#4A6076"
PRIMARY = "#13315C"
SURFACE = "#F3F7FB"
HAIRLINE = "#D3E0EA"
RESISTANT = "#B23A3A"
SUSCEPT = "#2E7D4F"
UNKNOWN = "#8DA9C4"
MONO = "ui-monospace, 'SF Mono', Menlo, Consolas, monospace"

# The two positions that decide the ciprofloxacin call. If either cannot be
# read, the app reports "cannot determine" rather than a prediction.
PRIMARY_SITES = ("gyrA_83", "parC_80")


def section(title: str) -> None:
    st.html(
        f'<div style="font-size:1.05rem;font-weight:600;color:{INK};'
        f'padding-bottom:.4rem;border-bottom:1px solid {HAIRLINE};'
        f'margin:1.8rem 0 .9rem 0">{title}</div>')


def muted(text: str) -> None:
    st.html(f'<p style="color:{INK_SOFT};font-size:.87rem;max-width:74ch;'
            f'line-height:1.55">{text}</p>')


# ── loading ───────────────────────────────────────────────────────────────────
@st.cache_resource
def load_models() -> tuple[dict, dict]:
    models = {p.stem: joblib.load(p) for p in C.MODELS.glob("*.joblib")}
    card_path = C.MODELS / "model_card.json"
    return models, (json.loads(card_path.read_text()) if card_path.exists() else {})


@st.cache_data
def load_cohort() -> pd.DataFrame | None:
    path = C.PROCESSED / f"folds_{C.PRIMARY_ANTIBIOTIC}.csv"
    if not path.exists():
        return None
    folds = pd.read_csv(path, dtype={"genome_id": str})
    residues = C.PROCESSED / "qrdr_residues.csv"
    if residues.exists():
        folds = folds.merge(pd.read_csv(residues, dtype={"genome_id": str}),
                            on="genome_id", how="left")
    return folds


def read_fasta(raw: bytes, name: str) -> str:
    if name.endswith(".gz"):
        raw = gzip.decompress(raw)
    text = raw.decode("utf-8", errors="ignore")
    return "".join(line.strip().upper()
                   for line in text.splitlines() if not line.startswith(">"))


def read_codon_context(sequence: str, flank: str, side: str,
                       context: int = 12) -> tuple | None:
    """Return (codon, amino acid, left context, right context) for one site.

    The conserved flank locates the position; the codon is whichever three bases
    sit beside it. Contigs come in arbitrary orientation, so if the flank is
    only present on the reverse strand a window around the hit is
    reverse-complemented and read forward.
    """
    def forward(seq: str):
        idx = seq.find(flank)
        if idx == -1:
            return None
        start = idx + len(flank) if side == "left" else idx - 3
        if start < 0:
            return None
        codon = seq[start:start + 3]
        if len(codon) != 3:
            return None
        return (codon, CODON_TABLE.get(codon),
                seq[max(0, start - context):start],
                seq[start + 3:start + 3 + context])

    hit = forward(sequence)
    if hit:
        return hit
    idx = sequence.find(_revcomp(flank))
    if idx == -1:
        return None
    window = sequence[max(0, idx - 200):idx + len(flank) + 200]
    return forward(_revcomp(window))


def analyse(sequence: str) -> dict:
    """QRDR genotype, per-site sequence context, and hexamer profile."""
    context, features = {}, np.zeros(len(QRDR_FEATURES), dtype=np.float32)
    for col, (site, (flank, side, wt, resistant)) in enumerate(QRDR_SITES.items()):
        residue = call_qrdr_site(sequence, flank, side)
        context[site] = read_codon_context(sequence, flank, side)
        if residue in resistant:
            features[col * 2] = 1
        elif residue == wt:
            features[col * 2 + 1] = 1

    counts = count_kmers(sequence)
    total = counts.sum()
    hexamers = (counts / total if total else counts).astype(np.float32)
    return {"context": context, "qrdr": features, "kmer": hexamers,
            "length": len(sequence)}


# ── texts in both languages ───────────────────────────────────────────────────
TEXT = {
    "en": {
        "title": "From DNA to Diagnosis",
        "subtitle": "Predicting ciprofloxacin resistance in <i>Escherichia coli</i> by reading the "
                    "quinolone resistance-determining regions straight from a genome assembly.",
        "view": "View", "views": ["Predict", "Results", "Method"],
        "genomes": "Genomes", "resistant_share": "Resistant", "models_loaded": "{n} fitted models loaded.",
        "genome": "Genome", "source": ["Cohort genome", "Upload assembly"],
        "uploader": "FASTA (.fna, .fasta, .fa, optionally gzipped)",
        "no_cohort": "No cohort found. Run the pipeline, or upload an assembly.",
        "no_genomes": "No assemblies found under data/genomes.",
        "available": "{n} assemblies available", "lab_phenotype": "Laboratory phenotype",
        "choose": "Choose an assembly to read its resistance loci.",
        "too_short": "This sequence is only {n} bp long. That is a fragment, not a whole genome, "
                     "so the resistance positions cannot be found reliably. Please upload a complete assembly.",
        "no_model": "No fitted model found. Run `python src/s10_build_app.py`, then reload this page.",
        "reading": "Reading resistance loci",
        "resistant": "Resistant", "susceptible": "Susceptible",
        "verdict_r": "Ciprofloxacin resistant", "verdict_s": "Ciprofloxacin susceptible",
        "undetermined": "Cannot determine",
        "undetermined_why": "{sites} could not be read from this assembly. These positions decide the "
                            "prediction, so the app does not make a call. This usually means an "
                            "incomplete assembly or a genome that is not <i>E. coli</i>.",
        "probability": "Predicted probability of resistance {p}.",
        "carries": "{site} carries {aa}", "no_allele": "No known resistance change at the four positions",
        "partial": "Note: {sites} could not be read; the prediction uses the other positions.",
        "lab_is": "The laboratory result for this genome is {r}.",
        "differs": "This differs from the laboratory result. On families held out of training the model "
                   "finds {sens} of resistant and {spec} of susceptible genomes, so some disagreements are "
                   "expected, often for bacteria that are resistant for reasons the model does not read.",
        "matches": "Matches the laboratory result.",
        "meta": "Sequence type {st}, isolated in {country}.",
        "loci": "Resistance loci, read from the assembly",
        "codon": "{gene} codon {pos}", "not_resolved": "not found",
        "not_resolved_note": "The reference sequence next to this position is missing from the assembly, "
                             "so the codon could not be found.",
        "call_r": "linked to resistance", "call_wt": "normal", "call_other": "unknown variant",
        "note_changed": "Normally {wt} at this position; this assembly reads {aa}.",
        "note_wt": "Normal amino acid {wt}, as expected in a susceptible bacterium.",
        "note_untranslated": "The codon could not be translated.",
        "loci_note": "Each strip shows the DNA around one of the four positions, with the codon that "
                     "decides the result highlighted. A short reference sequence finds the position, "
                     "and the three letters next to it are translated.",
        "contrast": "The same genome, two feature sets",
        "mech_model": "Mechanism model", "comp_model": "Composition model",
        "mech_help": "Eight features describing the four positions above.",
        "auroc_note": "AUROC {a} on families held out of training.",
        "contrast_note": "The composition model reads the frequencies of all 4,096 six-letter sequences "
                         "in the genome. It loses accuracy on new families, because it mostly recognises "
                         "the family rather than the cause of resistance. When the two numbers differ "
                         "on one genome, that is the main finding of the thesis in miniature.",
        "report_btn": "Generate report", "report_help": "Download a short, printable explanation of this result.",
        # report
        "r_title": "Resistance report", "r_genome": "Genome", "r_date": "Date",
        "r_result": "Result", "r_what": "What was checked",
        "r_what_text": "The app looked at four places in two genes, gyrA and parC. Small changes at these "
                       "places are the main cause of resistance to ciprofloxacin.",
        "r_pos": "Position", "r_found": "What was found", "r_meaning": "What it means",
        "r_m_r": "Changed. This change is linked to resistance.",
        "r_m_wt": "Normal. No sign of resistance here.",
        "r_m_other": "An unusual variant. Its effect is not known.",
        "r_m_nf": "Not found in this file.",
        "r_means": "What this means",
        "r_means_r": "Ciprofloxacin is unlikely to work against this bacterium. Changes linked to "
                     "resistance were found at the positions listed above.",
        "r_means_s": "No resistance changes were found at the four positions, so ciprofloxacin is likely "
                     "to work. The app does not read every possible cause of resistance, so a laboratory "
                     "test remains the final answer.",
        "r_means_u": "The app could not make a prediction, because an important position could not be "
                     "found. The file may be incomplete, or the bacterium may not be E. coli.",
        "r_prob": "Probability of resistance: {p}",
        "r_lab": "Laboratory result", "r_lab_match": "The prediction matches the laboratory result.",
        "r_lab_diff": "The prediction does not match the laboratory result.",
        "r_model": "About the model",
        "r_model_text": "On bacterial families it had never seen, the model correctly found {sens} of "
                        "resistant and {spec} of susceptible genomes (AUROC {a}).",
        "r_note": "This report comes from a research tool built for a bachelor thesis. It is not a medical test.",
        "uploaded_file": "uploaded file",
        # results
        "results_intro": "These are the main results of the thesis, in the order they build on each other. "
                         "AUROC measures how well a model separates resistant from susceptible genomes: "
                         "1.0 is perfect and 0.5 is guessing.",
        "no_figures": "No figures yet. Run `python src/s09_make_figures.py`.",
        "tables": "Tables",
        "table_names": ["Main results, lineage-grouped", "All validation designs",
                        "QRDR features on the other drugs", "Number of genomes per antibiotic"],
    },
    "sq": {
        "title": "Nga ADN-ja te diagnoza",
        "subtitle": "Parashikimi i rezistencës ndaj ciprofloksacinës te <i>Escherichia coli</i>, duke lexuar "
                    "rajonet përcaktuese të rezistencës ndaj kinoloneve drejtpërdrejt nga sekuenca e gjenomit.",
        "view": "Pamja", "views": ["Parashikimi", "Rezultatet", "Metoda"],
        "genomes": "Gjenome", "resistant_share": "Rezistente", "models_loaded": "{n} modele të ngarkuara.",
        "genome": "Gjenomi", "source": ["Gjenom nga kohorta", "Ngarko skedar"],
        "uploader": "FASTA (.fna, .fasta, .fa, mund të jetë .gz)",
        "no_cohort": "Kohorta nuk u gjet. Ekzekutoni rrjedhën ose ngarkoni një skedar.",
        "no_genomes": "Nuk u gjet asnjë gjenom në data/genomes.",
        "available": "{n} gjenome në dispozicion", "lab_phenotype": "Rezultati laboratorik",
        "choose": "Zgjidhni një gjenom për të lexuar pozicionet e rezistencës.",
        "too_short": "Kjo sekuencë ka vetëm {n} baza. Është një fragment dhe jo një gjenom i plotë, prandaj "
                     "pozicionet e rezistencës nuk mund të gjenden me siguri. Ngarkoni një gjenom të plotë.",
        "no_model": "Modeli nuk u gjet. Ekzekutoni `python src/s10_build_app.py` dhe rifreskoni faqen.",
        "reading": "Duke lexuar pozicionet e rezistencës",
        "resistant": "Rezistent", "susceptible": "I ndjeshëm",
        "verdict_r": "Rezistent ndaj ciprofloksacinës", "verdict_s": "I ndjeshëm ndaj ciprofloksacinës",
        "undetermined": "Nuk mund të përcaktohet",
        "undetermined_why": "{sites} nuk u lexua nga ky gjenom. Këto pozicione e vendosin parashikimin, "
                            "prandaj aplikacioni nuk jep përgjigje. Zakonisht kjo do të thotë që skedari "
                            "është i paplotë ose që gjenomi nuk është <i>E. coli</i>.",
        "probability": "Probabiliteti i parashikuar i rezistencës {p}.",
        "carries": "{site} ka {aa}", "no_allele": "Asnjë ndryshim i njohur rezistence në katër pozicionet",
        "partial": "Shënim: {sites} nuk u lexua; parashikimi përdor pozicionet e tjera.",
        "lab_is": "Rezultati laboratorik për këtë gjenom është {r}.",
        "differs": "Kjo nuk përputhet me rezultatin laboratorik. Te familjet që nuk ishin në trajnim, modeli "
                   "gjen {sens} të gjenomeve rezistente dhe {spec} të atyre të ndjeshme, prandaj disa "
                   "mospërputhje janë të pritshme, shpesh te bakteret që janë rezistente për arsye që modeli "
                   "nuk i lexon.",
        "matches": "Përputhet me rezultatin laboratorik.",
        "meta": "Tipi i sekuencës {st}, marrë në {country}.",
        "loci": "Pozicionet e rezistencës, të lexuara nga gjenomi",
        "codon": "Kodoni {pos} i {gene}", "not_resolved": "nuk u gjet",
        "not_resolved_note": "Sekuenca referuese pranë këtij pozicioni mungon në gjenom, prandaj kodoni "
                             "nuk u gjet.",
        "call_r": "i lidhur me rezistencën", "call_wt": "i zakonshëm", "call_other": "variant i panjohur",
        "note_changed": "Zakonisht {wt} në këtë pozicion; ky gjenom ka {aa}.",
        "note_wt": "Aminoacidi i zakonshëm {wt}, siç pritet te një bakter i ndjeshëm.",
        "note_untranslated": "Kodoni nuk mund të përkthehej.",
        "loci_note": "Secili rresht tregon ADN-në përreth njërit prej katër pozicioneve, me kodonin që e "
                     "vendos rezultatin të theksuar. Një sekuencë e shkurtër referuese e gjen pozicionin, "
                     "dhe tri shkronjat pranë saj përkthehen.",
        "contrast": "I njëjti gjenom, dy bashkësi veçorish",
        "mech_model": "Modeli i mekanizmit", "comp_model": "Modeli i përbërjes",
        "mech_help": "Tetë veçori që përshkruajnë katër pozicionet më sipër.",
        "auroc_note": "AUROC {a} te familjet që nuk ishin në trajnim.",
        "contrast_note": "Modeli i përbërjes lexon frekuencat e të 4,096 sekuencave me gjashtë shkronja në "
                         "gjenom. Ai e humb saktësinë te familjet e reja, sepse kryesisht njeh familjen dhe "
                         "jo shkakun e rezistencës. Kur dy numrat ndryshojnë për një gjenom, kjo është "
                         "gjetja kryesore e punimit në miniaturë.",
        "report_btn": "Gjenero raportin", "report_help": "Shkarkoni një shpjegim të shkurtër dhe të printueshëm të këtij rezultati.",
        "r_title": "Raport i rezistencës", "r_genome": "Gjenomi", "r_date": "Data",
        "r_result": "Rezultati", "r_what": "Çfarë u kontrollua",
        "r_what_text": "Aplikacioni shqyrtoi katër vende në dy gjene, gyrA dhe parC. Ndryshimet e vogla në "
                       "këto vende janë shkaku kryesor i rezistencës ndaj ciprofloksacinës.",
        "r_pos": "Pozicioni", "r_found": "Çfarë u gjet", "r_meaning": "Çfarë do të thotë",
        "r_m_r": "I ndryshuar. Ky ndryshim lidhet me rezistencën.",
        "r_m_wt": "I zakonshëm. Asnjë shenjë rezistence këtu.",
        "r_m_other": "Një variant i pazakontë. Efekti i tij nuk dihet.",
        "r_m_nf": "Nuk u gjet në këtë skedar.",
        "r_means": "Çfarë do të thotë",
        "r_means_r": "Ka shumë gjasa që ciprofloksacina të mos veprojë kundër këtij bakteri. Në pozicionet "
                     "e mësipërme u gjetën ndryshime të lidhura me rezistencën.",
        "r_means_s": "Në katër pozicionet nuk u gjet asnjë ndryshim rezistence, prandaj ka gjasa që "
                     "ciprofloksacina të veprojë. Aplikacioni nuk i lexon të gjitha shkaqet e mundshme të "
                     "rezistencës, prandaj përgjigjja përfundimtare mbetet testi laboratorik.",
        "r_means_u": "Aplikacioni nuk mund të jepte parashikim, sepse një pozicion i rëndësishëm nuk u gjet. "
                     "Skedari mund të jetë i paplotë, ose bakteri mund të mos jetë E. coli.",
        "r_prob": "Probabiliteti i rezistencës: {p}",
        "r_lab": "Rezultati laboratorik", "r_lab_match": "Parashikimi përputhet me rezultatin laboratorik.",
        "r_lab_diff": "Parashikimi nuk përputhet me rezultatin laboratorik.",
        "r_model": "Për modelin",
        "r_model_text": "Te familjet bakteriale që nuk i kishte parë kurrë, modeli gjeti saktë {sens} të "
                        "gjenomeve rezistente dhe {spec} të atyre të ndjeshme (AUROC {a}).",
        "r_note": "Ky raport vjen nga një mjet kërkimor i ndërtuar për një punim bachelor. Nuk është test mjekësor.",
        "uploaded_file": "skedar i ngarkuar",
        "results_intro": "Këto janë rezultatet kryesore të punimit, sipas radhës si ndërtohen mbi njëri-tjetrin. "
                         "AUROC mat sa mirë i dallon modeli gjenomet rezistente nga të ndjeshmet: 1.0 është e "
                         "përsosur dhe 0.5 është hamendësim.",
        "no_figures": "Ende nuk ka figura. Ekzekutoni `python src/s09_make_figures.py`.",
        "tables": "Tabelat",
        "table_names": ["Rezultatet kryesore, grupim sipas linjës", "Të gjitha skemat e vlerësimit",
                        "Veçoritë QRDR te antibiotikët e tjerë", "Numri i gjenomeve për antibiotik"],
    },
}

COUNTRY_SQ = {"United States": "SHBA", "United Kingdom": "Mbretërinë e Bashkuar", "Norway": "Norvegji",
              "Thailand": "Tajlandë", "Portugal": "Portugali", "Pakistan": "Pakistan", "India": "Indi",
              "Canada": "Kanada", "Germany": "Gjermani", "Netherlands": "Holandë", "Mexico": "Meksikë"}


def lang() -> str:
    return st.session_state.get("lang", "sq")


def t(key: str, **kw):
    value = TEXT[lang()][key]
    return value.format(**kw) if kw else value


def site_name(site: str) -> str:
    gene, pos = site.split("_")
    return f"{gene} {pos}"


# ── report ────────────────────────────────────────────────────────────────────
def build_report(genome: str, verdict: str, probability: float, result: dict,
                 truth: int | None, metrics: dict) -> bytes | None:
    """A short PDF report in plain language. Returns None if reportlab is missing."""
    try:
        from io import BytesIO
        from datetime import date
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    except ImportError:
        return None

    colour = colors.HexColor({"r": RESISTANT, "s": SUSCEPT, "u": UNKNOWN}[verdict])
    tint = colors.HexColor({"r": "#FBEEEE", "s": "#EDF6F0", "u": SURFACE}[verdict])
    ink, soft, line = colors.HexColor(INK), colors.HexColor(INK_SOFT), colors.HexColor(HAIRLINE)
    body = ParagraphStyle("body", fontName="Times-Roman", fontSize=11, leading=15, textColor=ink)
    small = ParagraphStyle("small", parent=body, fontSize=9.5, leading=13, textColor=soft)
    h1 = ParagraphStyle("h1", parent=body, fontName="Times-Bold", fontSize=18, leading=22,
                        spaceAfter=4)
    h2 = ParagraphStyle("h2", parent=body, fontName="Times-Bold", fontSize=12.5, leading=16,
                        spaceBefore=12, spaceAfter=4)
    cell = ParagraphStyle("cell", parent=body, fontSize=10, leading=13)
    head = ParagraphStyle("head", parent=cell, textColor=soft)

    word = {"r": t("verdict_r"), "s": t("verdict_s"), "u": t("undetermined")}[verdict]
    story = [Paragraph(t("r_title"), h1),
             Paragraph(f"{t('r_genome')}: {genome} &nbsp;&nbsp;|&nbsp;&nbsp; "
                       f"{t('r_date')}: {date.today().strftime('%d.%m.%Y')}", small),
             Paragraph(t("r_result"), h2)]
    verdict_style = ParagraphStyle("verdict", parent=body, fontName="Times-Bold", fontSize=14,
                                   leading=18, textColor=colour)
    box = Table([[Paragraph(word, verdict_style)]], colWidths=[16 * cm])
    box.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), tint),
                             ("LINEBEFORE", (0, 0), (0, -1), 4, colour),
                             ("LEFTPADDING", (0, 0), (-1, -1), 10),
                             ("TOPPADDING", (0, 0), (-1, -1), 8),
                             ("BOTTOMPADDING", (0, 0), (-1, -1), 8)]))
    story += [box, Spacer(1, 6)]
    if verdict != "u":
        story.append(Paragraph(t("r_prob", p=f"{probability:.1%}"), body))

    story += [Paragraph(t("r_what"), h2), Paragraph(t("r_what_text"), body), Spacer(1, 6)]
    rows = [[Paragraph(t("r_pos"), head), Paragraph(t("r_found"), head),
             Paragraph(t("r_meaning"), head)]]
    for site, (_, _, wt, resistant) in QRDR_SITES.items():
        hit = result["context"][site]
        if hit is None:
            found, meaning = "-", t("r_m_nf")
        else:
            codon, residue, _, _ = hit
            found = f"{codon} ({residue or '?'})"
            meaning = (t("r_m_r") if residue in resistant else t("r_m_wt") if residue == wt
                       else t("r_m_other"))
        rows.append([Paragraph(site_name(site), cell), Paragraph(found, cell),
                     Paragraph(meaning, cell)])
    table = Table(rows, colWidths=[3 * cm, 3.2 * cm, 9.8 * cm])
    table.setStyle(TableStyle([("LINEBELOW", (0, 0), (-1, -1), 0.5, line),
                               ("VALIGN", (0, 0), (-1, -1), "TOP"),
                               ("TOPPADDING", (0, 0), (-1, -1), 5),
                               ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
    story.append(table)

    means = {"r": t("r_means_r"), "s": t("r_means_s"), "u": t("r_means_u")}[verdict]
    story += [Paragraph(t("r_means"), h2), Paragraph(means, body)]
    if truth is not None:
        lab_word = (t("resistant") if truth else t("susceptible")).lower()
        text = t("lab_is", r=lab_word)
        if verdict != "u":
            text += " " + (t("r_lab_match") if (verdict == "r") == bool(truth) else t("r_lab_diff"))
        story += [Paragraph(t("r_lab"), h2), Paragraph(text, body)]
    if metrics.get("sensitivity_mlst_grouped") is not None:
        story += [Paragraph(t("r_model"), h2), Paragraph(t(
            "r_model_text", sens=f"{metrics['sensitivity_mlst_grouped']:.1%}",
            spec=f"{metrics['specificity_mlst_grouped']:.1%}",
            a=f"{metrics.get('cv_auroc_mlst_grouped', 0):.3f}"), body)]
    story += [Spacer(1, 18), Paragraph(t("r_note"), small)]

    buffer = BytesIO()
    SimpleDocTemplate(buffer, pagesize=A4, leftMargin=2.5 * cm, rightMargin=2.5 * cm,
                      topMargin=2.2 * cm, bottomMargin=2.2 * cm,
                      title=t("r_title"), author="From DNA to Diagnosis").build(story)
    return buffer.getvalue()


# ── prediction view ───────────────────────────────────────────────────────────
def view_predict(models: dict, card: dict, cohort: pd.DataFrame | None) -> None:
    section(t("genome"))
    # The labels change with the language, which makes Streamlit treat the radio
    # as a new widget. The chosen position is kept in session state instead, so
    # switching language does not change the user's choice.
    labels = t("source")
    picked = st.radio("source", labels, horizontal=True, label_visibility="collapsed",
                      index=st.session_state.get("source_idx", 0))
    st.session_state["source_idx"] = labels.index(picked)
    source = "cohort" if st.session_state["source_idx"] == 0 else "upload"
    sequence, truth, meta, genome_name = None, None, None, None

    if source == "upload":
        muted(t("uploader"))
        upload = st.file_uploader("fasta", type=["fna", "fasta", "fa", "gz", "txt"],
                                  key="fasta_upload", label_visibility="collapsed")
        if upload is not None:
            sequence = read_fasta(upload.getvalue(), upload.name)
            genome_name = upload.name
    else:
        if cohort is None:
            st.warning(t("no_cohort"))
            return
        available = [g for g in cohort["genome_id"] if (C.GENOMES / f"{g}.fna.gz").exists()]
        if not available:
            st.warning(t("no_genomes"))
            return
        left, right = st.columns([3, 2])
        with left:
            st.html(f'<p style="font-size:.88rem;color:{INK};margin:0 0 -.4rem 0">'
                    f'{t("available", n=f"{len(available):,}")}</p>')
            genome_id = st.selectbox("genome", available, key="genome_choice",
                                     label_visibility="collapsed")
        row = cohort[cohort["genome_id"] == genome_id].iloc[0]
        truth = int(row["label"])
        with right:
            st.metric(t("lab_phenotype"), t("resistant") if truth else t("susceptible"))
        country = row.get("isolation_country")
        if isinstance(country, str):
            country = COUNTRY_SQ.get(country, country) if lang() == "sq" else country
            meta = t("meta", st=row.get("sequence_type", "?"), country=country)
        sequence = read_fasta((C.GENOMES / f"{genome_id}.fna.gz").read_bytes(), f"{genome_id}.fna.gz")
        genome_name = genome_id

    if sequence is None:
        muted(t("choose"))
        return
    if len(sequence) < 100_000:
        st.error(t("too_short", n=f"{len(sequence):,}"))
        return

    with st.spinner(t("reading")):
        result = analyse(sequence)

    qrdr_model = models.get(f"qrdr_{C.PRIMARY_ANTIBIOTIC}")
    if qrdr_model is None:
        st.error(t("no_model"))
        return

    probability = float(qrdr_model.predict_proba(result["qrdr"].reshape(1, -1))[0, 1])
    resistant_call = probability >= 0.5

    def site_read(site: str) -> bool:
        hit = result["context"][site]
        _, _, wt, res = QRDR_SITES[site]
        return bool(hit) and (hit[1] == wt or hit[1] in res)

    unread = [site for site in QRDR_SITES if not site_read(site)]
    inconclusive = any(site in unread for site in PRIMARY_SITES)
    detected = [t("carries", site=site_name(site), aa=result["context"][site][1])
                for site, (_, _, _, res) in QRDR_SITES.items()
                if result["context"][site] and result["context"][site][1] in res]
    metrics = card.get("models", {}).get(f"qrdr_{C.PRIMARY_ANTIBIOTIC}", {})
    sensitivity = metrics.get("sensitivity_mlst_grouped")
    specificity = metrics.get("specificity_mlst_grouped")

    if inconclusive:
        verdict = "u"
        st.html(
            f'<div style="padding:1.05rem 1.25rem;border-left:4px solid {UNKNOWN};'
            f'background:{SURFACE};margin:.2rem 0 1.2rem 0">'
            f'<div style="font-weight:600;font-size:1.3rem;color:{INK}">{t("undetermined")}</div>'
            f'<div style="color:{INK_SOFT};font-size:.95rem;margin-top:.25rem">'
            f'{t("undetermined_why", sites=", ".join(site_name(s) for s in unread))}</div></div>')
        if truth is not None:
            muted(t("lab_is", r=(t("resistant") if truth else t("susceptible")).lower()))
    else:
        verdict = "r" if resistant_call else "s"
        accent = RESISTANT if resistant_call else SUSCEPT
        tint = "#FBEEEE" if resistant_call else "#EDF6F0"
        partial = (" " + t("partial", sites=", ".join(site_name(s) for s in unread))) if unread else ""
        st.html(
            f'<div style="padding:1.05rem 1.25rem;border-left:4px solid {accent};'
            f'background:{tint};margin:.2rem 0 1.2rem 0">'
            f'<div style="font-weight:600;font-size:1.3rem;color:{accent}">'
            f'{t("verdict_r") if resistant_call else t("verdict_s")}</div>'
            f'<div style="color:{INK_SOFT};font-size:.95rem;margin-top:.25rem">'
            f'{t("probability", p=f"{probability:.1%}")} '
            f'{"; ".join(detected) if detected else t("no_allele")}.{partial}</div></div>')
        if truth is not None and resistant_call != bool(truth) and sensitivity is not None:
            muted(t("differs", sens=f"{sensitivity:.1%}", spec=f"{specificity:.1%}"))
        elif truth is not None:
            muted(t("matches"))
    if meta:
        muted(meta)

    pdf = build_report(genome_name or t("uploaded_file"), verdict, probability, result, truth, metrics)
    if pdf is None:
        st.info("pip install reportlab")
    else:
        st.download_button(t("report_btn"), data=pdf, key="report_button",
                           file_name=f"raport_{(genome_name or 'gjenomi').split('.fna')[0]}.pdf",
                           mime="application/pdf", help=t("report_help"), type="primary")

    # ── the codons themselves ────────────────────────────────────────────────
    section(t("loci"))
    for site, (_, _, wt, resistant) in QRDR_SITES.items():
        gene, position = site.split("_")
        hit = result["context"][site]
        box = (f'border:1px solid {HAIRLINE};border-radius:3px;'
               f'padding:.7rem .9rem;margin-bottom:.55rem;background:#FFFFFF')
        title = t("codon", gene=gene, pos=position)
        if hit is None:
            st.html(
                f'<div style="{box}"><div style="display:flex;justify-content:space-between;'
                f'align-items:baseline;margin-bottom:.45rem">'
                f'<span style="font-weight:600;color:{INK};font-size:.95rem">{title}</span>'
                f'<span style="font-size:.8rem;color:{INK_SOFT}">{t("not_resolved")}</span></div>'
                f'<div style="color:{INK_SOFT};font-size:.78rem">{t("not_resolved_note")}</div></div>')
            continue
        codon, residue, before, after = hit
        is_resistant = residue in resistant
        is_wildtype = residue == wt
        call = t("call_r") if is_resistant else t("call_wt") if is_wildtype else t("call_other")
        call_colour = RESISTANT if is_resistant else SUSCEPT if is_wildtype else INK_SOFT
        codon_bg = RESISTANT if is_resistant else SUSCEPT if is_wildtype else PRIMARY
        note = (t("note_changed", wt=wt, aa=residue) if residue and not is_wildtype
                else t("note_wt", wt=wt) if is_wildtype else t("note_untranslated"))
        st.html(
            f'<div style="{box}"><div style="display:flex;justify-content:space-between;'
            f'align-items:baseline;margin-bottom:.45rem">'
            f'<span style="font-weight:600;color:{INK};font-size:.95rem">{title}</span>'
            f'<span style="font-size:.8rem;font-weight:500;color:{call_colour}">'
            f'{residue or "?"}, {call}</span></div>'
            f'<div style="font-family:{MONO};font-size:.9rem;letter-spacing:.055em;color:{INK_SOFT};'
            f'white-space:nowrap;overflow-x:auto">{before}'
            f'<span style="background:{codon_bg};color:#FFFFFF;font-weight:600;'
            f'padding:.13rem .3rem;border-radius:2px">{codon}</span>{after}</div>'
            f'<div style="color:{INK_SOFT};font-size:.78rem;margin-top:.4rem">{note}</div></div>')
    muted(t("loci_note"))

    if inconclusive:
        return

    # ── contrast between feature sets ────────────────────────────────────────
    section(t("contrast"))
    reported = card.get("models", {})
    col_1, col_2 = st.columns(2)
    with col_1:
        st.metric(t("mech_model"), f"{probability:.1%}", help=t("mech_help"))
        a = reported.get(f"qrdr_{C.PRIMARY_ANTIBIOTIC}", {}).get("cv_auroc_mlst_grouped")
        if a is not None:
            muted(t("auroc_note", a=f"{a:.3f}"))
    with col_2:
        kmer_model = models.get(f"kmer_{C.PRIMARY_ANTIBIOTIC}")
        if kmer_model is not None:
            kmer_probability = float(kmer_model.predict_proba(result["kmer"].reshape(1, -1))[0, 1])
            st.metric(t("comp_model"), f"{kmer_probability:.1%}",
                      delta=f"{kmer_probability - probability:+.1%}", delta_color="off")
            a = reported.get(f"kmer_{C.PRIMARY_ANTIBIOTIC}", {}).get("cv_auroc_mlst_grouped")
            if a is not None:
                muted(t("auroc_note", a=f"{a:.3f}"))
    muted(t("contrast_note"))


# ── results view ──────────────────────────────────────────────────────────────
# In the order of the story: what the data look like, why the usual check fails,
# what changes when it is done properly, why, and whether it holds on new data.
RESULTS = [
    ("06_qrdr_prevalence.png",
     ("1. What the four positions look like in the data",
      "For each of the four positions, the bars show how often each amino acid was found. Dark blue "
      "is the normal amino acid; red marks a change linked to resistance. About one genome in three "
      "carries the S83L change in gyrA, which matches what other studies report for E. coli."),
     ("1. Si duken katër pozicionet në të dhëna",
      "Për secilin nga katër pozicionet, shtyllat tregojnë sa shpesh u gjet secili aminoacid. Bluja e "
      "errët është aminoacidi i zakonshëm; e kuqja tregon një ndryshim të lidhur me rezistencën. Rreth "
      "një në tre gjenome ka ndryshimin S83L te gyrA, siç raportojnë edhe studimet e tjera për E. coli.")),
    ("03_cluster_validation.png",
     ("2. The grouping from the proposal does not separate families",
      "Bacteria come in families (lineages). A fair test must keep a whole family on one side, either "
      "training or testing. The proposal grouped genomes with k-means. Each bar shows how much of one "
      "family ended up in the same group. Most families are split, close to the dashed line that chance "
      "would give, so k-means does not do the job."),
     ("2. Grupimi i propozimit nuk i ndan familjet",
      "Bakteret ndahen në familje (linja). Një test i drejtë duhet ta mbajë një familje të tërë në një "
      "anë, ose në trajnim ose në testim. Propozimi i grupoi gjenomet me k-means. Secila shtyllë tregon "
      "sa nga një familje përfundoi në të njëjtin grup. Shumica e familjeve ndahen, afër vijës së "
      "ndërprerë që do të jepte rastësia, prandaj k-means nuk e kryen këtë punë.")),
    ("01_lineage_gap.png",
     ("3. The way we test changes the answer",
      "Each line is one type of feature, and each panel one model. Moving right, the test gets stricter. "
      "The QRDR features (the four positions) stay near 0.98 in every model. The k-mer features fall as "
      "soon as families are kept apart, because they were partly recognising the family, not the resistance."),
     ("3. Mënyra e testimit e ndryshon përgjigjen",
      "Secila vijë është një lloj veçorish, dhe secili panel një model. Djathtas, testi bëhet më i rreptë. "
      "Veçoritë QRDR (katër pozicionet) mbeten afër 0.98 te çdo model. Veçoritë k-mer bien sapo familjet "
      "ndahen, sepse pjesërisht njihnin familjen dhe jo rezistencën.")),
    ("04_cnn_comparison.png",
     ("4. What the input is matters more than the model",
      "The same neural network scores 0.98 when it reads the gyrA and parC genes, and 0.50, the level of "
      "guessing, when it reads random parts of the genome. The difference comes only from what the "
      "network is given to read."),
     ("4. Hyrja ka më shumë rëndësi se modeli",
      "I njëjti rrjet neural arrin 0.98 kur lexon gjenet gyrA dhe parC, dhe 0.50, nivelin e hamendësimit, "
      "kur lexon pjesë të rastësishme të gjenomit. Dallimi vjen vetëm nga ajo që i jepet rrjetit për ta lexuar.")),
    ("05_palindrome_signature.png",
     ("5. Why the k-mer model was misled",
      "Left: the 20 short sequences the k-mer model relied on most. 15 of them (blue) read the same on both "
      "DNA strands; these are cutting sites of bacterial defence enzymes. Right: each dot is one sequence. "
      "Almost all sit far below the line, meaning they tell us which family a bacterium belongs to, not "
      "whether it is resistant."),
     ("5. Pse modeli k-mer u mashtrua",
      "Majtas: 20 sekuencat e shkurtra mbi të cilat u mbështet më shumë modeli k-mer. 15 prej tyre (blu) "
      "lexohen njësoj në të dy vargjet e ADN-së; këto janë vende ku presin enzimat mbrojtëse të bakterit. "
      "Djathtas: çdo pikë është një sekuencë. Pothuajse të gjitha janë shumë nën vijë, që do të thotë se "
      "tregojnë familjen e bakterit dhe jo nëse ai është rezistent.")),
    ("07_external_validation.png",
     ("6. The test on completely new data",
      "Both models were tested on 7,383 genomes from NCBI, a different database they had never seen. A "
      "curve closer to the top-left corner is better. The QRDR model (AUROC 0.903) stays clearly ahead of "
      "the k-mer model (0.851)."),
     ("6. Testi me të dhëna krejt të reja",
      "Të dy modelet u testuan me 7,383 gjenome nga NCBI, një bazë tjetër të dhënash që nuk e kishin parë "
      "kurrë. Sa më afër këndit lart majtas të jetë kurba, aq më mirë. Modeli QRDR (AUROC 0.903) mbetet "
      "qartë para modelit k-mer (0.851).")),
]


def view_results() -> None:
    muted(t("results_intro"))
    shown = 0
    for name, english, albanian in RESULTS:
        path = C.FIGURES / name
        if not path.exists():
            continue
        title, text = albanian if lang() == "sq" else english
        section(title)
        _, middle, _ = st.columns([1, 6, 1])
        with middle:
            st.image(str(path), width="stretch")
        muted(text)
        shown += 1
    if not shown:
        st.warning(t("no_figures"))
        return
    section(t("tables"))
    files = ["T2_main_results.csv", "T3_cv_designs.csv", "T4_qrdr_specificity.csv", "T1_cohort.csv"]
    for name, title in zip(files, t("table_names")):
        path = C.TABLES / name
        if path.exists():
            with st.expander(title):
                st.dataframe(pd.read_csv(path), hide_index=True, width="stretch")


# ── method view ───────────────────────────────────────────────────────────────
METHOD = {
    "en": [
        ("What this app does",
         "Ciprofloxacin is a common antibiotic. Some *E. coli* bacteria are resistant to it. This app looks "
         "at the DNA of a bacterium and predicts whether the antibiotic will still work. A normal laboratory "
         "test takes one to three days; reading the DNA is much faster."),
        ("How it reads the DNA",
         "Resistance to ciprofloxacin is mostly caused by small changes at four known places in two genes, "
         "*gyrA* and *parC*. The app finds these four places in the genome, reads the three DNA letters at "
         "each one, and checks whether they show a change linked to resistance. A model then turns these four "
         "answers into a prediction.\n\nIf one of the two most important places cannot be found, the app does "
         "not guess. It says **\"Cannot determine\"** instead."),
        ("How we checked that it works",
         "The models were trained on 8,644 *E. coli* genomes with known laboratory results.\n\nBacteria come in "
         "families. If the same family appears in both the training and the test data, a model can look good "
         "just by recognising the family. To avoid this, the main test always keeps each family on one side "
         "only.\n\nFinally, the models were tested on 7,383 genomes from a different database (NCBI) that they "
         "had never seen. The four-position model still worked well there."),
        ("What it cannot do",
         "- It only reads four places. Bacteria that are resistant for other reasons, for example through extra "
         "genes, will be called susceptible.\n- A harmless spelling difference near one of the four places can "
         "stop the app from finding it.\n- It is a research tool for a thesis, not a medical test."),
    ],
    "sq": [
        ("Çfarë bën ky aplikacion",
         "Ciprofloksacina është një antibiotik i zakonshëm. Disa baktere *E. coli* janë rezistente ndaj saj. "
         "Ky aplikacion lexon ADN-në e një bakteri dhe parashikon nëse antibiotiku do të veprojë ende. Një test "
         "i zakonshëm laboratorik zgjat një deri në tri ditë; leximi i ADN-së është shumë më i shpejtë."),
        ("Si e lexon ADN-në",
         "Rezistenca ndaj ciprofloksacinës shkaktohet kryesisht nga ndryshime të vogla në katër vende të njohura "
         "të dy gjeneve, *gyrA* dhe *parC*. Aplikacioni i gjen këto katër vende në gjenom, lexon tri shkronjat e "
         "ADN-së në secilin prej tyre dhe kontrollon nëse kanë një ndryshim të lidhur me rezistencën. Pastaj një "
         "model i kthen këto katër përgjigje në një parashikim.\n\nNëse njëri nga dy vendet më të rëndësishme "
         "nuk gjendet, aplikacioni nuk hamendëson. Në vend të kësaj shkruan **\"Nuk mund të përcaktohet\"**."),
        ("Si u kontrollua që funksionon",
         "Modelet u trajnuan me 8,644 gjenome të *E. coli*-t me rezultate laboratorike të njohura.\n\nBakteret "
         "ndahen në familje. Nëse e njëjta familje gjendet edhe në trajnim edhe në testim, një model mund të duket "
         "i mirë vetëm duke e njohur familjen. Për ta shmangur këtë, testi kryesor e mban çdo familje vetëm në "
         "njërën anë.\n\nNë fund, modelet u testuan me 7,383 gjenome nga një bazë tjetër të dhënash (NCBI), që nuk "
         "i kishin parë kurrë. Modeli me katër pozicionet funksionoi mirë edhe atje."),
        ("Çfarë nuk mund të bëjë",
         "- Lexon vetëm katër vende. Bakteret që janë rezistente për arsye të tjera, për shembull përmes gjeneve "
         "shtesë, do të quhen të ndjeshme.\n- Një ndryshim i padëmshëm pranë njërit nga katër vendet mund ta "
         "pengojë aplikacionin ta gjejë atë.\n- Është një mjet kërkimor për një punim diplome, jo një test mjekësor."),
    ],
}

DEVIATIONS = {
    "en": (["Planned", "Done instead", "Why"], [
        ["Install the CARD tool locally", "Used ready-made CARD results from BV-BRC, plus our own reader for the four places", "The tool would have needed 70 to 140 hours of computing"],
        ["Search the top 20 short sequences with BLAST", "Measured whether they follow the family or the resistance", "A 6-letter sequence appears over 1,000 times in every genome"],
        ["Neural network on whole genomes", "Neural network on the gyrA and parC genes and on random pieces", "Whole genomes were too large for the available computer"],
        ["Group genomes with k-means", "Kept k-means and added grouping by family (MLST)", "k-means did not keep families together"],
        ["Remove sequences shorter than 100 kb", "Check every download is at least 95% of its full length", "The old rule let incomplete downloads through"],
        ["Test on NCBI data", "Done for ciprofloxacin, on 7,383 genomes", "CARD results do not exist for genomes outside BV-BRC"]]),
    "sq": (["Planifikuar", "U bë në vend të kësaj", "Pse"], [
        ["Instalimi lokal i mjetit CARD", "U përdorën rezultatet e gatshme të CARD nga BV-BRC, plus lexuesi ynë për katër vendet", "Mjeti do të kërkonte 70 deri në 140 orë llogaritje"],
        ["Kërkimi i 20 sekuencave kryesore me BLAST", "U mat nëse ato ndjekin familjen apo rezistencën", "Një sekuencë me 6 shkronja shfaqet mbi 1,000 herë në çdo gjenom"],
        ["Rrjet neural me gjenome të plota", "Rrjet neural me gjenet gyrA dhe parC dhe me pjesë të rastësishme", "Gjenomet e plota ishin shumë të mëdha për kompjuterin në dispozicion"],
        ["Grupimi i gjenomeve me k-means", "U mbajt k-means dhe u shtua grupimi sipas familjes (MLST)", "K-means nuk i mbajti bashkë familjet"],
        ["Heqja e sekuencave më të shkurtra se 100 kb", "Kontrollohet që çdo shkarkim të jetë të paktën 95% e gjatësisë së plotë", "Rregulli i vjetër lejonte shkarkime të paplota"],
        ["Testi me të dhëna nga NCBI", "U bë për ciprofloksacinën, me 7,383 gjenome", "Rezultatet CARD nuk ekzistojnë për gjenome jashtë BV-BRC"]]),
}


def view_method(card: dict) -> None:
    for title, body in METHOD[lang()]:
        section(title)
        st.markdown(body)
    cols, rows = DEVIATIONS[lang()]
    with st.expander("Changes from the approved proposal" if lang() == "en"
                     else "Ndryshimet nga propozimi i miratuar"):
        st.dataframe(pd.DataFrame(rows, columns=cols), hide_index=True, width="stretch")
    if card:
        with st.expander("Technical details of the models" if lang() == "en"
                         else "Detajet teknike të modeleve"):
            st.json(card)


# ── main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    models, card = load_models()
    cohort = load_cohort()

    with st.sidebar:
        choice = st.segmented_control("Language", ["Shqip", "English"], default="Shqip",
                                      label_visibility="collapsed", key="lang_choice")
        st.session_state["lang"] = "en" if choice == "English" else "sq"
        st.html(f'<div style="font-size:1.05rem;font-weight:600;color:{INK};'
                f'padding:.6rem 0 .4rem 0;border-bottom:1px solid {HAIRLINE};'
                f'margin-bottom:.9rem">{t("view")}</div>')
        labels = t("views")
        picked = st.radio("view", labels, label_visibility="collapsed",
                          index=st.session_state.get("view_idx", 0))
        st.session_state["view_idx"] = labels.index(picked)
        view = ["predict", "results", "method"][st.session_state["view_idx"]]
        st.write("")
        if cohort is not None:
            st.metric(t("genomes"), f"{len(cohort):,}")
            st.metric(t("resistant_share"), f"{cohort['label'].mean():.1%}")
        st.html(f'<p style="font-size:.85rem;color:{INK_SOFT}">'
                f'{t("models_loaded", n=len(models))}</p>')

    st.html(
        f'<div style="margin-bottom:.2rem">'
        f'<div style="font-size:2rem;font-weight:600;color:{INK};'
        f'line-height:1.15;letter-spacing:-.02em">{t("title")}</div>'
        f'<p style="color:{INK_SOFT};font-size:.95rem;max-width:62ch;'
        f'margin:.4rem 0 0 0;line-height:1.5">{t("subtitle")}</p>'
        f'<div style="height:3px;width:54px;background:{PRIMARY};'
        f'margin:1rem 0 .4rem 0"></div></div>')

    if view == "predict":
        view_predict(models, card, cohort)
    elif view == "results":
        view_results()
    else:
        view_method(card)


if __name__ == "__main__":
    main()
