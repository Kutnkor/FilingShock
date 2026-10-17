"""Interpretable text-change features.

No learned vocabulary or TF-IDF weights are used: every feature is a function of the event's own
current and reference sections plus versioned dictionaries, so nothing is fitted on other events.

Loughran-McDonald membership is applied point-in-time: a word counts for an event in year Y only
if the dictionary's year-added flag is <= Y and it had not been removed (negative flag) by Y.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import pandas as pd
import requests
import yaml

from filingshock.storage import DataLayout, sha256_file, write_bytes_atomic, write_json_atomic

LM_URL = "https://drive.google.com/uc?export=download&id=1iq2RUf8qGFEAk1g8wQntP3habOnR3fXF"
LM_PAGE = "https://sraf.nd.edu/loughranmcdonald-master-dictionary/"
LM_FILE = "Loughran-McDonald_MasterDictionary_1993-2025.csv"
LM_CATEGORIES = ["Negative", "Uncertainty", "Litigious", "Constraining"]
TOKEN = re.compile(r"[a-z]+(?:'[a-z]+)?")

TEXT_GROUPS = {
    "length": ["txt_mdna_log_len_ratio"],
    "dictionary": ["txt_mdna_uncertainty_rate", "txt_mdna_uncertainty_change", "txt_mdna_negative_rate",
                   "txt_mdna_negative_change", "txt_changed_topic_share"],
    "paragraph_change": ["txt_mdna_prop_added", "txt_mdna_prop_removed", "txt_mdna_prop_substantial",
                         "txt_mdna_lexical_distance", "txt_risk_prop_added"],
}
TEXT_FEATURES = [f for g in TEXT_GROUPS.values() for f in g]


def fetch_lm_dictionary(layout: DataLayout, user_agent: str) -> Path:
    path = layout.raw_resources / LM_FILE
    if path.exists():
        return path
    resp = requests.get(LM_URL, headers={"User-Agent": user_agent}, timeout=60)
    resp.raise_for_status()
    if not resp.content.startswith(b"Word,"):
        raise RuntimeError("Loughran-McDonald download did not return the expected CSV")
    write_bytes_atomic(path, resp.content)
    write_json_atomic(layout.raw_resources / "lm_manifest.json", {
        "source_page": LM_PAGE, "download_url": LM_URL, "file": LM_FILE,
        "retrieved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sha256": sha256_file(path),
        "terms": "Free for academic research; commercial use requires a licence from the authors "
                 "(loughranmcdonald@gmail.com). Not redistributed with this repository.",
    })
    return path


@dataclass
class PointInTimeLexicon:
    table: pd.DataFrame          # word, category, added, removed

    @classmethod
    def from_csv(cls, path: Path) -> "PointInTimeLexicon":
        raw = pd.read_csv(path, usecols=["Word"] + LM_CATEGORIES, keep_default_na=False)
        rows = []
        for cat in LM_CATEGORIES:
            s = raw[raw[cat] != 0]
            for w, flag in zip(s["Word"], s[cat]):
                flag = int(flag)
                # positive flag = year added; negative flag = year removed (added in the base list)
                added, removed = (flag, None) if flag > 0 else (2009, -flag)
                rows.append((str(w).lower(), cat.lower(), added, removed))
        return cls(pd.DataFrame(rows, columns=["word", "category", "added", "removed"]))

    @lru_cache(maxsize=64)
    def words(self, category: str, year: int) -> frozenset:
        t = self.table[(self.table["category"] == category) & (self.table["added"] <= year)]
        t = t[t["removed"].isna() | (t["removed"] > year)]
        return frozenset(t["word"])

    def __hash__(self):
        return id(self)


class TopicLexicon:
    def __init__(self, path: Path):
        doc = yaml.safe_load(Path(path).read_text())
        self.version = doc["version"]
        self.patterns = {t: re.compile(r"\b(?:" + "|".join(p) + r")\b", re.I) for t, p in doc["topics"].items()}

    def topics_in(self, text: str) -> list[str]:
        return [t for t, rx in self.patterns.items() if rx.search(text)]


def tokens(text: str) -> list[str]:
    return TOKEN.findall(text.lower())


def rate_per_1000(toks: list[str], words: frozenset) -> float | None:
    if not toks:
        return None
    return 1000.0 * sum(1 for t in toks if t in words) / len(toks)


def cosine_distance(a: Counter, b: Counter) -> float | None:
    if not a or not b:
        return None
    dot = sum(v * b.get(k, 0) for k, v in a.items())
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return 1.0 - dot / (na * nb) if na and nb else None


def section_text_features(cur_paras: list[dict], ref_paras: list[dict], align_pairs: pd.DataFrame,
                          align_summary: dict, lexicon: PointInTimeLexicon, topics: TopicLexicon,
                          year: int) -> tuple[dict, dict]:
    """MD&A features. *_paras are paragraph dicts with keys text, is_heading."""
    cur_body = [p["text"] for p in cur_paras if not p["is_heading"]]
    ref_body = [p["text"] for p in ref_paras if not p["is_heading"]]
    cur_tok = [t for p in cur_body for t in tokens(p)]
    ref_tok = [t for p in ref_body for t in tokens(p)]
    f: dict = {}
    body_pairs = align_pairs[~align_pairs["is_heading"]] if "is_heading" in align_pairs else align_pairs
    n_cur = max(sum(1 for _ in cur_body), 1)
    n_ref = max(sum(1 for _ in ref_body), 1)
    st = body_pairs["status"].value_counts()
    f["txt_mdna_prop_added"] = st.get("added", 0) / n_cur
    f["txt_mdna_prop_removed"] = st.get("removed", 0) / n_ref
    f["txt_mdna_prop_substantial"] = st.get("substantial_edit", 0) / n_cur
    f["txt_mdna_lexical_distance"] = cosine_distance(Counter(cur_tok), Counter(ref_tok))
    f["txt_mdna_log_len_ratio"] = math.log(len(cur_tok) / len(ref_tok)) if cur_tok and ref_tok else None
    unc, neg = lexicon.words("uncertainty", year), lexicon.words("negative", year)
    u_c, u_r = rate_per_1000(cur_tok, unc), rate_per_1000(ref_tok, unc)
    n_c, n_r = rate_per_1000(cur_tok, neg), rate_per_1000(ref_tok, neg)
    f["txt_mdna_uncertainty_rate"] = u_c
    f["txt_mdna_uncertainty_change"] = None if u_c is None or u_r is None else u_c - u_r
    f["txt_mdna_negative_rate"] = n_c
    f["txt_mdna_negative_change"] = None if n_c is None or n_r is None else n_c - n_r
    changed = body_pairs[body_pairs["status"].isin(["added", "substantial_edit"])]
    changed_texts = [cur_paras[int(j)]["text"] for j in changed["cur_ordinal"].dropna()]
    f["txt_changed_topic_share"] = (sum(1 for t in changed_texts if topics.topics_in(t)) / len(changed_texts)
                                    if changed_texts else 0.0)
    # descriptive (not model inputs): topic rates and newly introduced topics
    obs: dict = {"n_cur_words": len(cur_tok), "n_ref_words": len(ref_tok)}
    for topic, rx in topics.patterns.items():
        c_hits = sum(1 for p in cur_body if rx.search(p))
        r_hits = sum(1 for p in ref_body if rx.search(p))
        obs[f"topic_{topic}_cur_paras"] = c_hits
        obs[f"topic_{topic}_ref_paras"] = r_hits
        obs[f"topic_{topic}_in_changed_paras"] = sum(1 for t in changed_texts if rx.search(t))
    for cat in ("litigious", "constraining"):
        obs[f"lm_{cat}_rate_cur"] = rate_per_1000(cur_tok, lexicon.words(cat, year))
    obs["lm_dictionary_year_applied"] = year
    return f, obs


def risk_prop_added(cur_status: str, ref_status: str, pairs: pd.DataFrame | None, n_cur_body: int) -> float | None:
    """Defined only when BOTH risk sections are fully present; otherwise missing (not zero)."""
    if cur_status != "extracted" or ref_status != "extracted" or pairs is None or n_cur_body == 0:
        return None
    body = pairs[~pairs["is_heading"]] if "is_heading" in pairs else pairs
    return float((body["status"] == "added").sum() / n_cur_body)


def load_section(layout: DataLayout, accession: str) -> dict | None:
    p = layout.sections_dir / f"{accession}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())
