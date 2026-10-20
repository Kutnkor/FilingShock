"""Deterministic, explainable paragraph alignment between a reference and a current section.

Algorithm (fs-align-1.0):
  1. exact match on normalised text                     -> unchanged
  2. exact match after masking numbers                   -> numeric_only (numbers changed)
  3. greedy one-to-one matching on word-shingle Jaccard  -> minor_edit / substantial_edit
     (candidate pairs need Jaccard >= MIN_JACCARD; each paragraph is used at most once)
  4. unmatched current paragraphs -> added; unmatched reference paragraphs -> removed
  5. matched pairs outside the longest increasing subsequence of positions are flagged moved;
     a moved pair with unchanged text gets status 'moved'
Match scores are kept for every pair; weak matches carry low_confidence=True.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from filingshock import ALIGNER_VERSION

NUM_RE = re.compile(r"\$?\(?\d[\d,]*(?:\.\d+)?\)?%?")
WORD_RE = re.compile(r"[a-z#]+(?:'[a-z]+)?")
MIN_JACCARD = 0.40
LOW_CONF_JACCARD = 0.55
MINOR_EDIT_RATIO = 0.90

STATUSES = ["unchanged", "numeric_only", "moved", "minor_edit", "substantial_edit", "added", "removed"]


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def mask_numbers(text: str) -> str:
    return NUM_RE.sub("#", norm(text))


def numbers_in(text: str) -> list[str]:
    return [m.group(0).strip("$()%").replace(",", "") for m in NUM_RE.finditer(text)]


def _shingles(masked: str) -> set[str]:
    w = WORD_RE.findall(masked)
    return set(w) | {f"{a} {b}" for a, b in zip(w, w[1:])}


def _lis_member(seq: list[int]) -> list[bool]:
    """Boolean membership of each element in one longest strictly increasing subsequence."""
    n = len(seq)
    if n == 0:
        return []
    tails, tails_idx, prev = [], [], [-1] * n
    import bisect
    for i, v in enumerate(seq):
        k = bisect.bisect_left(tails, v)
        if k == len(tails):
            tails.append(v)
            tails_idx.append(i)
        else:
            tails[k] = v
            tails_idx[k] = i
        prev[i] = tails_idx[k - 1] if k > 0 else -1
    member = [False] * n
    i = tails_idx[-1]
    while i >= 0:
        member[i] = True
        i = prev[i]
    return member


@dataclass
class AlignmentResult:
    pairs: pd.DataFrame
    summary: dict


def align_paragraphs(ref: list[str], cur: list[str]) -> AlignmentResult:
    ref_n = [norm(t) for t in ref]
    cur_n = [norm(t) for t in cur]
    ref_m = [mask_numbers(t) for t in ref]
    cur_m = [mask_numbers(t) for t in cur]
    ref_used = [False] * len(ref)
    cur_match: dict[int, tuple[int, str, float, float]] = {}

    def exact_pass(ref_keys, cur_keys, status):
        index: dict[str, list[int]] = {}
        for i, k in enumerate(ref_keys):
            if not ref_used[i]:
                index.setdefault(k, []).append(i)
        for j, k in enumerate(cur_keys):
            if j in cur_match:
                continue
            cands = index.get(k)
            if cands:
                i = cands.pop(0)
                ref_used[i] = True
                cur_match[j] = (i, status, 1.0, 1.0)

    exact_pass(ref_n, cur_n, "unchanged")
    exact_pass(ref_m, cur_m, "numeric_only")

    rem_r = [i for i in range(len(ref)) if not ref_used[i]]
    rem_c = [j for j in range(len(cur)) if j not in cur_match]
    if rem_r and rem_c:
        rs = [_shingles(ref_m[i]) for i in rem_r]
        cs = [_shingles(cur_m[j]) for j in rem_c]
        cand = []
        for a, j in enumerate(rem_c):
            sj = cs[a]
            if not sj:
                continue
            for b, i in enumerate(rem_r):
                si = rs[b]
                if not si:
                    continue
                inter = len(sj & si)
                if inter == 0:
                    continue
                jac = inter / len(sj | si)
                if jac >= MIN_JACCARD:
                    cand.append((jac, -abs(j / max(len(cur), 1) - i / max(len(ref), 1)), j, i))
        cand.sort(reverse=True)
        for jac, _, j, i in cand:
            if j in cur_match or ref_used[i]:
                continue
            ratio = difflib.SequenceMatcher(None, WORD_RE.findall(ref_m[i]), WORD_RE.findall(cur_m[j]),
                                            autojunk=False).ratio()
            status = "minor_edit" if ratio >= MINOR_EDIT_RATIO else "substantial_edit"
            ref_used[i] = True
            cur_match[j] = (i, status, jac, ratio)

    rows = []
    matched_cur = sorted(cur_match)
    member = _lis_member([cur_match[j][0] for j in matched_cur])
    moved = {j: not m for j, m in zip(matched_cur, member)}
    for j in range(len(cur)):
        if j in cur_match:
            i, status, jac, ratio = cur_match[j]
            num_change = numbers_in(ref[i]) != numbers_in(cur[j])
            if moved[j] and status in ("unchanged", "numeric_only"):
                status = "moved"
            rows.append({"ref_ordinal": i, "cur_ordinal": j, "status": status, "jaccard": round(jac, 4),
                         "edit_ratio": round(ratio, 4), "moved": moved[j], "numeric_change": num_change,
                         "low_confidence": jac < LOW_CONF_JACCARD})
        else:
            rows.append({"ref_ordinal": None, "cur_ordinal": j, "status": "added", "jaccard": np.nan,
                         "edit_ratio": np.nan, "moved": False, "numeric_change": False,
                         "low_confidence": False})
    for i in range(len(ref)):
        if not ref_used[i]:
            rows.append({"ref_ordinal": i, "cur_ordinal": None, "status": "removed", "jaccard": np.nan,
                         "edit_ratio": np.nan, "moved": False, "numeric_change": False,
                         "low_confidence": False})
    pairs = pd.DataFrame(rows, columns=["ref_ordinal", "cur_ordinal", "status", "jaccard", "edit_ratio",
                                        "moved", "numeric_change", "low_confidence"])
    pairs["ref_ordinal"] = pairs["ref_ordinal"].astype("Int64")
    pairs["cur_ordinal"] = pairs["cur_ordinal"].astype("Int64")
    # display order: by current position, removed paragraphs placed after their reference predecessor
    pairs["sort_key"] = _display_order(pairs)
    pairs = pairs.sort_values("sort_key").reset_index(drop=True)
    counts = pairs["status"].value_counts().to_dict()
    n_matched = int(pairs["ref_ordinal"].notna().sum() - counts.get("removed", 0))
    n_low = int(pairs["low_confidence"].sum())
    summary = {
        "aligner_version": ALIGNER_VERSION,
        "n_ref": len(ref), "n_cur": len(cur),
        **{f"n_{s}": int(counts.get(s, 0)) for s in STATUSES},
        "n_numeric_change": int(pairs["numeric_change"].sum()),
        "n_low_confidence": n_low,
        "low_confidence_share": round(n_low / n_matched, 4) if n_matched else None,
        "alignment_quality": _quality_label(n_low, n_matched),
    }
    return AlignmentResult(pairs, summary)


def _quality_label(n_low: int, n_matched: int) -> str:
    if n_matched == 0:
        return "no_matches"
    share = n_low / n_matched
    return "good" if share <= 0.15 else ("fair" if share <= 0.35 else "weak")


def _display_order(pairs: pd.DataFrame) -> list[float]:
    ref_to_cur = {int(r): int(c) for r, c in zip(pairs["ref_ordinal"], pairs["cur_ordinal"])
                  if pd.notna(r) and pd.notna(c)}
    keys = []
    for r, c in zip(pairs["ref_ordinal"], pairs["cur_ordinal"]):
        if pd.notna(c):
            keys.append(float(c))
        else:
            prev = [ref_to_cur[k] for k in ref_to_cur if k < int(r)]
            base = max(prev) if prev else -1
            keys.append(base + 0.5 + int(r) * 1e-4)
    return keys


def word_diff(ref_text: str, cur_text: str) -> list[tuple[str, str]]:
    """Word-level diff as (op, text) tuples with op in {equal, delete, insert}. Text is raw;
    callers must escape it before rendering."""
    a, b = ref_text.split(), cur_text.split()
    out: list[tuple[str, str]] = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if op == "equal":
            out.append(("equal", " ".join(a[i1:i2])))
        else:
            if i2 > i1:
                out.append(("delete", " ".join(a[i1:i2])))
            if j2 > j1:
                out.append(("insert", " ".join(b[j1:j2])))
    return out
