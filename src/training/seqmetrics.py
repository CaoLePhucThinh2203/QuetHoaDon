"""Entity-level precision / recall / F1 for BIO tag sequences (pure Python).

An entity counts as correct only when its field **and** both boundaries match the gold entity. This is the main
training metric: token accuracy is meaningless here because about 78 % of the words are ``O``.
"""
from typing import Dict, List, Sequence, Set, Tuple

from src.evaluation.metrics import prf1
from src.extraction.schema import ANNOTATED_FIELDS

Span = Tuple[str, int, int]


def bio_spans(tags: Sequence[str]) -> Set[Span]:
    """``{(field, start, end)}`` with ``end`` exclusive. ``I-X`` not continuing an ``X`` entity starts a new one."""
    spans: Set[Span] = set()
    field, start = None, 0
    for i, tag in enumerate(list(tags) + ["O"]):
        if tag == "O" or "-" not in tag:
            new_field, prefix = None, ""
        else:
            prefix, new_field = tag.split("-", 1)
        continues = new_field is not None and new_field == field and prefix == "I"
        if field is not None and not continues:
            spans.add((field, start, i))
            field = None
        if new_field is not None and not continues:
            field, start = new_field, i
    return spans


def entity_scores(gold: Sequence[Sequence[str]], pred: Sequence[Sequence[str]],
                  fields: Sequence[str] = ANNOTATED_FIELDS) -> Dict[str, Dict[str, float]]:
    """Per-field and micro-averaged entity P/R/F1 over many receipts (``gold[i]`` and ``pred[i]`` are tag lists)."""
    counts: Dict[str, List[int]] = {f: [0, 0, 0] for f in fields}      # tp, fp, fn
    for g_tags, p_tags in zip(gold, pred):
        g, p = bio_spans(g_tags), bio_spans(p_tags)
        for f in fields:
            gf = {s for s in g if s[0] == f}
            pf = {s for s in p if s[0] == f}
            counts[f][0] += len(gf & pf)
            counts[f][1] += len(pf - gf)
            counts[f][2] += len(gf - pf)
    out = {f: prf1(*c) for f, c in counts.items()}
    out["micro"] = prf1(sum(c[0] for c in counts.values()), sum(c[1] for c in counts.values()),
                        sum(c[2] for c in counts.values()))
    return out
