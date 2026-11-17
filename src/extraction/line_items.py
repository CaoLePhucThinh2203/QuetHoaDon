"""Read the item lines (name, quantity, unit price, amount) of a receipt from the OCR words and their boxes.

Pure Python and rule based. MC-OCR has no item labels, so nothing here is trained: the rules were written by reading the
OCR text of TRAIN receipts only, and are scored by an arithmetic check instead of item labels (see ``score_items.py``):

* every item line is checked as ``quantity * unit price == amount``;
* the sum of the amounts is compared with the total printed on the receipt (``summary_amounts``) and, when scoring, with
  the annotated ``TOTAL_COST`` (a receipt whose items add up to the right total almost surely has every item amount right).

Steps: group words into lines by their y position -> find the item table (after the column header, before the first
"total" line) -> per line, read the amounts ("36 .500", "24 000", "8, 000", "110,000") and the quantity -> take the name
from the line, or from the line above when the numbers sit on their own line (the common two-line layout).
"""
import re
import unicodedata
from typing import Any, Dict, List, Optional, Sequence, Tuple

_SEP = r"\s?[.,]\s?"
_AMOUNT = re.compile(
    r"(?<![\d])(\d{1,3}(?:" + _SEP + r"\d{3})+|\d{1,3}\s\d00(?!\d))(?:" + _SEP + r"0{2}(?!\d))?")
_DECIMAL_TAIL = re.compile(_SEP + r"0{2}$")
_BARCODE = re.compile(r"\b\d{6,14}\b")
_SMALL_INT = re.compile(r"(?<![\d.,])(\d{1,3})(?![\d.,])")

_HEADER_WORDS = ("ten hang", "ten mon", "mat hang", "hang hoa", "sl", "so luong", "don gia", "thanh tien", "t tien",
                 "dvt", "gia", "tt")
_END_WORDS = ("tong", "cong", "thanh toan", "phai tra", "phai t toan")
_TOP_WORDS = ("hoa don", "ngay", "gio", "thu ngan", "quay", "so hd", "phieu", "nvbh", "ban hang")
_SKIP_START = ("gia dang ky", "gia niem yet", "gia goc")
_DISCOUNT_START = ("km", "khuyen mai", "giam", "chiet khau", "ck")
_NOT_A_NAME = ("ten mon", "ten hang", "mat hang", "thanh tien", "so luong", "don gia", "tong", "cong")
_MAX_UNCHECKED = 2_000_000          # an amount above this whose line fails quantity * price is a misread number
_UNITS = {"goi", "hop", "lo", "ch", "nt", "kg", "chai", "cai", "thung", "lon", "tui", "kien", "cap", "bich", "g", "vnd",
          "d", "vat", "x"}


def plain(text: str) -> str:
    """Lower case, no diacritics, letters and digits only, single spaces."""
    t = unicodedata.normalize("NFD", text.lower().replace("đ", "d").replace("Đ", "d"))
    t = "".join(c for c in t if unicodedata.category(c) != "Mn")
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", t)).strip()


def _has(plain_text: str, words: Sequence[str]) -> bool:
    padded = f" {plain_text} "
    return any(f" {w} " in padded for w in words)


# ----------------------------------------------------------------------------- lines
def group_lines(words: Sequence[str], boxes: Sequence[Sequence[float]]) -> List[str]:
    """Reading-order text lines: words whose vertical centre is within ~0.6 of a word height of the line's mean."""
    items = sorted(zip(words, boxes), key=lambda wb: (wb[1][1] + wb[1][3]) / 2)
    lines: List[List[Tuple[str, Sequence[float]]]] = []
    cur: List[Tuple[str, Sequence[float]]] = []
    for w, b in items:
        cy = (b[1] + b[3]) / 2
        if cur:
            mean_y = sum((c[1][1] + c[1][3]) / 2 for c in cur) / len(cur)
            mean_h = sum(c[1][3] - c[1][1] for c in cur) / len(cur)
            if abs(cy - mean_y) < max(8.0, 0.6 * mean_h):
                cur.append((w, b))
                continue
            lines.append(cur)
        cur = [(w, b)]
    if cur:
        lines.append(cur)
    return [" ".join(w for w, _ in sorted(line, key=lambda wb: wb[1][0])) for line in lines]


# ----------------------------------------------------------------------------- numbers
def parse_amounts(line: str) -> List[Tuple[int, int, int, bool]]:
    """Money amounts of a line as ``(value, start, end, negative)``; ``-3 .300`` is a discount of 3300."""
    out = []
    for m in _AMOUNT.finditer(line):
        digits = re.sub(r"\D", "", _DECIMAL_TAIL.sub("", m.group(0)))
        if not digits:
            continue
        before = line[max(0, m.start() - 2):m.start()]
        out.append((int(digits), m.start(), m.end(), before.rstrip().endswith("-")))
    return out


def _small_ints(line: str, amounts: Sequence[Tuple[int, int, int, bool]]) -> List[int]:
    covered = [(s, e) for _, s, e, _ in amounts]
    return [int(m.group(1)) for m in _SMALL_INT.finditer(line)
            if 0 < int(m.group(1)) < 1000 and not any(s <= m.start() < e for s, e in covered)]


def _name_of(line: str, amounts: Sequence[Tuple[int, int, int, bool]]) -> str:
    text = line
    for _, s, e, _ in sorted(amounts, key=lambda a: -a[1]):
        text = text[:s] + " " + text[e:]
    text = _BARCODE.sub(" ", text)
    toks = text.split()

    def junk(t: str) -> bool:
        return bool(re.fullmatch(r"[\d.,:/x+\-]+", t)) or plain(t) == "" or plain(t) in _UNITS

    while toks and junk(toks[0]):          # numbering, barcode leftovers and units only at the two ends of the name
        toks.pop(0)
    while toks and junk(toks[-1]):
        toks.pop()
    name = " ".join(toks).strip(" .,:;-")
    return name if sum(c.isalpha() for c in name) >= 3 else ""


def _quantity(amounts: Sequence[Tuple[int, int, int, bool]], smalls: Sequence[int]) -> Optional[Dict[str, Any]]:
    """Quantity, unit price and amount of one item line, chosen so that ``quantity * unit price == amount``."""
    values = [a[0] for a in amounts]
    total = values[-1]
    if len(values) == 1:
        if 1 in smalls or not smalls:
            return {"quantity": 1, "unit_price": total, "amount": total, "line_ok": True}
        q = smalls[0]
        ok = q > 0 and total % q == 0
        return {"quantity": q, "unit_price": total // q if ok else total, "amount": total, "line_ok": ok}
    for unit in reversed(values[:-1]):
        for q in smalls:
            if q * unit == total:
                return {"quantity": q, "unit_price": unit, "amount": total, "line_ok": True}
        if unit == total:
            return {"quantity": 1, "unit_price": unit, "amount": total, "line_ok": True}
        if unit and total % unit == 0 and 1 < total // unit < 1000:
            return {"quantity": total // unit, "unit_price": unit, "amount": total, "line_ok": True}
    return {"quantity": smalls[0] if smalls else 1, "unit_price": values[-2], "amount": total, "line_ok": False}


# ----------------------------------------------------------------------------- the table
def _header_index(lines_plain: Sequence[str]) -> Optional[int]:
    for i, p in enumerate(lines_plain):
        if sum(1 for w in _HEADER_WORDS if f" {w} " in f" {p} ") >= 2:
            return i
    return None


def _end_index(lines_plain: Sequence[str], start: int) -> Optional[int]:
    for i in range(start, len(lines_plain)):
        if _has(lines_plain[i], _END_WORDS):
            return i
    return None


def extract_items(words: Sequence[str], boxes: Sequence[Sequence[float]]) -> Dict[str, Any]:
    """Items of a receipt plus what was found around them.

    Returns ``items`` (list of ``name``, ``quantity``, ``unit_price``, ``amount``, ``line_ok``, ``discount``),
    ``summary_amounts`` (amounts on the "total" lines after the table), ``header_found`` and ``end_found``.
    """
    lines = group_lines(words, boxes)
    lp = [plain(x) for x in lines]
    header = _header_index(lp)
    start = header + 1 if header is not None else 0
    end = _end_index(lp, start)
    if header is None and end is not None:
        top = [i for i in range(end) if _has(lp[i], _TOP_WORDS)]
        start = (max(top) + 1) if top else 0
    stop = end if end is not None else len(lines)

    items: List[Dict[str, Any]] = []
    pending = ""
    for i in range(start, stop):
        line, p = lines[i], lp[i]
        if any(p.startswith(s) for s in _SKIP_START):
            continue
        amounts = parse_amounts(line)
        if not amounts:
            if sum(c.isalpha() for c in line) >= 3 and not _has(p, _TOP_WORDS):
                pending = _name_of(line, []) or pending
            continue
        parsed = _quantity(amounts, _small_ints(line, amounts))
        name = _name_of(line, amounts)
        if plain(name).startswith("gia ban"):          # "Giá bán: 12 HOP 6,500 78,000" sits under the product name
            name = ""
        if any(plain(name).startswith(w) for w in _NOT_A_NAME if name):
            continue
        name = name or pending
        if not parsed["line_ok"] and abs(parsed["amount"]) > _MAX_UNCHECKED:
            continue
        discount = any(p.startswith(d + " ") or p == d for d in _DISCOUNT_START) or amounts[-1][3]
        if discount:
            parsed.update(amount=-parsed["amount"], quantity=1, unit_price=-parsed["amount"], line_ok=True)
        elif not name:
            pending = ""
            continue
        items.append({"name": name, **parsed, "discount": bool(discount)})
        pending = ""

    summary: List[int] = []
    if end is not None:
        for j in range(end, min(len(lines), end + 8)):
            if _has(lp[j], _END_WORDS):
                summary += [a[0] for a in parse_amounts(lines[j])]
    return {"items": items, "summary_amounts": summary, "header_found": header is not None,
            "end_found": end is not None}


def check_items(items: Sequence[Dict[str, Any]], totals: Sequence[Optional[int]]) -> Dict[str, Any]:
    """Compare the sum of the item amounts with the candidate totals; ``verified`` when one of them is equal."""
    total_sum = sum(it["amount"] for it in items)
    gross = sum(it["amount"] for it in items if not it.get("discount"))
    wanted = [t for t in totals if t]
    status = "verified" if items and (total_sum in wanted or gross in wanted) else "unverified"
    return {"sum": total_sum, "sum_without_discounts": gross, "totals": wanted, "status": status,
            "lines_ok": sum(1 for it in items if it.get("line_ok")), "n_items": len(items)}
