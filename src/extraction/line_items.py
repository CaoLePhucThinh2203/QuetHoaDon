"""Read the item lines (name, quantity, unit price, amount) of a receipt from the OCR words and their boxes.

Pure Python and rule based. MC-OCR has no item labels, so nothing here is trained: the rules were written by reading the
OCR text of TRAIN receipts only, and are scored by an arithmetic check instead of item labels (see ``score_items.py``):

* every item line is checked as ``quantity * unit price == amount``;
* the sum of the amounts is compared with the total printed on the receipt (``summary_amounts``) and, when scoring, with
  the annotated ``TOTAL_COST`` (a receipt whose items add up to the right total almost surely has every item amount right).

Steps: group words into lines by their y position along the text slope, joining numbers split into two overlapping
boxes -> find the item table (after the column header, before the first
"total" line) -> per line, read the amounts ("36 .500", "24 000", "8, 000", "110,000") and the quantity -> take the name
from the line, or from the line above when the numbers sit on their own line (the common two-line layout).
"""
import re
import statistics
import unicodedata
from typing import Any, Dict, List, Optional, Sequence, Tuple

_SEP = r"\s?[.,]\s?"
_AMOUNT = re.compile(
    r"(?<![\d])(\d{1,3}(?:" + _SEP + r"\d{3})+|\d{1,3}\s\d00(?!\d))(?:" + _SEP + r"0{2}(?!\d))?")
_DECIMAL_TAIL = re.compile(_SEP + r"0{2}$")
_BARCODE = re.compile(r"\b\d{6,14}\b")
_SMALL_INT = re.compile(r"(?<![\d.,])(\d{1,3})(?![\d.,])")
_NUMBER_TOKEN = re.compile(r"[\d.,]+")
_DECIMAL_QTY = re.compile(r"(?<![\d.,])(\d{1,2})\s?[.,]\s?(\d{1,3})(?!\d)")
_PER_KG = re.compile(r"/\s*kg\b", re.IGNORECASE)
_WEIGHT = re.compile(r"(\d{1,3})\s?[.,]\s?(\d{1,3})\s*kg\b", re.IGNORECASE)
_WEIGHT_ONLY = re.compile(r"\s*[xX]?\s*\d{1,3}\s?[.,]\s?\d{1,3}\s*kg\s*", re.IGNORECASE)
_MIN_SLOPE_WORDS = 10
_SLOPE_STEP = 0.005                  # slopes tried: -0.2 .. 0.2 (about +-11 degrees)
_SLOPE_STEPS = 40
_SLOPE_GAIN = 1.10                   # a slope must stack the words 10 % better than horizontal to be used

_HEADER_WORDS = ("ten hang", "ten mon", "mat hang", "hang hoa", "sl", "so luong", "don gia", "thanh tien", "t tien",
                 "dvt", "gia", "tt")
_END_WORDS = ("tong", "cong", "thanh toan", "phai tra", "phai t toan")
_TOP_WORDS = ("hoa don", "ngay", "gio", "thu ngan", "quay", "so hd", "phieu", "nvbh", "ban hang")
_NAME_STOP = tuple(w for w in _TOP_WORDS if w != "gio")     # "giò heo" (pork) loses its accent and reads "gio"
_STRONG_HEADER = ("thanh tien", "don gia", "ten hang", "mat hang", "so luong")
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
def _profile_score(boxes: Sequence[Sequence[float]], k: float, bin_h: float) -> float:
    """Projection profile: sum of squared counts of the word centres binned along the direction of slope ``k``."""
    counts: Dict[int, int] = {}
    for b in boxes:
        key = int(((b[1] + b[3]) / 2 - k * (b[0] + b[2]) / 2) // bin_h)
        counts[key] = counts.get(key, 0) + 1
    return float(sum(c * c for c in counts.values()))


def text_slope(boxes: Sequence[Sequence[float]]) -> float:
    """Slope dy/dx of the text lines, by the projection profile (the slope that stacks the words into the fewest, fullest
    rows). Returns 0 unless a slope beats the horizontal clearly, so a straight receipt is grouped exactly as before."""
    if len(boxes) < _MIN_SLOPE_WORDS:
        return 0.0
    bin_h = max(1.0, 0.5 * statistics.median(b[3] - b[1] for b in boxes))
    flat = _profile_score(boxes, 0.0, bin_h)
    best_k, best = 0.0, flat
    for step in range(-_SLOPE_STEPS, _SLOPE_STEPS + 1):
        k = step * _SLOPE_STEP
        score = _profile_score(boxes, k, bin_h)
        if score > best:
            best_k, best = k, score
    return best_k if best >= _SLOPE_GAIN * flat else 0.0


def _is_number(text: str) -> bool:
    return bool(_NUMBER_TOKEN.fullmatch(text)) and any(c.isdigit() for c in text)


def merge_split_numbers(line: Sequence[Tuple[str, Sequence[float]]]) -> List[Tuple[str, Sequence[float]]]:
    """Join a number that OCR read as two overlapping boxes repeating the shared character.

    Examples from train OCR: ``"399"`` + ``"9,000"`` -> ``"399,000"``, ``"345,"`` + ``",000"`` -> ``"345,000"``. The two
    boxes must overlap and the last character of the left box must be the first character of the right one (the
    character both crops saw) and be a separator or be followed by one, so ``"10"`` + ``"000"`` (the amount "10 000"
    written with a space) stays as it is. ``line`` is in reading order (left to right).
    """
    out: List[Tuple[str, Sequence[float]]] = []
    for w, b in line:
        if out:
            pw, pb = out[-1]
            shared = pw[-1] == w[0] and (w[0] in ".," or (len(w) > 1 and w[1] in ".,"))
            if _is_number(pw) and _is_number(w) and b[0] < pb[2] <= b[2] and shared:
                out[-1] = (pw + w[1:], [pb[0], min(pb[1], b[1]), b[2], max(pb[3], b[3])])
                continue
        out.append((w, b))
    return out


def group_lines(words: Sequence[str], boxes: Sequence[Sequence[float]]) -> List[str]:
    """Reading-order text lines: words whose vertical centre is within ~0.6 of a word height of the line's mean.

    The vertical centre is measured along the text direction (``text_slope``), so a receipt photographed at a slant
    keeps the words of one printed line together. Split numbers are joined (``merge_split_numbers``).
    """
    k = text_slope(boxes)

    def level(b: Sequence[float]) -> float:
        return (b[1] + b[3]) / 2 - k * (b[0] + b[2]) / 2

    items = sorted(zip(words, boxes), key=lambda wb: level(wb[1]))
    lines: List[List[Tuple[str, Sequence[float]]]] = []
    cur: List[Tuple[str, Sequence[float]]] = []
    for w, b in items:
        cy = level(b)
        if cur:
            mean_y = sum(level(c[1]) for c in cur) / len(cur)
            mean_h = sum(c[1][3] - c[1][1] for c in cur) / len(cur)
            if abs(cy - mean_y) < max(8.0, 0.6 * mean_h):
                cur.append((w, b))
                continue
            lines.append(cur)
        cur = [(w, b)]
    if cur:
        lines.append(cur)
    return [" ".join(w for w, _ in merge_split_numbers(sorted(line, key=lambda wb: wb[1][0]))) for line in lines]


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


def _decimal_quantity(line: str, amounts: Sequence[Tuple[int, int, int, bool]]
                      ) -> Optional[Tuple[Dict[str, Any], List[Tuple[int, int, int, bool]]]]:
    """A weighed item without "/KG": "0.39 Kg 25,000 9,750", "0, 52 290,000 150,800", "1.214 20,500 24,887".

    A small decimal (0 < q < 100) is the quantity only when ``q * unit price == amount`` with the other amounts of the
    line; returns the parsed line and the amounts left once the decimal is removed (it may have been read as an amount,
    "1.214" -> 1214). ``None`` when no decimal passes the check.
    """
    for m in _DECIMAL_QTY.finditer(line):
        q = float(f"{m.group(1)}.{m.group(2)}")
        if not 0 < q < 100 or q == int(q):
            continue
        rest = [a for a in amounts if a[2] <= m.start() or a[1] >= m.end()]
        values = [a[0] for a in rest]
        if len(values) < 2:
            continue
        total = values[-1]
        for unit in reversed(values[:-1]):
            if unit and abs(q * unit - total) <= max(2.0, 0.005 * total):
                return {"quantity": q, "unit_price": unit, "amount": total, "line_ok": True}, rest
    return None


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
def _before_skip_phrase(line: str) -> Optional[str]:
    """The text before a "giá đăng ký" / "giá niêm yết" / "giá gốc" phrase inside the line, or None without one."""
    toks = line.split()
    for i in range(1, len(toks)):
        if any(plain(" ".join(toks[i:])).startswith(s) for s in _SKIP_START):
            return " ".join(toks[:i])
    return None


def _weight(text: str) -> Optional[float]:
    m = _WEIGHT.search(text)
    return float(f"{m.group(1)}.{m.group(2)}") if m else None


def _per_kg(line: str) -> Optional[Tuple[str, Optional[int], Optional[float]]]:
    """``(name, price per kg, weight)`` of a weighed-goods line such as "Bơ đặc biệt 52. 700/KG x 0,432 KG"."""
    m = _PER_KG.search(line)
    if m is None:
        return None
    before = line[:m.start()]
    amounts = parse_amounts(before)
    return _name_of(before, amounts), (amounts[-1][0] if amounts else None), _weight(line[m.end():])


def _header_index(lines: Sequence[str], lines_plain: Sequence[str]) -> Optional[int]:
    """First column-header line: two header words, or one strong header phrase on a line without amounts
    (OCR often garbles the other words: "S1. Đơn giú Thành tiền"). A "total" line is never the header
    ("Tổng số lượng hàng")."""
    for i, p in enumerate(lines_plain):
        if _has(p, _END_WORDS):
            continue
        if sum(1 for w in _HEADER_WORDS if f" {w} " in f" {p} ") >= 2:
            return i
        if _has(p, _STRONG_HEADER) and not parse_amounts(lines[i]):
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
    header = _header_index(lines, lp)
    start = header + 1 if header is not None else 0
    end = _end_index(lp, start)
    first = (start, end)
    while end is not None and not any(parse_amounts(lines[j]) for j in range(start, end)):
        # a "total" line with no amount line before it is not the end of the table (a heading, or a phrase the
        # fine-tuned OCR invents at the top of an unreadable crop): look for the end after it
        start, end = end + 1, _end_index(lp, end + 1)
    if end is None:                       # no later total line: keep the first one (a slip without an item table)
        start, end = first
    if header is None and end is not None:
        top = [i for i in range(start, end) if _has(lp[i], _TOP_WORDS)]
        start = (max(top) + 1) if top else start
    stop = end if end is not None else len(lines)

    items: List[Dict[str, Any]] = []
    pending = ""
    pending_kg: Optional[Tuple[Optional[int], Optional[float]]] = None
    for i in range(start, stop):
        line, p = lines[i], lp[i]
        if any(p.startswith(s) for s in _SKIP_START):
            continue
        kg = _per_kg(line)
        if kg is not None:                # "Rau muống 21 000/KG X 0, 464 KG": the amount is on the next line
            name, price, weight = kg
            pending, pending_kg = name or pending, (price, weight)
            continue
        if pending_kg is not None and _WEIGHT_ONLY.fullmatch(line):          # "0, 606 KG" on its own line
            pending_kg = (pending_kg[0], _weight(line))
            continue
        head = _before_skip_phrase(line)
        if head is not None:              # "Sữa Mộc Châu 180ml Giá đăng ký: 7,000": a slanted name joined the price line
            pending = _name_of(head, parse_amounts(head)) or pending
            continue
        amounts = parse_amounts(line)
        if not amounts:
            if sum(c.isalpha() for c in line) >= 3 and not _has(p, _NAME_STOP):
                pending = _name_of(line, []) or pending
            continue
        # a decimal quantity is tried first: "0.146 155,000 22,630" also passes as 155 x 146 = 22,630, because a weight
        # times a price per kg always has the digits of (price / 1000) x (weight x 1000)
        weighed = _decimal_quantity(line, amounts)
        if weighed is not None:
            parsed, amounts = weighed
        else:
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
            pending, pending_kg = "", None
            continue
        if pending_kg is not None and not discount:
            price, weight = pending_kg
            if price and weight and abs(price * weight - parsed["amount"]) <= max(2.0, 0.005 * parsed["amount"]):
                parsed.update(quantity=weight, unit_price=price, line_ok=True)
        items.append({"name": name, **parsed, "discount": bool(discount)})
        pending, pending_kg = "", None

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
