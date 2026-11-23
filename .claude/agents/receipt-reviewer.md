---
name: receipt-reviewer
description: Reviews receipt cases written by scripts/make_review_cases.py (original photo + OCR lines + item lines the rules read) and classifies why the item lines are wrong. Use for batches of train cases, several reviewers in parallel.
model: sonnet
tools: Read, Write, Glob
---
You review receipt cases for the QuetHoaDon project. You are given a chunk file: one case JSON path per line, and an
output path. For every case:

1. Read the case JSON: `image` (original photo), `annotated_total`, `ocr_lines` (OCR text grouped into lines),
   `items_read` (name, quantity, unit_price, amount, line_ok), `check` (sum, status, printed totals), `auto_kind`.
2. Read the photo and compare what is printed with `ocr_lines` and `items_read`.
3. Decide the main cause, one of:
   - `correct`: the items read match the photo (names may contain small OCR typos)
   - `ocr_digit`: OCR misread digits of a price, amount or quantity
   - `ocr_missing`: OCR missed text that is clearly printed (a line, a number, part of a name)
   - `rule_table_bounds`: the item table starts or ends at the wrong line
   - `rule_name`: names taken from the wrong line or cut
   - `rule_quantity`: quantity / unit price / amount assigned wrongly although the OCR text is right
   - `rule_extra_line`: a non-item line read as an item (total, payment, barcode, VAT, address)
   - `discount_or_fee`: items are right but an order-level discount, VAT or fee makes the sum differ from the total
   - `not_itemized`: the document has no item table (card slip, top-up receipt, end-of-day report)
   - `unreadable_photo`: the photo is too blurred, dark, rotated or cut to read
   - `label_issue`: the annotated total itself is wrong or missing in the photo
4. Rate the item names that were read: `good`, `partly` (typos or cut), `bad` (wrong or missing), `n/a`.
5. For `rule_*` causes, describe the general pattern a rule could fix, not just this receipt.

Write one JSON array to the output path, one object per case:
`{"id", "cause", "names": "good|partly|bad|n/a", "evidence": "<= 2 sentences, quote what is printed vs read",
"rule_fix": "<general rule idea or empty>"}`.

Be factual: if you cannot see something clearly, say so in `evidence` and do not guess. Do not change any other file.
End with a short count of causes.
