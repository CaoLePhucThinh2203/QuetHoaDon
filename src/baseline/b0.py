"""Baseline B0 = the Week-3 pipeline: frozen LayoutXLM features -> clustering -> clusters mapped to fields.

No gradient is used. Steps (fit on the train split only):
1. word embedding = frozen LayoutXLM hidden state of the word's first sub-token (sliding windows averaged);
2. standardise, PCA to ``pca_dim`` dimensions, append hand-crafted layout / format features (docx section 2:
   relative position, box size, amount / date shape, digit ratio, total keyword nearby);
3. K-means with ``k`` clusters; ``k`` is chosen by the Silhouette score on validation words;
4. each cluster gets the label distribution of its *training* words (majority mapping, smoothed), which is used as
   the word "probabilities", so B0 is scored by exactly the same code as the fine-tuned models
   (``src.training.field_eval.build_report``, including the keyword / format rules that fill empty fields).

The numerical parts are pure NumPy / scikit-learn; ``embed_records`` needs torch + transformers (GPU).
"""
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.extraction.decode import amount_candidates
from src.extraction.rules import fold
from src.preprocessing.token_dataset import LABEL2ID, LABELS

N_HANDCRAFTED = 10
_TOTAL_HINTS = ("tong", "thanhtoan", "total", "cong")
_DATE_CHARS = set("0123456789/-.:")


def handcrafted_features(rec: Dict[str, Any]) -> np.ndarray:
    """``[n_words, N_HANDCRAFTED]`` layout and format features of one receipt (all roughly in [0, 1])."""
    words: Sequence[str] = rec["words"]
    boxes: Sequence[Sequence[int]] = rec["boxes"]
    n = len(words)
    feats = np.zeros((n, N_HANDCRAFTED), dtype=np.float32)
    if n == 0:
        return feats
    in_amount = np.zeros(n, dtype=bool)
    for s, e, _, _ in amount_candidates(words):
        in_amount[s:e] = True
    folded = [fold(w) for w in words]
    for i, (w, b) in enumerate(zip(words, boxes)):
        x0, y0, x1, y1 = b
        digits = sum(c.isdigit() for c in w)
        letters = [c for c in w if c.isalpha()]
        recent = "".join(folded[max(0, i - 3):i])
        feats[i] = (
            (y0 + y1) / 2000.0,                                     # relative vertical position
            (x0 + x1) / 2000.0,                                     # relative horizontal position
            (x1 - x0) / 1000.0,                                     # box width
            (y1 - y0) / 1000.0,                                     # box height
            i / max(1, n - 1),                                      # reading-order rank
            float(in_amount[i]),                                    # amount-shaped (incl. OCR-split amounts)
            float(digits >= 4 and set(w) <= _DATE_CHARS and ("/" in w or "-" in w)),   # date-shaped
            digits / max(1, len(w)),                                # digit ratio
            sum(c.isupper() for c in letters) / max(1, len(letters)),                  # upper-case ratio
            float(any(h in recent for h in _TOTAL_HINTS)),          # total keyword in the 3 previous words
        )
    return feats


def receipt_matrix(embeddings: np.ndarray, rec: Dict[str, Any]) -> np.ndarray:
    return np.concatenate([embeddings.astype(np.float32), handcrafted_features(rec)], axis=1)


def fit_b0(train_emb: Sequence[np.ndarray], train_records: Sequence[Dict[str, Any]], k: int, pca_dim: int = 64,
           hand_weight: float = 2.0, seed: int = 42, alpha: float = 1.0) -> Dict[str, Any]:
    """Fit scaler + PCA + K-means on the training words and map every cluster to a label distribution."""
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    emb = np.concatenate(list(train_emb), axis=0).astype(np.float32)
    hand = np.concatenate([handcrafted_features(r) for r in train_records], axis=0)
    labels = np.array([LABEL2ID[t] for r in train_records for t in r["labels"]])
    scaler = StandardScaler().fit(emb)
    pca = PCA(n_components=pca_dim, random_state=seed).fit(scaler.transform(emb))
    hand_scaler = StandardScaler().fit(hand)
    model = {"scaler": scaler, "pca": pca, "hand_scaler": hand_scaler, "hand_weight": hand_weight,
             "k": k, "seed": seed}
    x = transform(model, emb, hand)
    km = MiniBatchKMeans(n_clusters=k, random_state=seed, n_init=5, batch_size=4096).fit(x)
    model["kmeans"] = km
    counts = np.zeros((k, len(LABELS)), dtype=np.float64)
    np.add.at(counts, (km.labels_, labels), 1.0)
    model["cluster_counts"] = counts
    model["cluster_probs"] = (counts + alpha / len(LABELS)) / (counts.sum(axis=1, keepdims=True) + alpha)
    return model


def transform(model: Dict[str, Any], emb: np.ndarray, hand: np.ndarray) -> np.ndarray:
    z = model["pca"].transform(model["scaler"].transform(emb.astype(np.float32)))
    h = model["hand_scaler"].transform(hand) * model["hand_weight"]
    return np.concatenate([z, h], axis=1).astype(np.float32)


def predict_b0(model: Dict[str, Any], emb: Sequence[np.ndarray], records: Sequence[Dict[str, Any]]
               ) -> Tuple[List[Dict[str, Any]], np.ndarray, np.ndarray]:
    """``merged`` (word probabilities per receipt, as the fine-tuned models produce), all features, cluster ids."""
    merged, xs, cs = [], [], []
    for e, rec in zip(emb, records):
        x = transform(model, e, handcrafted_features(rec))
        c = model["kmeans"].predict(x) if len(x) else np.zeros(0, dtype=np.int64)
        merged.append({"probs": model["cluster_probs"][c], "covered": np.ones(len(c), dtype=bool)})
        xs.append(x)
        cs.append(c)
    x_all = np.concatenate(xs, axis=0) if xs else np.zeros((0, 1))
    c_all = np.concatenate(cs, axis=0) if cs else np.zeros(0, dtype=np.int64)
    return merged, x_all, c_all


def silhouette(x: np.ndarray, clusters: np.ndarray, sample: int = 5000, seed: int = 42) -> Optional[float]:
    from sklearn.metrics import silhouette_score

    if len(np.unique(clusters)) < 2:
        return None
    return float(silhouette_score(x, clusters, sample_size=min(sample, len(x)), random_state=seed))


def cluster_summary(model: Dict[str, Any], top: int = 3) -> List[Dict[str, Any]]:
    """Size and dominant training labels of every cluster (for the report)."""
    out = []
    for c, row in enumerate(model["cluster_counts"]):
        order = np.argsort(-row)[:top]
        out.append({"cluster": c, "train_words": int(row.sum()),
                    "top_labels": {LABELS[i]: round(float(row[i] / max(1.0, row.sum())), 3) for i in order}})
    return out


# ---------------------------------------------------------------------------------- frozen encoder (GPU)
def embed_records(records: Sequence[Dict[str, Any]], image_dir: str, model_name: str = "microsoft/layoutxlm-base",
                  max_tokens: int = 480, overlap_tokens: int = 100, batch_size: int = 8, num_workers: int = 2
                  ) -> List[np.ndarray]:
    """Frozen LayoutXLM embedding of every word (first sub-token, averaged over overlapping windows)."""
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoProcessor, LayoutLMv2Model

    from src.training.data import WindowDataset, build_windows
    from src.training.windows import merge_window_probs

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(model_name, apply_ocr=False)
    encoder = LayoutLMv2Model.from_pretrained(model_name).to(device).eval()
    windows = build_windows(records, processor.tokenizer, max_tokens, overlap_tokens)
    ds = WindowDataset(records, windows, processor, image_dir)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    per_doc: Dict[int, List[Tuple[int, np.ndarray]]] = {}
    with torch.inference_mode():
        for batch in loader:
            labels = batch["labels"]
            inputs = {k: v.to(device) for k, v in batch.items() if k not in ("labels", "doc", "start")}
            with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
                hidden = encoder(**inputs).last_hidden_state
            hidden = hidden[:, :labels.shape[1]].float().cpu().numpy()      # text positions (visual tokens follow)
            for b in range(labels.shape[0]):
                pos = (labels[b] != -100).nonzero(as_tuple=True)[0].numpy()
                per_doc.setdefault(int(batch["doc"][b]), []).append((int(batch["start"][b]), hidden[b][pos]))
    dim = encoder.config.hidden_size
    out = []
    for d, rec in enumerate(records):
        emb, _ = merge_window_probs(len(rec["words"]), dim, per_doc.get(d, []))
        out.append(emb.astype(np.float32))
    return out
