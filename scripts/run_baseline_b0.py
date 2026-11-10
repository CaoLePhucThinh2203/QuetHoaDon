"""Baseline B0 (Week-3 pipeline): frozen LayoutXLM features -> K-means -> clusters mapped to fields (GPU for step 1).

1. embeds the train and the evaluated split with the frozen encoder (cached as .npz in ``--cache``);
2. for every ``k`` in ``--ks``: fits scaler + PCA + K-means on train words, maps clusters to labels, measures the
   Silhouette score on the evaluated split's words and the field F1 (lenient + soft) for information;
3. keeps the ``k`` with the best Silhouette (the Week-3 selection rule) and writes the same full report as
   ``evaluate_layoutxlm.py`` (``--out``), plus ``<out>.selection.json`` with every ``k``.

The test split needs ``--allow-test`` and reuses the ``k`` chosen on validation (``--k``).

Example:
    python scripts/run_baseline_b0.py --token-dir <token_dataset_v1> --image-dir <preprocessed_images> \
        --cache /kaggle/working/b0_cache --out /kaggle/working/eval/b0_val.json
"""
import argparse
import json
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from src.baseline.b0 import cluster_summary, embed_records, fit_b0, predict_b0, silhouette  # noqa: E402
from src.training.data import load_jsonl  # noqa: E402
from src.training.field_eval import build_report, format_report  # noqa: E402


def _json_default(o):
    return o.item() if hasattr(o, "item") else str(o)


def cached_embeddings(name, records, args):
    path = Path(args.cache) / f"{name}.npz"
    if path.exists():
        data = np.load(path)
        print(f"[b0] loaded cached embeddings {path}")
        return [data[str(i)] for i in range(len(records))]
    t0 = time.time()
    emb = embed_records(records, args.image_dir, batch_size=args.batch_size, num_workers=args.num_workers)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **{str(i): e.astype(np.float16) for i, e in enumerate(emb)})
    print(f"[b0] embedded {name}: {len(records)} receipts in {time.time() - t0:.0f}s -> {path}")
    return emb


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--token-dir", required=True)
    ap.add_argument("--image-dir", required=True)
    ap.add_argument("--cache", required=True, help="Folder for cached embeddings and the fitted B0 model")
    ap.add_argument("--out", required=True, help="JSON report of the selected k")
    ap.add_argument("--split", choices=["validation", "test"], default="validation")
    ap.add_argument("--allow-test", action="store_true")
    ap.add_argument("--ks", default="8,12,16,24,32,48", help="Cluster counts to try (validation)")
    ap.add_argument("--k", type=int, default=None, help="Use this k only (e.g. the one chosen on validation)")
    ap.add_argument("--pca-dim", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()
    if args.split == "test" and not args.allow_test:
        sys.exit("The test split is scored once, after the configuration is frozen. Re-run with --allow-test.")

    token_dir = Path(args.token_dir)
    train = load_jsonl(str(token_dir / "train.jsonl"))
    evald = load_jsonl(str(token_dir / f"{args.split}.jsonl"))
    emb_train = cached_embeddings("train", train, args)
    emb_eval = cached_embeddings(args.split, evald, args)
    emb_train = [e.astype(np.float32) for e in emb_train]
    emb_eval = [e.astype(np.float32) for e in emb_eval]

    ks = [args.k] if args.k else [int(k) for k in args.ks.split(",")]
    from src.training.field_eval import decode_predictions_soft, gold_fields, lenient_scores

    selection, best = [], None
    for k in ks:
        t0 = time.time()
        model = fit_b0(emb_train, train, k, pca_dim=args.pca_dim)
        merged, x, c = predict_b0(model, emb_eval, evald)
        sil = silhouette(x, c)
        scores = lenient_scores(decode_predictions_soft(evald, merged), gold_fields(evald))
        macro = float(np.mean([s["f1"] for s in scores.values()]))
        row = {"k": k, "silhouette": sil, "lenient_soft_macro_f1": macro,
               "field_f1": {f: s["f1"] for f, s in scores.items()}, "seconds": round(time.time() - t0, 1)}
        selection.append(row)
        print(f"[b0] k={k:>3}  silhouette={sil if sil is None else round(sil, 4)}  macro F1={macro:.3f}  "
              + "  ".join(f"{f}={s['f1']:.3f}" for f, s in scores.items()))
        if best is None or (sil or -1) > (best[0] or -1):
            best = (sil, k, model, merged)

    sil, k, model, merged = best
    print(f"[b0] selected k={k} (best silhouette {sil})")
    with open(Path(args.cache) / f"b0_k{k}.pkl", "wb") as f:
        pickle.dump(model, f)

    report = build_report(evald, merged, n_boot=args.n_boot)
    report.update({"split": args.split, "system": "B0", "k": k, "silhouette": sil, "selection": selection,
                   "clusters": cluster_summary(model)})
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=_json_default)
    print(format_report(report))
    print("full report:", args.out)


if __name__ == "__main__":
    main()
