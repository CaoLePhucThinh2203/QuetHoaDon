"""Training of the fine-tuned LayoutXLM token classifier (model M1b; M1a freezes the encoder).

``windows`` and ``seqmetrics`` are pure Python / NumPy and unit-tested anywhere. ``data`` and ``engine`` need
torch and transformers (Colab GPU) and are imported explicitly.
"""
