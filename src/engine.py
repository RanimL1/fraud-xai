"""Moteur d'inférence et d'explication partagé (utilisé par l'application Streamlit)."""
from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ART_DIR = Path(__file__).resolve().parents[1] / "artifacts"
OTHER = "(autre)"


class Engine:
    def __init__(self, art_dir: Path = ART_DIR):
        art_dir = Path(art_dir)
        if not (art_dir / "model.txt").exists():
            raise FileNotFoundError("Artefacts introuvables : lancez d'abord `python train.py` ou le notebook.")
        self.meta = json.loads((art_dir / "meta.json").read_text(encoding="utf-8"))
        self.feats: list[str] = self.meta["features"]
        self.enc: dict[str, list[str]] = self.meta["encoders"]
        self.metrics: dict = self.meta["metrics"]
        self.thr: float = self.metrics["threshold"]
        self.booster = lgb.Booster(model_file=str(art_dir / "model.txt"))
        self.sample = pd.read_parquet(art_dir / "sample.parquet")

    # -- valeurs lisibles <-> encodées
    def decode(self, col: str, v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return None
        if col in self.enc:
            i = int(v)
            return self.enc[col][i] if i < len(self.enc[col]) else OTHER
        return round(float(v), 3)

    def encode(self, col: str, v) -> float:
        if v is None or str(v).strip().lower() in ("", "none", "nan", "manquant"):
            return np.nan
        if col in self.enc:
            s = str(v)
            return float(self.enc[col].index(s)) if s in self.enc[col] else float(len(self.enc[col]))
        try:
            return float(v)
        except ValueError as e:
            raise ValueError(f"Valeur numérique attendue pour {col} : {v!r}") from e

    # -- explication TreeSHAP exacte (log-odds)
    def explain(self, row: pd.Series, top: int = 12, force: tuple = ()) -> dict:
        x = row[self.feats].to_numpy(dtype="float32").reshape(1, -1)
        contrib = self.booster.predict(x, pred_contrib=True)[0]
        base, phi = float(contrib[-1]), contrib[:-1]
        proba = float(1 / (1 + np.exp(-(base + phi.sum()))))
        idx = list(np.argsort(-np.abs(phi))[:top])
        idx += [self.feats.index(f) for f in force if f in self.feats and self.feats.index(f) not in idx]
        items = [{"feature": self.feats[i], "value": self.decode(self.feats[i], float(x[0, i])),
                  "shap": float(phi[i])} for i in idx]
        others = float(phi.sum() - sum(it["shap"] for it in items))
        return {"proba": proba, "base_proba": float(1 / (1 + np.exp(-base))), "flagged": proba >= self.thr,
                "items": items, "others": others}

    def explain_idx(self, idx: int, changes: dict | None = None) -> dict:
        row = self.sample.iloc[idx].copy()
        for col, v in (changes or {}).items():
            row[col] = self.encode(col, v)
        out = self.explain(row, force=tuple(changes or {}))
        out["fraud"] = int(self.sample.iloc[idx].isFraud)
        return out

    # -- liste de transactions
    def transactions(self, label="all", flagged="all", sort="proba", limit=40, seed=0) -> pd.DataFrame:
        d = self.sample
        if label in ("fraud", "legit"):
            d = d[d.isFraud == (1 if label == "fraud" else 0)]
        if flagged in ("yes", "no"):
            d = d[(d.proba >= self.thr) == (flagged == "yes")]
        if sort == "random":
            d = d.sample(min(limit, len(d)), random_state=seed)
        else:
            key = {"proba": "proba", "amount": "TransactionAmt", "time": "TransactionDT"}[sort]
            d = d.sort_values(key, ascending=(sort == "time")).head(limit)
        return pd.DataFrame({
            "idx": d.index, "ID": d.TransactionID.astype(int).to_numpy(),
            "Montant ($)": d.TransactionAmt.round(2).to_numpy(),
            "Produit": [self.decode("ProductCD", float(v)) for v in d.ProductCD],
            "Score": d.proba.round(3).to_numpy(),
            "Signalée": np.where(d.proba >= self.thr, "oui", "non"),
            "Réel": np.where(d.isFraud == 1, "fraude", "légitime"),
        }).reset_index(drop=True)
