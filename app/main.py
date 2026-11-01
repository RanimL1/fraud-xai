"""API FastAPI : exploration des transactions, explications SHAP locales, scénarios « et si ? »."""
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ART = Path(__file__).resolve().parents[1] / "artifacts"
STATIC = Path(__file__).parent / "static"
if not (ART / "model.txt").exists():
    raise SystemExit("Artefacts introuvables : lancez d'abord `python train.py` (ou le notebook).")

META = json.loads((ART / "meta.json").read_text(encoding="utf-8"))
FEATS: list[str] = META["features"]
ENC: dict[str, list[str]] = META["encoders"]
THR: float = META["metrics"]["threshold"]
BOOSTER = lgb.Booster(model_file=str(ART / "model.txt"))
SAMPLE = pd.read_parquet(ART / "sample.parquet")

app = FastAPI(title="Fraude explicable — IEEE-CIS")
app.mount("/static", StaticFiles(directory=STATIC), name="static")


def decode(col: str, v):
    """Valeur encodée -> valeur lisible."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    if col in ENC:
        i = int(v)
        return ENC[col][i] if i < len(ENC[col]) else "(autre)"
    return round(float(v), 3)


def encode(col: str, v):
    if v is None or str(v).strip() in ("", "None", "manquant"):
        return np.nan
    if col in ENC:
        s = str(v)
        return float(ENC[col].index(s)) if s in ENC[col] else float(len(ENC[col]))
    try:
        return float(v)
    except ValueError:
        raise HTTPException(400, f"Valeur numérique attendue pour {col} : {v!r}")


def explain(row: pd.Series, top: int = 12, force: tuple = ()):
    x = row[FEATS].to_numpy(dtype="float32").reshape(1, -1)
    contrib = BOOSTER.predict(x, pred_contrib=True)[0]  # TreeSHAP exact, en log-odds
    base, phi = float(contrib[-1]), contrib[:-1]
    proba = float(1 / (1 + np.exp(-(base + phi.sum()))))
    idx = list(np.argsort(-np.abs(phi))[:top])
    idx += [FEATS.index(f) for f in force if f in FEATS and FEATS.index(f) not in idx]
    items = [{"feature": FEATS[i], "value": decode(FEATS[i], float(x[0, i])), "shap": float(phi[i])} for i in idx]
    others = float(phi.sum() - sum(it["shap"] for it in items))
    return {"proba": proba, "base_proba": float(1 / (1 + np.exp(-base))), "flagged": proba >= THR,
            "threshold": THR, "items": items, "others": others}


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/summary")
def summary():
    return {"metrics": META["metrics"], "importance": META["importance"][:20],
            "n_sample": len(SAMPLE), "note": META["sample_note"]}


@app.get("/api/transactions")
def transactions(label: str = "all", flagged: str = "all", sort: str = "proba", limit: int = 30):
    d = SAMPLE
    if label in ("fraud", "legit"):
        d = d[d.isFraud == (1 if label == "fraud" else 0)]
    if flagged in ("yes", "no"):
        d = d[(d.proba >= THR) == (flagged == "yes")]
    limit = max(1, min(limit, 200))
    if sort == "random":
        d = d.sample(min(limit, len(d)), random_state=None)
    else:
        key = {"proba": "proba", "amount": "TransactionAmt", "time": "TransactionDT"}.get(sort, "proba")
        d = d.sort_values(key, ascending=(sort == "time")).head(limit)
    return [{"idx": int(i), "id": int(r.TransactionID), "amount": round(float(r.TransactionAmt), 2),
             "product": decode("ProductCD", float(r.ProductCD)), "proba": float(r.proba),
             "flagged": bool(r.proba >= THR), "fraud": int(r.isFraud)} for i, r in d.iterrows()]


@app.get("/api/explain/{idx}")
def explain_one(idx: int):
    if not 0 <= idx < len(SAMPLE):
        raise HTTPException(404, "Transaction inconnue")
    row = SAMPLE.iloc[idx]
    return {**explain(row), "fraud": int(row.isFraud), "amount": float(row.TransactionAmt)}


class WhatIf(BaseModel):
    idx: int
    changes: dict[str, str | float | None]


@app.post("/api/whatif")
def whatif(req: WhatIf):
    if not 0 <= req.idx < len(SAMPLE):
        raise HTTPException(404, "Transaction inconnue")
    row = SAMPLE.iloc[req.idx].copy()
    for col, v in req.changes.items():
        if col not in FEATS:
            raise HTTPException(400, f"Variable inconnue : {col}")
        row[col] = encode(col, v)
    return {**explain(row, force=tuple(req.changes)), "fraud": int(SAMPLE.iloc[req.idx].isFraud)}
