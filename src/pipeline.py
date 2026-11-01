"""Pipeline de détection de fraude (IEEE-CIS / Vesta) : chargement, variables,
découpage temporel, encodage, entraînement LightGBM, évaluation, export des artefacts."""
from __future__ import annotations

import gc
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             precision_recall_curve, roc_auc_score)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
ART_DIR = ROOT / "artifacts"
FP_COST = 5.0  # coût d'une fausse alerte (revue manuelle + friction client), en $

PARAMS = dict(
    objective="binary", n_estimators=3000, learning_rate=0.03, num_leaves=127,
    min_child_samples=50, subsample=0.8, subsample_freq=1, colsample_bytree=0.5,
    reg_lambda=1.0, n_jobs=-1, random_state=42, verbose=-1,
)


# --------------------------------------------------------------------------- données
def _read_light(path: Path, nrows: int | None = None) -> pd.DataFrame:
    """Lecture par blocs avec réduction des types (float64 -> float32, int64 -> int32) :
    évite de matérialiser tout le fichier en 64 bits (≈ 2x moins de mémoire)."""
    parts = []
    for ch in pd.read_csv(path, nrows=nrows, chunksize=100_000):
        types = {}
        for col, dt in ch.dtypes.items():
            if dt == np.float64 and col != "TransactionAmt":
                types[col] = "float32"
            elif dt == np.int64:
                types[col] = "int32"
        parts.append(ch.astype(types))
    return pd.concat(parts, ignore_index=True)


def _light_dtypes(path: Path) -> dict:
    """Types économes (float32 / int32) déduits d'un petit échantillon du CSV."""
    head = pd.read_csv(path, nrows=2000)
    out = {}
    for c, t in head.dtypes.items():
        if t == "float64" and c != "TransactionAmt":
            out[c] = "float32"
        elif t == "int64" and c not in ("TransactionDT", "TransactionID"):
            out[c] = "int32"
    return out


def load_data(data_dir: Path = DATA_DIR, nrows: int | None = None) -> pd.DataFrame:
    """Fusionne transactions + identité en types légers (~2x moins de RAM), trié par temps."""
    data_dir = Path(data_dir)
    p_tr, p_id = data_dir / "train_transaction.csv", data_dir / "train_identity.csv"
    tr = pd.read_csv(p_tr, nrows=nrows, dtype=_light_dtypes(p_tr))
    idt = pd.read_csv(p_id, dtype=_light_dtypes(p_id))
    df = tr.merge(idt, on="TransactionID", how="left")
    del tr, idt
    if not df["TransactionDT"].is_monotonic_increasing:   # le fichier Kaggle est déjà trié : évite une copie
        df = df.sort_values("TransactionDT", kind="stable").reset_index(drop=True)
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Variables dérivées. Celles qui agrègent l'historique n'utilisent que le passé
    (le tableau est trié par temps), donc pas de fuite d'information vers le futur."""
    dt = df["TransactionDT"]
    df["hour"] = ((dt // 3600) % 24).astype("float32")
    df["weekday"] = ((dt // 86400) % 7).astype("float32")
    df["amt_log"] = np.log1p(df["TransactionAmt"]).astype("float32")
    df["amt_cents"] = ((df["TransactionAmt"] * 100).round() % 100).astype("float32")
    both = df["P_emaildomain"].notna() & df["R_emaildomain"].notna()
    df["email_match"] = (df["P_emaildomain"] == df["R_emaildomain"]).astype("float32").where(both)

    g = df.groupby("card1")
    n_prev = g.cumcount()
    prev_mean = (g["TransactionAmt"].cumsum() - df["TransactionAmt"]) / n_prev.replace(0, np.nan)
    df["card_n_prev"] = n_prev.astype("float32")
    df["amt_vs_card_mean"] = (df["TransactionAmt"] / prev_mean).astype("float32")
    df["card_dt_since_last"] = g["TransactionDT"].diff().astype("float32")
    return df


def temporal_split(df: pd.DataFrame, test_frac: float = 0.2, val_frac: float = 0.1):
    """Train / validation / test dans l'ordre chronologique (jamais de mélange aléatoire)."""
    n = len(df)
    train_end = int(n * (1 - test_frac))
    val_start = int(train_end * (1 - val_frac))
    return df.iloc[:val_start], df.iloc[val_start:train_end], df.iloc[train_end:]


# ---------------------------------------------------------------------- encodage
def _is_text(s: pd.Series) -> bool:
    return s.dtype == object or pd.api.types.is_string_dtype(s)


def fit_encoders(train: pd.DataFrame, min_count: int = 100) -> dict[str, list[str]]:
    """Pour chaque colonne texte : catégories fréquentes (>= min_count) vues en train.
    Les autres valeurs sont regroupées dans une modalité « autre » (code = len(liste))."""
    enc = {}
    for c in train.columns:
        if _is_text(train[c]):
            vc = train[c].value_counts()
            enc[c] = [str(v) for v in vc[vc >= min_count].index]
    return enc


def apply_encoders(df: pd.DataFrame, enc: dict[str, list[str]]) -> pd.DataFrame:
    out = df.copy()
    for c, keep in enc.items():
        s = out[c].astype("object")
        codes = s.map({v: i for i, v in enumerate(keep)})
        codes = codes.where(~(codes.isna() & s.notna()), len(keep))
        out[c] = codes.astype("float32")
    return out


def prepare(train, val, test):
    drop = {"isFraud", "TransactionID", "TransactionDT"}
    feats = [c for c in train.columns if c not in drop]
    enc = fit_encoders(train[feats])
    mats = [apply_encoders(d[feats], enc).astype("float32") for d in (train, val, test)]
    return feats, enc, mats


# -------------------------------------------------------------------- modèle / seuils
def train_model(Xtr, ytr, Xva, yva, cat_cols, params: dict | None = None):
    model = lgb.LGBMClassifier(**{**PARAMS, **(params or {})})
    model.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric="auc",
              categorical_feature=list(cat_cols),
              callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)])
    return model


def best_f1_threshold(y, p) -> float:
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-12)
    return float(thr[int(np.argmax(f1))])


def total_cost(y, p, amount, thr, fp_cost: float = FP_COST) -> float:
    """Coût métier : fraude non détectée = montant perdu ; fausse alerte = fp_cost."""
    pred = p >= thr
    fn = (~pred) & (y == 1)
    fp = pred & (y == 0)
    return float(amount[fn].sum() + fp_cost * fp.sum())


def cost_threshold(y, p, amount, fp_cost: float = FP_COST) -> float:
    grid = np.linspace(0.01, 0.99, 99)
    costs = [total_cost(y, p, amount, t, fp_cost) for t in grid]
    return float(grid[int(np.argmin(costs))])


def evaluate(y, p, amount, thr_f1: float, thr_cost: float, fp_cost: float = FP_COST) -> dict:
    y, p, amount = np.asarray(y), np.asarray(p), np.asarray(amount)
    pred = p >= thr_f1
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    cost_none = float(amount[y == 1].sum())
    cost_model = total_cost(y, p, amount, thr_cost, fp_cost)
    return {
        "roc_auc": float(roc_auc_score(y, p)), "pr_auc": float(average_precision_score(y, p)),
        "prevalence": float(y.mean()), "threshold": float(thr_f1), "threshold_cost": float(thr_cost),
        "precision": float(prec), "recall": float(rec), "f1": float(2 * prec * rec / max(prec + rec, 1e-12)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "cost_no_model": cost_none, "cost_with_model": cost_model,
        "cost_reduction": float(1 - cost_model / cost_none) if cost_none else 0.0,
    }


# ------------------------------------------------------------------------- export
def export_artifacts(model, enc, feats, X_test, df_test, metrics, art_dir: Path = ART_DIR,
                     n_sample: int = 20000, seed: int = 42) -> Path:
    """Sauvegarde modèle, métadonnées, importance globale SHAP et un échantillon de test
    (enrichi en fraudes pour la démo) que l'application web explore."""
    art_dir = Path(art_dir)
    art_dir.mkdir(parents=True, exist_ok=True)
    booster = model.booster_
    booster.save_model(str(art_dir / "model.txt"))

    rng = np.random.default_rng(seed)
    y = df_test["isFraud"].to_numpy()
    fraud, legit = np.where(y == 1)[0], np.where(y == 0)[0]
    n_f = min(len(fraud), n_sample // 5)
    pick = np.sort(np.concatenate([
        rng.choice(fraud, n_f, replace=False),
        rng.choice(legit, min(len(legit), n_sample - n_f), replace=False)]))

    proba = model.predict_proba(X_test)[:, 1]
    sample = X_test.iloc[pick].reset_index(drop=True)
    sample["isFraud"] = y[pick]
    sample["proba"] = proba[pick].astype("float32")
    sample["TransactionID"] = df_test["TransactionID"].to_numpy()[pick]
    sample["TransactionDT"] = df_test["TransactionDT"].to_numpy()[pick]
    sample.to_parquet(art_dir / "sample.parquet")

    ridx = rng.choice(len(X_test), min(5000, len(X_test)), replace=False)
    contrib = booster.predict(X_test.iloc[ridx].to_numpy(dtype="float32"), pred_contrib=True)[:, :-1]
    imp = np.abs(contrib).mean(0)
    order = np.argsort(-imp)[:30]
    importance = [{"feature": feats[i], "mean_abs_shap": float(imp[i])} for i in order]

    meta = {"features": feats, "encoders": enc, "metrics": metrics, "importance": importance,
            "sample_note": "Échantillon de test enrichi en fraudes (~20 %) pour la démonstration."}
    (art_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    return art_dir
