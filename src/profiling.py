"""Profil du projet pour l'application Streamlit : statistiques, courbes et échantillon SHAP.

Aucune donnée brute n'est conservée : seulement des agrégats (histogrammes, taux, courbes), ce qui permet
de publier l'application sans publier les CSV Kaggle. Appelé par train.py et par le notebook."""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve

import pipeline as P

FAMILIES = [(r"V\d+", "V (Vesta)"), (r"C\d+", "C (comptages)"), (r"D\d+", "D (délais)"), (r"M\d", "M (correspondances)"),
            (r"id_\d+", "id (identité)"), (r"card\d", "card"), (r"addr\d", "addr"), (r"dist\d", "dist")]


def family(c: str) -> str:
    for pat, name in FAMILIES:
        if re.fullmatch(pat, c):
            return name
    return "autres"


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return None if not np.isfinite(o) else round(float(o), 6)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def _hist(v, bins=40, rng=None):
    c, e = np.histogram(np.asarray(v, dtype="float64"), bins=bins, range=rng)
    return {"edges": e.tolist(), "counts": c.tolist()}


def _hist_by_class(v, y, bins=40, clip=(0.01, 0.99)):
    v = np.asarray(v, dtype="float64")
    ok = ~np.isnan(v)
    lo, hi = np.quantile(v[ok], clip)
    if hi <= lo:
        hi = lo + 1.0
    edges = np.linspace(lo, hi, bins + 1)
    out = {"edges": edges.tolist()}
    for name, cls in (("legit", 0), ("fraud", 1)):
        x = np.clip(v[ok & (y == cls)], lo, hi)
        d, _ = np.histogram(x, bins=edges, density=True) if len(x) else (np.zeros(bins), None)
        out[name] = d.tolist()
    return out


def _thin(n, k=300):
    return np.unique(np.linspace(0, n - 1, min(k, n)).astype(int))


# ------------------------------------------------------------------------------ sections
def _sources(df, data_dir, has_id, n_cols_tx, n_cols_id, n_id_rows):
    return {
        "tables": [{"table": "train_transaction", "lignes": len(df), "colonnes": n_cols_tx},
                   {"table": "train_identity", "lignes": n_id_rows, "colonnes": n_cols_id}],
        "coverage": {"avec": int(has_id.sum()), "sans": int((~has_id).sum()),
                     "fraude_avec": float(df.loc[has_id, "isFraud"].mean() * 100) if has_id.any() else 0.0,
                     "fraude_sans": float(df.loc[~has_id, "isFraud"].mean() * 100) if (~has_id).any() else 0.0},
    }


def _types(df):
    is_text = lambda t: t == object or pd.api.types.is_string_dtype(t)
    kinds = df.dtypes.map(lambda t: "texte (catégorielle)" if is_text(t) else str(t)).value_counts()
    mem_light = df.memory_usage(deep=True).sum() / 1e9
    n_light = sum(str(df[c].dtype) in ("float32", "int32") for c in df.columns)
    return {"kinds": kinds.to_dict(), "mem_gb_default": mem_light + n_light * len(df) * 4 / 1e9, "mem_gb_light": mem_light}


def _missing(df, cols, S, y_S):
    miss = df[cols].isna().mean()
    fam = pd.Series({c: family(c) for c in cols})
    tab = (pd.DataFrame({"famille": fam, "manquant": miss * 100}).groupby("famille")
           .agg(colonnes=("manquant", "size"), manquant_moyen=("manquant", "mean")).reset_index())
    info = []
    for c in cols:
        if 0.1 < miss[c] < 0.9:
            isn = S[c].isna().to_numpy()
            if isn.any() and (~isn).any():
                a, b = y_S[isn].mean() * 100, y_S[~isn].mean() * 100
                info.append({"col": c, "si_manquant": a, "si_present": b, "ecart": abs(a - b)})
    info = sorted(info, key=lambda r: -r["ecart"])[:12]
    top = miss.sort_values(ascending=False).head(25)
    return {"families": tab.to_dict("records"), "hist": _hist(miss * 100, 20, (0, 100)),
            "informative": info, "top_missing": [{"col": c, "pct": float(v * 100)} for c, v in top.items()]}


def _anomalies(df, S, cols):
    quasi = []
    for c in cols:
        vc = S[c].value_counts(normalize=True, dropna=False)
        if len(vc) and vc.iloc[0] > 0.99:
            quasi.append(c)
    amt = df["TransactionAmt"].to_numpy(dtype="float64")
    q = np.quantile(amt, [.5, .9, .99, .999])
    return {
        "id_unique": bool(df["TransactionID"].is_unique), "chrono": bool(df["TransactionDT"].is_monotonic_increasing),
        "amt_nonpos": int((amt <= 0).sum()), "dup_ids": int(df["TransactionID"].duplicated().sum()),
        "n_quasi": len(quasi), "quasi_examples": quasi[:20],
        "amount_stats": {"moyenne": amt.mean(), "écart-type": amt.std(), "min": amt.min(), "médiane": q[0],
                         "90e centile": q[1], "99e centile": q[2], "99,9e centile": q[3], "max": amt.max()},
        "amount_hist_raw": _hist(np.clip(amt, None, np.quantile(amt, .995)), 50),
        "amount_hist_log": _hist(np.log1p(amt), 50),
    }


def _explore(df, S, raw_cols, y):
    day = (df["TransactionDT"] // 86400).to_numpy()
    daily = df.groupby(day)["isFraud"].mean() * 100
    hour = ((df["TransactionDT"] // 3600) % 24).to_numpy()
    cd = [c for c in raw_cols if re.fullmatch(r"[CD]\d+", c)]
    num = [c for c in raw_cols if c not in ("isFraud", "TransactionID", "TransactionDT")
           and pd.api.types.is_numeric_dtype(S[c])]
    cor = S[num].corrwith(S["isFraud"]).dropna()
    best = cor.reindex(cor.abs().sort_values(ascending=False).head(20).index)
    out = {
        "classes": {"legit": int((y == 0).sum()), "fraud": int((y == 1).sum())},
        "daily_fraud": [{"jour": int(d), "taux": float(v)} for d, v in daily.items()],
        "amount_by_class": _hist_by_class(np.log1p(df["TransactionAmt"].to_numpy()), y, 40, (0, 1)),
        "hour": [{"heure": int(h), "taux": float(v)} for h, v in pd.Series(y).groupby(hour).mean().mul(100).items()],
        "corr_target": [{"feature": c, "corr": float(v)} for c, v in best.items()],
    }
    for name, col in (("product", "ProductCD"), ("device", "DeviceType")):
        if col in df:
            g = df.groupby(df[col].fillna("inconnu"))["isFraud"].agg(["mean", "size"])
            out[name] = [{"valeur": str(k), "taux": float(r["mean"] * 100), "n": int(r["size"])} for k, r in g.iterrows()]
    if len(cd) > 1:
        out["corr_cd"] = {"cols": cd, "matrix": S[cd].corr().fillna(0).round(2).to_numpy().tolist()}
    return out


def _prep(df, new_cols, train, val, test, enc, feats, y, n_raw):
    shown = [c for c in ("hour", "amt_log", "amt_vs_card_mean", "card_dt_since_last") if c in new_cols]
    dist = {}
    for c in shown:
        dist[c] = _hist_by_class(df[c].to_numpy(), y, 24 if c == "hour" else 40, (0, 1) if c == "hour" else (0.01, 0.99))
    parts = []
    for name, d in (("train", train), ("val", val), ("test", test)):
        parts.append({"partie": name, "lignes": len(d), "jour_min": int(d["TransactionDT"].min() // 86400),
                      "jour_max": int(d["TransactionDT"].max() // 86400), "taux_fraude": float(d["isFraud"].mean() * 100)})
    counts = df.groupby(df["TransactionDT"] // 86400).size()
    card = sorted(({"col": c, "avant": int(train[c].nunique()),
                    "apres": len(enc[c]) + int(train[c].nunique() > len(enc[c]))} for c in enc), key=lambda r: -r["avant"])[:15]
    examples = {c: enc[c][:12] + (["(autre)"] if len(enc[c]) < 12 else []) for c in ("ProductCD", "card4", "card6", "P_emaildomain") if c in enc}
    recap = [{"étape": "Colonnes brutes après jointure", "colonnes": n_raw},
             {"étape": "+ variables dérivées", "colonnes": len(new_cols)},
             {"étape": "− identifiant, cible, horodatage (non utilisés comme variables)", "colonnes": -3},
             {"étape": "= variables fournies au modèle", "colonnes": len(feats)},
             {"étape": "dont catégorielles encodées", "colonnes": len(enc)}]
    return {"new_cols": new_cols, "features_dist": dist, "split": parts,
            "daily_counts": [{"jour": int(d), "n": int(n)} for d, n in counts.items()],
            "cardinality": card, "encoder_examples": examples, "recap": recap,
            "n_features": len(feats), "n_cat": len(enc)}


def _model(model, feats, rows):
    ev = model.evals_result_.get("valid_0", {})
    curves = {}
    for k, v in ev.items():
        idx = _thin(len(v), 400)
        curves[k] = [{"iteration": int(i), "valeur": float(v[i])} for i in idx]
    g = model.booster_.feature_importance("gain")
    g = g / g.sum() * 100
    order = np.argsort(-g)[:15]
    return {"params": {k: v for k, v in P.PARAMS.items()}, "best_iteration": int(model.best_iteration_),
            "curves": curves, "gain": [{"feature": feats[i], "gain_pct": float(g[i])} for i in order],
            "rows": rows}


def _eval(test, pte, metrics):
    y = test["isFraud"].to_numpy()
    amt = test["TransactionAmt"].to_numpy(dtype="float64")
    fpr, tpr, _ = roc_curve(y, pte)
    pr, rc, _ = precision_recall_curve(y, pte)
    i, j = _thin(len(fpr)), _thin(len(pr))
    grid = np.round(np.linspace(0.01, 0.99, 99), 2)
    sweep = []
    for t in grid:
        pred = pte >= t
        sweep.append({"seuil": float(t), "tp": int((pred & (y == 1)).sum()), "fp": int((pred & (y == 0)).sum()),
                      "fn": int((~pred & (y == 1)).sum()), "tn": int((~pred & (y == 0)).sum()),
                      "fn_montant": float(amt[~pred & (y == 1)].sum())})
    wk = ((test["TransactionDT"] - test["TransactionDT"].min()) // (7 * 86400)).to_numpy()
    weekly = [{"semaine": int(w), "auc": float(roc_auc_score(y[wk == w], pte[wk == w]))}
              for w in np.unique(wk) if len(np.unique(y[wk == w])) == 2]
    return {"metrics": metrics, "roc": {"fpr": fpr[i], "tpr": tpr[i]}, "pr": {"recall": rc[j], "precision": pr[j]},
            "scores": _hist_by_class(pte, y, 40, (0, 1)), "sweep": sweep, "weekly_auc": weekly,
            "cost_no_model": float(amt[y == 1].sum()), "prevalence": float(y.mean())}


def _shap(model, X_test, feats, enc, art_dir, seed, n=1500):
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(X_test), min(n, len(X_test)), replace=False))
    Xs = X_test.iloc[idx].to_numpy(dtype="float32")
    contrib = model.booster_.predict(Xs, pred_contrib=True)[:, :-1]
    imp = np.abs(contrib).mean(0)
    order = np.argsort(-imp)
    importance = [{"feature": feats[i], "mean_abs_shap": float(imp[i])} for i in order[:25]]
    frames = []
    for j in order[:15]:
        v = Xs[:, j].astype("float64")
        ok = ~np.isnan(v)
        rank = np.full(len(v), -1.0)
        if ok.sum() > 1:
            rank[ok] = pd.Series(v[ok]).rank(pct=True).to_numpy()
        f = feats[j]
        if f in enc:
            labels = np.array(enc[f] + ["(autre)"], dtype=object)
            lab = np.where(ok, labels[np.nan_to_num(v, nan=0).astype(int).clip(0, len(labels) - 1)], "manquant")
        else:
            lab = np.where(ok, np.round(v, 3).astype(str), "manquant")
        frames.append(pd.DataFrame({"row": np.arange(len(v)), "feature": f, "value": v, "label": lab,
                                    "shap": contrib[:, j].astype("float64"), "rank": rank, "categorical": f in enc}))
    pd.concat(frames, ignore_index=True).to_parquet(Path(art_dir) / "shap_sample.parquet", index=False)
    return {"importance": importance, "n_sample": int(len(idx))}


# ------------------------------------------------------------------------------ entrée
def profile_data(*, data_dir, df, new_cols, train, val, test, enc, feats, seed=0) -> dict:
    """Statistiques qui ne dépendent que des données (à appeler avant de libérer le gros tableau)."""
    data_dir = Path(data_dir)
    raw_cols = [c for c in df.columns if c not in new_cols]
    cols = [c for c in raw_cols if c != "isFraud"]
    y = df["isFraud"].to_numpy()
    S = df.sample(min(200_000, len(df)), random_state=seed)
    y_S = S["isFraud"].to_numpy()

    p_tr, p_id = data_dir / "train_transaction.csv", data_dir / "train_identity.csv"
    ids = pd.read_csv(p_id, usecols=["TransactionID"])["TransactionID"]
    has_id = pd.Series(df["TransactionID"].isin(ids).to_numpy(), index=df.index)
    n_cols_tx, n_cols_id = pd.read_csv(p_tr, nrows=0).shape[1], pd.read_csv(p_id, nrows=0).shape[1]
    return {
        "meta": {"n_rows": len(df), "n_fraud": int(y.sum()), "prevalence": float(y.mean()),
                 "jour_min": int(df["TransactionDT"].min() // 86400), "jour_max": int(df["TransactionDT"].max() // 86400)},
        "sources": _sources(df, data_dir, has_id, n_cols_tx, n_cols_id, len(ids)),
        "types": _types(df),
        "missing": _missing(df, cols, S, y_S),
        "anomalies": _anomalies(df, S, raw_cols),
        "explore": _explore(df, S, raw_cols, y),
        "prep": _prep(df, new_cols, train, val, test, enc, feats, y, n_cols_tx + n_cols_id - 1),
    }


def finish_profile(prof: dict, *, model, feats, enc, X_test, test, pte, metrics, art_dir=P.ART_DIR, seed=0) -> Path:
    """Ajoute les sections dépendant du modèle, puis écrit profile.json et shap_sample.parquet."""
    art_dir = Path(art_dir)
    art_dir.mkdir(parents=True, exist_ok=True)
    rows = {p["partie"]: p["lignes"] for p in prof["prep"]["split"]}
    prof["model"] = _model(model, feats, rows)
    prof["eval"] = _eval(test, pte, metrics)
    prof["shap"] = _shap(model, X_test, feats, enc, art_dir, seed)
    out = art_dir / "profile.json"
    out.write_text(json.dumps(_clean(prof), ensure_ascii=False), encoding="utf-8")
    return out


def build_profile(*, data_dir, df, new_cols, train, val, test, enc, feats, model, X_test, pte, metrics,
                  art_dir=P.ART_DIR, seed=0) -> Path:
    """Les deux temps d'un coup (notebook, où tous les tableaux sont déjà en mémoire)."""
    prof = profile_data(data_dir=data_dir, df=df, new_cols=new_cols, train=train, val=val, test=test,
                        enc=enc, feats=feats, seed=seed)
    return finish_profile(prof, model=model, feats=feats, enc=enc, X_test=X_test, test=test, pte=pte,
                          metrics=metrics, art_dir=art_dir, seed=seed)
