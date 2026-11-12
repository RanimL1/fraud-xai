"""Partie non supervisée : détection d'anomalies (Isolation Forest) et segmentation (PCA + K-Means).

Les étiquettes `isFraud` ne servent JAMAIS à ajuster ces modèles : elles ne sont utilisées qu'après coup,
pour mesurer si les anomalies / segments trouvés recoupent la fraude, et pour comparer au modèle supervisé.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

ART_DIR = Path(__file__).resolve().parents[1] / "artifacts"
MAX_MISSING = 0.5     # on écarte les variables manquantes à plus de 50 % (peu informatives sans étiquette)
N_CLUSTERS = 8


def select_features(X: pd.DataFrame, cat_cols=(), max_missing: float = MAX_MISSING) -> list[str]:
    """Variables numériques peu manquantes. Les catégories encodées en entiers sont exclues :
    un code (0, 1, 2…) n'a pas de sens géométrique pour une distance ou une ACP."""
    num = [c for c in X.columns if c not in set(cat_cols)]
    return [c for c in num if X[c].isna().mean() < max_missing]


def _prep(X_fit: pd.DataFrame, X_eval: pd.DataFrame, cols: list[str]):
    """Imputation par la médiane (apprise sur X_fit), puis standardisation robuste (valeurs écrêtées à ±5 σ)."""
    med = X_fit[cols].median()
    A = X_fit[cols].fillna(med).to_numpy("float32")
    B = X_eval[cols].fillna(med).to_numpy("float32")
    sc = StandardScaler().fit(A)
    return A, B, np.clip(sc.transform(A), -5, 5), np.clip(sc.transform(B), -5, 5)


def _top_stats(y: np.ndarray, score: np.ndarray, fracs=(0.01, 0.05, 0.10)) -> list[dict]:
    """Parmi les k % de transactions les plus anormales : précision, rappel, « lift » (précision / taux de base)."""
    order = np.argsort(-score)
    base, n_fraud = y.mean(), max(int(y.sum()), 1)
    out = []
    for f in fracs:
        k = max(int(len(y) * f), 1)
        tp = int(y[order[:k]].sum())
        out.append({"top_pct": f, "precision": tp / k, "recall": tp / n_fraud, "lift": (tp / k) / base if base else 0.0})
    return out


def run(X_fit: pd.DataFrame, X_eval: pd.DataFrame, y_eval, p_sup=None, thr_sup: float | None = None,
        cat_cols=(), n_estimators: int = 200, max_samples: int | str = 4096, n_clusters: int = N_CLUSTERS,
        seed: int = 42, source: str = "", max_fit: int = 100_000) -> dict:
    """Ajuste sur X_fit (sans étiquettes), évalue sur X_eval. Retourne un dictionnaire sérialisable (JSON)."""
    y = np.asarray(y_eval).astype(int)
    if len(X_fit) > max_fit:                           # sous-échantillon : K-Means sur 470 k lignes serait inutilement lourd
        X_fit = X_fit.sample(max_fit, random_state=seed)
    cols = select_features(X_fit, cat_cols)
    A, B, As, Bs = _prep(X_fit, X_eval, cols)

    # 1. Isolation Forest : plus un point est isolé vite par des coupures aléatoires, plus il est anormal
    iso = IsolationForest(n_estimators=n_estimators, max_samples=max_samples, random_state=seed, n_jobs=-1).fit(A)
    s_iso = -iso.score_samples(B)                      # grand = anormal

    # 2. ACP (2 axes pour la carte) + K-Means sur l'espace standardisé
    pca = PCA(n_components=2, random_state=seed).fit(As)
    Z = pca.transform(Bs)
    km = KMeans(n_clusters=n_clusters, n_init=5, random_state=seed).fit(As)
    lab = km.predict(Bs)
    dist = np.linalg.norm(Bs - km.cluster_centers_[lab], axis=1)   # distance au centre de son segment

    base = float(y.mean())
    clusters = []
    for k in range(n_clusters):
        m = lab == k
        if m.sum():
            clusters.append({"cluster": k, "n": int(m.sum()), "fraud_rate": float(y[m].mean()),
                             "lift": float(y[m].mean() / base) if base else 0.0, "mean_iso": float(s_iso[m].mean())})
    clusters.sort(key=lambda c: -c["fraud_rate"])

    methods = {"Isolation Forest": s_iso, "Distance au centre K-Means": dist}
    res = {name: {"roc_auc": float(roc_auc_score(y, s)), "pr_auc": float(average_precision_score(y, s)),
                  "top": _top_stats(y, s)} for name, s in methods.items()}

    out = {"source": source, "n_fit": int(len(X_fit)), "n_eval": int(len(X_eval)), "n_features": len(cols),
           "prevalence": base, "methods": res, "clusters": clusters,
           "pca_variance": [float(v) for v in pca.explained_variance_ratio_]}

    # nuage ACP (sous-échantillon, fraudes sur-représentées pour rester lisible)
    rng = np.random.default_rng(seed)
    fr, lg = np.where(y == 1)[0], np.where(y == 0)[0]
    pick = np.concatenate([rng.choice(fr, min(len(fr), 600), replace=False), rng.choice(lg, min(len(lg), 1400), replace=False)])
    out["pca_points"] = [{"x": float(Z[i, 0]), "y": float(Z[i, 1]), "fraude": int(y[i]), "cluster": int(lab[i]),
                          "iso": float(s_iso[i])} for i in pick]
    out["iso_hist"] = _hist(s_iso, y)

    # 3. Comparaison / complémentarité avec le modèle supervisé (mêmes lignes)
    if p_sup is not None:
        p = np.asarray(p_sup, dtype=float)
        rk = lambda v: pd.Series(v).rank(pct=True).to_numpy()
        blend = (rk(p) + rk(s_iso)) / 2
        sup = {"roc_auc": float(roc_auc_score(y, p)), "pr_auc": float(average_precision_score(y, p)), "top": _top_stats(y, p)}
        out["supervised"] = sup
        out["blend"] = {"roc_auc": float(roc_auc_score(y, blend)), "pr_auc": float(average_precision_score(y, blend)),
                        "top": _top_stats(y, blend)}
        if thr_sup is not None:
            missed = (y == 1) & (p < thr_sup)
            top5 = s_iso >= np.quantile(s_iso, 0.95)
            out["complement"] = {"missed": int(missed.sum()), "missed_in_top5": int((missed & top5).sum()),
                                 "share": float((missed & top5).sum() / max(missed.sum(), 1)),
                                 "legit_in_top5": float(top5[y == 0].mean())}
    return out


def _hist(s: np.ndarray, y: np.ndarray, bins: int = 30) -> dict:
    edges = np.linspace(np.quantile(s, 0.001), np.quantile(s, 0.999), bins + 1)
    h = lambda v: (np.histogram(np.clip(v, edges[0], edges[-1]), bins=edges)[0] / max(len(v), 1)).tolist()
    return {"edges": edges.tolist(), "legit": h(s[y == 0]), "fraud": h(s[y == 1])}


def save(result: dict, art_dir: Path = ART_DIR) -> Path:
    art_dir = Path(art_dir)
    art_dir.mkdir(parents=True, exist_ok=True)
    p = art_dir / "unsupervised.json"
    p.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    return p
