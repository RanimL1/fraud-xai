"""Entraînement bout en bout : python train.py [--nrows N]"""
import argparse
import gc
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
import pipeline as P  # noqa: E402
import profiling  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(P.DATA_DIR))
    ap.add_argument("--nrows", type=int, default=None, help="sous-échantillon pour tester vite")
    ap.add_argument("--sample-size", type=int, default=20000, help="taille de l'échantillon de démo (5000 pour un déploiement léger)")
    a = ap.parse_args()

    df = P.load_data(Path(a.data_dir), a.nrows)
    cols_before = list(df.columns)
    df = P.add_features(df)
    new_cols = [c for c in df.columns if c not in cols_before]
    train, val, test = P.temporal_split(df)
    feats, enc, (Xtr, Xva, Xte) = P.prepare(train, val, test)
    prof = profiling.profile_data(data_dir=Path(a.data_dir), df=df, new_cols=new_cols, train=train, val=val,
                                  test=test, enc=enc, feats=feats)
    # libère le gros tableau d'origine : on ne garde que les colonnes utiles à l'évaluation
    keep = ["isFraud", "TransactionID", "TransactionDT", "TransactionAmt"]
    train, val, test = (d[keep].copy() for d in (train, val, test))
    del df
    gc.collect()
    model = P.train_model(Xtr, train.isFraud, Xva, val.isFraud, list(enc))

    pva = model.predict_proba(Xva)[:, 1]
    pte = model.predict_proba(Xte)[:, 1]
    thr_f1 = P.best_f1_threshold(val.isFraud.to_numpy(), pva)
    thr_cost = P.cost_threshold(val.isFraud.to_numpy(), pva, val.TransactionAmt.to_numpy())
    metrics = P.evaluate(test.isFraud.to_numpy(), pte, test.TransactionAmt.to_numpy(), thr_f1, thr_cost)
    for k, v in metrics.items():
        print(f"{k:16s} {v:.4f}" if isinstance(v, float) else f"{k:16s} {v}")
    print("Artefacts :", P.export_artifacts(model, enc, feats, Xte, test, metrics, n_sample=a.sample_size))
    print("Profil    :", profiling.finish_profile(prof, model=model, feats=feats, enc=enc, X_test=Xte, test=test,
                                                    pte=pte, metrics=metrics))

if __name__ == "__main__":
    main()
