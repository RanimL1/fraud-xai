import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from app_pages import common as C

P = C.profile()
Ev = P["eval"]
M = Ev["metrics"]
C.header("6 ·", "Évaluation", "Performance sur la période la plus récente, choix du seuil et coût métier.")

c = st.columns(4)
c[0].metric("ROC-AUC", f"{M['roc_auc']:.3f}")
c[1].metric("PR-AUC", f"{M['pr_auc']:.3f}", help=f"Référence au hasard : {Ev['prevalence']:.3f}")
c[2].metric("Seuil F1 (validation)", f"{M['threshold']:.2f}")
c[3].metric("Seuil coût (validation)", f"{M['threshold_cost']:.2f}")

a, b = st.columns(2)
with a:
    roc = pd.DataFrame(Ev["roc"])
    diag = alt.Chart(pd.DataFrame({"fpr": [0, 1], "tpr": [0, 1]})).mark_line(color=C.GREY, strokeDash=[4, 4]).encode(x="fpr:Q", y="tpr:Q")
    line = alt.Chart(roc).mark_line(color=C.BLUE).encode(x=alt.X("fpr:Q", title="Taux de faux positifs"), y=alt.Y("tpr:Q", title="Taux de vrais positifs"),
                                                       tooltip=[alt.Tooltip("fpr:Q", format=".3f"), alt.Tooltip("tpr:Q", format=".3f")])
    C.show((line + diag).properties(title=f"Courbe ROC (AUC = {M['roc_auc']:.3f})", height=300))
with b:
    pr = pd.DataFrame(Ev["pr"])
    base = alt.Chart(pd.DataFrame({"y": [Ev["prevalence"]]})).mark_rule(color=C.GREY, strokeDash=[4, 4]).encode(y="y:Q")
    line = alt.Chart(pr).mark_line(color=C.RED).encode(x=alt.X("recall:Q", title="Rappel"), y=alt.Y("precision:Q", title="Précision"),
                                                      tooltip=[alt.Tooltip("recall:Q", format=".3f"), alt.Tooltip("precision:Q", format=".3f")])
    C.show((line + base).properties(title=f"Précision-rappel (AP = {M['pr_auc']:.3f})", height=300))

C.show(C.class_hist(Ev["scores"], "Distribution des scores : le modèle sépare-t-il bien les deux classes ?", "Score de fraude prédit"))

st.subheader("Choisir le seuil et mesurer le coût")
st.markdown("Une transaction est signalée si son score dépasse le seuil. Un seuil bas détecte plus de fraudes mais génère plus de fausses alertes.")
sweep = pd.DataFrame(Ev["sweep"])
x, y = st.columns(2)
thr = x.slider("Seuil d'alerte", 0.01, 0.99, float(round(M["threshold"], 2)), 0.01)
fp_cost = y.number_input("Coût d'une fausse alerte ($)", 0.0, 500.0, 5.0, 1.0,
                         help="Revue manuelle et friction client. Une fraude non détectée coûte le montant de la transaction.")
sweep["cout"] = sweep["fn_montant"] + fp_cost * sweep["fp"]
r = sweep.iloc[(sweep["seuil"] - thr).abs().argmin()]
prec = r.tp / max(r.tp + r.fp, 1)
rec = r.tp / max(r.tp + r.fn, 1)
f1 = 2 * prec * rec / max(prec + rec, 1e-12)
cost_none = Ev["cost_no_model"]
k = st.columns(5)
k[0].metric("Précision", f"{prec:.1%}")
k[1].metric("Rappel", f"{rec:.1%}")
k[2].metric("F1", f"{f1:.3f}")
k[3].metric("Coût avec modèle", f"{r.cout:,.0f} $")
k[4].metric("Coût évité", f"{1 - r.cout / cost_none:.0%}", help=f"Sans modèle, les fraudes coûtent {cost_none:,.0f} $.")

a, b = st.columns([1, 1.6])
with a:
    st.markdown("**Matrice de confusion**")
    st.dataframe(pd.DataFrame([[int(r.tn), int(r.fp)], [int(r.fn), int(r.tp)]], index=["Réel légitime", "Réel fraude"],
                              columns=["Prédit légitime", "Prédit fraude"]), width="stretch")
with b:
    best = sweep.iloc[sweep["cout"].argmin()]
    curve = alt.Chart(sweep).mark_line(color=C.BLUE).encode(x=alt.X("seuil:Q", title="Seuil"), y=alt.Y("cout:Q", title="Coût total ($)"),
                                                          tooltip=[alt.Tooltip("seuil:Q", format=".2f"), alt.Tooltip("cout:Q", format=",.0f")])
    cur = alt.Chart(pd.DataFrame({"x": [thr]})).mark_rule(color=C.RED).encode(x="x:Q")
    opt = alt.Chart(pd.DataFrame({"x": [float(best.seuil)]})).mark_rule(color=C.GREEN, strokeDash=[4, 4]).encode(x="x:Q")
    C.show((curve + cur + opt).properties(height=240, title=f"Coût selon le seuil (rouge : choisi ; vert : optimal = {best.seuil:.2f})"))

st.subheader("Stabilité dans le temps")
wk = pd.DataFrame(Ev["weekly_auc"])
if len(wk):
    ch = alt.Chart(wk).mark_bar(color=C.BLUE).encode(x=alt.X("semaine:O", title="Semaine du jeu de test", axis=alt.Axis(labelAngle=0)),
                                                      y=alt.Y("auc:Q", scale=alt.Scale(domain=[max(0.5, wk.auc.min() - 0.05), 1]), title="AUC"),
                                                      tooltip=["semaine", alt.Tooltip("auc:Q", format=".3f")])
    C.show(ch.properties(height=240, title="AUC par semaine : une baisse signalerait une dérive de la fraude"))
