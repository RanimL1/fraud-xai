import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from app_pages import common as C

P = C.profile()
Ev = P["eval"]
M = Ev["metrics"]
C.header("6 ·", "Évaluation", "Performance sur la période la plus récente, choix du seuil, coût métier et comparaison avec le non supervisé.")

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

# ------------------------------------------------------------------ non supervisé
st.divider()
st.header("Et sans étiquettes ? Comparaison avec le non supervisé")
_up = C.ART / "unsupervised.json"
if not _up.exists():
    st.info("Résultats non supervisés introuvables (`artifacts/unsupervised.json`). Relancez `python train.py` ou la section 10 du notebook.")
    st.stop()
U = json.loads(_up.read_text(encoding="utf-8"))

IF, KM = U["methods"]["Isolation Forest"], U["methods"]["Distance au centre K-Means"]
SUP, BL, CP = U.get("supervised"), U.get("blend"), U.get("complement")

st.markdown("""
Le modèle supervisé (pages 5 et 6 ci-dessus) a besoin d'étiquettes, or une fraude n'est confirmée qu'après coup et les nouveaux types de fraude
n'ont encore aucun exemple. L'approche non supervisée part d'une autre hypothèse : **une fraude ressemble rarement à une transaction normale**.
Deux méthodes, ajustées **sans étiquettes** ; `isFraud` ne sert qu'à vérifier ensuite si ce qu'elles trouvent recoupe la fraude.

- **Isolation Forest** : des coupures aléatoires isolent les points ; plus un point est isolé vite, plus son score d'anomalie est élevé.
- **ACP + K-Means** : on projette les transactions, on les regroupe en segments de comportement, et on regarde la part de fraude par segment.
""")
st.caption(f"Source des résultats : {U['source'] or 'jeu de test'} · {U['n_features']} variables numériques (manquants < 50 %, médiane pour le reste) · "
           f"taux de fraude de référence {U['prevalence']:.1%}.")

c = st.columns(4)
c[0].metric("ROC-AUC · Isolation Forest", f"{IF['roc_auc']:.3f}", help="0,5 = hasard. Calculé sans que le modèle ait vu une seule étiquette.")
c[1].metric("Lift top 5 %", f"×{IF['top'][1]['lift']:.1f}", help="Parmi les 5 % de transactions les plus anormales, la fraude est ×N fois plus fréquente que dans l'ensemble.")
c[2].metric("ROC-AUC · supervisé", f"{SUP['roc_auc']:.3f}" if SUP else "—")
c[3].metric("ROC-AUC · mélange", f"{BL['roc_auc']:.3f}" if BL else "—", help="Moyenne des rangs du score supervisé et du score d'anomalie.")

st.subheader("1 · Les fraudes sont-elles plus « anormales » ?")
C.show(C.class_hist(U["iso_hist"], "Score d'anomalie (Isolation Forest) par classe", "Score d'anomalie (plus grand = plus inhabituel)"))
st.caption("Si les deux courbes se recouvrent presque, l'anomalie seule ne suffit pas : c'est le cas ici, les fraudes imitent des achats normaux.")

st.subheader("2 · Comparaison avec le modèle supervisé")
rows = [("Isolation Forest (non supervisé)", IF), ("Distance K-Means (non supervisé)", KM)]
if SUP:
    rows += [("LightGBM (supervisé)", SUP), ("Mélange des rangs", BL)]
tab = pd.DataFrame([{"Méthode": n, "ROC-AUC": v["roc_auc"], "PR-AUC": v["pr_auc"],
                     "Précision top 1 %": v["top"][0]["precision"], "Précision top 5 %": v["top"][1]["precision"],
                     "Rappel top 10 %": v["top"][2]["recall"]} for n, v in rows])
st.dataframe(tab.style.format({"ROC-AUC": "{:.3f}", "PR-AUC": "{:.3f}", "Précision top 1 %": "{:.1%}",
                               "Précision top 5 %": "{:.1%}", "Rappel top 10 %": "{:.1%}"}), width="stretch", hide_index=True)
st.caption("« Top k % » : on ne retient que les k % de transactions au score le plus élevé. La PR-AUC dépend du taux de fraude : à comparer "
           f"au hasard ({U['prevalence']:.1%}), et seulement entre méthodes évaluées sur les mêmes lignes.")
if CP:
    st.info(f"**Complémentarité.** Le modèle supervisé laisse passer {CP['missed']:,} fraudes au seuil choisi ; "
            f"{CP['missed_in_top5']:,} d'entre elles ({CP['share']:.1%}) figurent parmi les 5 % de transactions les plus anormales, "
            f"contre {CP['legit_in_top5']:.1%} des transactions légitimes. Le signal d'anomalie rattrape un peu de fraude manquée, mais reste faible.", icon="🔗")

st.subheader("3 · Segments de comportement (ACP + K-Means)")
pts = pd.DataFrame(U["pca_points"]).assign(Classe=lambda d: d.fraude.map({0: "Légitime", 1: "Fraude"}))
v = U["pca_variance"]
scat = alt.Chart(pts).mark_circle(size=22, opacity=0.5).encode(
    x=alt.X("x:Q", title=f"Axe 1 ({v[0]:.0%} de la variance)"), y=alt.Y("y:Q", title=f"Axe 2 ({v[1]:.0%})"),
    color=alt.Color("Classe:N", scale=C.CLASS_COLORS), tooltip=["Classe", "cluster", alt.Tooltip("iso:Q", format=".3f", title="anomalie")])
C.show(scat.properties(height=340, title="Projection ACP (sous-échantillon ; la couleur n'a pas servi à construire les axes)"))
cl = pd.DataFrame(U["clusters"]).assign(segment=lambda d: "Segment " + d.cluster.astype(str))
base_u = alt.Chart(pd.DataFrame({"y": [U["prevalence"]]})).mark_rule(color=C.GREY, strokeDash=[4, 4]).encode(y="y:Q")
bars = alt.Chart(cl).mark_bar(color=C.RED).encode(
    x=alt.X("segment:N", sort="-y", title=None, axis=alt.Axis(labelAngle=0)), y=alt.Y("fraud_rate:Q", title="Taux de fraude", axis=alt.Axis(format="%")),
    tooltip=["segment", alt.Tooltip("n:Q", title="transactions"), alt.Tooltip("fraud_rate:Q", format=".1%"), alt.Tooltip("lift:Q", format=".1f", title="lift")])
C.show((bars + base_u).properties(height=260, title="Taux de fraude par segment (pointillé : taux global)"))
st.dataframe(cl[["segment", "n", "fraud_rate", "lift"]].rename(columns={"n": "Transactions", "fraud_rate": "Taux de fraude", "lift": "Lift"})
             .style.format({"Transactions": "{:,}", "Taux de fraude": "{:.1%}", "Lift": "×{:.1f}"}), width="stretch", hide_index=True)

C.decision("Le non supervisé ne remplace pas le modèle supervisé (AUC bien plus faible) ; il sert de **filet complémentaire** : repérer des "
           "comportements atypiques sans étiquette, et prioriser les segments les plus risqués. Les résultats sont à lire comme des pistes, pas des preuves.")
st.warning("Limites : les variables `V*`, `C*`, `D*` sont anonymisées, donc un segment ou une anomalie ne s'interprète pas métier ; les catégories "
           "encodées sont exclues (distance non pertinente) ; les résultats dépendent du nombre de segments (8) choisi a priori ; un échantillon "
           "enrichi en fraudes (~20 %) donne des précisions plus élevées que le trafic réel.")
