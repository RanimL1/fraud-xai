import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from app_pages import common as C

P = C.profile()
C.header("7 ·", "Explicabilité globale (SHAP)", "Quelles variables comptent le plus, et dans quel sens poussent-elles le score ?")
st.markdown("""
Les valeurs de **Shapley** répartissent la décision du modèle entre les variables : chaque transaction reçoit, pour chaque variable,
une contribution en **log-odds** (positive = pousse vers « fraude », négative = vers « légitime »). Pour les arbres, leur calcul
est **exact** (TreeSHAP).
""")
st.caption(f"Calculées sur {P['shap']['n_sample']:,} transactions du jeu de test.")

imp = pd.DataFrame(P["shap"]["importance"])
n = st.slider("Nombre de variables affichées", 5, min(25, len(imp)), 15)
C.show(C.bar_h(imp.head(n).rename(columns={"feature": "variable"}), "mean_abs_shap", "variable", "Importance globale : moyenne des |SHAP|", height=24 * n + 40, fmt=".3f"))

ss = C.get_shap_sample()
order = [f for f in imp["feature"] if f in set(ss["feature"])]
st.subheader("Effet des valeurs sur le score")
st.caption("Chaque point est une transaction. À droite de 0 : la variable pousse vers « fraude ». Couleur : valeur de la variable "
           "(bleu = basse, rouge = haute, gris = manquante).")
k = st.slider("Variables du graphique", 3, len(order), min(10, len(order)))
d = ss[ss["feature"].isin(order[:k])].copy()
rng = np.random.default_rng(0)
d["jitter"] = np.clip(rng.normal(0, 0.35, len(d)), -1, 1)
pts = alt.Chart(d).mark_circle(size=16, opacity=0.65).encode(
    x=alt.X("shap:Q", title="Valeur SHAP (log-odds)"), y=alt.Y("jitter:Q", axis=None, scale=alt.Scale(domain=[-1.3, 1.3])),
    color=alt.condition(alt.datum["rank"] < 0, alt.value(C.GREY),
                        alt.Color("rank:Q", scale=alt.Scale(domain=[0, 1], range=[C.BLUE, C.RED]), legend=None)),
    tooltip=["feature", alt.Tooltip("label:N", title="valeur"), alt.Tooltip("shap:Q", format="+.3f")])
facet = pts.properties(width=640, height=26).facet(row=alt.Row("feature:N", sort=order[:k], title=None, header=alt.Header(labelAngle=0, labelAlign="left", labelFontSize=11)))
st.altair_chart(facet.resolve_scale(y="shared"), width="content")

st.subheader("Dépendance : effet d'une variable selon sa valeur")
f = st.selectbox("Variable", order)
dd = ss[ss["feature"] == f]
if dd["categorical"].iloc[0]:
    ch = alt.Chart(dd).mark_boxplot(color=C.BLUE).encode(x=alt.X("label:N", title=f), y=alt.Y("shap:Q", title="Valeur SHAP (log-odds)"))
else:
    dd = dd[dd["value"].notna()]
    ch = alt.Chart(dd).mark_circle(size=22, opacity=0.5, color=C.BLUE).encode(
        x=alt.X("value:Q", title=f), y=alt.Y("shap:Q", title="Valeur SHAP (log-odds)"),
        tooltip=[alt.Tooltip("value:Q", format=",.3~g"), alt.Tooltip("shap:Q", format="+.3f")])
    ch = ch + alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color=C.GREY, strokeDash=[4, 4]).encode(y="y:Q")
C.show(ch.properties(height=320))
st.warning("Beaucoup de variables (`V*`, `C*`, `D*`) sont anonymisées par Vesta : l'explication dit **quelle** variable compte, "
           "pas toujours **ce qu'elle signifie**. Elle est statistique, non causale.")
