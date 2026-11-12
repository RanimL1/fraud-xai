import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from app_pages import common as C

P = C.profile()
X = P["explore"]
C.header("3 ·", "Exploration", "Qu'est-ce qui distingue une fraude d'une transaction légitime ?")

st.subheader("Déséquilibre et évolution dans le temps")
a, b = st.columns([1, 2])
with a:
    d = pd.DataFrame({"classe": ["Légitime", "Fraude"], "n": [X["classes"]["legit"], X["classes"]["fraud"]]})
    ch = alt.Chart(d).mark_bar().encode(x=alt.X("classe:N", title=None, axis=alt.Axis(labelAngle=0)),
                                         y=alt.Y("n:Q", scale=alt.Scale(type="log"), title="Nombre (échelle log)"),
                                         color=alt.Color("classe:N", scale=C.CLASS_COLORS, legend=None), tooltip=["classe", "n"])
    C.show(ch.properties(title="Classes", height=260))
with b:
    C.show(C.line_chart(pd.DataFrame(X["daily_fraud"]), "jour", "taux", "Taux de fraude par jour (%)", xtitle="Jour depuis l'origine"))
st.info("La fraude est rare : l'exactitude (*accuracy*) est trompeuse. On évalue donc avec PR-AUC, précision/rappel et un coût métier (page 6).")

st.subheader("Montant, produit, heure, appareil")
a, b = st.columns(2)
with a:
    C.show(C.class_hist(X["amount_by_class"], "Montant (log1p), densité par classe", "log(1 + montant)"))
with b:
    if "product" in X:
        C.show(C.bar_v(pd.DataFrame(X["product"]), "valeur", "taux", "Taux de fraude par ProductCD (%)", color=C.ORANGE))
a, b = st.columns(2)
with a:
    C.show(C.line_chart(pd.DataFrame(X["hour"]), "heure", "taux", "Taux de fraude par heure (%)", xtitle="Heure (déduite de TransactionDT)").mark_line(point=True))
with b:
    if "device" in X:
        C.show(C.bar_v(pd.DataFrame(X["device"]), "valeur", "taux", "Taux de fraude par type d'appareil (%)", color=C.GREEN))

st.subheader("Corrélations")
if "corr_cd" in X:
    st.markdown("Les variables `C*` et `D*` sont souvent redondantes entre elles. Les arbres n'en souffrent pas, mais l'importance "
                "SHAP peut alors se répartir entre variables voisines.")
    cols, mat = X["corr_cd"]["cols"], np.array(X["corr_cd"]["matrix"])
    long = pd.DataFrame([(cols[i], cols[j], mat[i, j]) for i in range(len(cols)) for j in range(len(cols))], columns=["x", "y", "corr"])
    ch = alt.Chart(long).mark_rect().encode(
        x=alt.X("x:N", sort=cols, title=None), y=alt.Y("y:N", sort=cols, title=None),
        color=alt.Color("corr:Q", scale=alt.Scale(domain=[-1, 0, 1], range=[C.BLUE, "#ffffff", C.RED]), title="Corrélation"),
        tooltip=["x", "y", alt.Tooltip("corr:Q", format=".2f")]).properties(height=460, title="Corrélation entre variables C* et D*")
    C.show(ch)
ct = pd.DataFrame(X["corr_target"])
ch = alt.Chart(ct).mark_bar().encode(
    x=alt.X("corr:Q", title="Corrélation de Pearson avec la fraude"), y=alt.Y("feature:N", sort=alt.EncodingSortField("corr", order="ascending"), title=None),
    color=alt.condition(alt.datum.corr > 0, alt.value(C.RED), alt.value(C.BLUE)),
    tooltip=["feature", alt.Tooltip("corr:Q", format=".3f")]).properties(title="Corrélation linéaire avec la fraude : top 20", height=460)
C.show(ch)
C.decision("les corrélations linéaires avec la fraude sont faibles : elle se détecte par des **interactions** entre variables, "
           "ce qui justifie un modèle à base d'arbres plutôt qu'un modèle linéaire.")
