import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import altair as alt
import pandas as pd
import streamlit as st

from app_pages import common as C

P = C.profile()
Mo = P["model"]
C.header("5 ·", "Modélisation", "LightGBM (arbres de décision boostés) avec arrêt précoce sur la validation.")

st.markdown("""
**Pourquoi LightGBM ?** Il gère nativement les valeurs manquantes et les variables catégorielles, capte les interactions entre
variables (les corrélations linéaires sont faibles, page 3), s'entraîne vite sur 590 000 lignes, et ses arbres permettent un calcul
**exact** des valeurs de Shapley (TreeSHAP) pour l'explicabilité.
""")
c = st.columns(4)
c[0].metric("Lignes d'entraînement", f"{Mo['rows']['train']:,}")
c[1].metric("Lignes de validation", f"{Mo['rows']['val']:,}")
c[2].metric("Lignes de test", f"{Mo['rows']['test']:,}")
c[3].metric("Itération retenue", f"{Mo['best_iteration']:,}")

st.subheader("Courbe d'apprentissage (validation)")
curves = Mo["curves"]
name = st.radio("Métrique", [k for k in ("auc", "binary_logloss") if k in curves], horizontal=True,
                format_func={"auc": "AUC (à maximiser)", "binary_logloss": "Log-loss (à minimiser)"}.get)
d = pd.DataFrame(curves[name])
line = alt.Chart(d).mark_line(color=C.BLUE).encode(x=alt.X("iteration:Q", title="Itération"), y=alt.Y("valeur:Q", title=name, scale=alt.Scale(zero=False)),
                                                  tooltip=["iteration", alt.Tooltip("valeur:Q", format=".4f")])
rule = alt.Chart(pd.DataFrame({"x": [Mo["best_iteration"]]})).mark_rule(color=C.RED, strokeDash=[5, 4]).encode(x="x:Q")
C.show((line + rule).properties(height=300, title=f"Arrêt à l'itération {Mo['best_iteration']} (100 itérations sans amélioration)"))
st.caption("L'arrêt précoce évite le sur-apprentissage : l'entraînement s'interrompt quand la validation ne s'améliore plus.")

a, b = st.columns(2)
with a:
    st.subheader("Hyperparamètres")
    st.dataframe(pd.DataFrame({"valeur": {k: str(v) for k, v in Mo["params"].items()}}), width="stretch", height=420)
with b:
    st.subheader("Importance « gain » (intégrée au modèle)")
    C.show(C.bar_h(pd.DataFrame(Mo["gain"]).rename(columns={"feature": "variable"}), "gain_pct", "variable", height=420, fmt=".1f"))
st.info("L'importance « gain » mesure la réduction d'erreur apportée par les coupures sur chaque variable. Elle est globale et peut "
        "être biaisée ; on la compare à l'importance SHAP (page 7), plus fiable et capable d'expliquer chaque décision.")
