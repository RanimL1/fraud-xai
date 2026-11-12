import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import streamlit as st

from app_pages import common as C

E = C.get_engine()
M = E.metrics
C.header("🛡️", "Détection de fraude explicable", "Challenge 2 · jeu IEEE-CIS (Vesta) · LightGBM · explications SHAP · application interactive")

st.markdown("""
Cette application présente **le projet de bout en bout**, une phase par page (menu de gauche) :
des données brutes à l'explication d'une décision individuelle. Un modèle de fraude n'est utile que si un analyste
peut **comprendre pourquoi** une transaction est signalée : d'où l'accent mis sur l'explicabilité.
""")

c = st.columns(5)
c[0].metric("ROC-AUC (test)", f"{M['roc_auc']:.3f}")
c[1].metric("PR-AUC (test)", f"{M['pr_auc']:.3f}", help=f"Référence au hasard : {M['prevalence']:.3f}")
c[2].metric("Précision", f"{M['precision']:.1%}")
c[3].metric("Rappel", f"{M['recall']:.1%}")
c[4].metric("Coût évité", f"{M['cost_reduction']:.0%}")
st.caption("Métriques calculées sur la période la plus récente (jeu de test chronologique), jamais vue à l'entraînement.")

st.subheader("Déroulé du projet")
st.graphviz_chart("""
digraph G {
  rankdir=LR; node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11, color="#555555"];
  a [label="1 · Données\\n(jointure, types)", fillcolor="#c6dbef"];
  b [label="2 · Qualité\\n(manquants, anomalies)", fillcolor="#fdd0a2"];
  c [label="3 · Exploration\\n(corrélations)", fillcolor="#fdd0a2"];
  d [label="4 · Préparation\\n(variables, découpage,\\nencodage)", fillcolor="#c7e9c0"];
  e [label="5 · Modélisation\\n(LightGBM)", fillcolor="#dadaeb"];
  f [label="6 · Évaluation\\n(seuils, coût)", fillcolor="#dadaeb"];
  g [label="7-8 · Explicabilité\\n(SHAP global et local)", fillcolor="#fcbba1"];
  a -> b -> c -> d -> e -> f -> g;
}
""")

st.subheader("Guide des pages")
st.markdown("""
| Page | Question à laquelle elle répond |
|---|---|
| 1 · Données | De quoi dispose-t-on ? Comment les deux tables sont-elles reliées ? |
| 2 · Qualité | Que manque-t-il, que faut-il nettoyer ? |
| 3 · Exploration | Qu'est-ce qui distingue une fraude d'une transaction légitime ? |
| 4 · Préparation | Comment passe-t-on des données brutes à celles du modèle ? |
| 5 · Modélisation | Quel modèle, quels réglages, comment a-t-il appris ? |
| 6 · Évaluation | Est-il bon, avec quel seuil, pour quel coût ? (curseur interactif) |
| 7 · Explicabilité globale | Quelles variables comptent, et dans quel sens ? |
| 8 · Explicabilité locale | Pourquoi *cette* transaction est-elle signalée ? Et si on la modifiait ? |
""")

with st.expander("Limites et pistes d'amélioration"):
    st.markdown("""
- **Anonymisation** : beaucoup de variables (`V*`, `C*`, `D*`) sont opaques ; l'explication est statistique, pas causale.
- **Dérive temporelle** : la fraude évolue ; la stabilité de l'AUC par semaine (page 6) permet de la surveiller. En production, un ré-entraînement périodique serait nécessaire.
- **Coût asymétrique** : le seuil dépend du métier (coût d'une fausse alerte supposé à 5 $, modifiable en page 6).
- **Équité et vie privée** : appareil, domaine e-mail, etc. peuvent introduire des biais ; à auditer avant tout usage réel.
- **Pistes** : calibration des probabilités, modèle de graphe (cartes et appareils partagés), SHAP par groupes de variables, comparaison avec un modèle interprétable de base.
""")
