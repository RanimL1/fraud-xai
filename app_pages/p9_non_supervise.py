import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import altair as alt
import json
import pandas as pd
import streamlit as st

from app_pages import common as C

C.header("9 ·", "Apprentissage non supervisé", "Détecter l'inhabituel et segmenter les transactions sans jamais utiliser l'étiquette « fraude ».")

path = C.ART / "unsupervised.json"
if not path.exists():
    st.error("Résultats non supervisés introuvables (`artifacts/unsupervised.json`). Relancez `python train.py` ou la section 10 du notebook.")
    st.stop()
U = json.loads(path.read_text(encoding="utf-8"))
IF, KM = U["methods"]["Isolation Forest"], U["methods"]["Distance au centre K-Means"]
SUP, BL, CP = U.get("supervised"), U.get("blend"), U.get("complement")

st.markdown("""
Le modèle supervisé (pages 5-6) a besoin d'étiquettes, or une fraude n'est confirmée qu'après coup et les nouveaux types de fraude
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
base = alt.Chart(pd.DataFrame({"y": [U["prevalence"]]})).mark_rule(color=C.GREY, strokeDash=[4, 4]).encode(y="y:Q")
bars = alt.Chart(cl).mark_bar(color=C.RED).encode(
    x=alt.X("segment:N", sort="-y", title=None, axis=alt.Axis(labelAngle=0)), y=alt.Y("fraud_rate:Q", title="Taux de fraude", axis=alt.Axis(format="%")),
    tooltip=["segment", alt.Tooltip("n:Q", title="transactions"), alt.Tooltip("fraud_rate:Q", format=".1%"), alt.Tooltip("lift:Q", format=".1f", title="lift")])
C.show((bars + base).properties(height=260, title="Taux de fraude par segment (pointillé : taux global)"))
st.dataframe(cl[["segment", "n", "fraud_rate", "lift"]].rename(columns={"n": "Transactions", "fraud_rate": "Taux de fraude", "lift": "Lift"})
             .style.format({"Transactions": "{:,}", "Taux de fraude": "{:.1%}", "Lift": "×{:.1f}"}), width="stretch", hide_index=True)

C.decision("Le non supervisé ne remplace pas le modèle supervisé (AUC bien plus faible) ; il sert de **filet complémentaire** : repérer des "
           "comportements atypiques sans étiquette, et prioriser les segments les plus risqués. Les résultats sont à lire comme des pistes, pas des preuves.")
st.warning("Limites : les variables `V*`, `C*`, `D*` sont anonymisées, donc un segment ou une anomalie ne s'interprète pas métier ; les catégories "
           "encodées sont exclues (distance non pertinente) ; les résultats dépendent du nombre de segments (8) choisi a priori ; un échantillon "
           "enrichi en fraudes (~20 %) donne des précisions plus élevées que le trafic réel.")
