import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import altair as alt
import pandas as pd
import streamlit as st

from app_pages import common as C

P = C.profile()
Q = P["prep"]
C.header("4 ·", "Préparation des données", "Variables dérivées, découpage temporel, encodage et récapitulatif des décisions.")

t1, t2, t3, t4 = st.tabs(["Variables dérivées", "Découpage temporel", "Encodage", "Récapitulatif"])

with t1:
    st.markdown("""
| Variable | Définition | Pourquoi |
|---|---|---|
| `hour`, `weekday` | heure et jour de semaine déduits de `TransactionDT` | rythmes d'activité |
| `amt_log` | log(1 + montant) | atténue l'asymétrie |
| `amt_cents` | centimes du montant | les montants « ronds » sont typiques de certaines fraudes |
| `email_match` | e-mail acheteur = e-mail destinataire | cohérence de l'identité |
| `card_n_prev` | nombre de transactions **antérieures** de la carte | ancienneté d'usage |
| `amt_vs_card_mean` | montant / montant moyen **passé** de la carte | écart au comportement habituel |
| `card_dt_since_last` | délai depuis la dernière transaction de la carte | rafales de transactions |
""")
    st.warning("Les trois dernières n'utilisent **que le passé** de chaque carte (le tableau est trié par temps) : aucune fuite d'information du futur.")
    avail = list(Q["features_dist"])
    if avail:
        f = st.selectbox("Distribution par classe", avail)
        C.show(C.class_hist(Q["features_dist"][f], f"{f} : légitime contre fraude (densité, 1er–99e centile)", f))

with t2:
    st.markdown("Pas de découpage aléatoire : en production, le modèle prédit des transactions **futures**. On entraîne sur le passé, "
                "on règle l'arrêt précoce et les seuils sur la période suivante, et on évalue sur la plus récente.")
    sp = pd.DataFrame(Q["split"])
    names = {"train": "Entraînement", "val": "Validation", "test": "Test"}
    sp["nom"] = sp["partie"].map(names)
    daily = pd.DataFrame(Q["daily_counts"])
    line = alt.Chart(daily).mark_line(color="#555").encode(x=alt.X("jour:Q", title="Jour"), y=alt.Y("n:Q", title="Transactions / jour"))
    spans = alt.Chart(sp).mark_rect(opacity=0.25).encode(
        x="jour_min:Q", x2="jour_max:Q",
        color=alt.Color("nom:N", scale=alt.Scale(domain=["Entraînement", "Validation", "Test"], range=[C.BLUE, C.ORANGE, C.RED]), title=None),
        tooltip=["nom", "lignes", "jour_min", "jour_max", alt.Tooltip("taux_fraude:Q", format=".2f")])
    C.show((spans + line).properties(title="Découpage chronologique", height=300))
    a, b = st.columns(2)
    with a:
        st.dataframe(sp[["nom", "lignes", "jour_min", "jour_max", "taux_fraude"]].rename(columns={
            "nom": "Partie", "lignes": "Lignes", "jour_min": "Jour début", "jour_max": "Jour fin", "taux_fraude": "% fraude"}).set_index("Partie").round(2),
            width="stretch")
    with b:
        C.show(C.bar_v(sp.rename(columns={"taux_fraude": "taux"}), "nom", "taux", "Taux de fraude par partie (%)", color=C.RED, height=200))

with t3:
    st.markdown("Les modalités sont apprises **sur l'entraînement uniquement**. Celles qui apparaissent moins de 100 fois, ou qui "
                "n'existent pas dans l'entraînement, sont regroupées dans une modalité « autre ». Les valeurs manquantes restent manquantes.")
    card = pd.DataFrame(Q["cardinality"])
    long = card.melt(id_vars="col", value_vars=["avant", "apres"], var_name="moment", value_name="modalités")
    long["moment"] = long["moment"].map({"avant": "avant", "apres": "après regroupement"})
    ch = alt.Chart(long).mark_bar().encode(
        y=alt.Y("col:N", sort=list(card["col"]), title=None), x=alt.X("modalités:Q", scale=alt.Scale(type="log"), title="Nombre de modalités (échelle log)"),
        yOffset="moment:N", color=alt.Color("moment:N", scale=alt.Scale(domain=["avant", "après regroupement"], range=[C.GREY, C.GREEN]), title=None),
        tooltip=["col", "moment", "modalités"]).properties(height=420, title="Cardinalité des variables textuelles")
    C.show(ch)
    ex = Q["encoder_examples"]
    if ex:
        c = st.selectbox("Exemple de codage", list(ex))
        st.dataframe(pd.DataFrame({"code": range(len(ex[c])), "modalité": ex[c]}).set_index("code").T, width="stretch")

with t4:
    st.dataframe(pd.DataFrame(Q["recap"]).set_index("étape"), width="stretch")
    st.markdown("""
| Étape | Décision | Justification |
|---|---|---|
| Jointure | à gauche, sur `TransactionID` | garder toutes les transactions |
| Types | décimaux en 32 bits | mémoire ÷ 2 |
| Valeurs manquantes | conservées (pas d'imputation) | le manquant est informatif, LightGBM le gère |
| Montants | pas de troncature, ajout de `amt_log` | arbres insensibles aux extrêmes, gros montants = gros risque |
| Variables dérivées | uniquement à partir du passé | pas de fuite du futur |
| Découpage | chronologique 72 / 8 / 20 | simule la production |
| Catégorielles | modalités apprises sur le train, rares regroupées | robustesse, pas de fuite |
| Déséquilibre | pas de sur-échantillonnage ; évaluation PR-AUC et coût | garde des probabilités fidèles |
""")
