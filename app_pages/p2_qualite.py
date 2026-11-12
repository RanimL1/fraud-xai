import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import altair as alt
import pandas as pd
import streamlit as st

from app_pages import common as C

P = C.profile()
C.header("2 ·", "Qualité des données", "Valeurs manquantes, anomalies, doublons, colonnes quasi constantes, montants.")

tab1, tab2, tab3 = st.tabs(["Valeurs manquantes", "Anomalies et doublons", "Montants"])

with tab1:
    m = P["missing"]
    st.markdown("Les colonnes sont regroupées par famille (`V*`, `C*`, `D*`, `M*`, `id_*`, `card*`, `addr*`, `dist*`).")
    fam = pd.DataFrame(m["families"]).sort_values("manquant_moyen")
    a, b = st.columns(2)
    with a:
        ch = alt.Chart(fam).mark_bar(color=C.ORANGE).encode(
            x=alt.X("manquant_moyen:Q", title=None), y=alt.Y("famille:N", sort="x", title=None),
            tooltip=["famille", "colonnes", alt.Tooltip("manquant_moyen:Q", format=".1f")]).properties(
            title="Valeurs manquantes moyennes par famille (%)", height=260)
        C.show(ch)
    with b:
        C.show(C.hist_chart(m["hist"], "Répartition du taux de manquants par colonne", "% de valeurs manquantes"))
    st.dataframe(fam.rename(columns={"famille": "Famille", "colonnes": "Colonnes", "manquant_moyen": "% manquant moyen"})
                 .round(1).set_index("Famille"), width="stretch")

    st.markdown("**Colonnes les plus incomplètes**")
    C.show(C.bar_h(pd.DataFrame(m["top_missing"]).rename(columns={"col": "colonne"}), "pct", "colonne", height=420))
    C.show_figure("04_carte_manquants.png", "Carte des valeurs manquantes (noir = manquant), échantillon de 400 transactions")

    st.markdown("**Le manquant est-il informatif ?** Taux de fraude quand la valeur est absente et quand elle est présente.")
    inf = pd.DataFrame(m["informative"])
    if len(inf):
        long = inf.melt(id_vars="col", value_vars=["si_manquant", "si_present"], var_name="cas", value_name="taux")
        long["cas"] = long["cas"].map({"si_manquant": "si manquant", "si_present": "si présent"})
        ch = alt.Chart(long).mark_bar().encode(
            y=alt.Y("col:N", sort=list(inf["col"]), title=None), x=alt.X("taux:Q", title="Taux de fraude (%)"),
            yOffset="cas:N", color=alt.Color("cas:N", scale=alt.Scale(domain=["si manquant", "si présent"], range=[C.RED, C.BLUE]), title=None),
            tooltip=["col", "cas", alt.Tooltip("taux:Q", format=".2f")]).properties(height=380)
        C.show(ch)
    C.decision("on **ne supprime pas** ces colonnes et on **n'impute pas** : LightGBM gère nativement les valeurs manquantes, "
               "et le fait qu'une valeur manque porte de l'information. Imputer par la moyenne ou la médiane détruirait ce signal.")

with tab2:
    a = P["anomalies"]
    c = st.columns(4)
    c[0].metric("Identifiants uniques", "oui" if a["id_unique"] else "non")
    c[1].metric("Ordre chronologique", "respecté" if a["chrono"] else "non")
    c[2].metric("Montants ≤ 0", f"{a['amt_nonpos']:,}")
    c[3].metric("Doublons d'identifiant", f"{a['dup_ids']:,}")
    st.markdown(f"**Colonnes quasi constantes** (une valeur représente plus de 99 % des lignes) : **{a['n_quasi']}**")
    if a["quasi_examples"]:
        st.code(", ".join(a["quasi_examples"]) + (" …" if a["n_quasi"] > len(a["quasi_examples"]) else ""))
    C.decision("identifiants uniques et chronologie respectée : rien à corriger. Les colonnes quasi constantes sont laissées au modèle "
               "(elles ne sont jamais choisies par les coupures des arbres et n'alourdissent pas l'entraînement de façon sensible).")

with tab3:
    a = P["anomalies"]
    st.dataframe(pd.DataFrame({"statistique": list(a["amount_stats"]), "valeur ($)": [round(v, 2) for v in a["amount_stats"].values()]})
                 .set_index("statistique"), width="stretch")
    x, y = st.columns(2)
    with x:
        C.show(C.hist_chart(a["amount_hist_raw"], "Montant brut (tronqué au 99,5e centile pour l'affichage)", "Montant ($)"))
    with y:
        C.show(C.hist_chart(a["amount_hist_log"], "Montant après log1p", "log(1 + montant)", color=C.GREEN))
    C.decision("les montants sont très asymétriques : on conserve la valeur brute et on ajoute `amt_log`. **Pas de troncature** des "
               "valeurs extrêmes : les arbres y sont insensibles et les gros montants sont ceux qui coûtent le plus cher en cas de fraude.")
