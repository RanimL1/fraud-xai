import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
import streamlit as st

from app_pages import common as C

P = C.profile()
C.header("1 ·", "Compréhension des données", "Sources, jointure, types de données et empreinte mémoire.")

meta, src = P["meta"], P["sources"]
c = st.columns(4)
c[0].metric("Transactions", f"{meta['n_rows']:,}")
c[1].metric("Fraudes", f"{meta['n_fraud']:,}")
c[2].metric("Taux de fraude", f"{meta['prevalence']:.2%}")
c[3].metric("Période couverte", f"{meta['jour_max'] - meta['jour_min']} jours")

st.subheader("Sources et jointure")
st.markdown("""
Deux tables liées par `TransactionID` : les **transactions** (montant, produit, carte, adresse, variables anonymisées
`V*`, `C*`, `D*`, `M*`) et l'**identité numérique** (appareil, navigateur, réseau), disponible seulement pour une
partie des transactions. Jointure **à gauche** : on garde toutes les transactions, l'identité est manquante quand elle n'existe pas.
""")
st.dataframe(pd.DataFrame(src["tables"]).set_index("table"), width="stretch")

cov = src["coverage"]
a, b = st.columns(2)
with a:
    d = pd.DataFrame({"groupe": ["avec identité", "sans identité"], "transactions": [cov["avec"], cov["sans"]]})
    C.show(C.bar_v(d, "groupe", "transactions", "Couverture de l'identité", color=C.BLUE))
with b:
    d = pd.DataFrame({"groupe": ["avec identité", "sans identité"], "taux": [cov["fraude_avec"], cov["fraude_sans"]]})
    C.show(C.bar_v(d, "groupe", "taux", "Taux de fraude selon la présence d'identité (%)", color=C.RED))
st.info("Les transactions qui ont une identité sont **plus souvent frauduleuses** : l'absence d'information est elle-même un signal. "
        "C'est pourquoi on ne supprime ni n'impute les valeurs manquantes (page 2).")

st.subheader("Types de données et mémoire")
t = P["types"]
a, b = st.columns(2)
with a:
    d = pd.DataFrame({"type": list(t["kinds"]), "colonnes": list(t["kinds"].values())})
    C.show(C.bar_h(d, "colonnes", "type", "Types des colonnes"))
with b:
    d = pd.DataFrame({"mode": ["types 64 bits (défaut)", "types allégés (utilisés)"], "Go": [t["mem_gb_default"], t["mem_gb_light"]]})
    C.show(C.bar_v(d, "mode", "Go", "Mémoire du tableau (Go)", color=C.GREEN))
C.decision("les colonnes décimales sont lues en 32 bits (mémoire divisée par deux environ, sans perte utile ici). "
           "`TransactionAmt` reste en 64 bits pour garder les centimes exacts.")

st.subheader("Familles de variables")
st.markdown("""
| Famille | Contenu |
|---|---|
| `TransactionAmt`, `ProductCD` | montant en dollars, code produit |
| `card1`–`card6`, `addr*`, `dist*` | carte (émetteur, type), adresses, distances |
| `P_emaildomain`, `R_emaildomain` | domaine e-mail de l'acheteur et du destinataire |
| `C1`–`C14` | comptages (adresses associées à la carte, etc.), signification masquée |
| `D1`–`D15` | délais entre événements |
| `M1`–`M9` | correspondances (noms, adresses), valeurs vrai/faux |
| `V1`–`V339` | variables de Vesta (ingénierie propriétaire), anonymisées |
| `id_*`, `DeviceType`, `DeviceInfo` | identité numérique (navigateur, système, appareil) |
""")
