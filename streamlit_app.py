"""Application Streamlit multi-pages — détection de fraude explicable (IEEE-CIS).
Lancer : streamlit run streamlit_app.py"""
import streamlit as st

st.set_page_config(page_title="Fraude explicable — IEEE-CIS", page_icon="🛡️", layout="wide")

try:
    from app_pages import common  # noqa: F401  (vérifie la présence des artefacts avant d'afficher le menu)
    common.get_engine()
except FileNotFoundError as err:
    st.error(str(err))
    st.stop()

pages = {
    "": [st.Page("app_pages/accueil.py", title="Accueil", icon="🏠", default=True)],
    "Démarche": [
        st.Page("app_pages/p1_donnees.py", title="1 · Données", icon="🗂️", url_path="donnees"),
        st.Page("app_pages/p2_qualite.py", title="2 · Qualité", icon="🧹", url_path="qualite"),
        st.Page("app_pages/p3_exploration.py", title="3 · Exploration", icon="🔎", url_path="exploration"),
        st.Page("app_pages/p4_preparation.py", title="4 · Préparation", icon="🛠️", url_path="preparation"),
        st.Page("app_pages/p5_modelisation.py", title="5 · Modélisation", icon="🤖", url_path="modelisation"),
        st.Page("app_pages/p6_evaluation.py", title="6 · Évaluation", icon="📈", url_path="evaluation"),
    ],
    "Explicabilité": [
        st.Page("app_pages/p7_explicabilite_globale.py", title="7 · Globale (SHAP)", icon="🌍", url_path="shap-global"),
        st.Page("app_pages/p8_explicabilite_locale.py", title="8 · Locale et « et si ? »", icon="🔬", url_path="shap-local"),
    ],
}
st.navigation(pages).run()
