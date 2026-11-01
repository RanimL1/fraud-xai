"""Application Streamlit — détection de fraude explicable (IEEE-CIS). Lancer : streamlit run streamlit_app.py"""
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))
from engine import Engine  # noqa: E402

st.set_page_config(page_title="Fraude explicable — IEEE-CIS", page_icon="🛡️", layout="wide")


@st.cache_resource(show_spinner="Chargement du modèle…")
def get_engine() -> Engine:
    return Engine()


try:
    E = get_engine()
except FileNotFoundError as err:
    st.error(str(err))
    st.stop()

M = E.metrics
st.title("🛡️ Détection de fraude explicable")
st.caption("IEEE-CIS (Vesta) · LightGBM · explications SHAP exactes · scénarios « et si ? »")

tab_overview, tab_tx = st.tabs(["Vue d'ensemble", "Transactions et explications"])

# ----------------------------------------------------------------------- vue d'ensemble
with tab_overview:
    c = st.columns(6)
    c[0].metric("ROC-AUC", f"{M['roc_auc']:.3f}")
    c[1].metric("PR-AUC", f"{M['pr_auc']:.3f}", help=f"Référence au hasard : {M['prevalence']:.3f}")
    c[2].metric("Précision", f"{M['precision']:.1%}")
    c[3].metric("Rappel", f"{M['recall']:.1%}")
    c[4].metric("F1", f"{M['f1']:.3f}")
    c[5].metric("Coût évité", f"{M['cost_reduction']:.0%}",
                help=f"{M['cost_no_model']:,.0f} $ sans modèle → {M['cost_with_model']:,.0f} $ avec (seuil {M['threshold_cost']:.2f})")
    st.caption(f"Métriques calculées sur tout le jeu de test temporel, seuil d'alerte {M['threshold']:.2f}.")

    left, right = st.columns([1.4, 1])
    with left:
        st.subheader("Importance globale (moyenne des |SHAP|)")
        imp = pd.DataFrame(E.meta["importance"][:20])
        st.altair_chart(alt.Chart(imp).mark_bar(color="#3b73c4").encode(
            x=alt.X("mean_abs_shap:Q", title="Moyenne des |SHAP| (log-odds)"),
            y=alt.Y("feature:N", sort="-x", title=None)).properties(height=480), width="stretch")
    with right:
        st.subheader("Matrice de confusion")
        st.dataframe(pd.DataFrame([[M["tn"], M["fp"]], [M["fn"], M["tp"]]],
                                  index=["Réel légitime", "Réel fraude"],
                                  columns=["Prédit légitime", "Prédit fraude"]), width="stretch")
        st.info("Beaucoup de variables (V*, C*, D*) sont anonymisées par Vesta : l'explication indique "
                "quelles variables comptent, pas toujours ce qu'elles signifient. Elle est statistique, non causale.")

# --------------------------------------------------------------------------- transactions
with tab_tx:
    f = st.columns([1, 1, 1, 1, 0.8])
    label = f[0].selectbox("Étiquette réelle", ["all", "fraud", "legit"],
                           format_func={"all": "Toutes", "fraud": "Fraudes", "legit": "Légitimes"}.get)
    flagged = f[1].selectbox("Alerte", ["all", "yes", "no"],
                             format_func={"all": "Toutes", "yes": "Signalées", "no": "Non signalées"}.get)
    sort = f[2].selectbox("Tri", ["proba", "random", "amount", "time"],
                          format_func={"proba": "Score décroissant", "random": "Aléatoire",
                                       "amount": "Montant décroissant", "time": "Date"}.get)
    limit = f[3].slider("Lignes", 10, 200, 40, step=10)
    if "seed" not in st.session_state:
        st.session_state.seed = 0
    if f[4].button("🎲 Retirer", width="stretch"):
        st.session_state.seed += 1

    view = E.transactions(label, flagged, sort, limit, st.session_state.seed)
    st.caption("Échantillon de test enrichi en fraudes (~20 %) pour la démonstration. Cliquer sur une ligne pour l'expliquer.")
    ev = st.dataframe(view.drop(columns="idx"), hide_index=True, width="stretch", height=300,
                      on_select="rerun", selection_mode="single-row",
                      column_config={"Score": st.column_config.ProgressColumn(min_value=0.0, max_value=1.0, format="%.3f")})
    if not ev.selection.rows:
        st.info("Sélectionnez une transaction pour afficher son explication.")
        st.stop()

    idx = int(view.iloc[ev.selection.rows[0]]["idx"])
    base = E.explain_idx(idx)
    wi_key = f"wi_{idx}"
    res = st.session_state.get(wi_key)       # dernier scénario « et si ? » pour cette transaction
    shown = res["out"] if res else base

    k = st.columns(4)
    k[0].metric("Score", f"{shown['proba']:.3f}",
                delta=f"{shown['proba'] - base['proba']:+.3f} vs initial" if res else None, delta_color="inverse")
    k[1].metric("Décision", "🚨 Signalée" if shown["flagged"] else "✅ Non signalée")
    k[2].metric("Réalité", "Fraude" if base["fraud"] else "Légitime")
    k[3].metric("Risque moyen du modèle", f"{base['base_proba']:.1%}")

    items = sorted(shown["items"], key=lambda i: -abs(i["shap"]))
    d = pd.DataFrame({
        "label": [f"{i['feature']} = {i['value'] if i['value'] is not None else 'manquant'}" for i in items] + ["autres variables"],
        "shap": [i["shap"] for i in items] + [shown["others"]]})
    st.subheader("Pourquoi ce score ?")
    st.caption("Rouge : pousse vers « fraude ». Bleu : pousse vers « légitime ». Unité : log-odds (SHAP).")
    st.altair_chart(alt.Chart(d).mark_bar().encode(
        x=alt.X("shap:Q", title="Contribution SHAP"), y=alt.Y("label:N", sort=None, title=None),
        color=alt.condition(alt.datum.shap > 0, alt.value("#d64545"), alt.value("#3b73c4")),
        tooltip=["label", alt.Tooltip("shap:Q", format="+.3f")]).properties(height=380), width="stretch")

    st.subheader("Et si… ?")
    st.caption("Modifiez des valeurs (ou écrivez « manquant »), puis recalculez le score.")
    with st.form(f"form_{idx}"):
        grid = st.columns(3)
        entries, original = {}, {}
        for n, it in enumerate(base["items"]):
            feat, val = it["feature"], it["value"]
            cur = res["changes"].get(feat) if res and feat in res["changes"] else val
            txt = "manquant" if cur is None else str(cur)
            original[feat] = "manquant" if val is None else str(val)
            if feat in E.enc:
                opts = ["manquant"] + E.enc[feat] + ["(autre)"]
                entries[feat] = grid[n % 3].selectbox(feat, opts, index=opts.index(txt) if txt in opts else 0)
            else:
                entries[feat] = grid[n % 3].text_input(feat, value=txt)
        c1, c2 = st.columns([1, 5])
        go = c1.form_submit_button("Recalculer", type="primary")
        reset = c2.form_submit_button("Réinitialiser")
    if reset:
        st.session_state.pop(wi_key, None)
        st.rerun()
    if go:
        changes = {f_: v for f_, v in entries.items() if v != original[f_]}
        if not changes:
            st.warning("Aucune valeur modifiée.")
        else:
            try:
                st.session_state[wi_key] = {"changes": changes, "out": E.explain_idx(idx, changes)}
                st.rerun()
            except ValueError as err:
                st.error(str(err))
    if res:
        st.success(f"Score initial {base['proba']:.3f} → nouveau score {res['out']['proba']:.3f} "
                   f"(modifications : {', '.join(res['changes'])})")
