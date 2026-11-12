import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import altair as alt
import pandas as pd
import streamlit as st

from app_pages import common as C

E = C.get_engine()
C.header("8 ·", "Explicabilité locale et scénarios « et si ? »",
         "Pourquoi cette transaction est-elle signalée ? Que se passerait-il si on modifiait certaines valeurs ?")
st.caption("Échantillon de test enrichi en fraudes (~20 %) pour la démonstration ; les métriques des autres pages portent sur tout le jeu de test.")

# ------------------------------------------------------------------ choix de la transaction
s, thr = E.sample, E.thr
cases = {}
for name, sel, pick in (("Fraude détectée", (s.isFraud == 1) & (s.proba >= thr), "max"),
                        ("Fausse alerte (faux positif)", (s.isFraud == 0) & (s.proba >= thr), "max"),
                        ("Fraude manquée (faux négatif)", (s.isFraud == 1) & (s.proba < thr), "min")):
    sub = s[sel]
    if len(sub):
        cases[name] = int(sub.proba.idxmax() if pick == "max" else sub.proba.idxmin())

mode = st.radio("Choisir une transaction", ["Cas typiques", "Parcourir la liste"], horizontal=True)
idx = None
if mode == "Cas typiques":
    if not cases:
        st.info("Aucun cas typique disponible dans l'échantillon.")
        st.stop()
    name = st.selectbox("Cas", list(cases))
    idx = cases[name]
    st.caption({"Fraude détectée": "Une vraie fraude correctement signalée, avec le score le plus élevé.",
                "Fausse alerte (faux positif)": "Une transaction légitime signalée à tort : l'explication aide l'analyste à la débloquer.",
                "Fraude manquée (faux négatif)": "Une vraie fraude non signalée : l'explication révèle ce qui a trompé le modèle."}[name])
else:
    f = st.columns([1, 1, 1, 1, 0.8])
    label = f[0].selectbox("Étiquette réelle", ["all", "fraud", "legit"], format_func={"all": "Toutes", "fraud": "Fraudes", "legit": "Légitimes"}.get)
    flagged = f[1].selectbox("Alerte", ["all", "yes", "no"], format_func={"all": "Toutes", "yes": "Signalées", "no": "Non signalées"}.get)
    sort = f[2].selectbox("Tri", ["proba", "random", "amount", "time"],
                          format_func={"proba": "Score décroissant", "random": "Aléatoire", "amount": "Montant décroissant", "time": "Date"}.get)
    limit = f[3].slider("Lignes", 10, 200, 40, step=10)
    if "seed" not in st.session_state:
        st.session_state.seed = 0
    if f[4].button("🎲 Retirer", width="stretch"):
        st.session_state.seed += 1
    view = E.transactions(label, flagged, sort, limit, st.session_state.seed)
    ev = st.dataframe(view.drop(columns="idx"), hide_index=True, width="stretch", height=300, on_select="rerun", selection_mode="single-row",
                      column_config={"Score": st.column_config.ProgressColumn(min_value=0.0, max_value=1.0, format="%.3f")})
    if not ev.selection.rows:
        st.info("Cliquez sur une ligne pour afficher son explication.")
        st.stop()
    idx = int(view.iloc[ev.selection.rows[0]]["idx"])

# ------------------------------------------------------------------ explication
base = E.explain_idx(idx)
wi_key = f"wi_{idx}"
res = st.session_state.get(wi_key)
shown = res["out"] if res else base

k = st.columns(4)
k[0].metric("Score", f"{shown['proba']:.3f}", delta=f"{shown['proba'] - base['proba']:+.3f} vs initial" if res else None, delta_color="inverse")
k[1].metric("Décision", "🚨 Signalée" if shown["flagged"] else "✅ Non signalée", help=f"Seuil d'alerte : {thr:.2f}")
k[2].metric("Réalité", "Fraude" if base["fraud"] else "Légitime")
k[3].metric("Risque moyen du modèle", f"{base['base_proba']:.1%}")

items = sorted(shown["items"], key=lambda i: -abs(i["shap"]))
d = pd.DataFrame({"label": [f"{i['feature']} = {i['value'] if i['value'] is not None else 'manquant'}" for i in items] + ["autres variables"],
                  "shap": [i["shap"] for i in items] + [shown["others"]]})
st.subheader("Pourquoi ce score ?")
st.caption("Rouge : pousse vers « fraude ». Bleu : pousse vers « légitime ». Unité : log-odds (SHAP). "
           "Le score part du risque moyen du modèle et s'ajuste variable par variable.")
st.altair_chart(alt.Chart(d).mark_bar().encode(
    x=alt.X("shap:Q", title="Contribution SHAP"), y=alt.Y("label:N", sort=None, title=None),
    color=alt.condition(alt.datum.shap > 0, alt.value(C.RED), alt.value(C.BLUE)),
    tooltip=["label", alt.Tooltip("shap:Q", format="+.3f")]).properties(height=380), width="stretch")

# ------------------------------------------------------------------ et si
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
    st.success(f"Score initial {base['proba']:.3f} → nouveau score {res['out']['proba']:.3f} (modifications : {', '.join(res['changes'])})")
