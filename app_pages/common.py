"""Éléments partagés par les pages : chargement des artefacts, couleurs, graphiques réutilisables."""
import json
import sys
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
FIG = ART / "figures"
sys.path.insert(0, str(ROOT / "src"))
from engine import Engine  # noqa: E402

BLUE, RED, ORANGE, GREEN, GREY = "#3b73c4", "#d64545", "#F58518", "#54A24B", "#bbbbbb"
CLASS_COLORS = alt.Scale(domain=["Légitime", "Fraude"], range=[BLUE, RED])


@st.cache_resource(show_spinner="Chargement du modèle…")
def get_engine() -> Engine:
    return Engine(ART)


@st.cache_data(show_spinner=False)
def _load_profile():
    p = ART / "profile.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


@st.cache_data(show_spinner=False)
def get_shap_sample() -> pd.DataFrame:
    return pd.read_parquet(ART / "shap_sample.parquet")


def profile() -> dict:
    prof = _load_profile()
    if prof is None:
        st.error("Profil du projet introuvable (`artifacts/profile.json`). Relancez `python train.py` ou le notebook "
                 "(section « Export ») pour le générer.")
        st.stop()
    return prof


def header(num: str, title: str, goal: str):
    st.title(f"{num} {title}" if num else title)
    st.caption(goal)


def decision(text: str):
    st.success("**Décision.** " + text, icon="✅")


# ---------------------------------------------------------------- graphiques
def centers(edges):
    e = np.asarray(edges, dtype=float)
    return (e[:-1] + e[1:]) / 2


def bar_h(df, x, y, title=None, color=BLUE, height=None, fmt=None):
    enc = [alt.Tooltip(f"{x}:Q", format=fmt or ",.3~g"), alt.Tooltip(f"{y}:N")]
    ch = alt.Chart(df).mark_bar(color=color).encode(x=alt.X(f"{x}:Q", title=None), y=alt.Y(f"{y}:N", sort="-x", title=None), tooltip=enc)
    return ch.properties(title=title or "", **({"height": height} if height else {}))


def bar_v(df, x, y, title=None, color=BLUE, height=260):
    return alt.Chart(df).mark_bar(color=color).encode(
        x=alt.X(f"{x}:N", sort=None, title=None, axis=alt.Axis(labelAngle=0)), y=alt.Y(f"{y}:Q", title=None),
        tooltip=[x, alt.Tooltip(f"{y}:Q", format=",.3~g")]).properties(title=title or "", height=height)


def hist_chart(h, title=None, xtitle=None, color=BLUE, height=260):
    e = np.asarray(h["edges"], dtype=float)
    d = pd.DataFrame({"x0": e[:-1], "x1": e[1:], "n": h["counts"]})
    return alt.Chart(d).mark_bar(color=color).encode(
        x=alt.X("x0:Q", title=xtitle), x2="x1:Q", y=alt.Y("n:Q", title="Nombre"),
        tooltip=[alt.Tooltip("x0:Q", format=",.3~g", title="de"), alt.Tooltip("x1:Q", format=",.3~g", title="à"), "n"]
    ).properties(title=title or "", height=height)


def class_hist(h, title=None, xtitle=None, height=260):
    c = centers(h["edges"])
    d = pd.concat([pd.DataFrame({"x": c, "densité": h["legit"], "classe": "Légitime"}),
                   pd.DataFrame({"x": c, "densité": h["fraud"], "classe": "Fraude"})])
    return alt.Chart(d).mark_area(opacity=0.45, line=True, interpolate="step-after").encode(
        x=alt.X("x:Q", title=xtitle), y=alt.Y("densité:Q"), color=alt.Color("classe:N", scale=CLASS_COLORS, title=None),
        tooltip=["classe", alt.Tooltip("x:Q", format=",.3~g"), alt.Tooltip("densité:Q", format=".4f")]).properties(title=title or "", height=height)


def line_chart(df, x, y, title=None, color=BLUE, height=260, xtitle=None, ytitle=None):
    return alt.Chart(df).mark_line(color=color).encode(
        x=alt.X(f"{x}:Q", title=xtitle), y=alt.Y(f"{y}:Q", title=ytitle),
        tooltip=[alt.Tooltip(f"{x}:Q", format=",.3~g"), alt.Tooltip(f"{y}:Q", format=",.4~g")]).properties(title=title or "", height=height)


def show(chart):
    st.altair_chart(chart, width="stretch")


def show_figure(name: str, caption: str = ""):
    p = FIG / name
    if p.exists():
        st.image(str(p), caption=caption or None)
