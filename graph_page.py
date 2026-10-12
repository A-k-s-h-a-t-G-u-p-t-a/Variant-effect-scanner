"""Protein similarity graph view for the Streamlit app (rendered by app.py)."""
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import py3Dmol
import streamlit as st

from src.evaluate import RESULTS
from src.presets import GROUPS
from src.structures import parse_ca

ROOT = RESULTS.parent
GROUP_COLORS = {"Case study": "#2a78d6", "Benchmark": "#eb6834", "Enzyme": "#1baf7a"}  # slots 1-3
EDGE_COLOR = "#8a8984"
# AlphaFold's own confidence bands
PLDDT_BANDS = [(90, 101, "#0053d6", "very high (90+)"), (70, 90, "#65cbf3", "confident (70-90)"),
               (50, 70, "#ffdb13", "low (50-70)"), (0, 50, "#ff7d45", "very low (<50)")]
VIEWS = {"embedding": "Sequence embedding: what the model thinks each protein is",
         "substitution": "Substitution grammar: how each protein tolerates change"}


@st.cache_data
def load_graph():
    path = RESULTS / "graph.json"
    return json.loads(path.read_text()) if path.exists() else None


@st.cache_data
def load_pairs():
    return pd.read_csv(RESULTS / "protein_pairs.csv", keep_default_na=False)


@st.cache_data
def benchmark_spearman():
    return pd.read_csv(RESULTS / "proteingym_spearman.csv").set_index("DMS_id")["spearman"].to_dict()


def graph_figure(g, view, shown):
    v = g["views"][view]
    pos = np.array(v["pos"])
    z = np.array(g["alignment_z"])
    fig = go.Figure()
    for found, name, dash in [(True, "edge: alignment also finds this pair", "solid"),
                              (False, "edge: only the model links this pair", "dot")]:
        xs, ys = [], []
        for i, j in v["edges"]:
            if i in shown and j in shown and (z[i][j] >= g["align_finds_z"]) == found:
                xs += [pos[i, 0], pos[j, 0], None]
                ys += [pos[i, 1], pos[j, 1], None]
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name=name, hoverinfo="skip",
                                 line={"color": EDGE_COLOR, "width": 2 if found else 1.2, "dash": dash}))
    for group, color in GROUP_COLORS.items():
        idx = [i for i in sorted(shown) if g["nodes"][i]["group"] == group]
        if not idx:
            continue
        nodes = [g["nodes"][i] for i in idx]
        fig.add_trace(go.Scatter(
            x=pos[idx, 0], y=pos[idx, 1], mode="markers+text", name=GROUPS[group],
            text=[n["short"] for n in nodes], textposition="top center", textfont={"size": 10},
            customdata=idx,
            hovertext=[f"<b>{n['short']}</b>: {n['name']}<br>{n['organism']}<br>"
                       f"{n['scanned_length']} residues scanned"
                       + (f"<br>AlphaFold mean confidence {n['plddt']:.0f}" if n["plddt"] else "")
                       for n in nodes],
            hovertemplate="%{hovertext}<extra></extra>",
            marker={"color": color, "size": 14, "line": {"color": "white", "width": 2}},
            selected={"marker": {"opacity": 1}}, unselected={"marker": {"opacity": 1}}))
    sel = st.session_state.get("graph_node")
    if sel in shown:
        fig.add_trace(go.Scatter(x=[pos[sel, 0]], y=[pos[sel, 1]], mode="markers", hoverinfo="skip",
                                 showlegend=False,
                                 marker={"size": 26, "color": "rgba(0,0,0,0)",
                                         "line": {"color": "#0b0b0b", "width": 2.5}}))
    fig.update_layout(height=600, margin={"l": 10, "r": 10, "t": 10, "b": 10}, clickmode="event+select",
                      legend={"orientation": "h", "y": -0.02, "yanchor": "top"},
                      xaxis={"visible": False}, yaxis={"visible": False, "scaleanchor": "x"},
                      dragmode="pan")
    return fig


def structure_html(pdb_path, height=420):
    _, plddt = parse_ca(pdb_path)
    view = py3Dmol.view(width="100%", height=height)
    view.addModel((ROOT / pdb_path).read_text(), "pdb")
    for lo, hi, color, _ in PLDDT_BANDS:
        resi = [i + 1 for i, b in enumerate(plddt) if lo <= b < hi]
        if resi:
            view.setStyle({"resi": resi}, {"cartoon": {"color": color}})
    view.zoomTo()
    view.spin(False)
    return view._make_html()


def node_panel(g, view, i, open_protein):
    n = g["nodes"][i]
    st.subheader(f"{n['short']}: {n['name']}")
    st.caption(f"{n['organism']} · UniProt {n['accession']} · {GROUPS[n['group']]}")

    a, b, c = st.columns(3)
    a.metric("Residues scanned", n["scanned_length"])
    if n["dms_id"]:
        b.metric("ProteinGym Spearman", f"{benchmark_spearman()[n['dms_id']]:.3f}")
    elif n["site_auc"] is not None:
        b.metric("Functional-site AUC", f"{n['site_auc']:.3f}")
    if n["plddt"]:
        c.metric("AlphaFold confidence", f"{n['plddt']:.0f} / 100")

    if n["structure"]:
        st.iframe(structure_html(n["structure"]), height=440)
        st.caption("AlphaFold predicted structure (drag to rotate, scroll to zoom), coloured by AlphaFold's "
                   "own per-residue confidence: " + " · ".join(f"{label}" for *_, label in PLDDT_BANDS)
                   + ". Orange floppy regions are low-confidence predictions, not real biology.")
        if n["uniprot_length"] != n["scanned_length"]:
            st.caption(f"The structure is the full UniProt protein ({n['uniprot_length']} residues); "
                       f"the scan covers the {n['scanned_length']}-residue region used in the experiment.")
    else:
        st.info("No AlphaFold structure available for this protein.")

    S = np.array(g["views"][view]["similarity"])[i]
    z, ident = np.array(g["alignment_z"])[i], np.array(g["identity"])[i]
    order = [j for j in np.argsort(-S) if j != i][:5]
    st.markdown(f"**Five most similar proteins** ({'embedding' if view == 'embedding' else 'substitution grammar'} view)")
    st.dataframe(pd.DataFrame({
        "Protein": [f"{g['nodes'][j]['short']}: {g['nodes'][j]['name']}" for j in order],
        "Cosine similarity": [round(float(S[j]), 3) for j in order],
        "Alignment z-score": [round(float(z[j]), 1) for j in order],
        "Identity (%)": [round(float(ident[j]), 1) for j in order],
        "Related by alignment": ["yes" if z[j] >= g["align_finds_z"] else "" for j in order],
    }), hide_index=True, width="stretch")
    st.button("Open this protein's mutation scan", on_click=open_protein, args=(n,), type="primary")


def analysis_section(g, view):
    st.divider()
    st.subheader("Does the model see relationships that sequence alignment misses?")
    st.markdown(
        "For every pair of proteins we compared the model's similarity with a **sequence alignment test**: "
        "the local alignment score against scores for 50 shuffled copies of the sequence. A **z-score of 8 "
        "or more** means the sequences are related beyond doubt; below 5 means alignment sees nothing.")
    pairs = load_pairs().copy()
    col = f"{view}_cosine"
    pairs["kind"] = np.where(pairs.alignment_z >= g["align_finds_z"], "related by alignment",
                             np.where(pairs.shared_interpro != "", "same superfamily, no alignment signal",
                                      "no known relationship"))
    fig = go.Figure()
    styles = {"no known relationship": ("#2a78d6", 6, 0.35),
              "related by alignment": ("#eb6834", 12, 1.0),
              "same superfamily, no alignment signal": ("#1baf7a", 12, 1.0)}
    for kind, (color, size, opacity) in styles.items():
        d = pairs[pairs.kind == kind]
        fig.add_trace(go.Scatter(
            x=d.alignment_z, y=d[col], mode="markers", name=kind,
            text=d.protein_a + " / " + d.protein_b,
            hovertemplate="%{text}<br>alignment z %{x:.1f}<br>cosine %{y:.3f}<extra></extra>",
            marker={"color": color, "size": size, "opacity": opacity,
                    "line": {"color": "white", "width": 1 if size > 6 else 0}}))
    fig.add_vline(x=g["align_finds_z"], line={"color": EDGE_COLOR, "dash": "dot", "width": 1})
    fig.update_layout(height=420, margin={"l": 10, "r": 10, "t": 10, "b": 10},
                      xaxis={"title": "sequence alignment z-score (shuffle test)"},
                      yaxis={"title": f"cosine similarity ({view} view)"},
                      legend={"orientation": "h", "y": -0.18, "yanchor": "top"})
    st.plotly_chart(fig, width="stretch")

    rank = f"{view}_rank"
    notable = pairs[pairs.kind != "no known relationship"].sort_values("alignment_z", ascending=False)
    st.markdown("**The only related pairs in this set, and where this view ranks them (out of 595 pairs):**")
    st.dataframe(notable[["protein_a", "protein_b", "name_a", "name_b", "alignment_z", rank, "kind"]].rename(
        columns={"protein_a": "A", "protein_b": "B", "name_a": "Protein A", "name_b": "Protein B",
                 "alignment_z": "Alignment z", rank: "Rank in this view", "kind": "Relationship"}),
        hide_index=True, width="stretch",
        column_config={"Alignment z": st.column_config.NumberColumn(format="%.1f")})

    top = pairs[pairs.alignment_z < g["no_signal_z"]].nlargest(5, col)
    st.markdown("**Top five pairs this view links despite no alignment signal:**")
    st.dataframe(top[["protein_a", "protein_b", "name_a", "name_b", "alignment_z", col, "shared_interpro"]]
                 .rename(columns={"protein_a": "A", "protein_b": "B", "name_a": "Protein A",
                                  "name_b": "Protein B", "alignment_z": "Alignment z",
                                  col: "Cosine", "shared_interpro": "Shared InterPro entries"}),
                 hide_index=True, width="stretch",
                 column_config={"Alignment z": st.column_config.NumberColumn(format="%.1f"),
                                "Cosine": st.column_config.NumberColumn(format="%.3f")})

    s = g["stats"]
    st.markdown(
        f"""**What this shows (honestly):**
- The 35 proteins were chosen without regard to family, so almost all pairs are unrelated: alignment finds
  only **2 related pairs** (cathepsin L / cathepsin B, and granzyme B / prostate-specific antigen).
- The embedding view ranks the cathepsins **4th** of 595 pairs but the two serine proteases only **213th**;
  the substitution-grammar view ranks them **13th and 28th**, so it recovers both.
- One pair shares a structural superfamily with no detectable sequence similarity: **HRas and sulfotransferase
  1E1** (P-loop nucleoside triphosphate hydrolase fold). Neither view links them, so in this set **we find no
  evidence that the model detects remote relationships alignment misses**.
- The top "model-only" pairs share no InterPro entry: at this level, mean-pooled embeddings mostly say
  "both are compact enzymes". Agreement between the two views is modest (Spearman
  {s['spearman_embedding_substitution']:.2f}).
- A proper remote-homology test needs a protein set built for it (for example structural-classification
  superfamilies with under 20% identity)."""
    )


def render(open_protein):
    g = load_graph()
    st.title("Protein similarity graph")
    if g is None:
        st.info("Graph data not built yet. Run `.\\.venv\\Scripts\\python.exe scripts\\build_graph.py`.")
        return
    st.markdown("Each dot is one of the 35 pre-scanned proteins; each protein is linked to its "
                f"{g['k']} most similar proteins. Dots are placed so that proteins high on each other's "
                "similarity lists sit close together. **Click a dot** to see its 3D structure.")

    c1, c2 = st.columns([3, 2])
    view = c1.radio("Similarity based on", list(VIEWS), format_func=VIEWS.get, key="graph_view")
    groups = c2.multiselect("Collections shown", list(GROUPS), default=list(GROUPS),
                            format_func=GROUPS.get, key="graph_groups")
    shown = {i for i, n in enumerate(g["nodes"]) if n["group"] in groups}

    # Read the click before drawing, so the selection ring moves on the same rerun. The chart keeps
    # re-reporting its last selection, so only a new click should change the selected protein.
    event = st.session_state.get(f"graph_{view}")
    points = [p for p in (event.selection.points if event else []) if "customdata" in p]
    clicked = int(points[0]["customdata"]) if points else None
    if clicked is not None and clicked != st.session_state.get("graph_last_click"):
        st.session_state["graph_node"] = clicked
    st.session_state["graph_last_click"] = clicked

    left, right = st.columns([3, 2], gap="large")
    with left:
        st.plotly_chart(graph_figure(g, view, shown), width="stretch", key=f"graph_{view}",
                        on_select="rerun", selection_mode="points", config={"scrollZoom": True})
        labels = {i: f"{n['short']}: {n['name']}" for i, n in enumerate(g["nodes"])}
        if st.session_state.get("graph_node") not in labels:
            st.session_state["graph_node"] = 0
        st.selectbox("Or choose a protein", list(labels), format_func=labels.get, key="graph_node")
    with right:
        node_panel(g, view, st.session_state["graph_node"], open_protein)

    analysis_section(g, view)
