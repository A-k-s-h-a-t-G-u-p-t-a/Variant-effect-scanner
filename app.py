"""Variant Effect Scanner: interactive front end.

Run with:  .\\.venv\\Scripts\\python.exe -m streamlit run app.py
"""
import hashlib
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import spearmanr

import graph_page
from src.evaluate import PG_DIR, RESULTS, cached_logprobs, predict_additive
from src.presets import BENCHMARK_NAMES, GROUPS, list_proteins
from src.score import AAS, entropy_from_logprobs, logratio_from_logprobs

st.set_page_config(page_title="Variant Effect Scanner", layout="wide")

# Diverging red (damaging) <-> neutral gray (0) <-> blue (tolerated); zero sits at 10/12 of [-10, 2].
RED_BLUE = [[0.0, "#5e1010"], [0.25, "#a82a2a"], [0.5, "#e34948"], [0.7, "#f2a99f"],
            [10 / 12, "#f0efec"], [1.0, "#2a78d6"]]
SERIES = "#2a78d6"     # categorical slot 1
HIGHLIGHT = "#eb6834"  # categorical slot 2
WILD_TYPE = "#8a8984"
WINDOW = 60
SECONDS_PER_RESIDUE = 1.5
MAX_ENTROPY = float(np.log(20))

# ---------------------------------------------------------------- data loading

@st.cache_data
def load_presets() -> list[dict]:
    return list_proteins()


@st.cache_resource(show_spinner="Loading the ESM Cambrian 300M model (first time only)...")
def get_model():
    from src.model import load_model
    return load_model()


@st.cache_data(show_spinner=False)
def analyse(key: str, sequence: str):
    """Log-ratio matrix, per-position entropy, and the within-protein percentile of each cell."""
    lp = cached_logprobs(None, None, key, sequence)
    lr = logratio_from_logprobs(lp, sequence)
    ent = entropy_from_logprobs(lp)
    wt_mask = np.zeros_like(lr, dtype=bool)
    wt_mask[np.arange(len(sequence)), [AAS.index(a) for a in sequence]] = True
    others = np.sort(lr[~wt_mask])
    # percentage of this protein's substitutions that score higher (are less damaging)
    pct = 100 * (others.size - np.searchsorted(others, lr, side="right")) / others.size
    pct[wt_mask] = np.nan
    return lr, ent, pct, wt_mask


@st.cache_data
def benchmark_table() -> pd.DataFrame:
    df = pd.read_csv(RESULTS / "proteingym_spearman.csv")
    df.insert(0, "Protein", df["DMS_id"].map(BENCHMARK_NAMES))
    return df


def substitutions_frame(sequence, lr, pct, wt_mask) -> pd.DataFrame:
    pos, col = np.nonzero(~wt_mask)
    return pd.DataFrame({
        "Mutation": [f"{sequence[p]}{p + 1}{AAS[c]}" for p, c in zip(pos, col)],
        "Position": pos + 1,
        "Wild type": [sequence[p] for p in pos],
        "Mutant": [AAS[c] for c in col],
        "Score": lr[pos, col].round(3),
        "More damaging than (%)": pct[pos, col].round(1),
    })


def describe_percentile(p: float) -> str:
    if p >= 90:
        return "Among the 10% most damaging substitutions the model predicts for this protein."
    if p >= 50:
        return "More damaging than a typical substitution in this protein."
    if p >= 10:
        return "Less damaging than a typical substitution in this protein."
    return "Among the 10% most tolerated substitutions the model predicts for this protein."


# ---------------------------------------------------------------- sidebar: choose a protein

def open_protein(node: dict):
    """Jump from the similarity graph to that protein's single-protein view (runs before widgets)."""
    st.session_state["mode"] = "Single protein"
    st.session_state["source"] = "Preset protein"
    st.session_state["collection"] = node["group"]
    st.session_state["protein_label"] = next(p["label"] for p in load_presets() if p["key"] == node["key"])


st.sidebar.title("Variant Effect Scanner")
st.sidebar.caption("Zero-shot mutation effect prediction with the ESM Cambrian 300M protein language model.")

MODES = {"Single protein": "Mutation scanner: one protein",
         "Protein similarity graph": "Protein similarity graph: all 35 proteins, with 3D structures"}
st.session_state.setdefault("mode", "Single protein")
mode = st.segmented_control("Choose a view", list(MODES), format_func=MODES.get, key="mode",
                            required=True, width="stretch")
if mode == "Protein similarity graph":
    st.sidebar.divider()
    st.sidebar.warning("Research tool only. Scores are model predictions, not a clinical diagnosis.")
    graph_page.render(open_protein)
    st.stop()

source = st.sidebar.radio("Protein source", ["Preset protein", "Paste a sequence"], key="source")
protein = None

if source == "Preset protein":
    presets = load_presets()
    group = st.sidebar.selectbox("Collection", list(GROUPS), format_func=GROUPS.get, key="collection")
    in_group = {p["label"]: p for p in presets if p["group"] == group}
    if st.session_state.get("protein_label") not in in_group:
        st.session_state["protein_label"] = next(iter(in_group))
    protein = in_group[st.sidebar.selectbox("Protein", list(in_group), key="protein_label")]
    st.sidebar.caption(f"{len(presets)} preset proteins, all pre-scanned: they load instantly.")
else:
    name = st.sidebar.text_input("Name (optional)", value="My protein")
    raw = st.sidebar.text_area("Protein sequence (one-letter codes, FASTA accepted)", height=150)
    lines = [l for l in raw.splitlines() if not l.startswith(">")]
    seq = re.sub(r"\s+", "", "".join(lines)).upper()
    bad = sorted(set(seq) - set(AAS))
    if not seq:
        st.sidebar.info("Paste a sequence to begin.")
    elif bad:
        st.sidebar.error(f"Not standard amino acids: {', '.join(bad)}")
    elif len(seq) < 10:
        st.sidebar.error("Sequence is too short (minimum 10 residues).")
    else:
        key = "custom_" + hashlib.sha1(seq.encode()).hexdigest()[:12]
        protein = {"label": f"{name} (pasted, {len(seq)} residues)", "key": key, "sequence": seq,
                   "group": "Custom"}
        if not (RESULTS / "logprobs" / f"{key}.npy").exists():
            minutes = len(seq) * SECONDS_PER_RESIDUE / 60
            st.sidebar.write(f"{len(seq)} residues: a full scan takes about **{minutes:.0f} minute(s)** "
                             "on this laptop. The result is saved, so it only runs once.")
            if len(seq) > 1000:
                st.sidebar.warning("Long sequences are slow and memory-hungry; consider scanning a domain.")
            if st.sidebar.button("Scan this protein", type="primary"):
                tok, model = get_model()
                bar = st.sidebar.progress(0.0, text="Scanning...")
                cached_logprobs(tok, model, key, seq,
                                progress=lambda f: bar.progress(f, text=f"Scanning... {f:.0%}"))
                st.rerun()
            protein = None

st.sidebar.divider()
st.sidebar.warning("Research tool only. Scores are model predictions, not a clinical diagnosis.")

if protein is None:
    st.title("Variant Effect Scanner")
    st.info("Paste a sequence and press **Scan this protein** in the sidebar, "
            "or switch to **Preset protein**: those load instantly.")
    st.stop()

# ---------------------------------------------------------------- main area

seq, key = protein["sequence"], protein["key"]
L = len(seq)
lr, ent, pct, wt_mask = analyse(key, seq)

st.title(protein["label"])
c1, c2, c3 = st.columns(3)
c1.metric("Length", f"{L} residues")
c2.metric("Substitutions scored", f"{L * 19:,}")
c3.metric("Median substitution score", f"{np.median(lr[~wt_mask]):.2f}")

tabs = st.tabs(["Heatmap", "Mutation lookup", "Most damaging", "Position importance", "Validation"])

# ---- 1. heatmap
with tabs[0]:
    st.markdown("Every possible single substitution. **Red = predicted damaging, blue = tolerated**, "
                "gray = neutral. Each column is a position (labelled with its wild-type amino acid); "
                "each row is the amino acid substituted in. Hover a cell for details.")
    starts = list(range(1, L + 1, WINDOW))
    start = starts[0] if len(starts) == 1 else st.select_slider(
        "Positions shown", options=starts, format_func=lambda s: f"{s} to {min(s + WINDOW - 1, L)}")
    end = min(start + WINDOW - 1, L)
    idx = range(start - 1, end)
    hover = [[(f"<b>{seq[i]}{i + 1}{aa}</b><br>wild type (score 0)" if seq[i] == aa else
               f"<b>{seq[i]}{i + 1}{aa}</b><br>score {lr[i, j]:.2f}<br>"
               f"more damaging than {pct[i, j]:.0f}% of substitutions")
              for i in idx] for j, aa in enumerate(AAS)]
    fig = go.Figure(go.Heatmap(
        z=lr[start - 1:end].T, x=[f"{seq[i]}{i + 1}" for i in idx], y=AAS,
        colorscale=RED_BLUE, zmin=-10, zmax=2, xgap=1, ygap=1,
        text=hover, hovertemplate="%{text}<extra></extra>",
        colorbar={"title": {"text": "log-ratio<br>(negative =<br>damaging)"}}))
    fig.update_layout(height=560, margin={"l": 10, "r": 10, "t": 10, "b": 10},
                      xaxis={"title": "position", "tickangle": -90, "tickfont": {"size": 10}},
                      yaxis={"title": "substituted amino acid", "autorange": "reversed"})
    st.plotly_chart(fig, width="stretch")
    st.caption("Scores below -10 are drawn at full red. Wild-type cells are exactly 0 by construction.")

# ---- 2. single mutation lookup
with tabs[1]:
    subs = substitutions_frame(seq, lr, pct, wt_mask)
    default = "H93Y" if key == "P68871" else subs.loc[subs["Score"].idxmin(), "Mutation"]
    query = st.text_input("Mutation (wild type, position, mutant; for example H93Y)", value=default)
    m = re.fullmatch(r"([A-Z])(\d+)([A-Z])", query.strip().upper())
    if not m:
        st.error("Write the mutation as wild-type letter, position, mutant letter, for example E7V.")
    else:
        wt, pos, mut = m.group(1), int(m.group(2)), m.group(3)
        if wt not in AAS or mut not in AAS:
            st.error("Use the 20 standard one-letter amino acid codes.")
        elif not 1 <= pos <= L:
            st.error(f"Position {pos} is outside this protein (1 to {L}).")
        elif seq[pos - 1] != wt:
            near = [p for p in range(max(1, pos - 3), min(L, pos + 3) + 1) if seq[p - 1] == wt]
            msg = f"Position {pos} is **{seq[pos - 1]}**, not {wt}."
            if near:
                msg += (" Did you mean " + ", ".join(f"**{wt}{p}{mut}**" for p in near) + "? "
                        "Clinical numbering sometimes skips the starting Methionine, "
                        "while this sequence counts it as position 1.")
            st.error(msg)
        elif wt == mut:
            st.info("That is the wild-type amino acid: score 0 by definition.")
        else:
            j = AAS.index(mut)
            a, b, c = st.columns(3)
            a.metric("Score (log-ratio)", f"{lr[pos - 1, j]:.2f}")
            b.metric("More damaging than", f"{pct[pos - 1, j]:.0f}% of substitutions")
            c.metric("Model uncertainty at this position", f"{ent[pos - 1]:.2f} of {MAX_ENTROPY:.2f} nats")
            st.write(describe_percentile(pct[pos - 1, j]))

            row = lr[pos - 1]
            colors = [HIGHLIGHT if aa == mut else WILD_TYPE if aa == wt else SERIES for aa in AAS]
            bars = go.Figure(go.Bar(
                x=AAS, y=row, marker={"color": colors, "cornerradius": 4},
                hovertemplate=f"{wt}{pos}%{{x}}: %{{y:.2f}}<extra></extra>"))
            bars.update_layout(height=360, margin={"l": 10, "r": 10, "t": 40, "b": 10},
                               title={"text": f"All 20 amino acids at position {pos} "
                                              f"(gray = wild type {wt}, orange = {mut})",
                                      "font": {"size": 14}},
                               xaxis={"title": "amino acid"}, yaxis={"title": "log-ratio"}, bargap=0.25)
            bars.add_annotation(x=wt, y=0, text="wild type", showarrow=False, yshift=10,
                                font={"color": WILD_TYPE, "size": 11})
            st.plotly_chart(bars, width="stretch")
            st.caption("Bars near 0 are amino acids the model finds about as plausible as the wild type; "
                       "deep negative bars are ones it rejects.")

# ---- 3. most damaging / most tolerated
with tabs[2]:
    subs = substitutions_frame(seq, lr, pct, wt_mask)
    left, right = st.columns(2)
    left.subheader("20 most damaging")
    left.dataframe(subs.nsmallest(20, "Score"), hide_index=True, width="stretch")
    right.subheader("20 most tolerated")
    right.dataframe(subs.nlargest(20, "Score"), hide_index=True, width="stretch")
    st.download_button("Download all substitutions (CSV)", subs.to_csv(index=False).encode(),
                       file_name=f"scan_{key}.csv", mime="text/csv")

# ---- 4. position importance
with tabs[3]:
    sites = protein.get("sites") or []
    st.markdown("**Entropy** measures how unsure the model is about which amino acid belongs at each "
                f"position: 0 means only one amino acid fits; {MAX_ENTROPY:.2f} means all 20 are "
                "equally likely. Dips mark positions the model treats as essential.")
    if sites:
        a, b = st.columns(2)
        a.metric("Experimentally proven functional sites", len(sites))
        b.metric("Area under the curve (entropy predicts sites)", f"{protein['site_auc']:.3f}")
    line = go.Figure()
    for s in sites:
        line.add_vrect(x0=s - 0.5, x1=s + 0.5, fillcolor=HIGHLIGHT, opacity=0.45, line_width=0)
    line.add_trace(go.Scatter(
        x=np.arange(1, L + 1), y=ent, mode="lines", line={"color": SERIES, "width": 2},
        text=[f"{a}{i + 1}" + (" (functional site)" if i + 1 in sites else "") for i, a in enumerate(seq)],
        hovertemplate="%{text}<br>entropy %{y:.2f} nats<extra></extra>", showlegend=False))
    if sites:
        line.add_trace(go.Scatter(
            x=sites, y=ent[np.array(sites) - 1], mode="markers", name="experimental functional site",
            marker={"color": HIGHLIGHT, "size": 10, "line": {"color": "white", "width": 2}},
            hoverinfo="skip", showlegend=False))
    line.update_layout(height=380, margin={"l": 10, "r": 10, "t": 10, "b": 10}, hovermode="x unified",
                       xaxis={"title": "residue position", "range": [0.5, L + 0.5]},
                       yaxis={"title": "entropy (nats)", "range": [0, MAX_ENTROPY + 0.05]})
    st.plotly_chart(line, width="stretch")
    if sites:
        st.caption("Orange dots and bands: active-site and binding-site residues proven by experiment "
                   "(UniProt evidence codes ECO:0000269 or ECO:0007744). The model never saw these labels.")

    order = np.argsort(ent)[:15]
    top = pd.DataFrame({"Position": order + 1, "Amino acid": [seq[i] for i in order],
                        "Entropy (nats)": ent[order].round(3)})
    if sites:
        top["Experimental functional site"] = ["yes" if i + 1 in sites else "" for i in order]
    st.subheader("15 positions where the model is most confident")
    st.dataframe(top, hide_index=True, width="stretch")

# ---- 5. validation against laboratory data
with tabs[4]:
    if "dms_id" in protein:
        dms_id = protein["dms_id"]
        df = pd.read_csv(PG_DIR / f"{dms_id}.csv")
        preds = predict_additive(lr, seq, df["mutant"])
        rho = spearmanr(preds, df["DMS_score"]).correlation
        published = benchmark_table().set_index("DMS_id").loc[dms_id, "pub ESM-C (300M)"]
        a, b, c = st.columns(3)
        a.metric("Spearman correlation (this app)", f"{rho:.3f}")
        b.metric("Published, same model", f"{published:.3f}")
        c.metric("Laboratory-measured variants", f"{len(df):,}")
        scatter = go.Figure(go.Scattergl(
            x=preds, y=df["DMS_score"], mode="markers", text=df["mutant"],
            marker={"color": SERIES, "size": 6, "opacity": 0.35},
            hovertemplate="%{text}<br>predicted %{x:.2f}<br>measured %{y:.2f}<extra></extra>"))
        scatter.update_layout(height=440, margin={"l": 10, "r": 10, "t": 10, "b": 10},
                              xaxis={"title": "predicted score (model)"},
                              yaxis={"title": "measured fitness (laboratory)"})
        st.plotly_chart(scatter, width="stretch")
        st.caption("Each dot is one laboratory-measured variant. Multi-mutants are scored as the sum of "
                   "their single-mutant scores, as ProteinGym does.")
    else:
        st.info("Laboratory measurements exist only for the nine ProteinGym benchmark proteins. "
                "Choose one under **ProteinGym benchmark proteins** to compare predictions with experiment.")

    st.subheader("Benchmark summary: our implementation against published results")
    t = benchmark_table()
    show = t[["Protein", "selection_type", "seq_len", "n_mutants", "spearman",
              "pub ESM-C (300M)", "pub ESM2 (650M)", "pub ESM-1v (single)"]].rename(columns={
        "selection_type": "Measurement", "seq_len": "Length", "n_mutants": "Variants",
        "spearman": "Ours", "pub ESM-C (300M)": "Published ESM Cambrian 300M",
        "pub ESM2 (650M)": "Published ESM-2 650M", "pub ESM-1v (single)": "Published ESM-1v"})
    show["Measurement"] = show["Measurement"].replace({"OrganismalFitness": "Organism fitness"})
    st.dataframe(show, hide_index=True, width="stretch", column_config={
        c: st.column_config.NumberColumn(format="%.3f") for c in show.columns[4:]})
    means = show[["Ours", "Published ESM Cambrian 300M", "Published ESM-2 650M", "Published ESM-1v"]].mean()
    st.write("Mean Spearman: " + " · ".join(f"{k} **{v:.3f}**" for k, v in means.items()))
