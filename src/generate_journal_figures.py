# -*- coding: utf-8 -*-
"""
Publication-Grade Vector PDF / PNG / 600-DPI RGB TIFF Figure Generator for:
"Stage-Aware Attribution Subspace Clustering for Group Counterfactual Recourse
in Student Early-Warning Systems"

Generates:
- Main Figures 1–4 (`fig1_framework`, `fig2_prediction_progression`,
  `fig3_attribution_drift`, `fig4_subspace_ablation`) as vector PDF + 300-DPI PNG
  in `交付文件/manuscript_scirep/figs/` and 600-DPI RGB TIFF (`Figure1.tif`–`Figure4.tif`)
  in `交付文件/SciRep_Submission/Figures/`.
- Supplementary Figures S1–S3 (`supp_fig1_cohort_taxonomy`, `supp_fig2_pareto_frontier`,
  `supp_fig3_risk_archetypes`) as vector PDF + 300-DPI PNG in `交付文件/manuscript_scirep/figs/`.
- Standard archive PNGs (`fig1_overall_framework.png` .. `fig7_risk_archetypes_heatmap.png`)
  in `实验/figures/`.
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10,
    "axes.titlesize": 10.8,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.5,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXP_DIR = os.path.join(BASE_DIR, "实验")
RES_JSON = os.path.join(EXP_DIR, "results", "final_experiment_results.json")
FIG_DIR_EXP = os.path.join(EXP_DIR, "figures")
FIG_DIR_SCIREP = os.path.join(BASE_DIR, "交付文件", "manuscript_scirep", "figs")
FIG_DIR_SUB_TIF = os.path.join(BASE_DIR, "交付文件", "SciRep_Submission", "Figures")

os.makedirs(FIG_DIR_EXP, exist_ok=True)
os.makedirs(FIG_DIR_SCIREP, exist_ok=True)
os.makedirs(FIG_DIR_SUB_TIF, exist_ok=True)

sys.path.insert(0, os.path.join(EXP_DIR, "src"))
from defensive_data_loader import load_uci697, load_xapi_edu, load_oulad_stage
from sa_aasc_recourse import OOFTabularEnsemble, cluster_at_risk_students, compute_iqr_scales

with open(RES_JSON, "r", encoding="utf-8") as f:
    RES = json.load(f)


def _save_outputs(fig, exp_png_name, scirep_stem, tif_name=None):
    """Save PNG in 实验/figures, vector PDF + RGB PNG in manuscript_scirep/figs, and optional 600-DPI RGB TIFF."""
    p_exp = os.path.join(FIG_DIR_EXP, exp_png_name)
    fig.savefig(p_exp, bbox_inches="tight", dpi=300, facecolor="white")

    p_pdf = os.path.join(FIG_DIR_SCIREP, f"{scirep_stem}.pdf")
    p_png = os.path.join(FIG_DIR_SCIREP, f"{scirep_stem}.png")
    fig.savefig(p_pdf, bbox_inches="tight", facecolor="white")
    fig.savefig(p_png, bbox_inches="tight", dpi=300, facecolor="white")

    # Convert PNG to strict RGB (no alpha channel)
    with Image.open(p_png) as im:
        rgb = im.convert("RGB")
        rgb.save(p_png, dpi=(300, 300))

    if tif_name is not None:
        tmp_hi = os.path.join(FIG_DIR_SUB_TIF, f"_tmp_{tif_name}.png")
        p_tif = os.path.join(FIG_DIR_SUB_TIF, tif_name)
        fig.savefig(tmp_hi, bbox_inches="tight", dpi=600, facecolor="white")
        with Image.open(tmp_hi) as im_hi:
            rgb_hi = im_hi.convert("RGB")
            rgb_hi.save(p_tif, format="TIFF", compression="tiff_lzw", dpi=(600, 600))
        if os.path.exists(tmp_hi):
            os.remove(tmp_hi)

    plt.close(fig)
    print(f"[Saved] {scirep_stem}.pdf / .png" + (f" + {tif_name} (600 DPI RGB)" if tif_name else ""))


# ==============================================================================
# MAIN FIGURE 1: Layered Architecture Diagram
# ==============================================================================
def make_fig1_framework():
    df4 = pd.DataFrame(RES["table_04_exp3_clustering_recourse_ablation"])
    df5 = pd.DataFrame(RES["table_05_exp4_advising_capacity_pareto"])

    def _get_mid(ds_name):
        return df4[(df4["dataset"] == ds_name) & (df4["clustering_method"] == "Proposed_SA_AASC") & (df4["constraint_variant"] == "A4_budget@mid")].iloc[0]

    r_uci = _get_mid("UCI #697")
    r_xapi = _get_mid("xAPI-Edu")
    r_oul = _get_mid("OULAD (40% drop)")
    icdh_min = min(r_uci["icdh"], r_xapi["icdh"], r_oul["icdh"])
    icdh_max = max(r_uci["icdh"], r_xapi["icdh"], r_oul["icdh"])

    k5_rows = df5[(df5["advising_capacity_K"] == 5) & (df5["clustering_method"] == "Proposed_SA_AASC")]
    hyb_min = k5_rows["hybrid_validity"].min() * 100
    hyb_max = k5_rows["hybrid_validity"].max() * 100

    fig, ax = plt.subplots(figsize=(13.6, 8.4))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    def draw_band(y_bottom, height, title, border_color, bg_color):
        rect = FancyBboxPatch(
            (1.5, y_bottom), 97.0, height,
            boxstyle="round,pad=0.4,rounding_size=1.2",
            linewidth=1.5, linestyle="--", edgecolor=border_color, facecolor=bg_color, zorder=1
        )
        ax.add_patch(rect)
        hdr = FancyBboxPatch(
            (3.0, y_bottom + height - 3.6), 94.0, 3.1,
            boxstyle="round,pad=0.2,rounding_size=0.6",
            linewidth=1.0, edgecolor=border_color, facecolor=border_color, alpha=0.18, zorder=2
        )
        ax.add_patch(hdr)
        ax.text(50, y_bottom + height - 2.05, title, ha="center", va="center",
                fontsize=10.8, fontweight="bold", color="#1A2530", zorder=3)

    def draw_box(x, y, w, h, title, body, edge_c="#2C5E8A", face_c="#FFFFFF", title_c="#10375C"):
        box = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.3,rounding_size=0.8",
            linewidth=1.3, edgecolor=edge_c, facecolor=face_c, zorder=4
        )
        ax.add_patch(box)
        ax.text(x + w/2, y + h - 1.0, title, ha="center", va="top",
                fontsize=9.3, fontweight="bold", color=title_c, zorder=5)
        ax.plot([x + 1.0, x + w - 1.0], [y + h - 3.3, y + h - 3.3], color=edge_c, lw=0.6, alpha=0.45, zorder=5)
        ax.text(x + w/2, y + (h - 3.4)/2, body, ha="center", va="center",
                fontsize=8.3, color="#2C3E50", zorder=5, linespacing=1.24)

    def draw_arrow(x1, y1, x2, y2):
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle="-|>,head_width=0.5,head_length=0.7", lw=1.6, color="#34495E"),
            zorder=6
        )

    # Stage I
    draw_band(76.5, 21.5,
              "Stage I: Longitudinal Multi-Stage Data Representation & Active-Survivor Cohort Filtering",
              "#2B5B84", "#F4F8FB")
    draw_box(3.5, 78.2, 28.5, 14.5,
             "Multi-Stage Educational Logs",
             "• OULAD ($N=32,593$; 20/40/60/80%)\n• UCI #697 ($N=4,424$; Sem 1 $\\rightarrow$ Sem 2)\n• xAPI-Edu ($N=480$; Behavioral Logs)\n• Stage Sequence $\\{(x_i^{(t)}, m_i^{(t)})\\}_{t=1}^S$",
             edge_c="#2B5B84")
    draw_box(35.5, 78.2, 29.0, 14.5,
             "Cohort Contamination Filter",
             "• Raw Cohort: Retains pre-cutoff exits\n  ($72.5\\%$ of all withdrawals occur by 40%)\n• Active-Survivor Cohort ($\\mathcal{I}_s^{\\mathrm{drop}}$):\n  Enforces $t_{\\mathrm{unreg},i} > t_s$ ($N_{40\\%}=25,224$)",
             edge_c="#2B5B84")
    draw_box(68.0, 78.2, 28.5, 14.5,
             "Five-Tier Mutability Taxonomy",
             "• $\\mathcal{F}_{\\mathrm{beh}}$: Modifiable Behavior; $\\mathcal{F}_{\\mathrm{out}}$: Outcomes\n• $\\mathcal{F}_{\\mathrm{inst}}$: Institutional / Financial Context\n• $\\mathcal{F}_{\\mathrm{imm}}$: Immutable Demographics\n• $\\mathcal{F}_{\\mathrm{der}}$: Derived Missingness Flags",
             edge_c="#2B5B84")
    draw_arrow(32.2, 85.4, 35.3, 85.4)
    draw_arrow(64.7, 85.4, 67.8, 85.4)

    # Stage II
    draw_band(51.5, 22.0,
              "Stage II: Mask-Aware BiLSTM & GBDT Late-Fusion Predictor (MA-TempNet+GBDT)",
              "#1E8449", "#F4FBF7")
    draw_box(3.5, 53.2, 29.5, 15.0,
             "Mask-Aware BiLSTM (MA-TempNet)",
             "• Step-Masked Input $\\tilde{x}_i^{(t)} = m_i^{(t)} x_i^{(t)}$\n• Mask-Gated Temporal Self-Attention:\n  $\\tilde{e}_{i,t} = e_{i,t} + (1 - m_i^{(t)})(-10^9)$\n• Bottleneck Embedding $e_i^{(s)} \\in \\mathbb{R}^{12}$",
             edge_c="#1E8449", title_c="#145A32")
    draw_box(36.5, 53.2, 27.5, 15.0,
             "Out-of-Fold Stacking & Screening",
             "• 3-Fold Inner CV Stacks $[x_i^{\\mathrm{tab}} \\Vert \\hat{p}_{\\mathrm{TempNet}} \\Vert e_i^{(s)}]$\n• 5-Fold Out-of-Fold Tabular Scorer $f_{\\mathrm{tab}}^{(-k(i))}$\n• Zero In-Sample Leakage for Recourse\n  Screening ($f_{\\mathrm{tab}}^{(-k(i))}(x_i) \\geq \\tau_s$)",
             edge_c="#1E8449", title_c="#145A32")
    draw_box(67.5, 53.2, 29.0, 15.0,
             "OOF Coalitional Log-Odds Attribution",
             "• Held-Out Fold Logit $\\ell^{(-k(i))}(x_i)$\n• Blended Marginal & Paired Coalitions\n  ($n_{\\mathrm{coal}}=8, n_r=4, M=2$)\n• Stage Attribution Share Across Tiers",
             edge_c="#1E8449", title_c="#145A32")
    draw_arrow(18.0, 78.0, 18.0, 68.5)
    draw_arrow(50.0, 78.0, 50.0, 68.5)
    draw_arrow(82.0, 78.0, 82.0, 68.5)
    draw_arrow(33.2, 60.7, 36.3, 60.7)
    draw_arrow(64.2, 60.7, 67.3, 60.7)

    # Stage III
    draw_band(26.5, 22.0,
              "Stage III: Stage-Aware Actionable Attribution Subspace Clustering (SA-AASC)",
              "#B9770E", "#FEF9E7")
    draw_box(3.5, 28.2, 29.5, 15.0,
             "Why Unconstrained Spaces Misalign",
             "• $B_1$ (Raw $X$) & $B_3$ (Full SHAP $\\Phi^{(s)}$)\n  are dominated by locked covariates\n  (tuition compliance, demographics)\n• Incompatible actionable sensitivities",
             edge_c="#B9770E", title_c="#7E5109")
    draw_box(36.5, 28.2, 28.5, 15.0,
             "SA-AASC 3-Block Projection",
             "• Restricts to Stage-Unlocked $\\mathcal{A}_s$\n• Positive Deficit & Headroom Cone:\n  $d_{i,j}^{(s)} = \\max(0, \\phi_{i,j}^{(s)})(0.35 + 0.65 r_{i,j})$\n• Coordinates $[1.65 \\tilde{d}_i \\Vert 0.85 r_i \\Vert 0.35 \\tilde{g}_i]$",
             edge_c="#B9770E", title_c="#7E5109")
    draw_box(68.5, 28.2, 28.0, 15.0,
             "Directional Homogeneity (ICDH)",
             f"• Fine-Grained $K=20$ & Capacity $K_c$ Sweep\n  ($K_c \\in \\{{5,10,15,20\\}}$)\n• Highest Intra-Cluster Agreement:\n  $\\mathrm{{ICDH}} = {icdh_min:.4f}$–${icdh_max:.4f}$ across cohorts",
             edge_c="#B9770E", title_c="#7E5109")
    draw_arrow(18.0, 53.0, 18.0, 43.5)
    draw_arrow(50.8, 53.0, 50.8, 43.5)
    draw_arrow(82.0, 53.0, 82.0, 43.5)
    draw_arrow(33.2, 35.7, 36.3, 35.7)
    draw_arrow(65.2, 35.7, 68.3, 35.7)

    # Stage IV
    draw_band(1.5, 22.0,
              "Stage IV: Matched 96-Candidate Water-Filling Optimization & Bi-Stage Hybrid Recourse",
              "#884EA0", "#F9F3FC")
    draw_box(3.5, 2.6, 29.5, 16.2,
             "Strict Feasibility & Budget Caps",
             "• $A_2$: Tier Lock ($\\delta_{g,j}=0$ for $j \\notin \\mathcal{A}_s$)\n• $A_3$: Monotonic Box ($s_j \\delta_{g,j} \\geq 0$)\n• $A_4$: Strict Effort & Sparsity Cap\n  ($\\mathcal{C}(\\delta) \\leq B, \\; \\Vert \\delta \\Vert_0 \\leq 3$ for every learner)",
             edge_c="#884EA0", title_c="#512E5F")
    draw_box(36.5, 3.2, 28.5, 15.0,
             "Stage 1: Shared Group Template $\\delta_g^*$",
             f"• Equal 96-Schedule Water-Filling Grid\n  Across All Clustering Spaces ($K=20$)\n• OOF Group Validity ($A_4@\\mathrm{{mid}}$):\n  {r_uci['group_validity']*100:.1f}% (UCI), {r_xapi['group_validity']*100:.1f}% (xAPI), {r_oul['group_validity']*100:.1f}% (OULAD)",
             edge_c="#884EA0", title_c="#512E5F")
    draw_box(68.5, 3.2, 28.0, 15.0,
             "Stage 2: 1-Feature Hybrid Residual $\\eta_i^*$",
             f"• Unflipped Learners Receive $\\Vert \\eta_i^* \\Vert_0 \\leq 1$\n  Within Remaining Budget $B - \\mathcal{{C}}_i$\n• $K=20$ Hybrid Validity: $69.3\\%$–$100.0\\%$\n• Plan Count Reduced by $96.0\\%$–$98.3\\%$ at $K_c=5$",
             edge_c="#884EA0", title_c="#512E5F")
    draw_arrow(18.0, 28.0, 18.0, 18.5)
    draw_arrow(50.8, 28.0, 50.8, 18.5)
    draw_arrow(82.0, 28.0, 82.0, 18.5)
    draw_arrow(33.2, 10.7, 36.3, 10.7)
    draw_arrow(65.2, 10.7, 68.3, 10.7)

    _save_outputs(fig, "fig1_overall_framework.png", "fig1_framework", tif_name="Figure1.tif")


# ==============================================================================
# SUPPLEMENTARY FIGURE S1: Longitudinal Cohort Timeline & 5-Tier Taxonomy
# ==============================================================================
def make_supp_fig1_cohort_timeline():
    fig = plt.figure(figsize=(13.2, 5.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.22, 1.0], wspace=0.28)

    ax1 = fig.add_subplot(gs[0, 0])
    stages = [20, 40, 60, 80]
    total_n = 32593
    drift_rows = RES["table_03_exp2_attribution_drift"]
    active_n = [r["active_survivors_n"] for r in drift_rows]
    already_exited = [total_n - n for n in active_n]
    survivor_pos = [int(round(r["active_survivors_n"] * r["survivor_pos_rate"])) for r in drift_rows]
    survivor_pass = [n - p for n, p in zip(active_n, survivor_pos)]

    x = np.arange(len(stages))
    w = 0.52
    ax1.bar(x, [v/1000 for v in survivor_pass], width=w, color="#2E86C1",
            edgecolor="white", label="Active Survivors — Eventual Pass/Distinction")
    ax1.bar(x, [v/1000 for v in survivor_pos], width=w,
            bottom=[v/1000 for v in survivor_pass], color="#E67E22",
            edgecolor="white", label="Active Survivors — Prospective At-Risk ($y=1$)")
    ax1.bar(x, [v/1000 for v in already_exited], width=w,
            bottom=[(s+p)/1000 for s, p in zip(survivor_pass, survivor_pos)],
            color="#884EA0", alpha=0.85, hatch="//", edgecolor="white",
            label="Pre-Cutoff Exits ($t_{\\mathrm{unreg}} \\leq t_s$, Target Contamination)")

    for idx, r in enumerate(drift_rows):
        ex_pct = (total_n - r["active_survivors_n"]) / total_n * 100
        pos_pct = r["survivor_pos_rate"] * 100
        ax1.text(idx, (survivor_pass[idx] + survivor_pos[idx]/2)/1000,
                 f"Risk: {pos_pct:.1f}%\n($n={survivor_pos[idx]:,}$)",
                 ha="center", va="center", fontsize=8.2, fontweight="bold", color="white")
        ax1.text(idx, (active_n[idx] + already_exited[idx]/2)/1000,
                 f"Exited: {ex_pct:.1f}%\n($n={already_exited[idx]:,}$)",
                 ha="center", va="center", fontsize=8.2, fontweight="bold", color="white")

    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{s}% Progress\n$N_{{\\mathrm{{surv}}}}={n:,}$" for s, n in zip(stages, active_n)], fontsize=8.5)
    ax1.set_ylabel("Registered Students ($\\times 10^3$, Total $N=32,593$)")
    ax1.set_ylim(0, 38.5)
    ax1.set_title("(a) OULAD Multi-Stage Cohort Decomposition & Contamination Audit", loc="left")
    ax1.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.92, fontsize=8.0)
    ax1.grid(axis="y", linestyle="--", alpha=0.35)

    # Panel (b): Complete Five-Tier Feature Taxonomy Composition (summing to exact d=36, 16, 31)
    ax2 = fig.add_subplot(gs[0, 1])
    u_data = load_uci697()
    x_data = load_xapi_edu()
    o_data = load_oulad_stage(0.40, "drop")
    ds_labels = [
        f"UCI #697\n($N=4,424, d={len(u_data['feature_names'])}$)",
        f"xAPI-Edu\n($N=480, d={len(x_data['feature_names'])}$)",
        f"OULAD (40% drop)\n($N=25,224, d={len(o_data['feature_names'])}$)"
    ]
    tiers = ["F_beh", "F_out", "F_inst", "F_imm", "F_der"]
    tier_names = [
        "$\\mathcal{F}_{\\mathrm{beh}}$ (Modifiable Behavior)",
        "$\\mathcal{F}_{\\mathrm{out}}$ (Assessment/Credit Outcome)",
        "$\\mathcal{F}_{\\mathrm{inst}}$ (Institutional/Macro Context)",
        "$\\mathcal{F}_{\\mathrm{imm}}$ (Immutable Prior/Demographic)",
        "$\\mathcal{F}_{\\mathrm{der}}$ (Derived Missingness Flags)"
    ]
    tier_colors = ["#1E8449", "#2874A6", "#D4AC0D", "#7F8C8D", "#5D6D7E"]

    counts_matrix = []
    for d_obj in [u_data, x_data, o_data]:
        t_dict = d_obj["tiers"]
        row_c = [len(t_dict.get(t, [])) for t in tiers]
        tot_d = len(d_obj["feature_names"])
        counts_matrix.append([c / tot_d * 100.0 for c in row_c])
    counts_matrix = np.array(counts_matrix)

    y_pos = np.arange(len(ds_labels))
    left_acc = np.zeros(len(ds_labels))
    for t_idx, (t_label, t_col) in enumerate(zip(tier_names, tier_colors)):
        vals = counts_matrix[:, t_idx]
        ax2.barh(y_pos, vals, left=left_acc, height=0.52, color=t_col, edgecolor="white", label=t_label)
        for i, v in enumerate(vals):
            if v >= 9.0:
                ax2.text(left_acc[i] + v/2, y_pos[i], f"{v:.1f}%",
                         ha="center", va="center", color="white", fontweight="bold", fontsize=8.4)
        left_acc += vals

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(ds_labels, fontsize=8.8)
    ax2.invert_yaxis()
    ax2.set_xlabel("Proportion of All $d$ Input Predictors (%)")
    ax2.set_xlim(0, 100)
    ax2.set_title("(b) Five-Tier Actionability Taxonomy Across All $d$ Predictors", loc="left")
    ax2.legend(loc="lower center", bbox_to_anchor=(0.5, -0.42), ncol=2, frameon=True, fontsize=8.0)
    ax2.grid(axis="x", linestyle="--", alpha=0.35)

    _save_outputs(fig, "fig2_dataset_cohort_timeline.png", "supp_fig1_cohort_taxonomy")


# ==============================================================================
# MAIN FIGURE 2: 8-Model Prediction Comparison & Longitudinal Curves
# ==============================================================================
def make_fig2_prediction_comparison():
    df1 = pd.DataFrame(RES["table_01_exp1_prediction_comparison"])
    df2 = pd.DataFrame(RES["table_02_exp2_stage_checkpoints"])

    fig = plt.figure(figsize=(14.2, 8.4))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.18, 1.0], hspace=0.40, wspace=0.30)

    datasets = ["UCI #697", "xAPI-Edu", "OULAD (40% raw)", "OULAD (40% drop)"]
    panel_tags = ["(a)", "(b)", "(c)", "(d)"]
    model_order = [
        "Unmasked-LSTM", "MLP", "Reg-TabNet", "LogReg",
        "RF", "GBDT", "MA-BiLSTM", "MA-TempNet+GBDT (Proposed)"
    ]
    short_names = [
        "Unmasked-LSTM", "MLP", "Reg-TabNet", "LogReg",
        "RF", "GBDT", "MA-BiLSTM", "MA-TempNet+GBDT*"
    ]

    for idx, (ds, ptag) in enumerate(zip(datasets, panel_tags)):
        ax = fig.add_subplot(gs[0, idx])
        sub = df1[df1["dataset"] == ds].set_index("model").reindex(model_order)
        y = np.arange(len(model_order))
        h = 0.37

        roc_vals = sub["auc_roc_mean"].values
        pr_vals = sub["auc_pr_mean"].values
        roc_std = sub["auc_roc_std"].values
        pr_std = sub["auc_pr_std"].values

        ax.barh(y - h/2, roc_vals, height=h, xerr=roc_std, capsize=1.8,
                color="#245888", alpha=0.92, label="AUC-ROC", error_kw=dict(lw=0.8, ecolor="#1B2631"))
        ax.barh(y + h/2, pr_vals, height=h, xerr=pr_std, capsize=1.8,
                color="#D95F02", alpha=0.92, label="AUC-PR", error_kw=dict(lw=0.8, ecolor="#1B2631"))

        for i, (rv, pv) in enumerate(zip(roc_vals, pr_vals)):
            ax.text(min(rv + roc_std[i] + 0.015, 0.99), y[i] - h/2, f"{rv:.3f}",
                    va="center", ha="left", fontsize=7.8, color="#1B4F72", fontweight="bold")
            ax.text(min(pv + pr_std[i] + 0.015, 0.99), y[i] + h/2, f"{pv:.3f}",
                    va="center", ha="left", fontsize=7.8, color="#A04000", fontweight="bold")

        ax.set_yticks(y)
        ax.set_yticklabels(short_names if idx == 0 else [""] * len(short_names), fontsize=8.8)
        ax.set_xlim(0.10, 1.16)
        ax.set_xlabel("Score (Mean $\\pm$ SD)", fontsize=9.2)
        ax.set_title(f"{ptag} {ds}", loc="left", fontsize=10.2)
        ax.grid(axis="x", linestyle="--", alpha=0.35)
        if idx == 0:
            ax.legend(loc="lower right", frameon=True, fontsize=8.0)

    ax_e = fig.add_subplot(gs[1, 0:2])
    ax_f = fig.add_subplot(gs[1, 2:4])

    models_stage = [
        ("MA-TempNet+GBDT (Proposed)", "MA-TempNet+GBDT (Proposed)", "#1E8449", "o", "-", 2.2),
        ("MA-BiLSTM", "MA-BiLSTM (Mask-Aware DL)", "#2874A6", "s", "--", 1.8),
        ("GBDT", "GBDT (Tabular Ensemble)", "#D35400", "^", "-.", 1.8),
        ("Unmasked-LSTM", "Unmasked-LSTM (Zero-Padded)", "#7D3C98", "x", ":", 1.8),
    ]
    checkpoints = [20, 40, 60, 80]

    for m_key, m_lbl, col, mk, ls, lw in models_stage:
        sub_raw = df2[(df2["coding"] == "raw") & (df2["model"] == m_key)].sort_values("cutoff_pct")
        sub_drop = df2[(df2["coding"] == "drop") & (df2["model"] == m_key)].sort_values("cutoff_pct")
        ax_e.plot(checkpoints, sub_raw["auc_roc"], marker=mk, color=col, linestyle=ls, linewidth=lw, label=f"{m_lbl} [raw]")
        ax_f.plot(checkpoints, sub_drop["auc_roc"], marker=mk, color=col, linestyle=ls, linewidth=lw, label=f"{m_lbl} [drop]")

    for c in checkpoints:
        row_prop = df2[(df2["coding"] == "raw") & (df2["model"] == "MA-TempNet+GBDT (Proposed)") & (df2["cutoff_pct"] == c)].iloc[0]
        ax_e.annotate(f"{row_prop['auc_roc']:.3f}", (c, row_prop["auc_roc"]),
                      textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8.2, fontweight="bold", color="#1E8449")
        row_drop = df2[(df2["coding"] == "drop") & (df2["model"] == "MA-TempNet+GBDT (Proposed)") & (df2["cutoff_pct"] == c)].iloc[0]
        ax_f.annotate(f"{row_drop['auc_roc']:.3f}", (c, row_drop["auc_roc"]),
                      textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8.2, fontweight="bold", color="#1E8449")

    ax_e.set_xticks(checkpoints)
    ax_e.set_xticklabels([f"{c}% Progress" for c in checkpoints])
    ax_e.set_ylim(0.52, 0.98)
    ax_e.set_ylabel("AUC-ROC")
    ax_e.set_title("(e) Longitudinal Trajectory on Contaminated Full Cohort (OULAD raw, $N_{\\mathrm{eval}}=4,500$)", loc="left")
    ax_e.legend(loc="lower right", frameon=True, fontsize=8.2)
    ax_e.grid(True, linestyle="--", alpha=0.35)

    ax_f.set_xticks(checkpoints)
    ax_f.set_xticklabels([f"{c}% Progress" for c in checkpoints])
    ax_f.set_ylim(0.52, 0.86)
    ax_f.set_ylabel("AUC-ROC")
    ax_f.set_title("(f) Longitudinal Trajectory on Active Survivors (OULAD drop, $N_{\\mathrm{eval}}=4,500$)", loc="left")
    ax_f.legend(loc="upper left", frameon=True, fontsize=8.2)
    ax_f.grid(True, linestyle="--", alpha=0.35)

    _save_outputs(fig, "fig3_prediction_and_temporal_progression.png", "fig2_prediction_progression", tif_name="Figure2.tif")


# ==============================================================================
# MAIN FIGURE 3: Stage-Aware OOF Attribution Drift & Colorblind-Safe SHAP Profiles
# ==============================================================================
def make_fig3_attribution_drift_and_shap():
    drift_df = pd.DataFrame(RES["table_03_exp2_attribution_drift"])

    fig = plt.figure(figsize=(14.2, 8.2))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.05, 1.15], hspace=0.44, wspace=0.38)

    ax_a = fig.add_subplot(gs[0, 0:2])
    stages = drift_df["cutoff_pct"].values
    x = np.arange(len(stages))
    s_beh = drift_df["share_F_beh"].values * 100
    s_out = drift_df["share_F_out"].values * 100
    s_inst = drift_df["share_F_inst"].values * 100
    s_imm = drift_df["share_F_imm"].values * 100
    s_der = drift_df["share_F_der"].values * 100
    rec_rate = drift_df["pure_beh_hybrid_validity"].values * 100

    w = 0.44
    ax_a.bar(x, s_beh, width=w, color="#1E8449", label="$\\mathcal{F}_{\\mathrm{beh}}$ (Modifiable Behavior %)")
    ax_a.bar(x, s_out, width=w, bottom=s_beh, color="#2874A6", label="$\\mathcal{F}_{\\mathrm{out}}$ (Assessment Outcome %)")
    ax_a.bar(x, s_inst, width=w, bottom=s_beh+s_out, color="#D4AC0D", label="$\\mathcal{F}_{\\mathrm{inst}}$ (Institutional Context %)")
    ax_a.bar(x, s_imm + s_der, width=w, bottom=s_beh+s_out+s_inst, color="#7F8C8D", label="$\\mathcal{F}_{\\mathrm{imm}}+\\mathcal{F}_{\\mathrm{der}}$ (Immutable + Missingness %)")

    for i in range(len(stages)):
        ax_a.text(x[i], s_beh[i]/2, f"{s_beh[i]:.1f}%", ha="center", va="center", color="white", fontweight="bold", fontsize=8.5)
        ax_a.text(x[i], s_beh[i] + s_out[i]/2, f"{s_out[i]:.1f}%", ha="center", va="center", color="white", fontweight="bold", fontsize=8.5)
        ax_a.text(x[i], s_beh[i] + s_out[i] + s_inst[i]/2, f"{s_inst[i]:.1f}%", ha="center", va="center", color="#1B2631", fontweight="bold", fontsize=8.2)
        ax_a.text(x[i], s_beh[i] + s_out[i] + s_inst[i] + (s_imm[i]+s_der[i])/2, f"{(s_imm[i]+s_der[i]):.1f}%", ha="center", va="center", color="white", fontweight="bold", fontsize=8.2)

    ax_a.set_xticks(x)
    ax_a.set_xticklabels([f"{s}% Checkpoint\n($N_{{\\mathrm{{drop}}}}={n:,}$)" for s, n in zip(stages, drift_df["active_survivors_n"])], fontsize=8.6)
    ax_a.set_ylabel("Out-of-Fold SHAP Attribution Share (%)")
    ax_a.set_ylim(0, 116)
    ax_a.set_title("(a) Stage-Aware OOF Attribution Mass Reallocation Across Active Survivors ($n \\leq 200$)", loc="left")
    ax_a.legend(loc="upper center", ncol=2, frameon=True, fontsize=8.1)
    ax_a.grid(axis="y", linestyle="--", alpha=0.35)

    ax_b = fig.add_subplot(gs[0, 2])
    ax_b_twin = ax_b.twinx()
    ax_b.spines["right"].set_visible(True)

    l1 = ax_b.plot(stages, s_beh, "-o", color="#1E8449", lw=2.2, markersize=6, label="$\\mathcal{F}_{\\mathrm{beh}}$ Attribution Share (%)")
    l2 = ax_b.plot(stages, drift_df["pure_beh_group_validity"]*100, "--s", color="#2874A6", lw=2.0, markersize=6, label="Pure-$\\mathcal{F}_{\\mathrm{beh}}$ Group Validity (%)")
    l3 = ax_b.plot(stages, rec_rate, "-.^", color="#8E44AD", lw=2.0, markersize=6, label="Pure-$\\mathcal{F}_{\\mathrm{beh}}$ Hybrid Validity (%)")
    l4 = ax_b_twin.plot(stages, drift_df["survivor_pos_rate"]*100, ":d", color="#D35400", lw=2.0, markersize=6, label="Active-Survivor Risk Rate (%)")

    ax_b.set_xticks(stages)
    ax_b.set_xticklabels([f"{s}%" for s in stages])
    ax_b.set_xlabel("Course Progress Checkpoint (%)")
    ax_b.set_ylabel("Attribution / Recourse Validity (%)")
    ax_b_twin.set_ylabel("Survivor Base Risk Rate (%)", color="#D35400")
    ax_b.set_ylim(10, 106)
    ax_b_twin.set_ylim(0, 22)
    ax_b.set_title("(b) Dynamic Actionability Window vs. Risk Base Rate", loc="left")
    lines = l1 + l2 + l3 + l4
    labels = [l.get_label() for l in lines]
    ax_b.legend(lines, labels, loc="center right", frameon=True, fontsize=7.8)
    ax_b.grid(True, linestyle="--", alpha=0.35)

    # Bottom Row (Panels c, d, e): Colorblind-safe dark-slate callouts + OOF SHAP
    tier_palette = {
        "F_beh": "#1E8449",
        "F_out": "#2874A6",
        "F_inst": "#D4AC0D",
        "F_imm": "#7F8C8D",
        "F_der": "#5D6D7E",
    }
    configs = [
        ("(c) UCI #697 Top Predictors ($n=400$)", load_uci697()),
        ("(d) xAPI-Edu Top Predictors ($n=400$)", load_xapi_edu()),
        ("(e) OULAD 40% drop Top Predictors ($n=400$)", load_oulad_stage(0.40, "drop", max_samples=4500)),
    ]
    for idx, (title_str, d_obj) in enumerate(configs):
        ax = fig.add_subplot(gs[1, idx])
        X_arr = d_obj["X_raw"]
        y_arr = d_obj["y"]
        f_names = d_obj["feature_names"]
        t_dict = d_obj["tiers"]
        feat_to_tier = {}
        for t_k, idx_list in t_dict.items():
            for j_i in idx_list:
                fname_key = f_names[j_i] if isinstance(j_i, int) else str(j_i)
                feat_to_tier[fname_key] = t_k

        oof_ens = OOFTabularEnsemble(X_arr, y_arr, n_splits=5, seed=42)
        rng = np.random.RandomState(42)
        eval_idx = np.sort(rng.choice(len(X_arr), min(400, len(X_arr)), replace=False))
        phi = oof_ens.compute_oof_shap(X_arr, eval_idx, n_coalitions=8, seed=42)

        mean_abs = np.mean(np.abs(phi), axis=0)
        order = np.argsort(mean_abs)[::-1][:8]
        top_feats = [f_names[j] for j in order][::-1]
        top_vals = [mean_abs[j] for j in order][::-1]
        top_cols = [tier_palette.get(feat_to_tier.get(f, "F_imm"), "#7F8C8D") for f in top_feats]

        y_p = np.arange(len(top_feats))
        ax.barh(y_p, top_vals, height=0.62, color=top_cols, edgecolor="white")
        max_v = max(top_vals)
        for yp, tv in zip(y_p, top_vals):
            # Colorblind-safe dark slate text (#1B2631) instead of red against green bars
            ax.text(tv + max_v * 0.02, yp, f"{tv:.3f}", va="center", ha="left",
                    fontsize=8.2, fontweight="bold", color="#1B2631")
        short_f = [f.replace("Curricular units ", "Units ")[:26] for f in top_feats]
        ax.set_yticks(y_p)
        ax.set_yticklabels(short_f, fontsize=8.3)
        ax.set_xlim(0, max_v * 1.28)
        ax.set_xlabel("Mean $|\\phi_{i,j}|$ (OOF Log-Odds Impact)")
        ax.set_title(title_str, loc="left", fontsize=10.0)
        ax.grid(axis="x", linestyle="--", alpha=0.35)

    _save_outputs(fig, "fig4_stage_attribution_drift_and_shap.png", "fig3_attribution_drift", tif_name="Figure3.tif")


# ==============================================================================
# MAIN FIGURE 4: 4-Space Subspace Geometry & Matched-96 Recourse Ablation
# ==============================================================================
def make_fig4_subspace_and_ablation():
    df4 = pd.DataFrame(RES["table_04_exp3_clustering_recourse_ablation"])

    fig = plt.figure(figsize=(14.4, 8.4))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.08], hspace=0.42, wspace=0.28)

    u_data = load_uci697()
    X_u, y_u, f_u, t_u = u_data["X_raw"], u_data["y"], u_data["feature_names"], u_data["tiers"]
    act_idx = sorted([f_u.index(j) for j in (t_u["F_beh"] + t_u["F_out"])])
    oof_u = OOFTabularEnsemble(X_u, y_u, n_splits=5, seed=42)
    r_idx = np.where(oof_u.oof_probs >= 0.50)[0]
    if len(r_idx) > 300:
        r_idx = np.sort(np.random.RandomState(42).choice(r_idx, 300, replace=False))
    sv = oof_u.compute_oof_shap(X_u, r_idx, n_coalitions=8, seed=42)

    icdh_b1 = df4[(df4["dataset"] == "UCI #697") & (df4["clustering_method"] == "B1_Raw_X") & (df4["constraint_variant"] == "A4_budget@mid")].iloc[0]["icdh"]
    icdh_sa = df4[(df4["dataset"] == "UCI #697") & (df4["clustering_method"] == "Proposed_SA_AASC") & (df4["constraint_variant"] == "A4_budget@mid")].iloc[0]["icdh"]

    top2_act = [f_u.index("Curricular units 2nd sem (approved)"), f_u.index("Curricular units 2nd sem (grade)")]
    sign_pattern = (sv[:, top2_act[0]] > np.median(sv[:, top2_act[0]])).astype(int) * 2 + (sv[:, top2_act[1]] > np.median(sv[:, top2_act[1]])).astype(int)
    pattern_labels = [
        "Regime I (Low App / Low Grd)",
        "Regime II (Low App / High Grd)",
        "Regime III (High App / Low Grd)",
        "Regime IV (High App / High Grd)"
    ]
    pattern_colors = ["#1E8449", "#2874A6", "#D35400", "#8E44AD"]

    ax_a = fig.add_subplot(gs[0, 0])
    Z_raw = PCA(n_components=2, random_state=42).fit_transform(StandardScaler().fit_transform(X_u[r_idx]))
    for p_val, p_lbl, p_col in zip([0, 1, 2, 3], pattern_labels, pattern_colors):
        m = sign_pattern == p_val
        if np.any(m):
            ax_a.scatter(Z_raw[m, 0], Z_raw[m, 1], c=p_col, s=26, alpha=0.80, edgecolors="white", linewidth=0.4, label=p_lbl)
    ax_a.set_xlabel("Principal Component 1 ($B_1$ Raw Space $X$)")
    ax_a.set_ylabel("Principal Component 2")
    ax_a.set_title(f"(a) $B_1$ Raw Space $X$: Interleaved Regimes\n(UCI #697, $n=300$, $\\mathrm{{ICDH}}={icdh_b1:.4f}$)", loc="left", fontsize=9.8)
    ax_a.legend(loc="upper right", frameon=True, fontsize=7.5)
    ax_a.grid(True, linestyle="--", alpha=0.35)

    ax_b = fig.add_subplot(gs[0, 1])
    sv_act = StandardScaler().fit_transform(sv[:, act_idx])
    Z_sa = PCA(n_components=2, random_state=42).fit_transform(sv_act)
    for p_val, p_lbl, p_col in zip([0, 1, 2, 3], pattern_labels, pattern_colors):
        m = sign_pattern == p_val
        if np.any(m):
            ax_b.scatter(Z_sa[m, 0], Z_sa[m, 1], c=p_col, s=26, alpha=0.82, edgecolors="white", linewidth=0.4, label=p_lbl)
    ax_b.set_xlabel("Principal Component 1 (SA-AASC Subspace)")
    ax_b.set_ylabel("Principal Component 2")
    ax_b.set_title(f"(b) Proposed SA-AASC: Separated Sensitivity Cones\n(UCI #697, $n=300$, $\\mathrm{{ICDH}}={icdh_sa:.4f}$)", loc="left", fontsize=9.8)
    ax_b.legend(loc="upper right", frameon=True, fontsize=7.5)
    ax_b.grid(True, linestyle="--", alpha=0.35)

    ax_c = fig.add_subplot(gs[0, 2])
    ds_list = ["UCI #697", "xAPI-Edu", "OULAD (40% drop)"]
    methods = ["B1_Raw_X", "B2_Actionable_X", "B3_Full_SHAP", "Proposed_SA_AASC"]
    m_labels = ["$B_1$: Raw $X$", "$B_2$: Actionable $X_{\\mathcal{A}_s}$", "$B_3$: Full SHAP $\\Phi$", "Proposed SA-AASC"]
    m_cols = ["#95A5A6", "#5DADE2", "#F5B041", "#1E8449"]

    x_ds = np.arange(len(ds_list))
    bw = 0.19
    for m_i, (m_k, m_l, m_c) in enumerate(zip(methods, m_labels, m_cols)):
        ratios, counts_txt = [], []
        for ds in ds_list:
            row = df4[(df4["dataset"] == ds) & (df4["clustering_method"] == m_k) & (df4["constraint_variant"] == "A4_budget@mid")].iloc[0]
            ratios.append(row["valid_cluster_ratio"] * 100)
            counts_txt.append(f"{int(row['valid_clusters'])}/{int(row['total_clusters'])}")
        rects = ax_c.bar(x_ds + (m_i - 1.5)*bw, ratios, width=bw, color=m_c, edgecolor="white", label=m_l)
        for r_box, c_str, val in zip(rects, counts_txt, ratios):
            ax_c.text(r_box.get_x() + r_box.get_width()/2, val + 1.8, c_str,
                      ha="center", va="bottom", fontsize=7.6, fontweight="bold")

    ax_c.set_xticks(x_ds)
    ax_c.set_xticklabels(["UCI #697\n($n=300$)", "xAPI-Edu\n($n=124$)", "OULAD 40%\n($n=300$)"], fontsize=8.5)
    ax_c.set_ylim(0, 118)
    ax_c.set_ylabel("Valid Cluster Coverage $M_{\\mathrm{valid}}/K$ (%)")
    ax_c.set_title("(c) Verified Cluster Coverage ($\\geq 50\\%$ Flip)\nUnder Matched $A_4@\\mathrm{mid}$ ($K=20$)", loc="left", fontsize=9.8)
    ax_c.legend(loc="upper left", frameon=True, fontsize=7.6)
    ax_c.grid(axis="y", linestyle="--", alpha=0.35)

    panel_bottom = [
        "(d) UCI #697 ($K=20, n=300, A_4@\\mathrm{mid}$)",
        "(e) xAPI-Edu ($K=20, n=124, A_4@\\mathrm{mid}$)",
        "(f) OULAD 40% drop ($K=20, n=300, A_4@\\mathrm{mid}$)"
    ]
    for idx, (ds, ptitle) in enumerate(zip(ds_list, panel_bottom)):
        ax = fig.add_subplot(gs[1, idx])
        sub = df4[(df4["dataset"] == ds) & (df4["constraint_variant"] == "A4_budget@mid")].set_index("clustering_method").reindex(methods)
        x = np.arange(len(methods))
        w = 0.26

        v_grp = sub["group_validity"].values
        v_hyb = sub["hybrid_validity"].values
        v_icdh = sub["icdh"].values

        ax.bar(x - w, v_grp, width=w, color="#2874A6", edgecolor="white", label="Group Template Validity")
        ax.bar(x, v_hyb, width=w, color="#1E8449", edgecolor="white", label="Bi-Stage Hybrid Validity")
        ax.bar(x + w, v_icdh, width=w, color="#D4AC0D", edgecolor="white", label="Directional Homogeneity (ICDH)")

        for i in range(len(methods)):
            ax.text(x[i] - w, v_grp[i] + 0.018, f"{v_grp[i]:.3f}", ha="center", va="bottom", fontsize=7.4, fontweight="bold", color="#1B4F72", rotation=90)
            ax.text(x[i], v_hyb[i] + 0.018, f"{v_hyb[i]:.3f}", ha="center", va="bottom", fontsize=7.4, fontweight="bold", color="#145A32", rotation=90)
            ax.text(x[i] + w, v_icdh[i] + 0.018, f"{v_icdh[i]:.3f}", ha="center", va="bottom", fontsize=7.4, fontweight="bold", color="#7E5109", rotation=90)

        ax.set_xticks(x)
        ax.set_xticklabels(["$B_1$ Raw $X$", "$B_2$ Act $X_{\\mathcal{A}_s}$", "$B_3$ Full $\\Phi$", "SA-AASC"], fontsize=8.8)
        ax.set_ylim(0, 1.18)
        ax.set_ylabel("Score / Validity Ratio")
        ax.set_title(ptitle, loc="left", fontsize=9.8)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        if idx == 0:
            ax.legend(loc="upper left", frameon=True, fontsize=7.6)

    _save_outputs(fig, "fig5_subspace_manifold_and_ablation.png", "fig4_subspace_ablation", tif_name="Figure4.tif")


# ==============================================================================
# SUPPLEMENTARY FIGURE S2: Advising Capacity Pareto Frontier (K_c in {5,10,15,20})
# ==============================================================================
def make_supp_fig2_pareto_frontier():
    df5 = pd.DataFrame(RES["table_05_exp4_advising_capacity_pareto"])

    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.6))
    ds_list = ["UCI #697", "xAPI-Edu", "OULAD (40% drop)"]
    ptags = ["(a)", "(b)", "(c)"]

    for idx, (ds, ptag) in enumerate(zip(ds_list, ptags)):
        ax = axes[idx]
        sub_b1 = df5[(df5["dataset"] == ds) & (df5["clustering_method"] == "B1_Raw_X")].sort_values("advising_capacity_K")
        sub_b3 = df5[(df5["dataset"] == ds) & (df5["clustering_method"] == "B3_Full_SHAP")].sort_values("advising_capacity_K")
        sub_sa = df5[(df5["dataset"] == ds) & (df5["clustering_method"] == "Proposed_SA_AASC")].sort_values("advising_capacity_K")

        K = sub_sa["advising_capacity_K"].values
        g_b1 = sub_b1["group_validity"].values
        g_b3 = sub_b3["group_validity"].values
        g_sa = sub_sa["group_validity"].values
        h_sa = sub_sa["hybrid_validity"].values
        n_ev = int(sub_sa.iloc[0]["evaluated_at_risk_n"])

        # Exact fill_between envelope between Pure Group curves (SA-AASC vs B1_Raw_X)
        ax.fill_between(K, g_b1, g_sa, color="#1E8449", alpha=0.14, label="SA-AASC Group Gain over $B_1$ Raw $X$")
        ax.plot(K, h_sa, ":o", color="#145A32", lw=2.0, markersize=6, label="SA-AASC (Bi-Stage Hybrid)")
        ax.plot(K, g_sa, "-s", color="#1E8449", lw=2.4, markersize=6.5, label="SA-AASC (Pure Group)")
        ax.plot(K, g_b3, "-.^", color="#2874A6", lw=2.0, markersize=6, label="$B_3$ Full SHAP (Pure Group)")
        ax.plot(K, g_b1, "--d", color="#D35400", lw=2.0, markersize=6, label="$B_1$ Raw $X$ (Pure Group)")

        for k_val, gs_v, gb_v in zip(K, g_sa, g_b1):
            ax.annotate(f"{gs_v:.3f}", (k_val, gs_v), textcoords="offset points", xytext=(0, 7),
                        ha="center", fontsize=8.0, fontweight="bold", color="#145A32")
            ax.annotate(f"{gb_v:.3f}", (k_val, gb_v), textcoords="offset points", xytext=(0, -13),
                        ha="center", fontsize=8.0, fontweight="bold", color="#A04000")

        red_k5 = sub_sa.iloc[0]["plan_reduction_ratio"] * 100
        gain_b1_k20 = (g_sa[-1] - g_b1[-1]) * 100
        gain_b3_k20 = (g_sa[-1] - g_b3[-1]) * 100
        y_box = 0.34 if idx == 0 else (0.10 if idx == 1 else 0.04)
        ax.text(0.52, y_box,
                f"$K_c=5$: Plan Count $-{red_k5:.1f}\\%$\n$K_c=20$ Gain: {gain_b1_k20:+.1f} pp ($B_1$), {gain_b3_k20:+.1f} pp ($B_3$)",
                transform=ax.transAxes, ha="center", va="bottom", fontsize=7.9, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.32", facecolor="#F4FBF7", edgecolor="#1E8449", alpha=0.95))

        ax.set_xticks(K)
        ax.set_xticklabels([f"$K_c={k}$" for k in K])
        ax.set_ylim(0.0, 1.15)
        ax.set_xlabel("Instructor Advising Capacity (Clusters $K_c$)")
        ax.set_ylabel("Out-of-Fold Counterfactual Validity")
        ax.set_title(f"{ptag} {ds} ($n={n_ev}, A_4@\\mathrm{{mid}}$)", loc="left")
        ax.grid(True, linestyle="--", alpha=0.35)
        if idx == 1:
            ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.36), ncol=3, frameon=True, fontsize=8.2)

    _save_outputs(fig, "fig6_advising_capacity_pareto.png", "supp_fig2_pareto_frontier")


# ==============================================================================
# SUPPLEMENTARY FIGURE S3: Discovered Student Risk Archetypes
# ==============================================================================
def make_supp_fig3_risk_archetypes():
    df6 = pd.DataFrame(RES["table_06_exp5_risk_archetypes"])

    fig = plt.figure(figsize=(14.2, 6.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.10, 1.40], wspace=0.26)

    ax1 = fig.add_subplot(gs[0, 0])
    labels = [f"{r['dataset'].split(' ')[0]} — {r['archetype_id']}" for _, r in df6.iterrows()]
    shares = df6["cohort_share_pct"].values
    risks = df6["mean_initial_risk"].values * 100
    shap_mag = df6["primary_actionable_shap"].values

    y = np.arange(len(labels))
    ds_colors = ["#2874A6" if "UCI" in r["dataset"] else ("#1E8449" if "xAPI" in r["dataset"] else "#8E44AD") for _, r in df6.iterrows()]

    ax1.barh(y, shares, height=0.62, color=ds_colors, alpha=0.88, edgecolor="white")
    for i, (sh, rk, sm) in enumerate(zip(shares, risks, shap_mag)):
        ax1.text(sh + 0.8, y[i], f"{sh:.1f}% (Risk: {rk:.1f}%, |SHAP|: {sm:.2f})",
                 va="center", ha="left", fontsize=8.3, fontweight="bold", color="#1B2631")

    ax1.set_yticks(y)
    ax1.set_yticklabels(labels, fontsize=8.8)
    ax1.invert_yaxis()
    ax1.set_xlim(0, max(shares) + 34)
    ax1.set_xlabel("Share of Evaluated At-Risk Cohort (%)")
    ax1.set_title("(a) SA-AASC Archetypes ($K=4$): Cohort Share & OOF Risk", loc="left")
    ax1.grid(axis="x", linestyle="--", alpha=0.35)

    ax2 = fig.add_subplot(gs[0, 1])
    y2 = np.arange(len(df6))
    shap_vals = df6["primary_actionable_shap"].values
    ax2.hlines(y=y2, xmin=0, xmax=shap_vals, color=ds_colors, alpha=0.75, linewidth=2.6)
    ax2.scatter(shap_vals, y2, color=ds_colors, s=72, zorder=4, edgecolors="white", linewidth=0.8)

    max_sv = max(shap_vals)
    for i, r in df6.iterrows():
        p_drv = r["primary_actionable_driver"].replace("Curricular units ", "Units ")
        s_drv = r["secondary_actionable_driver"].replace("Curricular units ", "Units ")
        l_ctx = r["dominant_locked_context"]
        sv = r["primary_actionable_shap"]
        ax2.text(sv + max_sv * 0.02, y2[i],
                 f"+{sv:.2f} [{p_drv}; 2nd: {s_drv} | Locked: {l_ctx}]",
                 va="center", ha="left", fontsize=8.2, fontweight="bold", color="#1B2631")

    ax2.set_yticks(y2)
    ax2.set_yticklabels([""] * len(y2))
    ax2.invert_yaxis()
    ax2.set_xlim(0, max_sv * 2.55)
    ax2.set_xlabel("Primary Actionable Deficit Attribution $|\\phi_{i,j}|$ (OOF Log-Odds)")
    ax2.set_title("(b) Actionable Prescription Targets & Locked Contextual Barriers", loc="left")
    ax2.grid(axis="x", linestyle="--", alpha=0.35)

    _save_outputs(fig, "fig7_risk_archetypes_heatmap.png", "supp_fig3_risk_archetypes")


if __name__ == "__main__":
    # Remove stale Figure5.tif / Figure6.tif and stale fig*.png in manuscript_scirep/figs
    for old_f in os.listdir(FIG_DIR_SUB_TIF):
        if not old_f.startswith(".") and old_f.endswith(".tif"):
            try:
                os.remove(os.path.join(FIG_DIR_SUB_TIF, old_f))
            except FileNotFoundError:
                pass
    for old_f in os.listdir(FIG_DIR_SCIREP):
        if not old_f.startswith("."):
            try:
                os.remove(os.path.join(FIG_DIR_SCIREP, old_f))
            except FileNotFoundError:
                pass

    make_fig1_framework()
    make_fig2_prediction_comparison()
    make_fig3_attribution_drift_and_shap()
    make_fig4_subspace_and_ablation()
    make_supp_fig1_cohort_timeline()
    make_supp_fig2_pareto_frontier()
    make_supp_fig3_risk_archetypes()
    print("ALL 4 MAIN + 3 SUPPLEMENTARY FIGURES GENERATED SUCCESSFULLY.")
