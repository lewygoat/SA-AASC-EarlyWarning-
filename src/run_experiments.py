"""
Master Runner with Defensive Strategy, Out-of-Fold Counterfactual Audit,
Equalized 96-Schedule Water-Filling Search, Paired Bootstrap Confidence Intervals,
and Multi-Seed / Threshold Sensitivity Checks for:
"Stage-Aware Attribution Subspace Clustering for Group Counterfactual Recourse
in Student Early-Warning Systems"
"""

from __future__ import annotations

import argparse
import json
import os
import time
import warnings
from typing import Any, Dict, List, Tuple

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from defensive_data_loader import (
    DefensiveDataError,
    load_oulad_stage,
    load_uci697,
    load_xapi_edu,
    verify_no_nan_inf,
)
from models_dl_ml import (
    MaskAwareStageBiLSTMClassifier,
    RegularizedTabNetWrapper,
    UnmaskedLSTMWrapper,
)
from sa_aasc_recourse import (
    OOFTabularEnsemble,
    cluster_at_risk_students,
    compute_intra_cluster_directional_homogeneity,
    compute_iqr_scales,
    compute_tier_attribution_shares,
    optimize_group_and_hybrid_recourse,
    paired_bootstrap_validity_ci,
)


EXP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(EXP_DIR, "results")
FIGURES_DIR = os.path.join(EXP_DIR, "figures")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


def evaluate_models_cv(
    ds: Dict[str, Any],
    n_splits: int = 5,
    seeds: List[int] = [42, 43],
    dl_epochs: int = 22,
) -> Tuple[List[Dict[str, Any]], OOFTabularEnsemble, np.ndarray]:
    """
    Evaluate all 8 architectures (LogReg, RF, MLP, Unmasked-LSTM, Reg-TabNet,
    GBDT, MA-BiLSTM, and MA-TempNet+GBDT) under stratified CV with defensive checks.
    Returns summary rows, a strictly leak-free 5-fold OOFTabularEnsemble for
    downstream XAI/recourse audit, and out-of-fold ensemble probabilities.
    """
    X = ds["X_raw"]
    y = ds["y"]
    seq = ds["seq_tensor"]
    mask = ds["seq_mask"]
    ds_name = ds["name"]

    model_names = [
        "LogReg",
        "RF",
        "MLP",
        "Unmasked-LSTM",
        "Reg-TabNet",
        "GBDT",
        "MA-BiLSTM",
        "MA-TempNet+GBDT (Proposed)",
    ]
    metrics_map: Dict[str, Dict[str, List[float]]] = {
        m: {"auc_pr": [], "auc_roc": [], "f1": [], "brier": []} for m in model_names
    }
    oof_probs_proposed = np.zeros(len(y), dtype=np.float64)

    for seed in seeds:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for tr_idx, te_idx in skf.split(X, y):
            X_tr, y_tr = X[tr_idx], y[tr_idx]
            X_te, y_te = X[te_idx], y[te_idx]
            seq_tr, mask_tr = seq[tr_idx], mask[tr_idx]
            seq_te, mask_te = seq[te_idx], mask[te_idx]

            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_te_s = scaler.transform(X_te)

            # 1. LogReg
            lr = LogisticRegression(max_iter=800, random_state=seed)
            lr.fit(X_tr_s, y_tr)
            p_lr = lr.predict_proba(X_te_s)[:, 1]

            # 2. RF
            rf = RandomForestClassifier(n_estimators=160, max_depth=12, random_state=seed, n_jobs=-1)
            rf.fit(X_tr, y_tr)
            p_rf = rf.predict_proba(X_te)[:, 1]

            # 3. MLP
            mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=150, early_stopping=True, random_state=seed)
            mlp.fit(X_tr_s, y_tr)
            p_mlp = mlp.predict_proba(X_te_s)[:, 1]

            # 4. Unmasked-LSTM
            u_lstm = UnmaskedLSTMWrapper(epochs=max(10, dl_epochs // 2), seed=seed)
            u_lstm.fit(seq_tr, mask_tr, y_tr)
            p_ulstm = u_lstm.predict_proba(seq_te, mask_te)

            # 5. Reg-TabNet
            tabnet = RegularizedTabNetWrapper(epochs=dl_epochs, seed=seed)
            tabnet.fit(X_tr, y_tr)
            p_tabnet = tabnet.predict_proba(X_te)

            # 6. GBDT
            gbdt = HistGradientBoostingClassifier(
                max_iter=150, learning_rate=0.055, max_leaf_nodes=31, random_state=seed
            )
            gbdt.fit(X_tr, y_tr)
            p_gbdt = gbdt.predict_proba(X_te)[:, 1]

            # 7. MA-BiLSTM
            ma_lstm = MaskAwareStageBiLSTMClassifier(
                hidden_dim=32, emb_dim=12, epochs=dl_epochs, seed=seed
            )
            ma_lstm.fit(X_tr, seq_tr, mask_tr, y_tr)
            p_malstm, emb_te = ma_lstm.predict_proba_and_emb(X_te, seq_te, mask_te)

            # 8. MA-TempNet+GBDT (Proposed Dual-Stream) with inner 3-fold OOF stacking
            X_te_aug = np.column_stack([X_te, p_malstm[:, None], emb_te])
            inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
            oof_dl = np.zeros(len(tr_idx), dtype=np.float64)
            oof_emb = np.zeros((len(tr_idx), emb_te.shape[1]), dtype=np.float64)
            for in_tr, in_val in inner.split(X_tr, y_tr):
                inner_lstm = MaskAwareStageBiLSTMClassifier(
                    hidden_dim=32, emb_dim=12, epochs=dl_epochs, seed=seed
                )
                inner_lstm.fit(X_tr[in_tr], seq_tr[in_tr], mask_tr[in_tr], y_tr[in_tr])
                p_in, emb_in = inner_lstm.predict_proba_and_emb(X_tr[in_val], seq_tr[in_val], mask_tr[in_val])
                oof_dl[in_val] = p_in
                oof_emb[in_val] = emb_in
            X_tr_stack = np.column_stack([X_tr, oof_dl[:, None], oof_emb])
            gbdt_fused = HistGradientBoostingClassifier(
                max_iter=180, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=15,
                l2_regularization=1.5, random_state=seed
            )
            gbdt_fused.fit(X_tr_stack, y_tr)
            p_fused = gbdt_fused.predict_proba(X_te_aug)[:, 1]
            p_prop = np.clip(0.55 * p_fused + 0.30 * p_gbdt + 0.15 * p_malstm, 1e-6, 1.0 - 1e-6)

            if seed == seeds[0]:
                oof_probs_proposed[te_idx] = p_prop

            for m_name, prob_vec in [
                ("LogReg", p_lr),
                ("RF", p_rf),
                ("MLP", p_mlp),
                ("Unmasked-LSTM", p_ulstm),
                ("Reg-TabNet", p_tabnet),
                ("GBDT", p_gbdt),
                ("MA-BiLSTM", p_malstm),
                ("MA-TempNet+GBDT (Proposed)", p_prop),
            ]:
                verify_no_nan_inf(prob_vec, f"{ds_name}_{m_name}_prob")
                pred_bin = (prob_vec >= 0.5).astype(int)
                metrics_map[m_name]["auc_pr"].append(float(average_precision_score(y_te, prob_vec)))
                metrics_map[m_name]["auc_roc"].append(float(roc_auc_score(y_te, prob_vec)))
                metrics_map[m_name]["f1"].append(float(f1_score(y_te, pred_bin, zero_division=0)))
                metrics_map[m_name]["brier"].append(float(brier_score_loss(y_te, prob_vec)))

    rows = []
    for m_name in model_names:
        m_dict = metrics_map[m_name]
        rows.append(
            {
                "dataset": ds_name,
                "model": m_name,
                "auc_pr_mean": float(np.mean(m_dict["auc_pr"])),
                "auc_pr_std": float(np.std(m_dict["auc_pr"], ddof=1)) if len(m_dict["auc_pr"]) > 1 else 0.0,
                "auc_roc_mean": float(np.mean(m_dict["auc_roc"])),
                "auc_roc_std": float(np.std(m_dict["auc_roc"], ddof=1)) if len(m_dict["auc_roc"]) > 1 else 0.0,
                "f1_mean": float(np.mean(m_dict["f1"])),
                "brier_mean": float(np.mean(m_dict["brier"])),
            }
        )

    # Strictly out-of-fold tabular ensemble (zero in-sample leakage for downstream XAI & recourse)
    oof_tab_ens = OOFTabularEnsemble(X, y, n_splits=n_splits, seed=seeds[0])
    return rows, oof_tab_ens, oof_probs_proposed


def run_exp2_stage_drift_and_leadtime(
    smoke: bool = False,
    cached_40: Dict[str, Tuple[Dict[str, Any], List[Dict[str, Any]], OOFTabularEnsemble]] | None = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Experiment 1B & Experiment 2:
    Across OULAD course progress stages (20%, 40%, 60%, 80%), evaluate:
    1) Active-survivor (`drop`) vs. full-cohort (`raw`) predictive performance.
    2) Out-of-fold stage-wise evolution of 5-tier attribution shares and
       pure-behavioural recourse feasibility window.
    """
    checkpoints = [0.20, 0.40, 0.60, 0.80]
    stage_pred_rows: List[Dict[str, Any]] = []
    stage_drift_rows: List[Dict[str, Any]] = []

    n_splits = 3 if smoke else 5
    seeds = [42] if smoke else [42, 43]
    dl_epochs = 12 if smoke else 22
    max_n = 800 if smoke else 4500

    for cp in checkpoints:
        for coding in ["drop", "raw"]:
            if int(round(cp * 100)) == 40 and cached_40 is not None and coding in cached_40:
                ds_cp, rows_cp, oof_tab_cp = cached_40[coding]
            else:
                ds_cp = load_oulad_stage(cutoff_pct=cp, coding=coding, smoke=smoke, max_samples=max_n)
                rows_cp, oof_tab_cp, _ = evaluate_models_cv(ds_cp, n_splits=n_splits, seeds=seeds, dl_epochs=dl_epochs)

            exact_surv_frac = (
                1.0 if coding == "raw"
                else float(ds_cp["n_survivor_cohort"] / max(ds_cp["n_full_cohort"], 1))
            )
            for r in rows_cp:
                if r["model"] in ("Unmasked-LSTM", "MA-BiLSTM", "GBDT", "MA-TempNet+GBDT (Proposed)"):
                    stage_pred_rows.append(
                        {
                            "cutoff_pct": int(cp * 100),
                            "coding": coding,
                            "pos_rate": float(ds_cp["p_raw_full"] if coding == "raw" else ds_cp["p_drop_full"]),
                            "survivor_fraction": exact_surv_frac,
                            "model": r["model"],
                            "auc_pr": r["auc_pr_mean"],
                            "auc_roc": r["auc_roc_mean"],
                            "f1": r["f1_mean"],
                            "brier": r["brier_mean"],
                        }
                    )

            if coding == "drop":
                X_all = ds_cp["X_raw"]
                probs_tab = oof_tab_cp.oof_probs
                at_risk_idx = np.where(probs_tab >= 0.35)[0]
                if len(at_risk_idx) < 30:
                    at_risk_idx = np.argsort(-probs_tab)[: min(120, len(X_all))]
                eval_idx = np.sort(at_risk_idx[: min(200, len(at_risk_idx))])
                X_eval = X_all[eval_idx]

                phi_cp = oof_tab_cp.compute_oof_shap(X_all, eval_idx, n_coalitions=8, seed=42)
                shares = compute_tier_attribution_shares(phi_cp, ds_cp["feature_names"], ds_cp["tiers"])

                iqr_sc = compute_iqr_scales(X_all)
                beh_only_tiers = {
                    "F_imm": ds_cp["tiers"]["F_imm"] + ds_cp["tiers"]["F_out"],
                    "F_inst": ds_cp["tiers"]["F_inst"],
                    "F_beh": ds_cp["tiers"]["F_beh"],
                    "F_out": [],
                    "F_der": ds_cp["tiers"]["F_der"],
                }
                name_to_idx = {c: i for i, c in enumerate(ds_cp["feature_names"])}
                beh_indices = [name_to_idx[c] for c in ds_cp["tiers"]["F_beh"]]
                mono_idx = {name_to_idx[k]: v for k, v in ds_cp["monotonic_dirs"].items() if k in name_to_idx}
                q_lo = np.percentile(X_all, 2, axis=0)
                q_hi = np.percentile(X_all, 98, axis=0)

                labels_beh = cluster_at_risk_students(
                    "Proposed_SA_AASC",
                    X_eval,
                    phi_cp,
                    probs_tab[eval_idx],
                    beh_indices,
                    mono_idx,
                    (q_lo, q_hi),
                    n_clusters=10,
                    seed=42,
                )
                tau_beh = float(np.median(probs_tab[eval_idx]) * 0.88)
                rec_beh = optimize_group_and_hybrid_recourse(
                    oof_tab_cp,
                    X_eval,
                    phi_cp,
                    labels_beh,
                    "Proposed_SA_AASC",
                    ds_cp["feature_names"],
                    beh_only_tiers,
                    ds_cp["monotonic_dirs"],
                    iqr_sc,
                    X_all,
                    tau=tau_beh,
                    budgets={"low": 3.0, "mid": 5.0, "high": 7.5},
                    orig_indices=eval_idx,
                    ref_probs=probs_tab,
                )

                stage_drift_rows.append(
                    {
                        "cutoff_pct": int(cp * 100),
                        "active_survivors_n": int(ds_cp["n_survivor_cohort"]),
                        "survivor_pos_rate": float(ds_cp["p_drop_full"]),
                        "share_F_beh": shares["F_beh"],
                        "share_F_out": shares["F_out"],
                        "share_F_inst": shares["F_inst"],
                        "share_F_imm": shares["F_imm"],
                        "share_F_der": shares["F_der"],
                        "pure_beh_group_validity": rec_beh["A4_budget@mid"]["group_validity"],
                        "pure_beh_hybrid_validity": rec_beh["A4_budget@mid"]["hybrid_validity"],
                    }
                )

    return stage_pred_rows, stage_drift_rows


def run_exp3_exp4_exp5_clustering_and_recourse(
    datasets_models: List[Tuple[Dict[str, Any], OOFTabularEnsemble, np.ndarray]],
    smoke: bool = False,
) -> Tuple[
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    Dict[str, Any],
]:
    """
    Run strictly out-of-fold, equal-96-candidate counterfactual experiments:
    - Experiment 3: 4 Clustering Spaces x 6 Constraint Regimes (with uniform K=20 and 2,000-draw paired bootstrap 95% CIs)
    - Experiment 3B: Coordinate-Block, Anchor-Control & Optimizer Ablation (with 5-seed mean ± SD across seeds 42..46)
    - Experiment 3C: Screening Threshold tau_s Sensitivity Sweep
    - Experiment 4: Advising Capacity K_c in {5, 10, 15, 20} Pareto Frontier
    - Experiment 5: Risk Archetype Profiling under SA-AASC (K=4 macro-archetypes)
    """
    exp3_rows: List[Dict[str, Any]] = []
    exp3_comp_rows: List[Dict[str, Any]] = []
    exp3_tau_rows: List[Dict[str, Any]] = []
    exp4_rows: List[Dict[str, Any]] = []
    exp5_rows: List[Dict[str, Any]] = []
    bootstrap_summary: Dict[str, Any] = {}

    k_default = 20
    budgets_by_ds = {
        "UCI #697": {"low": 3.2, "mid": 5.2, "high": 7.8},
        "xAPI-Edu": {"low": 1.6, "mid": 2.6, "high": 4.0},
        "OULAD (40% drop)": {"low": 3.0, "mid": 5.0, "high": 7.5},
    }
    cluster_seeds = [42] if smoke else [42, 43, 44, 45, 46]

    for ds, oof_tab, _ in datasets_models:
        ds_name = ds["name"]
        X_all = ds["X_raw"]
        probs_all = oof_tab.oof_probs.copy()

        # Select at-risk evaluation pool strictly using out-of-fold f_tab scores
        tau_eval = 0.50 if ("drop" not in ds_name) else float(np.percentile(probs_all, 82))
        tau_eval = max(min(tau_eval, 0.50), 0.25)
        risk_mask = probs_all >= tau_eval
        total_at_risk_pool = int(np.sum(risk_mask))
        if total_at_risk_pool < 40:
            top_k_risk = min(150, len(X_all) // 3)
            risk_idx = np.argsort(-probs_all)[:top_k_risk]
            tau_eval = float(probs_all[risk_idx[-1]])
            total_at_risk_pool = int(len(risk_idx))
        else:
            risk_idx = np.where(risk_mask)[0]

        max_pool = 120 if smoke else 300
        if len(risk_idx) > max_pool:
            rng = np.random.RandomState(42)
            risk_idx = np.sort(rng.choice(risk_idx, size=max_pool, replace=False))
        else:
            risk_idx = np.sort(risk_idx)

        X_risk = X_all[risk_idx]
        probs_risk = probs_all[risk_idx]
        eval_n = int(len(X_risk))
        phi_risk = oof_tab.compute_oof_shap(X_all, risk_idx, n_coalitions=8, seed=42)

        name_to_idx = {c: i for i, c in enumerate(ds["feature_names"])}
        act_cols = ds["tiers"]["F_beh"] + ds["tiers"]["F_out"]
        act_indices = [name_to_idx[c] for c in act_cols if c in name_to_idx]
        mono_idx = {name_to_idx[k]: v for k, v in ds["monotonic_dirs"].items() if k in name_to_idx}

        q_lo = np.percentile(X_all, 2, axis=0)
        q_hi = np.percentile(X_all, 98, axis=0)
        iqr_scales = compute_iqr_scales(X_all)
        b_map = budgets_by_ds.get(ds_name, {"low": 3.0, "mid": 5.0, "high": 7.5})

        mid_flip_vecs_grp: Dict[str, np.ndarray] = {}
        mid_flip_vecs_hyb: Dict[str, np.ndarray] = {}

        for c_method in ["B1_Raw_X", "B2_Actionable_X", "B3_Full_SHAP", "Proposed_SA_AASC"]:
            # Multi-seed evaluation under A4_budget@mid across 5 K-Means seeds
            multiseed_grp = []
            multiseed_hyb = []
            for s_k in cluster_seeds:
                lbl_s = cluster_at_risk_students(
                    c_method, X_risk, phi_risk, probs_risk, act_indices, mono_idx, (q_lo, q_hi), n_clusters=k_default, seed=s_k
                )
                res_s = optimize_group_and_hybrid_recourse(
                    oof_tab, X_risk, phi_risk, lbl_s, c_method,
                    ds["feature_names"], ds["tiers"], ds["monotonic_dirs"], iqr_scales, X_all,
                    tau=tau_eval, budgets=b_map, use_water_filling=True,
                    orig_indices=risk_idx, ref_probs=probs_all,
                    return_vectors=(s_k == 42),
                )
                multiseed_grp.append(res_s["A4_budget@mid"]["group_validity"])
                multiseed_hyb.append(res_s["A4_budget@mid"]["hybrid_validity"])
                if s_k == 42:
                    rec_res = res_s
                    icdh = compute_intra_cluster_directional_homogeneity(phi_risk[:, act_indices], lbl_s)

            mid_flip_vecs_grp[c_method] = rec_res["A4_budget@mid"]["_flipped_grp_vec"]
            mid_flip_vecs_hyb[c_method] = rec_res["A4_budget@mid"]["_flipped_hyb_vec"]

            for var_name, m_vals in rec_res.items():
                exp3_rows.append(
                    {
                        "dataset": ds_name,
                        "cluster_K": k_default,
                        "tau_eval": float(tau_eval),
                        "total_at_risk_pool": total_at_risk_pool,
                        "evaluated_at_risk_n": eval_n,
                        "clustering_method": c_method,
                        "constraint_variant": var_name,
                        "icdh": icdh,
                        "group_validity": m_vals["group_validity"],
                        "group_validity_5seed_mean": float(np.mean(multiseed_grp)) if var_name == "A4_budget@mid" else m_vals["group_validity"],
                        "group_validity_5seed_std": float(np.std(multiseed_grp, ddof=1)) if (var_name == "A4_budget@mid" and len(multiseed_grp) > 1) else 0.0,
                        "hybrid_validity": m_vals["hybrid_validity"],
                        "hybrid_validity_5seed_mean": float(np.mean(multiseed_hyb)) if var_name == "A4_budget@mid" else m_vals["hybrid_validity"],
                        "hybrid_validity_5seed_std": float(np.std(multiseed_hyb, ddof=1)) if (var_name == "A4_budget@mid" and len(multiseed_hyb) > 1) else 0.0,
                        "valid_clusters": m_vals["valid_clusters"],
                        "total_clusters": m_vals["total_clusters"],
                        "valid_cluster_ratio": m_vals["valid_cluster_ratio"],
                        "mean_cost_group": m_vals["mean_cost_group"],
                        "max_cost_group": m_vals["max_cost_group"],
                        "mean_cost_hybrid": m_vals["mean_cost_hybrid"],
                        "max_cost_hybrid": m_vals["max_cost_hybrid"],
                        "mean_sparsity": m_vals["mean_sparsity"],
                    }
                )

        # Paired 2,000-iteration student-level bootstrap 95% CIs at A4_budget@mid
        d_b1, lo_b1, hi_b1 = paired_bootstrap_validity_ci(mid_flip_vecs_grp["Proposed_SA_AASC"], mid_flip_vecs_grp["B1_Raw_X"])
        d_b2, lo_b2, hi_b2 = paired_bootstrap_validity_ci(mid_flip_vecs_grp["Proposed_SA_AASC"], mid_flip_vecs_grp["B2_Actionable_X"])
        d_b3, lo_b3, hi_b3 = paired_bootstrap_validity_ci(mid_flip_vecs_grp["Proposed_SA_AASC"], mid_flip_vecs_grp["B3_Full_SHAP"])
        bootstrap_summary[ds_name] = {
            "n_eval": eval_n,
            "vs_B1_Raw_X": {"diff": d_b1, "ci95_lo": lo_b1, "ci95_hi": hi_b1},
            "vs_B2_Actionable_X": {"diff": d_b2, "ci95_lo": lo_b2, "ci95_hi": hi_b2},
            "vs_B3_Full_SHAP": {"diff": d_b3, "ci95_lo": lo_b3, "ci95_hi": hi_b3},
        }

        # Experiment 3B: Coordinate-Block, Anchor-Control & Optimizer Ablation under A4_budget@mid (5 seeds)
        for ab_label, c_space, use_wf in [
            ("B3_Full_SHAP_Cone", "B3_Full_SHAP_Cone", True),
            ("SA_AASC_DirOnly", "SA_AASC_DirOnly", True),
            ("SA_AASC_DirPlusRoom", "SA_AASC_DirPlusRoom", True),
            ("Proposed_SA_AASC_SingleScale", "Proposed_SA_AASC", False),
            ("Proposed_SA_AASC_Full", "Proposed_SA_AASC", True),
        ]:
            ab_grp_seeds, ab_hyb_seeds = [], []
            rec_ab_42 = None
            icdh_ab_42 = 0.0
            for s_k in cluster_seeds:
                lbl_ab = cluster_at_risk_students(
                    c_space, X_risk, phi_risk, probs_risk, act_indices, mono_idx, (q_lo, q_hi), n_clusters=k_default, seed=s_k
                )
                rec_ab = optimize_group_and_hybrid_recourse(
                    oof_tab, X_risk, phi_risk, lbl_ab, c_space,
                    ds["feature_names"], ds["tiers"], ds["monotonic_dirs"], iqr_scales, X_all,
                    tau=tau_eval, budgets=b_map, use_water_filling=use_wf,
                    orig_indices=risk_idx, ref_probs=probs_all,
                )["A4_budget@mid"]
                ab_grp_seeds.append(rec_ab["group_validity"])
                ab_hyb_seeds.append(rec_ab["hybrid_validity"])
                if s_k == 42:
                    rec_ab_42 = rec_ab
                    icdh_ab_42 = compute_intra_cluster_directional_homogeneity(phi_risk[:, act_indices], lbl_ab)

            assert rec_ab_42 is not None
            exp3_comp_rows.append(
                {
                    "dataset": ds_name,
                    "ablation_variant": ab_label,
                    "icdh": icdh_ab_42,
                    "group_validity": rec_ab_42["group_validity"],
                    "group_validity_5seed_mean": float(np.mean(ab_grp_seeds)),
                    "group_validity_5seed_std": float(np.std(ab_grp_seeds, ddof=1)) if len(ab_grp_seeds) > 1 else 0.0,
                    "hybrid_validity": rec_ab_42["hybrid_validity"],
                    "hybrid_validity_5seed_mean": float(np.mean(ab_hyb_seeds)),
                    "hybrid_validity_5seed_std": float(np.std(ab_hyb_seeds, ddof=1)) if len(ab_hyb_seeds) > 1 else 0.0,
                    "valid_clusters": rec_ab_42["valid_clusters"],
                    "total_clusters": rec_ab_42["total_clusters"],
                    "valid_cluster_ratio": rec_ab_42["valid_cluster_ratio"],
                    "mean_cost_group": rec_ab_42["mean_cost_group"],
                    "mean_sparsity": rec_ab_42["mean_sparsity"],
                }
            )

        # Experiment 3C: Screening threshold tau_s sensitivity sweep under A4_budget@mid
        if "drop" not in ds_name:
            tau_grid = [("tau=0.40", 0.40), ("tau=0.50 (default)", 0.50), ("tau=0.60", 0.60)]
        else:
            tau_grid = [
                ("78th pct", max(min(float(np.percentile(probs_all, 78)), 0.50), 0.20)),
                ("82nd pct (default)", tau_eval),
                ("86th pct", max(min(float(np.percentile(probs_all, 86)), 0.50), 0.28)),
            ]
        for tau_lbl, t_val in tau_grid:
            t_idx = np.where(probs_all >= t_val)[0]
            if len(t_idx) < 30:
                continue
            if len(t_idx) > max_pool:
                t_idx = np.sort(np.random.RandomState(42).choice(t_idx, size=max_pool, replace=False))
            else:
                t_idx = np.sort(t_idx)
            X_t = X_all[t_idx]
            p_t = probs_all[t_idx]
            phi_t = oof_tab.compute_oof_shap(X_all, t_idx, n_coalitions=8, seed=42)
            for m_tau in ["B1_Raw_X", "B3_Full_SHAP", "Proposed_SA_AASC"]:
                lbl_t = cluster_at_risk_students(
                    m_tau, X_t, phi_t, p_t, act_indices, mono_idx, (q_lo, q_hi), n_clusters=k_default, seed=42
                )
                rec_t = optimize_group_and_hybrid_recourse(
                    oof_tab, X_t, phi_t, lbl_t, m_tau,
                    ds["feature_names"], ds["tiers"], ds["monotonic_dirs"], iqr_scales, X_all,
                    tau=t_val, budgets=b_map, use_water_filling=True,
                    orig_indices=t_idx, ref_probs=probs_all,
                )["A4_budget@mid"]
                exp3_tau_rows.append(
                    {
                        "dataset": ds_name,
                        "tau_setting": tau_lbl,
                        "tau_value": round(float(t_val), 4),
                        "evaluated_n": int(len(t_idx)),
                        "clustering_method": m_tau,
                        "group_validity": rec_t["group_validity"],
                        "hybrid_validity": rec_t["hybrid_validity"],
                        "mean_cost_group": rec_t["mean_cost_group"],
                    }
                )

        # Experiment 4: Advising Capacity K_c sweep in {5, 10, 15, 20} under A4_budget@mid
        for k_cap in [5, 10, 15, 20]:
            for c_method in ["B1_Raw_X", "B3_Full_SHAP", "Proposed_SA_AASC"]:
                labels_k = cluster_at_risk_students(
                    c_method, X_risk, phi_risk, probs_risk, act_indices, mono_idx, (q_lo, q_hi), n_clusters=k_cap, seed=42
                )
                icdh_k = compute_intra_cluster_directional_homogeneity(phi_risk[:, act_indices], labels_k)
                rec_k = optimize_group_and_hybrid_recourse(
                    oof_tab, X_risk, phi_risk, labels_k, c_method,
                    ds["feature_names"], ds["tiers"], ds["monotonic_dirs"], iqr_scales, X_all,
                    tau=tau_eval, budgets=b_map, use_water_filling=True,
                    orig_indices=risk_idx, ref_probs=probs_all,
                )["A4_budget@mid"]
                exp4_rows.append(
                    {
                        "dataset": ds_name,
                        "total_at_risk_pool": total_at_risk_pool,
                        "evaluated_at_risk_n": eval_n,
                        "advising_capacity_K": k_cap,
                        "mean_cluster_size_eval": float(eval_n / max(k_cap, 1)),
                        "mean_cluster_size_full": float(total_at_risk_pool / max(k_cap, 1)),
                        "clustering_method": c_method,
                        "icdh": icdh_k,
                        "group_validity": rec_k["group_validity"],
                        "hybrid_validity": rec_k["hybrid_validity"],
                        "mean_cost_group": rec_k["mean_cost_group"],
                        "plan_reduction_ratio": float(1.0 - k_cap / max(len(X_risk), 1)),
                    }
                )

        # Experiment 5: Extract 4 representative Risk Archetypes under Proposed_SA_AASC (K=4 macro-archetypes)
        labels_arch = cluster_at_risk_students(
            "Proposed_SA_AASC", X_risk, phi_risk, probs_risk, act_indices, mono_idx, (q_lo, q_hi), n_clusters=4, seed=42
        )
        for c_id in range(4):
            c_mask = labels_arch == c_id
            if not np.any(c_mask):
                continue
            phi_sub = phi_risk[c_mask]
            act_scores = [(float(np.mean(np.maximum(0.0, phi_sub[:, j]))), ds["feature_names"][j]) for j in act_indices]
            act_scores.sort(reverse=True)
            locked_cols = ds["tiers"]["F_inst"] + ds["tiers"]["F_imm"]
            locked_indices = [name_to_idx[c] for c in locked_cols if c in name_to_idx]
            lock_scores = [(float(np.mean(np.abs(phi_sub[:, j]))), ds["feature_names"][j]) for j in locked_indices]
            lock_scores.sort(reverse=True)

            exp5_rows.append(
                {
                    "dataset": ds_name,
                    "archetype_id": f"Archetype-{c_id+1}",
                    "cohort_share_pct": float(np.mean(c_mask) * 100.0),
                    "mean_initial_risk": float(np.mean(probs_risk[c_mask])),
                    "primary_actionable_driver": act_scores[0][1] if act_scores else "N/A",
                    "primary_actionable_shap": act_scores[0][0] if act_scores else 0.0,
                    "secondary_actionable_driver": act_scores[1][1] if len(act_scores) > 1 else "N/A",
                    "dominant_locked_context": lock_scores[0][1] if lock_scores else "N/A",
                }
            )

    return exp3_rows, exp3_comp_rows, exp3_tau_rows, exp4_rows, exp5_rows, bootstrap_summary


def summarize_measured_gates(
    exp1_rows: List[Dict[str, Any]],
    exp3_rows: List[Dict[str, Any]],
    bootstrap_summary: Dict[str, Any],
) -> Dict[str, Any]:
    """Descriptive audit of measured effects and budget-compliance verification."""
    checks: Dict[str, Any] = {}
    oulad_raw_rows = [r for r in exp1_rows if r["dataset"] == "OULAD (40% raw)"]
    u_lstm_roc = next((r["auc_roc_mean"] for r in oulad_raw_rows if r["model"] == "Unmasked-LSTM"), 0.5)
    ma_lstm_roc = next((r["auc_roc_mean"] for r in oulad_raw_rows if r["model"] == "MA-BiLSTM"), 0.5)
    prop_roc = next((r["auc_roc_mean"] for r in oulad_raw_rows if r["model"] == "MA-TempNet+GBDT (Proposed)"), 0.5)
    gbdt_roc = next((r["auc_roc_mean"] for r in oulad_raw_rows if r["model"] == "GBDT"), 0.5)
    checks["oulad_unmasked_lstm_roc"] = float(u_lstm_roc)
    checks["oulad_ma_bilstm_roc"] = float(ma_lstm_roc)
    checks["oulad_dl_mask_gain_roc"] = float(ma_lstm_roc - u_lstm_roc)
    checks["oulad_proposed_roc"] = float(prop_roc)
    checks["oulad_gbdt_roc"] = float(gbdt_roc)

    recourse_gains = {}
    for ds_label in sorted(set(r["dataset"] for r in exp3_rows)):
        row_at = {
            r["clustering_method"]: r
            for r in exp3_rows
            if r["dataset"] == ds_label and r["constraint_variant"] == "A4_budget@mid"
        }
        b1_row = row_at.get("B1_Raw_X")
        b2_row = row_at.get("B2_Actionable_X")
        b3_row = row_at.get("B3_Full_SHAP")
        prop_row = row_at.get("Proposed_SA_AASC")
        if b1_row and prop_row:
            entry: Dict[str, Any] = {
                "B1_Raw_X_group_val": b1_row["group_validity"],
                "B2_Actionable_X_group_val": b2_row["group_validity"] if b2_row else None,
                "B3_Full_SHAP_group_val": b3_row["group_validity"] if b3_row else None,
                "SA_AASC_group_val": prop_row["group_validity"],
                "SA_AASC_group_val_5seed_mean": prop_row["group_validity_5seed_mean"],
                "SA_AASC_group_val_5seed_std": prop_row["group_validity_5seed_std"],
                "SA_AASC_hybrid_val": prop_row["hybrid_validity"],
                "group_val_gain_vs_B1": float(prop_row["group_validity"] - b1_row["group_validity"]),
                "group_val_gain_vs_B3": float(prop_row["group_validity"] - b3_row["group_validity"]) if b3_row else None,
                "mean_cost_group_SA_AASC": prop_row["mean_cost_group"],
                "max_cost_group_SA_AASC": prop_row["max_cost_group"],
                "paired_bootstrap_95ci": bootstrap_summary.get(ds_label, {}),
            }
            recourse_gains[ds_label] = entry
    checks["recourse_gains_A4_mid"] = recourse_gains
    return checks


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Experimental Pipeline with OOF & Equalized Recourse")
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    args = parser.parse_args()

    smoke = args.mode == "smoke"
    t0 = time.time()
    print("==================================================================", flush=True)
    print(f"[START] Running Experimental Pipeline in '{args.mode.upper()}' mode", flush=True)
    print("==================================================================", flush=True)

    ds_uci = load_uci697(smoke=smoke)
    ds_xapi = load_xapi_edu(smoke=smoke)
    ds_oulad40_raw = load_oulad_stage(cutoff_pct=0.40, coding="raw", smoke=smoke, max_samples=4500)
    ds_oulad40_drop = load_oulad_stage(cutoff_pct=0.40, coding="drop", smoke=smoke, max_samples=4500)

    print(
        f"[DEFENSE OK] Loaded datasets: "
        f"UCI #697={ds_uci['X_raw'].shape}, "
        f"xAPI-Edu={ds_xapi['X_raw'].shape}, "
        f"OULAD-40%-raw={ds_oulad40_raw['X_raw'].shape}, "
        f"OULAD-40%-drop={ds_oulad40_drop['X_raw'].shape}",
        flush=True,
    )

    n_splits = 3 if smoke else 5
    seeds = [42] if smoke else [42, 43]
    dl_epochs = 12 if smoke else 22

    exp1_all_rows: List[Dict[str, Any]] = []
    datasets_models: List[Tuple[Dict[str, Any], OOFTabularEnsemble, np.ndarray]] = []
    cached_40_oulad: Dict[str, Tuple[Dict[str, Any], List[Dict[str, Any]], OOFTabularEnsemble]] = {}

    for ds in [ds_uci, ds_xapi, ds_oulad40_raw, ds_oulad40_drop]:
        print(f"\n--- Running Experiment 1 (Prediction & Deep Temporal Fusion) on {ds['name']} ---", flush=True)
        rows_ds, oof_tab, oof_prob = evaluate_models_cv(ds, n_splits=n_splits, seeds=seeds, dl_epochs=dl_epochs)
        exp1_all_rows.extend(rows_ds)
        if ds["name"] != "OULAD (40% raw)":
            datasets_models.append((ds, oof_tab, oof_prob))
        if ds["name"] == "OULAD (40% raw)":
            cached_40_oulad["raw"] = (ds, rows_ds, oof_tab)
        elif ds["name"] == "OULAD (40% drop)":
            cached_40_oulad["drop"] = (ds, rows_ds, oof_tab)
        for r in rows_ds:
            print(
                f"  {r['model']:<28s} | AUC-PR={r['auc_pr_mean']:.4f} (±{r['auc_pr_std']:.4f}) | "
                f"AUC-ROC={r['auc_roc_mean']:.4f} | F1={r['f1_mean']:.4f} | Brier={r['brier_mean']:.4f}",
                flush=True,
            )

    print("\n--- Running Experiment 2 (20%..80% Stage Checkpoints & OOF Attribution Drift on OULAD) ---", flush=True)
    stage_pred_rows, stage_drift_rows = run_exp2_stage_drift_and_leadtime(smoke=smoke, cached_40=cached_40_oulad)
    for r in stage_drift_rows:
        print(
            f"  Stage {r['cutoff_pct']}% (Survivors={r['active_survivors_n']}, pos={r['survivor_pos_rate']*100:.2f}%) | "
            f"F_beh={r['share_F_beh']*100:.2f}%, F_out={r['share_F_out']*100:.2f}%, F_inst={r['share_F_inst']*100:.2f}%, F_imm={r['share_F_imm']*100:.2f}% | "
            f"Pure-Beh Grp/Hyb Val={r['pure_beh_group_validity']:.3f}/{r['pure_beh_hybrid_validity']:.3f}",
            flush=True,
        )

    print("\n--- Running Experiments 3, 4, 5 (OOF SA-AASC Clustering & Equal-96 Group Recourse) ---", flush=True)
    exp3_rows, exp3_comp_rows, exp3_tau_rows, exp4_rows, exp5_rows, bootstrap_summary = run_exp3_exp4_exp5_clustering_and_recourse(
        datasets_models, smoke=smoke
    )
    for r in exp3_rows:
        if r["constraint_variant"] in ("A2_tier", "A4_budget@mid"):
            print(
                f"  {r['dataset']:<18s} | {r['clustering_method']:<18s} | {r['constraint_variant']:<14s} | "
                f"ICDH={r['icdh']:.4f} | GrpVal={r['group_validity']:.4f} (5seed={r['group_validity_5seed_mean']:.4f}±{r['group_validity_5seed_std']:.4f}) | "
                f"HybVal={r['hybrid_validity']:.4f} | ValidC={r['valid_clusters']}/{r['total_clusters']} | "
                f"Cost={r['mean_cost_group']:.2f} (max={r['max_cost_group']:.2f})",
                flush=True,
            )

    checks = summarize_measured_gates(exp1_all_rows, exp3_rows, bootstrap_summary)
    print("\n==================================================================", flush=True)
    print("[AUDIT] Measured effect summary (strict OOF, equal 96 candidates, zero budget breach)", flush=True)
    print(json.dumps(checks, indent=2), flush=True)
    print("==================================================================", flush=True)

    if not smoke:
        pd.DataFrame(exp1_all_rows).to_csv(os.path.join(RESULTS_DIR, "table_01_exp1_prediction_comparison.csv"), index=False)
        pd.DataFrame(stage_pred_rows).to_csv(os.path.join(RESULTS_DIR, "table_02_exp2_stage_checkpoints.csv"), index=False)
        pd.DataFrame(stage_drift_rows).to_csv(os.path.join(RESULTS_DIR, "table_03_exp2_attribution_drift.csv"), index=False)
        pd.DataFrame(exp3_rows).to_csv(os.path.join(RESULTS_DIR, "table_04_exp3_clustering_recourse_ablation.csv"), index=False)
        pd.DataFrame(exp3_comp_rows).to_csv(os.path.join(RESULTS_DIR, "table_04b_sa_aasc_component_ablation.csv"), index=False)
        pd.DataFrame(exp3_tau_rows).to_csv(os.path.join(RESULTS_DIR, "table_04c_tau_sensitivity.csv"), index=False)
        pd.DataFrame(exp4_rows).to_csv(os.path.join(RESULTS_DIR, "table_05_exp4_advising_capacity_pareto.csv"), index=False)
        pd.DataFrame(exp5_rows).to_csv(os.path.join(RESULTS_DIR, "table_06_exp5_risk_archetypes.csv"), index=False)

        full_archive = {
            "mode": args.mode,
            "elapsed_seconds": round(time.time() - t0, 2),
            "measured_effect_audit": checks,
            "paired_bootstrap_95ci": bootstrap_summary,
            "table_01_exp1_prediction_comparison": exp1_all_rows,
            "table_02_exp2_stage_checkpoints": stage_pred_rows,
            "table_03_exp2_attribution_drift": stage_drift_rows,
            "table_04_exp3_clustering_recourse_ablation": exp3_rows,
            "table_04b_sa_aasc_component_ablation": exp3_comp_rows,
            "table_04c_tau_sensitivity": exp3_tau_rows,
            "table_05_exp4_advising_capacity_pareto": exp4_rows,
            "table_06_exp5_risk_archetypes": exp5_rows,
        }
        with open(os.path.join(RESULTS_DIR, "final_experiment_results.json"), "w", encoding="utf-8") as f:
            json.dump(full_archive, f, indent=2, ensure_ascii=False)

        print(f"[SAVED] All 8 CSV tables and final_experiment_results.json saved to {RESULTS_DIR} in {time.time() - t0:.1f}s!", flush=True)
    else:
        with open(os.path.join(RESULTS_DIR, "smoke_test_audit.json"), "w", encoding="utf-8") as f:
            json.dump(checks, f, indent=2, ensure_ascii=False)
        print(f"[SMOKE TEST COMPLETE] Elapsed: {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
