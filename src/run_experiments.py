"""
Master Runner with Defensive Strategy, Smoke-Test Gate, Preset-Target Verification,
Auto-Adjustment Loop, and Result Archiving for Paper 2:
"Bridging Temporal Deep Learning and Actionable Group Recourse: Stage-Aware
Attribution Subspace Clustering for Student Early-Warning Systems"
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Dict, List

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
    DualStreamTempGBDTClassifier,
    MaskAwareStageBiLSTMClassifier,
    RegularizedTabNetWrapper,
    UnmaskedLSTMWrapper,
)
from sa_aasc_recourse import (
    cluster_at_risk_students,
    compute_intra_cluster_directional_homogeneity,
    compute_iqr_scales,
    compute_local_shap_attributions,
    compute_tier_attribution_shares,
    optimize_group_and_hybrid_recourse,
)


EXP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(EXP_DIR, "results")
FIGURES_DIR = os.path.join(EXP_DIR, "figures")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


def evaluate_models_cv(
    ds: Dict[str, Any],
    n_splits: int = 5,
    seeds: List[int] = [42],
    dl_epochs: int = 24,
) -> Tuple[List[Dict[str, Any]], DualStreamTempGBDTClassifier, np.ndarray]:
    """
    Evaluate all 7 architectures (LogReg, RF, GBDT, MLP, Reg-TabNet, Unmasked-LSTM,
    MA-BiLSTM, and Proposed MA-TempNet+GBDT) under stratified CV with defensive checks.
    Returns summary rows, a fitted DualStreamTempGBDTClassifier on full data for
    downstream XAI/recourse audit, and out-of-fold probabilities.
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

            # 8. MA-TempNet+GBDT (Proposed Dual-Stream) with inner out-of-fold stacking
            p_tr_dl, emb_tr = ma_lstm.predict_proba_and_emb(X_tr, seq_tr, mask_tr)
            X_tr_aug = np.column_stack([X_tr, p_tr_dl[:, None], emb_tr])
            X_te_aug = np.column_stack([X_te, p_malstm[:, None], emb_te])
            inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
            oof_dl = np.zeros(len(tr_idx), dtype=np.float64)
            oof_emb = np.zeros((len(tr_idx), emb_tr.shape[1]), dtype=np.float64)
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

    # Fit full DualStreamTempGBDTClassifier for downstream attribution & recourse audit
    full_model = DualStreamTempGBDTClassifier(seed=seeds[0], epochs=dl_epochs)
    full_model.fit(X, seq, mask, y)
    return rows, full_model, oof_probs_proposed


def run_exp2_stage_drift_and_leadtime(
    smoke: bool = False,
    cached_40: Dict[str, Tuple[ Dict[str, Any], List[Dict[str, Any]], DualStreamTempGBDTClassifier ]] | None = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Experiment 1B & Experiment 2:
    Across OULAD course progress stages (20%, 40%, 60%, 80%), evaluate:
    1) Active-survivor (`drop`) vs. full-cohort (`raw`) predictive performance of
       Unmasked-LSTM, MA-BiLSTM, GBDT, and MA-TempNet+GBDT under the unified
       cross-validation protocol (max_samples=4500, 5-fold CV, seeds=[42, 43]).
    2) Stage-wise evolution of 5-tier attribution shares (F_beh, F_out, F_inst, F_imm, F_der)
       and pure-behavioural recourse feasibility window.
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
                ds_cp, rows_cp, model_cp = cached_40[coding]
            else:
                ds_cp = load_oulad_stage(cutoff_pct=cp, coding=coding, smoke=smoke, max_samples=max_n)
                rows_cp, model_cp, _ = evaluate_models_cv(ds_cp, n_splits=n_splits, seeds=seeds, dl_epochs=dl_epochs)
            for r in rows_cp:
                if r["model"] in ("Unmasked-LSTM", "MA-BiLSTM", "GBDT", "MA-TempNet+GBDT (Proposed)"):
                    stage_pred_rows.append(
                        {
                            "cutoff_pct": int(cp * 100),
                            "coding": coding,
                            "pos_rate": float(np.mean(ds_cp["y"])),
                            "survivor_fraction": float(
                                (1.0 - ds_cp["p_raw_full"]) / max(1.0 - ds_cp["p_drop_full"], 1e-6)
                            ),
                            "model": r["model"],
                            "auc_pr": r["auc_pr_mean"],
                            "auc_roc": r["auc_roc_mean"],
                            "f1": r["f1_mean"],
                            "brier": r["brier_mean"],
                        }
                    )

            if coding == "drop":
                # Compute stage attribution drift on active survivors
                X_all = ds_cp["X_raw"]
                probs_tab = model_cp.predict_proba_tabular(X_all)
                at_risk_idx = np.where(probs_tab >= 0.35)[0]
                if len(at_risk_idx) < 30:
                    at_risk_idx = np.argsort(-probs_tab)[: min(120, len(X_all))]
                X_eval = X_all[at_risk_idx[: min(200, len(at_risk_idx))]]
                phi_cp = compute_local_shap_attributions(
                    model_cp.predict_proba_tabular, X_eval, X_all, n_coalitions=8, seed=42
                )
                shares = compute_tier_attribution_shares(phi_cp, ds_cp["feature_names"], ds_cp["tiers"])

                # Also compute pure-behavioural (F_beh only) vs F_beh+F_out recourse flip rate at this checkpoint
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
                    probs_tab[at_risk_idx[: len(X_eval)]],
                    beh_indices,
                    mono_idx,
                    (q_lo, q_hi),
                    n_clusters=10,
                    seed=42,
                )
                rec_beh = optimize_group_and_hybrid_recourse(
                    model_cp.predict_proba_tabular,
                    X_eval,
                    phi_cp,
                    labels_beh,
                    "Proposed_SA_AASC",
                    ds_cp["feature_names"],
                    beh_only_tiers,
                    ds_cp["monotonic_dirs"],
                    iqr_sc,
                    X_all,
                    tau=float(np.median(probs_tab[at_risk_idx[: len(X_eval)]]) * 0.88),
                )

                stage_drift_rows.append(
                    {
                        "cutoff_pct": int(cp * 100),
                        "active_survivors_n": ds_cp["n_survivor_cohort"],
                        "survivor_pos_rate": ds_cp["p_drop_full"],
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
    datasets_models: List[Tuple[Dict[str, Any], DualStreamTempGBDTClassifier, np.ndarray]],
    smoke: bool = False,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Run:
    - Experiment 3: 4 Clustering Spaces x 6 Constraint Stages Group & Hybrid Recourse
    - Experiment 3B: SA-AASC Coordinate-Block & Optimizer Ablation (DirOnly, DirPlusRoom, SingleScale, Full)
    - Experiment 4: Advising Capacity K_c in {5, 10, 15, 20} Pareto Frontier
    - Experiment 5: Risk Archetype Profiling under SA-AASC
    """
    exp3_rows: List[Dict[str, Any]] = []
    exp3_comp_rows: List[Dict[str, Any]] = []
    exp4_rows: List[Dict[str, Any]] = []
    exp5_rows: List[Dict[str, Any]] = []

    cluster_counts_default = {"UCI #697": 22, "xAPI-Edu": 20}
    budgets_by_ds = {
        "UCI #697": {"low": 3.2, "mid": 5.2, "high": 7.8},
        "xAPI-Edu": {"low": 1.6, "mid": 2.6, "high": 4.0},
    }

    for ds, model, oof_prob in datasets_models:
        ds_name = ds["name"]
        X_all = ds["X_raw"]
        probs_all = oof_prob.copy()

        # Select at-risk evaluation pool using out-of-fold scores
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

        X_risk = X_all[risk_idx]
        probs_risk = probs_all[risk_idx]
        eval_n = int(len(X_risk))
        phi_risk = compute_local_shap_attributions(
            model.predict_proba_tabular, X_risk, X_all, n_coalitions=8, seed=42
        )

        name_to_idx = {c: i for i, c in enumerate(ds["feature_names"])}
        act_cols = ds["tiers"]["F_beh"] + ds["tiers"]["F_out"]
        act_indices = [name_to_idx[c] for c in act_cols if c in name_to_idx]
        mono_idx = {name_to_idx[k]: v for k, v in ds["monotonic_dirs"].items() if k in name_to_idx}

        q_lo = np.percentile(X_all, 2, axis=0)
        q_hi = np.percentile(X_all, 98, axis=0)
        iqr_scales = compute_iqr_scales(X_all)
        k_default = cluster_counts_default.get(ds_name, 21)
        b_map = budgets_by_ds.get(ds_name, {"low": 3.0, "mid": 5.0, "high": 7.5})

        for c_method in ["B1_Raw_X", "B2_Actionable_X", "B3_Full_SHAP", "Proposed_SA_AASC"]:
            labels = cluster_at_risk_students(
                c_method,
                X_risk,
                phi_risk,
                probs_risk,
                act_indices,
                mono_idx,
                (q_lo, q_hi),
                n_clusters=k_default,
                seed=42,
            )
            icdh = compute_intra_cluster_directional_homogeneity(phi_risk[:, act_indices], labels)
            rec_res = optimize_group_and_hybrid_recourse(
                model.predict_proba_tabular,
                X_risk,
                phi_risk,
                labels,
                c_method,
                ds["feature_names"],
                ds["tiers"],
                ds["monotonic_dirs"],
                iqr_scales,
                X_all,
                tau=tau_eval,
                budgets=b_map,
                use_water_filling=True,
            )
            for var_name, m_vals in rec_res.items():
                exp3_rows.append(
                    {
                        "dataset": ds_name,
                        "tau_eval": float(tau_eval),
                        "total_at_risk_pool": total_at_risk_pool,
                        "evaluated_at_risk_n": eval_n,
                        "clustering_method": c_method,
                        "constraint_variant": var_name,
                        "icdh": icdh,
                        "group_validity": m_vals["group_validity"],
                        "hybrid_validity": m_vals["hybrid_validity"],
                        "valid_clusters": m_vals["valid_clusters"],
                        "total_clusters": m_vals["total_clusters"],
                        "valid_cluster_ratio": m_vals["valid_cluster_ratio"],
                        "mean_cost_group": m_vals["mean_cost_group"],
                        "mean_cost_hybrid": m_vals["mean_cost_hybrid"],
                        "mean_sparsity": m_vals["mean_sparsity"],
                    }
                )

        # Experiment 3B: Coordinate-Block & Optimizer Ablation under A4_budget@mid
        for ab_label, c_space, use_wf in [
            ("SA_AASC_DirOnly", "SA_AASC_DirOnly", True),
            ("SA_AASC_DirPlusRoom", "SA_AASC_DirPlusRoom", True),
            ("Proposed_SA_AASC_SingleScale", "Proposed_SA_AASC", False),
            ("Proposed_SA_AASC_Full", "Proposed_SA_AASC", True),
        ]:
            lbl_ab = cluster_at_risk_students(
                c_space,
                X_risk,
                phi_risk,
                probs_risk,
                act_indices,
                mono_idx,
                (q_lo, q_hi),
                n_clusters=k_default,
                seed=42,
            )
            icdh_ab = compute_intra_cluster_directional_homogeneity(phi_risk[:, act_indices], lbl_ab)
            rec_ab = optimize_group_and_hybrid_recourse(
                model.predict_proba_tabular,
                X_risk,
                phi_risk,
                lbl_ab,
                c_space,
                ds["feature_names"],
                ds["tiers"],
                ds["monotonic_dirs"],
                iqr_scales,
                X_all,
                tau=tau_eval,
                budgets=b_map,
                use_water_filling=use_wf,
            )["A4_budget@mid"]
            exp3_comp_rows.append(
                {
                    "dataset": ds_name,
                    "ablation_variant": ab_label,
                    "icdh": icdh_ab,
                    "group_validity": rec_ab["group_validity"],
                    "hybrid_validity": rec_ab["hybrid_validity"],
                    "valid_clusters": rec_ab["valid_clusters"],
                    "total_clusters": rec_ab["total_clusters"],
                    "valid_cluster_ratio": rec_ab["valid_cluster_ratio"],
                    "mean_cost_group": rec_ab["mean_cost_group"],
                    "mean_sparsity": rec_ab["mean_sparsity"],
                }
            )

        # Experiment 4: Advising Capacity K_c sweep in {5, 10, 15, 20} under A4_budget@mid
        for k_cap in [5, 10, 15, 20]:
            for c_method in ["B1_Raw_X", "Proposed_SA_AASC"]:
                labels_k = cluster_at_risk_students(
                    c_method,
                    X_risk,
                    phi_risk,
                    probs_risk,
                    act_indices,
                    mono_idx,
                    (q_lo, q_hi),
                    n_clusters=k_cap,
                    seed=42,
                )
                icdh_k = compute_intra_cluster_directional_homogeneity(phi_risk[:, act_indices], labels_k)
                rec_k = optimize_group_and_hybrid_recourse(
                    model.predict_proba_tabular,
                    X_risk,
                    phi_risk,
                    labels_k,
                    c_method,
                    ds["feature_names"],
                    ds["tiers"],
                    ds["monotonic_dirs"],
                    iqr_scales,
                    X_all,
                    tau=tau_eval,
                    budgets=b_map,
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
            "Proposed_SA_AASC",
            X_risk,
            phi_risk,
            probs_risk,
            act_indices,
            mono_idx,
            (q_lo, q_hi),
            n_clusters=4,
            seed=42,
        )
        rec_arch = optimize_group_and_hybrid_recourse(
            model.predict_proba_tabular,
            X_risk,
            phi_risk,
            labels_arch,
            "Proposed_SA_AASC",
            ds["feature_names"],
            ds["tiers"],
            ds["monotonic_dirs"],
            iqr_scales,
            X_all,
            tau=tau_eval,
            budgets=b_map,
        )
        for c_id in range(4):
            c_mask = labels_arch == c_id
            if not np.any(c_mask):
                continue
            phi_sub = phi_risk[c_mask]
            # Top-2 actionable deficit features and top-1 locked institutional/immutable driver
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

    return exp3_rows, exp3_comp_rows, exp4_rows, exp5_rows


def summarize_measured_gates(
    exp1_rows,
    exp3_rows,
):
    """Descriptive audit of measured effects. Reports observed values only; no target is imposed."""
    checks = {}
    oulad_raw_rows = [r for r in exp1_rows if r["dataset"] == "OULAD (40% raw)"]
    if not oulad_raw_rows:
        oulad_raw_rows = [r for r in exp1_rows if "OULAD" in r["dataset"]]
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
    for ds_label in set(r["dataset"] for r in exp3_rows):
        row_at = {}
        for r in exp3_rows:
            if r["dataset"] == ds_label and r["constraint_variant"] == "A4_budget@mid":
                row_at[r["clustering_method"]] = r
        b1_row = row_at.get("B1_Raw_X")
        b3_row = row_at.get("B3_Full_SHAP")
        prop_row = row_at.get("Proposed_SA_AASC")
        if b1_row and prop_row:
            entry = {
                "B1_Raw_X_group_val": b1_row["group_validity"],
                "SA_AASC_group_val": prop_row["group_validity"],
                "SA_AASC_hybrid_val": prop_row["hybrid_validity"],
                "group_val_gain_vs_B1": float(prop_row["group_validity"] - b1_row["group_validity"]),
                "B1_icdh": b1_row["icdh"],
                "SA_AASC_icdh": prop_row["icdh"],
                "mean_cost_group_SA_AASC": prop_row["mean_cost_group"],
            }
            if b3_row:
                entry["B3_Full_SHAP_group_val"] = b3_row["group_validity"]
                entry["group_val_gain_vs_B3"] = float(prop_row["group_validity"] - b3_row["group_validity"])
                entry["mean_cost_group_B3"] = b3_row["mean_cost_group"]
            recourse_gains[ds_label] = entry
    checks["recourse_gains_A4_mid"] = recourse_gains
    return checks


def generate_publication_figures(
    exp1_rows: List[Dict[str, Any]],
    stage_pred_rows: List[Dict[str, Any]],
    stage_drift_rows: List[Dict[str, Any]],
    exp3_rows: List[Dict[str, Any]],
    exp4_rows: List[Dict[str, Any]],
) -> None:
    """Generate 5 publication-grade 300-DPI figures summarizing all 5 experiments."""
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10})

    # Figure 1: Predictive comparison across datasets (Exp 1)
    ds_list_exp1 = ["UCI #697", "xAPI-Edu", "OULAD (40% raw)", "OULAD (40% drop)"]
    ds_list_exp1 = [d for d in ds_list_exp1 if any(r["dataset"] == d for r in exp1_rows)]
    fig, axes = plt.subplots(1, len(ds_list_exp1), figsize=(4.2 * len(ds_list_exp1), 4.2), dpi=300)
    if len(ds_list_exp1) == 1:
        axes = [axes]
    for ax, ds_name in zip(axes, ds_list_exp1):
        sub = [r for r in exp1_rows if r["dataset"] == ds_name]
        models = [r["model"].replace(" (Proposed)", "*") for r in sub]
        auc_prs = [r["auc_pr_mean"] for r in sub]
        auc_rocs = [r["auc_roc_mean"] for r in sub]
        y_pos = np.arange(len(models))
        ax.barh(y_pos - 0.18, auc_rocs, height=0.35, label="AUC-ROC", color="#2b5c8f")
        ax.barh(y_pos + 0.18, auc_prs, height=0.35, label="AUC-PR", color="#d95f02")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(models, fontsize=8.0)
        ax.set_xlim(0.1, 1.02)
        ax.set_title(ds_name)
        ax.grid(axis="x", linestyle="--", alpha=0.4)
    axes[0].legend(loc="lower right", fontsize=8.0)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig1_model_prediction_comparison.png"), dpi=300)
    plt.close()

    ds_list = ["UCI #697", "xAPI-Edu", "OULAD (40% drop)"]
    ds_list = [d for d in ds_list if any(r["dataset"] == d for r in exp3_rows)]

    # Figure 2: Stage-Aware Attribution Drift & Pure-Behavioural Recourse Window across 20%..80% (Exp 2)
    if stage_drift_rows:
        fig, ax1 = plt.subplots(figsize=(7.5, 4.5), dpi=300)
        cps = [r["cutoff_pct"] for r in stage_drift_rows]
        s_beh = [r["share_F_beh"] * 100 for r in stage_drift_rows]
        s_out = [r["share_F_out"] * 100 for r in stage_drift_rows]
        s_inst = [r["share_F_inst"] * 100 for r in stage_drift_rows]
        beh_val = [r["pure_beh_hybrid_validity"] * 100 for r in stage_drift_rows]

        ax1.plot(cps, s_beh, "o-", linewidth=2.2, label="Behavioural Attribution Share (F_beh %)", color="#1b9e77")
        ax1.plot(cps, s_out, "s--", linewidth=2.0, label="Assessment Outcome Share (F_out %)", color="#d95f02")
        ax1.plot(cps, s_inst, "^:", linewidth=1.8, label="Institutional Share (F_inst %)", color="#7570b3")
        ax1.plot(cps, beh_val, "D-", linewidth=2.4, label="Actionable Behavioural Recourse Rate (%)", color="#e7298a")
        ax1.set_xlabel("Course Progress Checkpoint (%)")
        ax1.set_ylabel("Percentage (%)")
        ax1.set_xticks(cps)
        ax1.set_title("Stage-Aware Attribution Drift & Actionable Window across Active Survivors")
        ax1.grid(True, linestyle="--", alpha=0.4)
        ax1.legend(loc="best", fontsize=8.5)
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, "fig2_stage_attribution_drift.png"), dpi=300)
        plt.close()

    # Figure 3: 4 Clustering Spaces Group & Hybrid Recourse Validity under A4_budget@mid (Exp 3)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), dpi=300)
    for ax, ds_name in zip(axes, ds_list):
        sub = [
            r for r in exp3_rows
            if r["dataset"] == ds_name and r["constraint_variant"] == "A4_budget@mid"
        ]
        methods = [r["clustering_method"].replace("Proposed_", "") for r in sub]
        grp_v = [r["group_validity"] for r in sub]
        hyb_v = [r["hybrid_validity"] for r in sub]
        icdh_v = [r["icdh"] for r in sub]
        x_p = np.arange(len(methods))
        ax.bar(x_p - 0.25, grp_v, width=0.25, label="Group Template Validity", color="#386cb0")
        ax.bar(x_p, hyb_v, width=0.25, label="Bi-Stage Hybrid Validity", color="#66a61e")
        ax.bar(x_p + 0.25, icdh_v, width=0.25, label="Directional Homogeneity (ICDH)", color="#e6ab02")
        ax.set_xticks(x_p)
        ax.set_xticklabels(methods, rotation=20, fontsize=8.5)
        ax.set_ylim(0.0, 1.05)
        ax.set_title(f"{ds_name} (A4_budget@mid)")
        ax.grid(axis="y", linestyle="--", alpha=0.4)
    axes[0].legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig3_clustering_recourse_ablation.png"), dpi=300)
    plt.close()

    # Figure 4: Advising Capacity K_c Pareto Frontier (Exp 4)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0), dpi=300)
    for ax, ds_name in zip(axes, ds_list):
        for m_name, col, mk in [("B1_Raw_X", "#d95f02", "s--"), ("Proposed_SA_AASC", "#1b9e77", "o-")]:
            sub = [r for r in exp4_rows if r["dataset"] == ds_name and r["clustering_method"] == m_name]
            ks = [r["advising_capacity_K"] for r in sub]
            g_val = [r["group_validity"] for r in sub]
            h_val = [r["hybrid_validity"] for r in sub]
            ax.plot(ks, g_val, mk, color=col, linewidth=2.0, label=f"{m_name} (Pure Group)")
            ax.plot(ks, h_val, mk, color=col, linewidth=1.4, alpha=0.65, linestyle=":", label=f"{m_name} (Hybrid +1 Feat)")
        ax.set_xlabel("Instructor Advising Capacity (Clusters K_c)")
        ax.set_ylabel("Valid Recourse Rate")
        ax.set_xticks([5, 10, 15, 20])
        ax.set_ylim(0.0, 1.02)
        ax.set_title(ds_name)
        ax.grid(True, linestyle="--", alpha=0.4)
    axes[0].legend(loc="lower right", fontsize=7.5)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig4_advising_capacity_pareto.png"), dpi=300)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Paper 2 Experiments with Defensive Strategy")
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    args = parser.parse_args()

    smoke = args.mode == "smoke"
    t0 = time.time()
    print(f"==================================================================")
    print(f"[START] Running Paper 2 Experimental Pipeline in '{args.mode.upper()}' mode")
    print(f"==================================================================")

    # Load primary datasets (including OULAD 40% raw & active-survivor drop)
    ds_uci = load_uci697(smoke=smoke)
    ds_xapi = load_xapi_edu(smoke=smoke)
    ds_oulad40_raw = load_oulad_stage(cutoff_pct=0.40, coding="raw", smoke=smoke, max_samples=4500)
    ds_oulad40_drop = load_oulad_stage(cutoff_pct=0.40, coding="drop", smoke=smoke, max_samples=4500)

    print(
        f"[DEFENSE OK] Loaded datasets: "
        f"UCI #697={ds_uci['X_raw'].shape}, "
        f"xAPI-Edu={ds_xapi['X_raw'].shape}, "
        f"OULAD-40%-raw={ds_oulad40_raw['X_raw'].shape}, "
        f"OULAD-40%-drop={ds_oulad40_drop['X_raw'].shape}"
    )

    n_splits = 3 if smoke else 5
    seeds = [42] if smoke else [42, 43]
    dl_epochs = 12 if smoke else 22

    # Experiment 1: Multi-Model Predictive Evaluation
    exp1_all_rows: List[Dict[str, Any]] = []
    datasets_models: List[Tuple[Dict[str, Any], DualStreamTempGBDTClassifier, np.ndarray]] = []
    cached_40_oulad: Dict[str, Tuple[Dict[str, Any], List[Dict[str, Any]], DualStreamTempGBDTClassifier]] = {}

    for ds in [ds_uci, ds_xapi, ds_oulad40_raw, ds_oulad40_drop]:
        print(f"\n--- Running Experiment 1 (Prediction & Deep Temporal Fusion) on {ds['name']} ---")
        rows_ds, fitted_model, oof_prob = evaluate_models_cv(ds, n_splits=n_splits, seeds=seeds, dl_epochs=dl_epochs)
        exp1_all_rows.extend(rows_ds)
        if ds["name"] != "OULAD (40% raw)":
            datasets_models.append((ds, fitted_model, oof_prob))
        if ds["name"] == "OULAD (40% raw)":
            cached_40_oulad["raw"] = (ds, rows_ds, fitted_model)
        elif ds["name"] == "OULAD (40% drop)":
            cached_40_oulad["drop"] = (ds, rows_ds, fitted_model)
        for r in rows_ds:
            print(
                f"  {r['model']:<28s} | AUC-PR={r['auc_pr_mean']:.4f} (±{r['auc_pr_std']:.4f}) | "
                f"AUC-ROC={r['auc_roc_mean']:.4f} | F1={r['f1_mean']:.4f} | Brier={r['brier_mean']:.4f}",
                flush=True,
            )

    # Experiment 2: Multi-Stage Progress Checkpoints (20%, 40%, 60%, 80%) & Attribution Drift
    print(f"\n--- Running Experiment 2 (20%..80% Stage Checkpoints & Attribution Drift on OULAD) ---", flush=True)
    stage_pred_rows, stage_drift_rows = run_exp2_stage_drift_and_leadtime(smoke=smoke, cached_40=cached_40_oulad)
    for r in stage_drift_rows:
        print(
            f"  Stage {r['cutoff_pct']}% (Survivors={r['active_survivors_n']}, pos={r['survivor_pos_rate']*100:.1f}%) | "
            f"F_beh={r['share_F_beh']*100:.1f}%, F_out={r['share_F_out']*100:.1f}%, F_inst={r['share_F_inst']*100:.1f}% | "
            f"Pure-Beh Hybrid Val={r['pure_beh_hybrid_validity']:.3f}",
            flush=True,
        )

    # Experiments 3, 4, 5: Subspace Clustering Recourse Ablation, Capacity Pareto, and Risk Archetypes
    print(f"\n--- Running Experiments 3, 4, 5 (SA-AASC Clustering & Constrained Group Recourse) ---", flush=True)
    exp3_rows, exp3_comp_rows, exp4_rows, exp5_rows = run_exp3_exp4_exp5_clustering_and_recourse(datasets_models, smoke=smoke)
    for r in exp3_rows:
        if r["constraint_variant"] in ("A2_tier", "A4_budget@mid"):
            print(
                f"  {r['dataset']:<18s} | {r['clustering_method']:<18s} | {r['constraint_variant']:<14s} | "
                f"ICDH={r['icdh']:.3f} | GrpVal={r['group_validity']:.3f} | HybVal={r['hybrid_validity']:.3f} | "
                f"ValidClusters={r['valid_clusters']}/{r['total_clusters']}",
                flush=True,
            )

    # Defensive Preset-Target Verification Gate
    checks = summarize_measured_gates(exp1_all_rows, exp3_rows)
    print(f"\n==================================================================")
    print("[AUDIT] Measured effect summary (no imposed target)")
    print(json.dumps(checks, indent=2), flush=True)
    print(f"==================================================================")

    if not smoke:
        # Save CSVs, JSON archive, and 300-DPI figures
        pd.DataFrame(exp1_all_rows).to_csv(os.path.join(RESULTS_DIR, "table_01_exp1_prediction_comparison.csv"), index=False)
        pd.DataFrame(stage_pred_rows).to_csv(os.path.join(RESULTS_DIR, "table_02_exp2_stage_checkpoints.csv"), index=False)
        pd.DataFrame(stage_drift_rows).to_csv(os.path.join(RESULTS_DIR, "table_03_exp2_attribution_drift.csv"), index=False)
        pd.DataFrame(exp3_rows).to_csv(os.path.join(RESULTS_DIR, "table_04_exp3_clustering_recourse_ablation.csv"), index=False)
        pd.DataFrame(exp3_comp_rows).to_csv(os.path.join(RESULTS_DIR, "table_04b_sa_aasc_component_ablation.csv"), index=False)
        pd.DataFrame(exp4_rows).to_csv(os.path.join(RESULTS_DIR, "table_05_exp4_advising_capacity_pareto.csv"), index=False)
        pd.DataFrame(exp5_rows).to_csv(os.path.join(RESULTS_DIR, "table_06_exp5_risk_archetypes.csv"), index=False)

        full_archive = {
            "mode": args.mode,
            "elapsed_seconds": round(time.time() - t0, 2),
            "measured_effect_audit": checks,
            "table_01_exp1_prediction_comparison": exp1_all_rows,
            "table_02_exp2_stage_checkpoints": stage_pred_rows,
            "table_03_exp2_attribution_drift": stage_drift_rows,
            "table_04_exp3_clustering_recourse_ablation": exp3_rows,
            "table_04b_sa_aasc_component_ablation": exp3_comp_rows,
            "table_05_exp4_advising_capacity_pareto": exp4_rows,
            "table_06_exp5_risk_archetypes": exp5_rows,
        }
        with open(os.path.join(RESULTS_DIR, "final_experiment_results.json"), "w", encoding="utf-8") as f:
            json.dump(full_archive, f, indent=2, ensure_ascii=False)

        generate_publication_figures(exp1_all_rows, stage_pred_rows, stage_drift_rows, exp3_rows, exp4_rows)
        print(f"[SAVED] All 7 CSV tables, final_experiment_results.json, and publication figures saved to {RESULTS_DIR} and {FIGURES_DIR}!", flush=True)
    else:
        with open(os.path.join(RESULTS_DIR, "smoke_test_audit.json"), "w", encoding="utf-8") as f:
            json.dump(checks, f, indent=2, ensure_ascii=False)
        print(f"[SMOKE TEST COMPLETE] Elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
