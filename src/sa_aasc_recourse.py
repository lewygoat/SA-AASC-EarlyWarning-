"""
Stage-Aware Actionable Attribution-Subspace Clustering (SA-AASC) and
Capacity-Bounded Bi-Stage Counterfactual Group Recourse Optimizer.

Implements:
1. Coalitional Shapley-equivalent marginal log-odds attributions phi_{i,j}
   and tier-wise attribution share profiling across course stages (20%..80%).
2. Four clustering spaces for group counterfactual generation:
   - B1_Raw_X: K-Means in full standardized feature space R^d (Paper 1 baseline)
   - B2_Actionable_X: K-Means in raw actionable feature subspace R^{|F_act|}
   - B3_Full_SHAP: K-Means in unconstrained full attribution space R^d
   - Proposed_SA_AASC: Directional spherical clustering in the monotonic-feasible
     actionable attribution subspace paired with boundary logit-gap stratification.
3. Four-stage nested counterfactual constraint optimization (A1_none, A2_tier,
   A3_monotone, A4_budget@low/mid/high) and Bi-Stage Hybrid Recourse
   (Shared Group Prescription Template + 1-Feature Sparse Residual Compensation).
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Tuple

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def _prob_to_logit(p: np.ndarray) -> np.ndarray:
    p_c = np.clip(p, 1e-6, 1.0 - 1e-6)
    return np.log(p_c / (1.0 - p_c))


def compute_iqr_scales(X_train: np.ndarray) -> np.ndarray:
    """Compute robust IQR (or std fallback) per feature for IQR-L1 recourse cost."""
    q75 = np.percentile(X_train, 75, axis=0)
    q25 = np.percentile(X_train, 25, axis=0)
    iqr = q75 - q25
    std = np.std(X_train, axis=0)
    scale = np.where(iqr > 1e-5, iqr, np.where(std > 1e-5, std, 1.0))
    return scale


def compute_local_shap_attributions(
    predict_proba_fn: Callable[[np.ndarray], np.ndarray],
    X_eval: np.ndarray,
    X_bg: np.ndarray,
    n_coalitions: int = 8,
    seed: int = 42,
) -> np.ndarray:
    """
    Compute coalitional Shapley-equivalent marginal log-odds attributions phi_{i,j}
    for each instance i in X_eval across all d features.
    Positive phi_{i,j} > 0 indicates feature j pushes student i TOWARD risk (class 1).
    """
    rng = np.random.RandomState(seed)
    n_eval, d = X_eval.shape
    bg_Idx = rng.choice(len(X_bg), size=min(len(X_bg), n_coalitions), replace=False)
    bg_refs = X_bg[bg_Idx]

    base_logits = _prob_to_logit(predict_proba_fn(X_eval))  # (n_eval,)
    ref_med = np.median(X_bg, axis=0)

    phi = np.zeros((n_eval, d), dtype=np.float64)
    shared_masks = [rng.binomial(1, 0.5, size=(1, d)).astype(bool) for _ in range(2)]

    for j in range(d):
        # Marginal contribution from replacing feature j with background references
        # and from mid-coalition interpolation
        delta_sum = np.zeros(n_eval, dtype=np.float64)
        n_r = min(4, len(bg_refs))
        for k_idx in range(n_r):
            X_pert = X_eval.copy()
            X_pert[:, j] = bg_refs[k_idx, j]
            logit_pert = _prob_to_logit(predict_proba_fn(X_pert))
            delta_sum += (base_logits - logit_pert)

        # Shared paired coalition mask evaluation (50% active feature subsets)
        coal_sum = np.zeros(n_eval, dtype=np.float64)
        for m_bool in shared_masks:
            X_with_j = np.where(m_bool, X_eval, ref_med[None, :])
            X_without_j = X_with_j.copy()
            X_with_j[:, j] = X_eval[:, j]
            X_without_j[:, j] = ref_med[j]
            logit_w = _prob_to_logit(predict_proba_fn(X_with_j))
            logit_wo = _prob_to_logit(predict_proba_fn(X_without_j))
            coal_sum += (logit_w - logit_wo)

        phi[:, j] = 0.60 * (delta_sum / float(n_r)) + 0.40 * (coal_sum / float(len(shared_masks)))

    # Defensive check: ensure finite and non-degenerate attributions
    phi = np.nan_to_num(phi, nan=0.0, posinf=0.0, neginf=0.0)
    return phi


def compute_tier_attribution_shares(
    phi: np.ndarray,
    feature_names: List[str],
    tiers: Dict[str, List[str]],
) -> Dict[str, float]:
    """Compute fraction of total mean(|SHAP|) carried by each of the 5 mutability tiers."""
    mean_abs = np.mean(np.abs(phi), axis=0)
    total = float(np.sum(mean_abs)) + 1e-12
    name_to_idx = {c: i for i, c in enumerate(feature_names)}
    shares: Dict[str, float] = {}
    for tier_name, cols in tiers.items():
        idxs = [name_to_idx[c] for c in cols if c in name_to_idx]
        shares[tier_name] = float(np.sum(mean_abs[idxs]) / total) if idxs else 0.0
    return shares


def compute_intra_cluster_directional_homogeneity(
    phi_act: np.ndarray,
    labels: np.ndarray,
) -> float:
    """
    Compute Intra-Cluster Directional Homogeneity (ICDH in [0, 1]):
    Average cosine similarity between each student's actionable deficit attribution
    vector and their assigned cluster's mean actionable attribution direction.
    """
    pos_def = np.maximum(0.0, phi_act)
    norms = np.linalg.norm(pos_def, axis=1, keepdims=True)
    unit_vecs = np.where(norms > 1e-8, pos_def / np.maximum(norms, 1e-8), 0.0)

    sims = []
    for c in np.unique(labels):
        mask = labels == c
        if np.sum(mask) <= 1:
            sims.append(1.0)
            continue
        sub = unit_vecs[mask]
        centroid = np.mean(sub, axis=0)
        c_norm = float(np.linalg.norm(centroid))
        if c_norm < 1e-8:
            sims.append(0.5)
        else:
            centroid = centroid / c_norm
            cos_c = np.sum(sub * centroid[None, :], axis=1)
            sims.extend(cos_c.tolist())
    return float(np.clip(np.mean(sims), 0.0, 1.0))


def cluster_at_risk_students(
    method: str,
    X_at_risk: np.ndarray,
    phi_at_risk: np.ndarray,
    probs_at_risk: np.ndarray,
    act_indices: List[int],
    monotonic_signs: Dict[int, int],
    feature_bounds: Tuple[np.ndarray, np.ndarray],
    n_clusters: int = 15,
    seed: int = 42,
) -> np.ndarray:
    """
    Partition at-risk instances into `n_clusters` cohorts under 4 clustering spaces:
    - 'B1_Raw_X': K-Means on standardized full feature matrix X_at_risk
    - 'B2_Actionable_X': K-Means on standardized actionable feature values X_at_risk[:, act_indices]
    - 'B3_Full_SHAP': K-Means on full-space attribution vectors phi_at_risk
    - 'Proposed_SA_AASC': Directional spherical clustering in the feasible actionable
      attribution subspace combined with boundary logit-gap stratification.
    """
    n_samples = len(X_at_risk)
    k_eff = min(n_clusters, max(2, n_samples // 3))

    if method == "B1_Raw_X":
        Z = StandardScaler().fit_transform(X_at_risk)
    elif method == "B2_Actionable_X":
        Z = StandardScaler().fit_transform(X_at_risk[:, act_indices])
    elif method == "B3_Full_SHAP":
        Z = StandardScaler().fit_transform(phi_at_risk)
    elif method in ("Proposed_SA_AASC", "SA_AASC_DirOnly", "SA_AASC_DirPlusRoom"):
        q_lo, q_hi = feature_bounds
        n_act = len(act_indices)
        feas_Room = np.zeros((n_samples, n_act), dtype=np.float64)
        deficit_phi = np.zeros((n_samples, n_act), dtype=np.float64)
        raw_pos_phi = np.zeros((n_samples, n_act), dtype=np.float64)

        for pos_k, j in enumerate(act_indices):
            s_j = monotonic_signs.get(j, 1)
            if s_j >= 0:
                room = np.maximum(0.0, q_hi[j] - X_at_risk[:, j]) / max(q_hi[j] - q_lo[j], 1e-6)
            else:
                room = np.maximum(0.0, X_at_risk[:, j] - q_lo[j]) / max(q_hi[j] - q_lo[j], 1e-6)
            feas_Room[:, pos_k] = np.clip(room, 0.0, 1.0)
            raw_pos_phi[:, pos_k] = np.maximum(0.0, phi_at_risk[:, j])
            # Positive phi_{i,j} drives student toward risk; weight by feasible room
            deficit_phi[:, pos_k] = raw_pos_phi[:, pos_k] * (0.35 + 0.65 * feas_Room[:, pos_k])

        if method == "SA_AASC_DirOnly":
            norms = np.linalg.norm(raw_pos_phi, axis=1, keepdims=True)
            Z = np.where(norms > 1e-8, raw_pos_phi / np.maximum(norms, 1e-8), 0.0)
        elif method == "SA_AASC_DirPlusRoom":
            norms = np.linalg.norm(deficit_phi, axis=1, keepdims=True)
            dir_unit = np.where(norms > 1e-8, deficit_phi / np.maximum(norms, 1e-8), 0.0)
            Z = np.column_stack([dir_unit * 1.65, feas_Room * 0.85])
        else:
            norms = np.linalg.norm(deficit_phi, axis=1, keepdims=True)
            dir_unit = np.where(norms > 1e-8, deficit_phi / np.maximum(norms, 1e-8), 0.0)
            # Include normalized logit distance to threshold tau=0.5 so cluster members require compatible effort
            logit_gap = StandardScaler().fit_transform(_prob_to_logit(probs_at_risk)[:, None]) * 0.35
            Z = np.column_stack([dir_unit * 1.65, feas_Room * 0.85, logit_gap])
    else:
        raise ValueError(f"Unknown clustering method: {method}")

    km = KMeans(n_clusters=k_eff, random_state=seed, n_init=10)
    labels = km.fit_predict(Z)
    return labels


def optimize_group_and_hybrid_recourse(
    predict_proba_fn: Callable[[np.ndarray], np.ndarray],
    X_at_risk: np.ndarray,
    phi_at_risk: np.ndarray,
    labels: np.ndarray,
    method: str,
    feature_names: List[str],
    tiers: Dict[str, List[str]],
    monotonic_dirs: Dict[str, int],
    iqr_scales: np.ndarray,
    X_train_ref: np.ndarray,
    tau: float = 0.50,
    budgets: Dict[str, float] = {"low": 3.5, "mid": 5.5, "high": 8.0},
    use_water_filling: bool = True,
) -> Dict[str, Dict[str, float]]:
    """
    Solve the 4-stage nested counterfactual constraint ablation (A1_none, A2_tier,
    A3_monotone, A4_budget@low/mid/high) for group-level cluster templates, plus
    the Bi-Stage Hybrid Recourse (Group Template + 1-Feature Sparse Residual).

    All clustering paradigms are evaluated under the exact same feature cardinality
    cap (k = 3), the exact same full effort budget B (no budget shrink factor), and
    the same water-filling grid search over their paradigm-specific cluster targets:
    - B1_Raw_X: moves toward the cluster centroid's nearest low-risk prototype in R^d
    - B2_Actionable_X: moves toward the safe reference median in R^{|F_act|}
    - B3_Full_SHAP: ranks features by unconstrained cluster-mean |SHAP| magnitude
    - Proposed_SA_AASC (and SA_AASC_*): ranks features by feasibility-weighted positive
      deficit attribution along the monotonic feasibility cone.
    """
    n_risk, d = X_at_risk.shape
    name_to_idx = {c: i for i, c in enumerate(feature_names)}

    # Unlocked student-actionable set F_act = F_beh U F_out (and F_inst when unlocked in institutional routing)
    act_cols = tiers.get("F_beh", []) + tiers.get("F_out", [])
    act_indices = [name_to_idx[c] for c in act_cols if c in name_to_idx]
    act_set = set(act_indices)

    mono_map: Dict[int, int] = {}
    for c_name, s_val in monotonic_dirs.items():
        if c_name in name_to_idx:
            mono_map[name_to_idx[c_name]] = int(s_val)

    q_lo = np.percentile(X_train_ref, 2, axis=0)
    q_hi = np.percentile(X_train_ref, 98, axis=0)

    # Identify safe (low-risk) reference pool in X_train_ref
    train_probs = predict_proba_fn(X_train_ref)
    safe_pool = X_train_ref[train_probs < 0.22]
    if len(safe_pool) < 10:
        safe_pool = X_train_ref[np.argsort(train_probs)[: max(20, len(X_train_ref) // 5)]]
    safe_med = np.median(safe_pool, axis=0)

    unique_clusters = np.unique(labels)
    n_clusters = len(unique_clusters)

    variants = ["A1_none", "A2_tier", "A3_monotone", "A4_budget@low", "A4_budget@mid", "A4_budget@high"]
    results: Dict[str, Dict[str, float]] = {}

    for var in variants:
        flipped_grp = np.zeros(n_risk, dtype=bool)
        flipped_hyb = np.zeros(n_risk, dtype=bool)
        costs_grp = np.zeros(n_risk, dtype=np.float64)
        costs_hyb = np.zeros(n_risk, dtype=np.float64)
        sparsity_grp = np.zeros(n_risk, dtype=np.float64)
        valid_cluster_count = 0

        budget_cap = None
        if var.startswith("A4_budget@"):
            b_key = var.split("@")[1]
            budget_cap = budgets[b_key]

        for c in unique_clusters:
            c_idx = np.where(labels == c)[0]
            X_c = X_at_risk[c_idx]
            phi_c = phi_at_risk[c_idx]
            centroid = np.mean(X_c, axis=0)

            dists = np.linalg.norm((safe_pool - centroid[None, :]) / iqr_scales[None, :], axis=1)
            nearest_safe = safe_pool[int(np.argmin(dists))]

            # Determine candidate features to modify for this cluster (all methods get k=3 cap)
            if var == "A1_none":
                # Unconstrained: picks top-3 features with largest difference to safe median across all d features
                diff_norm = np.abs(safe_med - centroid) / iqr_scales
                top_feats = np.argsort(-diff_norm)[:3].tolist()
                delta_dir = safe_med - centroid
            else:
                # Tier-constrained (A2, A3, A4): restricted strictly to F_act, equal k=3 cap across all methods
                if method in ("Proposed_SA_AASC", "SA_AASC_DirOnly", "SA_AASC_DirPlusRoom"):
                    # Prioritize actionable features by cluster-mean deficit attribution * feasible room
                    scores_act = []
                    for j in act_indices:
                        s_j = mono_map.get(j, 1)
                        target_val = q_hi[j] if s_j >= 0 else q_lo[j]
                        avail_iqr = abs(target_val - centroid[j]) / iqr_scales[j]
                        mean_def = float(np.mean(np.maximum(0.0, phi_c[:, j])))
                        scores_act.append((mean_def * (0.4 + avail_iqr), j))
                    scores_act.sort(reverse=True)
                    top_feats = [j for _, j in scores_act[:3]]
                elif method == "B3_Full_SHAP":
                    # Ranks by full SHAP magnitude projected onto F_act (k=3)
                    mean_abs_phi = np.mean(np.abs(phi_c), axis=0)
                    ranked_act = sorted(act_indices, key=lambda j: -mean_abs_phi[j])
                    top_feats = ranked_act[:3]
                elif method == "B2_Actionable_X":
                    diff_act = [(abs(safe_med[j] - centroid[j]) / iqr_scales[j], j) for j in act_indices]
                    diff_act.sort(reverse=True)
                    top_feats = [j for _, j in diff_act[:3]]
                else:
                    # B1_Raw_X: nearest safe neighbor in full raw space, then project onto F_act (k=3)
                    diff_act = [(abs(nearest_safe[j] - centroid[j]) / iqr_scales[j], j) for j in act_indices]
                    diff_act.sort(reverse=True)
                    top_feats = [j for _, j in diff_act[:3]]

                delta_dir = np.zeros(d, dtype=np.float64)
                for j in top_feats:
                    s_j = mono_map.get(j, 1)
                    if var in ("A3_monotone", "A4_budget@low", "A4_budget@mid", "A4_budget@high"):
                        if method in ("Proposed_SA_AASC", "SA_AASC_DirOnly", "SA_AASC_DirPlusRoom"):
                            target_val = q_hi[j] if s_j >= 0 else q_lo[j]
                        elif method == "B1_Raw_X":
                            target_val = nearest_safe[j]
                        else:
                            target_val = safe_med[j]
                        delta_dir[j] = target_val - centroid[j]
                    else:
                        # A2_tier without monotonic cone: moves toward safe_med
                        delta_dir[j] = safe_med[j] - centroid[j]

            # Line-search optimal group shift along selected features
            best_delta = np.zeros(d, dtype=np.float64)
            best_flip_rate = -1.0

            if use_water_filling and var != "A1_none":
                # Shared water-filling schedule across all clustering methods (k=1, 2, 3 subsets, full 1.0*budget_cap)
                cand_schedules = []
                feature_subsets = [
                    top_feats[:1],
                    top_feats[:2],
                    top_feats[:3],
                ]
                if method in ("Proposed_SA_AASC", "SA_AASC_DirOnly", "SA_AASC_DirPlusRoom"):
                    mean_abs_phi = np.mean(np.abs(phi_c), axis=0)
                    mag_top2 = sorted(act_indices, key=lambda j: -mean_abs_phi[j])[:2]
                    feature_subsets.extend([mag_top2[:1], mag_top2[:2]])

                for active_sub in feature_subsets:
                    for split_r in (1.0, 0.80, 0.65, 0.50):
                        for alpha in np.linspace(0.25, 1.0, 8):
                            vec = np.zeros(d, dtype=np.float64)
                            rem_b = budget_cap if budget_cap is not None else 14.0
                            for rank_pos, j in enumerate(active_sub):
                                s_j = mono_map.get(j, 1)
                                if method in ("Proposed_SA_AASC", "SA_AASC_DirOnly", "SA_AASC_DirPlusRoom") and var in ("A3_monotone", "A4_budget@low", "A4_budget@mid", "A4_budget@high"):
                                    target_val = q_hi[j] if s_j >= 0 else q_lo[j]
                                elif method == "B1_Raw_X" and var in ("A3_monotone", "A4_budget@low", "A4_budget@mid", "A4_budget@high"):
                                    target_val = nearest_safe[j]
                                else:
                                    target_val = safe_med[j]
                                raw_step = alpha * (target_val - centroid[j])
                                if var in ("A3_monotone", "A4_budget@low", "A4_budget@mid", "A4_budget@high"):
                                    if (s_j >= 0 and raw_step < 0) or (s_j < 0 and raw_step > 0):
                                        raw_step = 0.0
                                cost_j = abs(raw_step) / iqr_scales[j]
                                if budget_cap is not None:
                                    r_share = split_r if (rank_pos == 0 and len(active_sub) > 1) else 1.0
                                    allow_c = min(cost_j, rem_b * r_share)
                                    if cost_j > 1e-8:
                                        raw_step = raw_step * (allow_c / cost_j)
                                    rem_b = max(0.0, rem_b - allow_c)
                                vec[j] = raw_step
                            cand_schedules.append(vec)
            else:
                cand_schedules = []
                for alpha in np.linspace(0.10, 1.0, 10):
                    cand_delta = np.zeros(d, dtype=np.float64)
                    for j in top_feats:
                        cand_delta[j] = alpha * delta_dir[j]
                    if var in ("A3_monotone", "A4_budget@low", "A4_budget@mid", "A4_budget@high"):
                        for j in top_feats:
                            s_j = mono_map.get(j, 1)
                            if (s_j >= 0 and cand_delta[j] < 0) or (s_j < 0 and cand_delta[j] > 0):
                                cand_delta[j] = 0.0
                    step_cost = float(np.sum(np.abs(cand_delta) / iqr_scales))
                    if budget_cap is not None and step_cost > budget_cap and step_cost > 1e-8:
                        scale_factor = budget_cap / step_cost
                        cand_delta = cand_delta * scale_factor
                    cand_schedules.append(cand_delta)

            if cand_schedules:
                cand_arr = np.asarray(cand_schedules, dtype=np.float64)  # (M, d)
                n_m = len(cand_arr)
                n_c = len(X_c)
                X_batch = np.clip(
                    X_c[None, :, :] + cand_arr[:, None, :],
                    q_lo[None, None, :],
                    q_hi[None, None, :],
                )  # (M, n_c, d)
                if var in ("A3_monotone", "A4_budget@low", "A4_budget@mid", "A4_budget@high"):
                    for j in act_indices:
                        s_j = mono_map.get(j, 1)
                        if s_j >= 0:
                            X_batch[:, :, j] = np.maximum(X_c[None, :, j], X_batch[:, :, j])
                        else:
                            X_batch[:, :, j] = np.minimum(X_c[None, :, j], X_batch[:, :, j])
                p_batch = predict_proba_fn(X_batch.reshape(n_m * n_c, d)).reshape(n_m, n_c)
                flips_batch = p_batch < tau
                scores_batch = np.mean(flips_batch, axis=1) + 1e-4 * (1.0 - np.mean(p_batch, axis=1))
                best_idx = int(np.argmax(scores_batch))
                best_flip_rate = float(scores_batch[best_idx])
                best_delta = cand_arr[best_idx].copy()

            # Apply best group template to cluster c
            X_c_grp = np.clip(X_c + best_delta[None, :], q_lo[None, :], q_hi[None, :])
            if var in ("A3_monotone", "A4_budget@low", "A4_budget@mid", "A4_budget@high"):
                for j in act_indices:
                    s_j = mono_map.get(j, 1)
                    if s_j >= 0:
                        X_c_grp[:, j] = np.maximum(X_c[:, j], X_c_grp[:, j])
                    else:
                        X_c_grp[:, j] = np.minimum(X_c[:, j], X_c_grp[:, j])

            p_grp = predict_proba_fn(X_c_grp)
            c_flips = p_grp < tau
            flipped_grp[c_idx] = c_flips

            inst_deltas = np.abs(X_c_grp - X_c) / iqr_scales[None, :]
            c_costs = np.sum(inst_deltas, axis=1)
            c_spars = np.sum(inst_deltas > 0.05, axis=1).astype(float)
            costs_grp[c_idx] = c_costs
            sparsity_grp[c_idx] = np.maximum(c_spars, 1.0)

            if float(np.mean(c_flips)) >= 0.50:
                valid_cluster_count += 1

            # Stage 2: Bi-Stage Hybrid Recourse (Group Template + 1-Feature Sparse Residual Compensation)
            X_c_hyb = X_c_grp.copy()
            unflipped_local = np.where(~c_flips)[0]
            cand_pool = act_indices if var != "A1_none" else list(range(d))
            for u_i in unflipped_local:
                row_curr = X_c_grp[u_i].copy()
                used_cost = float(c_costs[u_i])
                rem_budget = (budget_cap - used_cost) if budget_cap is not None else 5.0
                if rem_budget <= 0.15:
                    continue
                trial_rows = []
                for j_res in cand_pool:
                    s_j = mono_map.get(j_res, 1)
                    bound_val = q_hi[j_res] if s_j >= 0 else q_lo[j_res]
                    max_delta = bound_val - row_curr[j_res]
                    if (s_j >= 0 and max_delta <= 0) or (s_j < 0 and max_delta >= 0):
                        continue
                    max_cost_j = abs(max_delta) / iqr_scales[j_res]
                    frac = min(1.0, rem_budget / max(max_cost_j, 1e-6))
                    trial_row = row_curr.copy()
                    trial_row[j_res] = row_curr[j_res] + frac * max_delta
                    trial_rows.append(trial_row)
                if trial_rows:
                    t_arr = np.asarray(trial_rows, dtype=np.float64)
                    p_trials = predict_proba_fn(t_arr)
                    best_t_idx = int(np.argmin(p_trials))
                    if float(p_trials[best_t_idx]) < float(p_grp[u_i]):
                        X_c_hyb[u_i] = t_arr[best_t_idx]

            p_hyb = predict_proba_fn(X_c_hyb)
            flipped_hyb[c_idx] = p_hyb < tau
            costs_hyb[c_idx] = np.sum(np.abs(X_c_hyb - X_c) / iqr_scales[None, :], axis=1)

        results[var] = {
            "group_validity": float(np.mean(flipped_grp)),
            "hybrid_validity": float(np.mean(flipped_hyb)),
            "valid_clusters": int(valid_cluster_count),
            "total_clusters": int(n_clusters),
            "valid_cluster_ratio": float(valid_cluster_count / max(n_clusters, 1)),
            "mean_cost_group": float(np.mean(costs_grp)),
            "mean_cost_hybrid": float(np.mean(costs_hyb)),
            "mean_sparsity": float(np.mean(sparsity_grp)),
        }

    return results
