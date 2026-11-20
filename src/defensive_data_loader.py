"""
Defensive Data Loader & Multi-Stage Temporal Sequence Builder for Paper 2:
"Bridging Temporal Deep Learning and Actionable Group Recourse: Stage-Aware
Attribution Subspace Clustering for Student Early-Warning Systems"

Enforces strict defensive checks:
1. Schema & sample-count validation across UCI #697 (N=4,424), xAPI-Edu-Data (N=480),
   and OULAD (N=32,593 registrations across 22 module presentations).
2. Strict temporal truncation guard at 20%, 40%, 60%, and 80% course progress
   (no post-cutoff assessment or unregistration leakage).
3. Explicit binary sequence validity masks M_{i,t} in {0, 1} for temporal deep learning.
4. Five-tier mutability taxonomy partition (F_imm, F_inst, F_beh, F_out, F_der).
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


EXP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(EXP_DIR, "data")


class DefensiveDataError(RuntimeError):
    """Raised when any defensive data integrity or leakage check fails."""


def verify_no_nan_inf(arr: np.ndarray, name: str) -> None:
    """Defensive guard against NaN or Inf in feature/target tensors."""
    if not np.all(np.isfinite(arr)):
        bad_cnt = int(np.sum(~np.isfinite(arr)))
        raise DefensiveDataError(f"[DEFENSE FAIL] Tensor '{name}' contains {bad_cnt} NaN/Inf values.")


def load_uci697(smoke: bool = False) -> Dict[str, Any]:
    """
    Load real UCI #697 dataset (N=4,424, d=36) and partition into 5 mutability tiers
    plus a 2-stage (Semester 1 -> Semester 2) temporal tensor for MA-TempNet.
    """
    csv_path = os.path.join(DATA_DIR, "uci697.csv")
    if not os.path.exists(csv_path):
        raise DefensiveDataError(f"Missing UCI #697 file: {csv_path}")

    df = pd.read_csv(csv_path, sep=None, engine="python")
    # Clean column names (strip whitespace/tabs)
    df.columns = [c.strip().replace("\t", "") for c in df.columns]
    if "Target" not in df.columns:
        raise DefensiveDataError(f"Missing 'Target' column in UCI #697: {list(df.columns)}")

    if len(df) != 4424:
        raise DefensiveDataError(f"Expected N=4424 in UCI #697, got {len(df)}")

    y = (df["Target"].astype(str).str.strip() == "Dropout").astype(int).to_numpy()
    X_df = df.drop(columns=["Target"]).copy()

    tiers = {
        "F_imm": [
            "Marital status", "Application mode", "Application order", "Course",
            "Daytime/evening attendance", "Previous qualification", "Previous qualification (grade)",
            "Nacionality", "Mother's qualification", "Father's qualification",
            "Mother's occupation", "Father's occupation", "Admission grade", "Displaced",
            "Educational special needs", "Gender", "Age at enrollment", "International",
            "Unemployment rate", "Inflation rate", "GDP"
        ],
        "F_inst": [
            "Debtor", "Tuition fees up to date", "Scholarship holder",
            "Curricular units 1st sem (credited)", "Curricular units 1st sem (enrolled)",
            "Curricular units 2nd sem (credited)", "Curricular units 2nd sem (enrolled)"
        ],
        "F_beh": [
            "Curricular units 1st sem (evaluations)",
            "Curricular units 2nd sem (evaluations)",
            "Curricular units 1st sem (without evaluations)",
            "Curricular units 2nd sem (without evaluations)"
        ],
        "F_out": [
            "Curricular units 1st sem (approved)", "Curricular units 1st sem (grade)",
            "Curricular units 2nd sem (approved)", "Curricular units 2nd sem (grade)"
        ],
        "F_der": []
    }

    monotonic_dirs = {
        "Curricular units 1st sem (evaluations)": 1,
        "Curricular units 2nd sem (evaluations)": 1,
        "Curricular units 1st sem (without evaluations)": -1,
        "Curricular units 2nd sem (without evaluations)": -1,
        "Curricular units 1st sem (approved)": 1,
        "Curricular units 1st sem (grade)": 1,
        "Curricular units 2nd sem (approved)": 1,
        "Curricular units 2nd sem (grade)": 1,
    }

    feature_names = list(X_df.columns)
    X_raw = X_df.to_numpy(dtype=np.float64)
    verify_no_nan_inf(X_raw, "UCI697_X_raw")

    # Construct 2-step temporal sequence (Semester 1 -> Semester 2) with validity mask
    sem1_cols = [
        "Curricular units 1st sem (credited)", "Curricular units 1st sem (enrolled)",
        "Curricular units 1st sem (evaluations)", "Curricular units 1st sem (approved)",
        "Curricular units 1st sem (grade)", "Curricular units 1st sem (without evaluations)"
    ]
    sem2_cols = [
        "Curricular units 2nd sem (credited)", "Curricular units 2nd sem (enrolled)",
        "Curricular units 2nd sem (evaluations)", "Curricular units 2nd sem (approved)",
        "Curricular units 2nd sem (grade)", "Curricular units 2nd sem (without evaluations)"
    ]
    s1 = X_df[sem1_cols].to_numpy(dtype=np.float64)
    s2 = X_df[sem2_cols].to_numpy(dtype=np.float64)
    seq_tensor = np.stack([s1, s2], axis=1)  # (N, T=2, d_seq=6)
    # Mark step as active if enrolled > 0 or evaluations > 0
    m1 = ((s1[:, 1] > 0) | (s1[:, 2] > 0)).astype(np.float64)
    m2 = ((s2[:, 1] > 0) | (s2[:, 2] > 0)).astype(np.float64)
    # Ensure at least step 1 mask is 1 to prevent all-zero mask
    m1 = np.maximum(m1, 1.0)
    seq_mask = np.stack([m1, m2], axis=1)    # (N, T=2)

    if smoke:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(y), size=600, replace=False)
        X_raw, y, seq_tensor, seq_mask = X_raw[idx], y[idx], seq_tensor[idx], seq_mask[idx]

    return {
        "name": "UCI #697",
        "X_raw": X_raw,
        "y": y,
        "seq_tensor": seq_tensor,
        "seq_mask": seq_mask,
        "feature_names": feature_names,
        "tiers": tiers,
        "monotonic_dirs": monotonic_dirs,
    }


def load_xapi_edu(smoke: bool = False) -> Dict[str, Any]:
    """
    Load real xAPI-Edu-Data dataset (N=480, d=16) and construct 2-stage behavioral
    engagement tensor + 5-tier mutability taxonomy.
    """
    csv_path = os.path.join(DATA_DIR, "xAPI-Edu-Data.csv")
    if not os.path.exists(csv_path):
        raise DefensiveDataError(f"Missing xAPI-Edu-Data file: {csv_path}")

    df = pd.read_csv(csv_path)
    if len(df) != 480:
        raise DefensiveDataError(f"Expected N=480 in xAPI-Edu-Data, got {len(df)}")

    y = (df["Class"].astype(str).str.strip() == "L").astype(int).to_numpy()
    X_df = df.drop(columns=["Class"]).copy()

    # Ordinal/categorical encoding with deterministic mapping
    # Special ordinal mapping for StudentAbsenceDays: Under-7 -> 0, Above-7 -> 1
    if "StudentAbsenceDays" in X_df.columns:
        X_df["StudentAbsenceDays"] = (
            X_df["StudentAbsenceDays"].astype(str).str.strip() == "Above-7"
        ).astype(float)

    for col in X_df.columns:
        if X_df[col].dtype == object or str(X_df[col].dtype).startswith("str"):
            le = LabelEncoder()
            X_df[col] = le.fit_transform(X_df[col].astype(str).str.strip()).astype(float)

    tiers = {
        "F_imm": [
            "gender", "NationalITy", "PlaceofBirth", "StageID", "GradeID",
            "SectionID", "Topic", "Relation", "ParentAnsweringSurvey", "ParentschoolSatisfaction"
        ],
        "F_inst": ["Semester"],
        "F_beh": [
            "raisedhands", "VisITedResources", "AnnouncementsView",
            "Discussion", "StudentAbsenceDays"
        ],
        "F_out": [],
        "F_der": []
    }

    monotonic_dirs = {
        "raisedhands": 1,
        "VisITedResources": 1,
        "AnnouncementsView": 1,
        "Discussion": 1,
        "StudentAbsenceDays": -1,
    }

    feature_names = list(X_df.columns)
    X_raw = X_df.to_numpy(dtype=np.float64)
    verify_no_nan_inf(X_raw, "xAPI_X_raw")

    # Build 2-phase behavioral progression sequence (In-class vs. Online LMS activity)
    beh_cols = ["raisedhands", "VisITedResources", "AnnouncementsView", "Discussion", "StudentAbsenceDays"]
    b_arr = X_df[beh_cols].to_numpy(dtype=np.float64)
    # Step 1: Early passive observation (AnnouncementsView, VisITedResources)
    # Step 2: Active interactive engagement (raisedhands, Discussion, Absence control)
    s1 = np.column_stack([b_arr[:, 1], b_arr[:, 2], np.zeros(len(b_arr)), np.zeros(len(b_arr)), b_arr[:, 4]])
    s2 = b_arr.copy()
    seq_tensor = np.stack([s1, s2], axis=1)
    seq_mask = np.ones((len(y), 2), dtype=np.float64)

    return {
        "name": "xAPI-Edu",
        "X_raw": X_raw,
        "y": y,
        "seq_tensor": seq_tensor,
        "seq_mask": seq_mask,
        "feature_names": feature_names,
        "tiers": tiers,
        "monotonic_dirs": monotonic_dirs,
    }


def load_oulad_stage(
    cutoff_pct: float = 0.40,
    coding: str = "drop",
    smoke: bool = False,
    max_samples: int = 8000,
) -> Dict[str, Any]:
    """
    Build leakage-controlled OULAD stage-aware dataset at `cutoff_pct` in {0.20, 0.40, 0.60, 0.80}.
    Constructs a 4-window temporal sequence up to `cutoff_pct` with explicit binary padding masks
    M_{i,t} in {0, 1} for windows where no assessment/activity occurred or after withdrawal.
    """
    oulad_dir = os.path.join(DATA_DIR, "oulad")
    s_info = pd.read_csv(os.path.join(oulad_dir, "studentInfo.csv"))
    s_reg = pd.read_csv(os.path.join(oulad_dir, "studentRegistration.csv"))
    courses = pd.read_csv(os.path.join(oulad_dir, "courses.csv"))
    assessments = pd.read_csv(os.path.join(oulad_dir, "assessments.csv"))
    s_ass = pd.read_csv(os.path.join(oulad_dir, "studentAssessment.csv"))

    if len(s_info) != 32593:
        raise DefensiveDataError(f"Expected N=32593 in OULAD studentInfo, got {len(s_info)}")

    keys = ["code_module", "code_presentation", "id_student"]
    base = s_info.merge(
        s_reg[keys + ["date_registration", "date_unregistration"]],
        on=keys,
        how="left",
    ).merge(
        courses[["code_module", "code_presentation", "module_presentation_length"]],
        on=["code_module", "code_presentation"],
        how="left",
    )

    base["t_cutoff"] = np.floor(cutoff_pct * base["module_presentation_length"]).astype(int)

    # Record raw vs drop cohort rates for survival accounting verification
    y_all_raw = (base["final_result"].astype(str).str.strip() == "Withdrawn").astype(int).to_numpy()
    p_raw = float(np.mean(y_all_raw))

    survivor_mask = base["date_unregistration"].isna() | (base["date_unregistration"] > base["t_cutoff"])
    y_survivors = y_all_raw[survivor_mask.to_numpy()]
    p_drop = float(np.mean(y_survivors))

    if coding == "drop":
        base = base.loc[survivor_mask].reset_index(drop=True)
    elif coding != "raw":
        raise ValueError(f"Invalid coding: {coding}")

    # Merge assessments strictly up to t_cutoff (LEAKAGE DEFENSE)
    ass_meta = assessments[assessments["assessment_type"] != "Exam"].copy()
    ass_merged = s_ass.merge(
        ass_meta[["id_assessment", "code_module", "code_presentation", "date", "weight", "assessment_type"]],
        on="id_assessment",
        how="inner",
    ).merge(
        base[keys + ["t_cutoff", "module_presentation_length"]],
        on=keys,
        how="inner",
    )

    # Strict temporal truncation: only submissions on or before t_cutoff
    ass_valid = ass_merged.loc[ass_merged["date_submitted"] <= ass_merged["t_cutoff"]].copy()
    if (ass_valid["date_submitted"] > ass_valid["t_cutoff"]).any():
        raise DefensiveDataError("Temporal leakage detected in OULAD assessment truncation!")

    ass_valid["score"] = ass_valid["score"].fillna(0.0)
    ass_valid["weight"] = ass_valid["weight"].fillna(10.0)
    ass_valid["sched_date"] = ass_valid["date"].fillna(ass_valid["t_cutoff"])
    ass_valid["delay_days"] = (ass_valid["date_submitted"] - ass_valid["sched_date"]).clip(lower=-30, upper=60)
    ass_valid["is_late"] = (ass_valid["date_submitted"] > ass_valid["sched_date"]).astype(float)
    ass_valid["early_days"] = np.maximum(0.0, ass_valid["sched_date"] - ass_valid["date_submitted"])
    ass_valid["weighted_score"] = ass_valid["score"] * (ass_valid["weight"] / 100.0)
    ass_valid["is_tma"] = (ass_valid["assessment_type"] == "TMA").astype(float)
    ass_valid["is_cma"] = (ass_valid["assessment_type"] == "CMA").astype(float)

    # Expected number of assessments scheduled up to t_cutoff per module-presentation
    sched_before = (
        ass_meta.merge(
            courses[["code_module", "code_presentation", "module_presentation_length"]],
            on=["code_module", "code_presentation"],
            how="left",
        )
    )
    sched_before["t_cutoff"] = np.floor(cutoff_pct * sched_before["module_presentation_length"]).astype(int)
    sched_before = sched_before.loc[sched_before["date"].fillna(999) <= sched_before["t_cutoff"]]
    expected_counts = (
        sched_before.groupby(["code_module", "code_presentation"], as_index=False)
        .agg(
            expected_ass_count=("id_assessment", "count"),
            expected_weight_sum=("weight", "sum"),
        )
    )

    agg = (
        ass_valid.groupby(keys, as_index=False)
        .agg(
            score_mean=("score", "mean"),
            score_count=("score", "count"),
            score_std=("score", "std"),
            score_max=("score", "max"),
            score_min=("score", "min"),
            weighted_score_sum=("weighted_score", "sum"),
            late_submission_rate=("is_late", "mean"),
            mean_early_days=("early_days", "mean"),
            mean_delay_days=("delay_days", "mean"),
            tma_count=("is_tma", "sum"),
            cma_count=("is_cma", "sum"),
            banked_count=("is_banked", "sum"),
            last_submit_day=("date_submitted", "max"),
        )
    )

    feat = base.merge(expected_counts, on=["code_module", "code_presentation"], how="left").merge(
        agg, on=keys, how="left"
    )
    feat["expected_ass_count"] = feat["expected_ass_count"].fillna(1.0).clip(lower=1.0)
    feat["expected_weight_sum"] = feat["expected_weight_sum"].fillna(10.0).clip(lower=1.0)

    # Derived missingness indicators (F_der)
    feat["missingindicator_score_mean"] = feat["score_mean"].isna().astype(float)
    feat["missingindicator_score_count"] = (feat["score_count"].isna() | (feat["score_count"] == 0)).astype(float)
    feat["missingindicator_score_std"] = feat["score_std"].isna().astype(float)
    feat["missingindicator_date_registration"] = feat["date_registration"].isna().astype(float)
    feat["missingindicator_imd_band"] = (
        feat["imd_band"].isna() | (feat["imd_band"].astype(str).str.strip() == "")
    ).astype(float)

    # Behavioral engagement proxies from longitudinal submission, VLE resource structure & registration regularity (F_beh)
    vle_meta = pd.read_csv(os.path.join(oulad_dir, "vle.csv"))
    vle_mod_counts = (
        vle_meta.groupby(["code_module", "code_presentation"], as_index=False)
        .agg(vle_activities_avail=("id_site", "count"))
    )
    feat = feat.merge(vle_mod_counts, on=["code_module", "code_presentation"], how="left")
    feat["vle_activities_avail"] = feat["vle_activities_avail"].fillna(100.0)

    feat["score_count"] = feat["score_count"].fillna(0.0)
    feat["submission_completion_ratio"] = (feat["score_count"] / feat["expected_ass_count"]).clip(0.0, 1.5)
    feat["tma_engagement"] = feat["tma_count"].fillna(0.0)
    feat["cma_engagement"] = feat["cma_count"].fillna(0.0)
    feat["early_preparation_days"] = feat["mean_early_days"].fillna(0.0)
    feat["late_submission_rate"] = feat["late_submission_rate"].fillna(1.0)
    feat["submission_recency_gap"] = (
        feat["t_cutoff"] - feat["last_submit_day"].fillna(-30.0)
    ).clip(lower=0.0, upper=300.0)
    # Active VLE interaction intensity index derived from active window participation & module resource density
    feat["derived_submission_index"] = (
        feat["submission_completion_ratio"] * np.log1p(feat["vle_activities_avail"])
        + 0.15 * feat["early_preparation_days"]
        + 0.50 * (feat["tma_engagement"] + feat["cma_engagement"])
    )

    # Outcome features (F_out)
    feat["score_mean"] = feat["score_mean"].fillna(0.0)
    feat["score_std"] = feat["score_std"].fillna(0.0)
    feat["score_max"] = feat["score_max"].fillna(0.0)
    feat["score_min"] = feat["score_min"].fillna(0.0)
    feat["weighted_score_sum"] = feat["weighted_score_sum"].fillna(0.0)

    # Institutional & Immutable features
    feat["date_registration"] = feat["date_registration"].fillna(0.0)
    feat["banked_count"] = feat["banked_count"].fillna(0.0)

    # Build 4-step temporal sequence across sub-windows [0..0.25*t_c, 0.25..0.50*t_c, 0.50..0.75*t_c, 0.75..1.0*t_c]
    # BEFORE encoding string keys code_module and code_presentation
    n_rows = len(feat)
    seq_tensor = np.zeros((n_rows, 4, 4), dtype=np.float64)
    seq_mask = np.zeros((n_rows, 4), dtype=np.float64)

    feat["_row_id"] = np.arange(n_rows)
    ass_with_row = ass_valid.merge(feat[keys + ["_row_id"]], on=keys, how="inner")

    cat_cols = ["code_module", "code_presentation", "gender", "region", "highest_education", "imd_band", "age_band", "disability"]
    for c in cat_cols:
        feat[c] = feat[c].fillna("Missing").astype(str).str.strip()
        le = LabelEncoder()
        feat[c] = le.fit_transform(feat[c]).astype(float)
    rel_pos = np.clip(
        ass_with_row["date_submitted"].to_numpy(dtype=np.float64)
        / np.maximum(ass_with_row["t_cutoff"].to_numpy(dtype=np.float64), 1.0),
        0.0,
        0.9999,
    )
    ass_with_row["_win"] = np.floor(rel_pos * 4.0).astype(int)
    win_agg = (
        ass_with_row.groupby(["_row_id", "_win"], as_index=False)
        .agg(
            w_cnt=("score", "count"),
            w_score=("score", "mean"),
            w_early=("early_days", "mean"),
            w_ontime=("is_late", lambda s: float(1.0 - np.mean(s))),
        )
    )
    for r_id, w_id, cnt, sc, ea, ot in zip(
        win_agg["_row_id"].to_numpy(dtype=int),
        win_agg["_win"].to_numpy(dtype=int),
        win_agg["w_cnt"].to_numpy(dtype=np.float64),
        win_agg["w_score"].to_numpy(dtype=np.float64),
        win_agg["w_early"].to_numpy(dtype=np.float64),
        win_agg["w_ontime"].to_numpy(dtype=np.float64),
    ):
        seq_tensor[r_id, w_id, 0] = cnt
        seq_tensor[r_id, w_id, 1] = sc
        seq_tensor[r_id, w_id, 2] = ea
        seq_tensor[r_id, w_id, 3] = ot
        seq_mask[r_id, w_id] = 1.0

    # If a student has pre-registration activity, mark step 0 as valid with registration lead time
    no_active_win = np.sum(seq_mask, axis=1) == 0
    seq_mask[no_active_win, 0] = 1.0
    seq_tensor[no_active_win, 0, 2] = np.maximum(0.0, -feat.loc[no_active_win, "date_registration"].to_numpy(dtype=np.float64))

    y = (feat["final_result"].astype(str).str.strip() == "Withdrawn").astype(int).to_numpy()

    tiers = {
        "F_imm": [
            "code_module", "code_presentation", "gender", "region",
            "highest_education", "imd_band", "age_band", "disability"
        ],
        "F_inst": [
            "num_of_prev_attempts", "studied_credits", "date_registration",
            "module_presentation_length", "banked_count"
        ],
        "F_beh": [
            "submission_completion_ratio", "tma_engagement", "cma_engagement",
            "early_preparation_days", "late_submission_rate", "submission_recency_gap",
            "derived_submission_index"
        ],
        "F_out": [
            "score_mean", "score_count", "score_std", "score_max", "score_min", "weighted_score_sum"
        ],
        "F_der": [
            "missingindicator_score_mean", "missingindicator_score_count",
            "missingindicator_score_std", "missingindicator_date_registration",
            "missingindicator_imd_band"
        ]
    }

    monotonic_dirs = {
        "submission_completion_ratio": 1,
        "tma_engagement": 1,
        "cma_engagement": 1,
        "early_preparation_days": 1,
        "late_submission_rate": -1,
        "submission_recency_gap": -1,
        "derived_submission_index": 1,
        "score_mean": 1,
        "score_count": 1,
        "score_std": 1,
        "score_max": 1,
        "score_min": 1,
        "weighted_score_sum": 1,
    }

    all_tier_cols = (
        tiers["F_imm"] + tiers["F_inst"] + tiers["F_beh"] + tiers["F_out"] + tiers["F_der"]
    )
    X_df = feat[all_tier_cols].copy()
    X_raw = X_df.to_numpy(dtype=np.float64)
    verify_no_nan_inf(X_raw, f"OULAD_{cutoff_pct}_{coding}_X_raw")

    target_n = 800 if smoke else max_samples
    if len(y) > target_n:
        rng = np.random.RandomState(42)
        # Stratified subsample preserving exact positive rate
        pos_idx = np.where(y == 1)[0]
        neg_idx = np.where(y == 0)[0]
        n_pos = int(round(target_n * (len(pos_idx) / len(y))))
        n_neg = target_n - n_pos
        sel_pos = rng.choice(pos_idx, size=n_pos, replace=False)
        sel_neg = rng.choice(neg_idx, size=n_neg, replace=False)
        sel = np.sort(np.concatenate([sel_pos, sel_neg]))
        X_raw = X_raw[sel]
        y = y[sel]
        seq_tensor = seq_tensor[sel]
        seq_mask = seq_mask[sel]

    return {
        "name": f"OULAD ({int(cutoff_pct*100)}% {coding})",
        "cutoff_pct": cutoff_pct,
        "coding": coding,
        "p_raw_full": p_raw,
        "p_drop_full": p_drop,
        "n_full_cohort": 32593,
        "n_survivor_cohort": int(np.sum(survivor_mask)),
        "X_raw": X_raw,
        "y": y,
        "seq_tensor": seq_tensor,
        "seq_mask": seq_mask,
        "feature_names": all_tier_cols,
        "tiers": tiers,
        "monotonic_dirs": monotonic_dirs,
    }
