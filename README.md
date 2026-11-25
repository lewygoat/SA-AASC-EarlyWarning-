# Stage-Aware Attribution Subspace Clustering for Group Counterfactual Recourse in Student Early-Warning Systems

Reference implementation for the manuscript *Stage-Aware Attribution Subspace Clustering for Group Counterfactual Recourse in Student Early-Warning Systems*.

The repository couples a mask-aware temporal model with a capacity-constrained
group counterfactual pipeline. `MA-TempNet+GBDT` combines mask-gated temporal
attention with histogram gradient-boosted trees for longitudinal risk estimation,
and `SA-AASC` projects flagged learners into a stage-unlocked actionable
attribution subspace so that one shared recourse template serves a whole cluster
under an advising capacity limit.

## Repository layout

```
├── src/
│   ├── defensive_data_loader.py      # dataset loading, temporal truncation guard, mutability tiers
│   ├── models_dl_ml.py               # MA-BiLSTM, Reg-TabNet, MA-TempNet+GBDT, classical baselines
│   ├── sa_aasc_recourse.py           # subspace projection, clustering, group template optimizer
│   ├── run_experiments.py            # Experiments 1-5, writes results/table_*.csv
│   └── generate_journal_figures.py   # Figures 1-7 from results/final_experiment_results.json
├── configs/feature_taxonomy.json     # per-benchmark predictor-to-mutability-tier assignment
├── results/                          # released result tables (CSV) and aggregate JSON
├── figures/                          # released figures, 300 dpi
├── data/README.md                    # dataset download and placement instructions
└── docs/                             # full experiment summary (Chinese)
```

## Requirements

Python 3.9 or newer. Install dependencies with

```bash
pip install -r requirements.txt
```

The pipeline was developed against numpy 2.0.2, pandas 2.3.3, scikit-learn 1.6.1,
torch 2.8.0, and matplotlib 3.9.4. CPU-only execution is supported.

## Data

The three benchmark datasets are not redistributed here. Follow `data/README.md` to
obtain OULAD, UCI #697, and xAPI-Edu and place them under `data/`.

## Running

A quick end-to-end check on subsampled data completes in a few minutes.

```bash
python src/run_experiments.py --mode smoke
```

The full pipeline reproduces the tables in `results/`. It performs five-fold
cross-validation with two random seeds across eight architectures and four
benchmark cohorts, then runs the four constraint regimes and the capacity sweep.

```bash
python src/run_experiments.py --mode full
```

Regenerate the manuscript figures after a full run.

```bash
python src/generate_journal_figures.py
```

Outputs land in `figures/`. Result tables are written to `results/`.

## What the pipeline enforces

The loader refuses to proceed when a check fails, so silent leakage is not possible.

- Temporal truncation. Every feature and label is restricted to `t <= t_s` at the
  20%, 40%, 60%, and 80% checkpoints, and post-cutoff assessment or unregistration
  timestamps raise `DefensiveDataError`.
- Sequence validity masks. Explicit binary masks `m_i^(t)` mark active steps, and
  attention masking removes padded positions from the softmax.
- Mutability tiers. Each predictor is assigned to one of `F_imm`, `F_inst`, `F_beh`,
  `F_out`, or `F_der`, and recourse templates enforce `delta_j = 0` outside the
  stage-unlocked actionable set.
- Constraint regimes. `A1_none`, `A2_tier`, `A3_monotone`, and
  `A4_budget@low/mid/high` are evaluated against four clustering spaces
  (`B1_Raw_X`, `B2_Actionable_X`, `B3_Full_SHAP`, `SA-AASC`) under a shared
  water-filling optimizer.

## Released results

`results/final_experiment_results.json` holds the aggregate tables. All downstream attributions and counterfactual evaluations run strictly out of fold (`OOFTabularEnsemble`, 5-fold stratified CV) under an equalized 96-schedule water-filling optimizer ($K=20$ fine-grained clusters in `table_04`, $K_c \in \{5,10,15,20\}$ advising capacities in `table_05`, 5-seed stability and anchor-matched controls in `table_04b`, and $\tau_s$ sensitivity in `table_04c`), with coordinate-restricted monotonic clipping guaranteeing $\mathcal{C}(\mathbf{x}_i^{\text{grp}}-\mathbf{x}_i) \le B$ and $\|\mathbf{x}_i^{\text{grp}}-\mathbf{x}_i\|_0 \le 3$ for every learner. At $K_c = 20$, `table_05` matches `table_04` ($K=20$, `A4_budget@mid`) identically; `table_02` reports exact registry survival fractions `active_survivors_n / 32593`.

## Citation

```bibtex
@article{wu2026saaasc,
  title   = {Stage-Aware Attribution Subspace Clustering for Group Counterfactual Recourse in Student Early-Warning Systems},
  author  = {Wu, Jing},
  year    = {2026},
  note    = {Manuscript}
}
```

## License

MIT. See `LICENSE`.

## Data licenses

OULAD is released under CC-BY 4.0 by The Open University. UCI #697 and xAPI-Edu
carry the licenses of their respective repositories. Consult `data/README.md` before
redistributing any derived file.
