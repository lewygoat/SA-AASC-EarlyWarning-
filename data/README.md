# Data

This repository does not redistribute the benchmark datasets. Download them from
their original providers and place the files exactly as shown below.

```
data/
├── uci697.csv                    # UCI #697, semicolon-separated, 4,424 rows
├── xAPI-Edu-Data.csv             # xAPI-Edu (Kalboard 360), 480 rows
└── oulad/
    ├── studentInfo.csv
    ├── studentRegistration.csv
    ├── courses.csv
    ├── assessments.csv
    ├── studentAssessment.csv
    └── vle.csv
```

## 1. OULAD (32,593 registrations, 22 module presentations)

Kuzilek J, Hlosta M, Zdrahal Z. Open University Learning Analytics dataset.
*Scientific Data*, 2017, 4: 170171. CC-BY 4.0.

- Primary source: https://research.stem.open.ac.uk/ouanalyse/dataset/
- Mirror: https://archive.ics.uci.edu/dataset/349/open+university+learning+analytics+dataset

Extract `studentInfo.csv`, `studentRegistration.csv`, `courses.csv`,
`assessments.csv`, `studentAssessment.csv`, and `vle.csv` into `data/oulad/`.

## 2. UCI #697 (4,424 students, 36 enrollment-time predictors)

Realinho H, Machado J, Baptista L, Martins M V. Predicting student dropout and
academic success. *Data*, 2022, 7(11): 146.

- https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success

Save the training split as `data/uci697.csv`. The loader auto-detects the
delimiter, so both comma- and semicolon-separated exports work.

## 3. xAPI-Edu (480 students, 16 features)

Amrieh E A, Hamtini T, Aljarah I. Preprocessing and analyzing educational data
set using X-API for improving student's performance. 2015.

- https://www.kaggle.com/datasets/aljarah/xAPI-Edu-Data

Save as `data/xAPI-Edu-Data.csv`.

## Verification

After placing the files, confirm the row counts before running experiments.

```bash
wc -l data/uci697.csv data/xAPI-Edu-Data.csv data/oulad/studentInfo.csv
# expect 4425, 481, 32594 (header row included)
```
