# FedICU — Federated Learning Dashboard v3

> Adaptive Federated Learning Architecture for Clinical Data  
> *Nilesh J*

---

## What This Is

FedICU is a browser-based federated learning (FL) research dashboard that demonstrates privacy-preserving predictive modelling across multiple institutions without sharing raw patient records. It simulates federated learning across geographic regions and hospital sites, compares multiple FL strategies against a centralized oracle, and visualises the results in an interactive web UI.

The system supports **three datasets across two domains**, showing that the same federated architecture generalises beyond a single dataset:

| Dataset | Domain | Clients | Partition | Download |
|---|---|---|---|---|---|
| eICU CRD Demo | Healthcare — ICU readmission | 5 | US geographic region | [PhysioNet](https://physionet.org/content/eicu-crd-demo/2.0.1/) |
| UCI Heart Disease | Healthcare — cardiac diagnosis | 4 | Hospital site (Cleveland / Hungarian / Switzerland / VA) | [UCI ML Repository](https://archive.ics.uci.edu/dataset/45/heart+disease) |
| Bank Customer Churn | Finance — churn prediction | 3 | Geography (France / Germany / Spain) | [Kaggle](https://www.kaggle.com/datasets/saurabhbadole/bank-customer-churn-prediction-dataset) |

**Key result:** FL global AUROC is within **1.5% of centralized** across all 3 datasets and both algorithms (LR + MLP), without sharing any raw records between clients.

---

## Quick Start

```bash
# 1. Extract the project
cd fedicu_updated

# 2. Install dependencies
pip install flask pandas numpy scikit-learn

# 3. Run
python run.py
```

The browser opens automatically at **http://localhost:5050**

---

## Prerequisites

- Python **3.9 – 3.12** (64-bit)
- pip (bundled with Python)
- 4 GB RAM minimum · 8 GB recommended for MLP on Bank Churn
- Any modern browser (Chrome 110+, Firefox 110+, Edge 110+, Safari 16+)

---

## Dataset Setup

### UCI Heart Disease + Bank Churn — bundled, nothing to do

Both are pre-packaged inside the `datasets/` subfolder.
Original sources for reference:

- **UCI Heart Disease** — [https://archive.ics.uci.edu/dataset/45/heart+disease](https://archive.ics.uci.edu/dataset/45/heart+disease)  
  Creators: Robert Detrano, M.D., Ph.D. · Cleveland Clinic Foundation · UCI ML Repository
- **Bank Customer Churn** — [https://www.kaggle.com/datasets/saurabhbadole/bank-customer-churn-prediction-dataset](https://www.kaggle.com/datasets/saurabhbadole/bank-customer-churn-prediction-dataset)  
  Published on Kaggle · Originally based on a European banking simulation dataset

```
datasets/
├── heart_disease/
│   ├── processed.cleveland.data
│   ├── processed.hungarian.data
│   ├── processed.switzerland.data
│   └── processed.va.data
└── bank_churn/
    └── Churn_Modelling.csv
```

### eICU Collaborative Research Database Demo — user download required

The eICU dataset requires a free PhysioNet account and a data-use agreement.

1. Register at **https://physionet.org/register/**
2. Complete the required CITI training (~30 min)
3. Download from **https://physionet.org/content/eicu-crd-demo/2.0.1/**
4. Extract so the folder name is exactly:

```
datasets/eicu-collaborative-research-database-demo-2.0/
```

The folder must contain: `patient.csv.gz`, `hospital.csv.gz`, `apachePatientResult.csv.gz`, `diagnosis.csv.gz`, `medication.csv.gz`, `pastHistory.csv.gz`

> **Note:** The dashboard fully works for Heart Disease and Bank Churn while the eICU folder is absent. A warning (not an error) is printed at startup.

---

## File Structure

```
fedicu_updated/
├── run.py                      ← startup launcher — run this
├── app.py                      ← Flask web server + SSE job queue
├── templates/
│   └── index.html              ← single-page dashboard UI (Chart.js)
├── federated_icu/
│   ├── config.py               ← experiment configuration + Config dataclass
│   ├── data.py                 ← eICU feature engineering + data split
│   ├── data_loaders.py         ← Heart Disease + Bank Churn loaders
│   ├── engine.py               ← run_all() — orchestrates all experiments
│   ├── client.py               ← FLClient — local training per site
│   ├── server.py               ← FLServer — FedAvg aggregation
│   ├── strategies.py           ← S1–S5 strategy implementations
│   ├── evaluate.py             ← AUROC, bootstrap CI, checkpoint
│   ├── models.py               ← LR + MLP model factory + weight injection
│   └── logger.py               ← RunLogger + SSE push
├── datasets/
│   ├── README.txt
│   ├── heart_disease/          ← bundled
│   └── bank_churn/             ← bundled
├── logs/                       ← auto-created on first run
└── checkpoints/                ← auto-created on first run
```

---

## Command-Line Options

```bash
python run.py                    # default: port 5050, opens browser
python run.py --port 8080        # use a different port
python run.py --no-browser       # start server without opening browser
python run.py --skip-data-check  # skip dataset folder verification
```

---

## FL Strategies

| # | Name | Description |
|---|---|---|
| S1 | Centralized Baseline | All data pooled — oracle upper bound. No privacy. |
| S2 | FedAvg Region | Standard FedAvg across geographic/site clients |
| S3 | Quality-Gated Hospital | Only hospitals above AUROC threshold participate *(eICU only)* |
| S4 | FedProx Region | FedAvg + proximal term μ‖w−w_global‖² to limit client drift |
| S5 | Non-IID Simulation | Synthetic Dirichlet(α) label-skewed clients *(eICU only)* |

> Heart Disease and Bank Churn run **S1 + S2 only** — S3/S4/S5 are not applicable to 4 and 3 clients respectively.

---

## Results Summary

### eICU — ICU 30-Day Readmission

| Strategy | LR AUROC | MLP AUROC |
|---|---|---|
| S1 Centralized (oracle) | **0.7540** | **0.7627** |
| S2 FedAvg Region | 0.7401 | 0.7613 |
| S3 Quality-Gated | 0.6867 | 0.6822 |
| S4 FedProx | 0.7405 | 0.7368 |
| S5 Non-IID | 0.7112 | 0.7667 |

FL gap vs centralized: **1.39% (LR) · 0.14% (MLP)**

### UCI Heart Disease — Cardiac Diagnosis

| Strategy | LR AUROC | MLP AUROC |
|---|---|---|
| S1 Centralized | **0.8902** | **0.8771** |
| S2 FedAvg | 0.8894 | 0.8746 |

FL gap: **0.08% (LR) · 0.25% (MLP)**

### Bank Customer Churn

| Strategy | LR AUROC | MLP AUROC |
|---|---|---|
| S1 Centralized | **0.7532** | **0.8206** |
| S2 FedAvg | 0.7483 | 0.8160 |

FL gap: **0.49% (LR) · 0.46% (MLP)**

> All AUROC values are bootstrap means over 200 resamples of a stratified 20% held-out test set. FL gap < 1.5% in every case.

---

## Regional Comparison (eICU)

The 6-experiment regional analysis tests whether global FL knowledge transfers to individual regions:

| Region | Standalone | FL Global | Uplift | Best |
|---|---|---|---|---|
| Midwest | 0.7143 | 0.7948 | +0.0805 | FL Global |
| Northeast | 0.8864 | 0.9545 | +0.0681 | Centralized W |
| South | 0.6238 | 0.7076 | +0.0838 | FL Global |
| Unknown ⚠ | 0.1071 | 0.9643 | +0.8572 | FL Global |
| West | 0.8176 | 0.7790 | −0.0386 | Standalone |

> ⚠ Unknown region (n=29, 3.5% positive): standalone AUROC is highly unstable due to the very small test set. The observed uplift should be interpreted cautiously.

---

## Architecture

```
Browser (Chart.js UI)
        │
        │  POST /run  { config JSON }
        ▼
app.py (Flask) ─────────────────────────────────────────────
        │                                                   │
        │  background worker thread                         │  SSE /poll?job=<id>
        ▼                                                   │  streams log + progress
   engine.run_all()                                         │
        │                                                   │
        ├── load dataset (data.py / data_loaders.py)        │
        ├── stratified 80/20 global test split              │
        ├── S1  centralized CV  (sklearn)           ────────┤
        ├── S2  FedAvg          (client + server)   ────────┤ push progress
        ├── S3/S4/S5  …                             ────────┤
        ├── regional comparison (strategies.py)     ────────┤
        └── return results JSON ─────────────────────────────┘
                                                            │
                                                   Browser renders charts
```

**FedAvg protocol:**
1. Seed global model (2-row warm-start to establish parameter shape)
2. Each client fits on local training data starting from global weights
3. Server aggregates: `w_global = Σ (n_i / N) × w_i` (sample-weighted average)
4. Repeat for the configured number of FL rounds
5. Evaluate global model on the unified held-out test set

---

## Configuration

All defaults live in `federated_icu/config.py`. Key fields:

```python
dataset_type    = "eicu"                # "eicu" | "heart" | "bank"
algorithm       = "logistic_regression" # or "mlp"
fl_rounds       = 3
cv_folds        = 5
fedprox_mu      = 0.01
dirichlet_alpha = 0.5
random_state    = 42
global_test_size = 0.20
augment_data    = False
```

---

## Reproducibility

| Parameter | Value |
|---|---|
| Random seed | 42 |
| Train/test split | 80/20 stratified |
| FL rounds | 3 |
| Aggregation | FedAvg (sample-weighted) |
| LR regularisation | C=0.01 L2 |
| Feature scaling | StandardScaler fitted on training data only |
| AUROC | Bootstrap mean (200 resamples) |

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: flask` | `pip install flask pandas numpy scikit-learn` |
| Port 5050 in use | `python run.py --port 8080` |
| eICU folder not found | Place unzipped eICU in `datasets/eicu-collaborative-research-database-demo-2.0/` |
| Feature column error after switching datasets | Fixed in v3. Make sure you are using the v3 zip. |
| Run stuck at "Importing FL engine" | Normal on first run — wait up to 30 s for JIT compilation |
| AUROC = 0.5000 | Too few positives in training split — verify dataset folder path |
| Browser shows blank page | Server may still be starting. Wait 3 s and reload. |

---

## Dependencies

```
flask>=2.0
pandas>=1.5
numpy>=1.23
scikit-learn>=1.1
```

```bash
pip install flask pandas numpy scikit-learn
```

---

## Privacy Note

FedICU simulates federated learning on a **single machine** for research purposes. No data leaves the local process. In a real multi-institution deployment, only model weight tensors (24 scalar values per round per client) would be transmitted — never raw patient records. However, model updates can still carry indirect privacy risks; real deployments would require differential privacy, secure aggregation, and appropriate institutional governance.

---

## Citation

```bibtex
@misc{fedicu2026,
  title  = {FedICU: Adaptive Federated Learning Architecture for Clinical Data},
  author = {Nilesh J},
  year   = {2026},
  note   = {M.Tech Dissertation}
}
```

---

## Dataset Credits

### eICU Collaborative Research Database Demo v2.0.1
- **Source:** PhysioNet — https://physionet.org/content/eicu-crd-demo/2.0.1/
- **DOI:** 10.13026/C2WM1R
- **Citation:** Pollard, T.J., Johnson, A.E.W., Raffa, J.D., Mark, R.G. (2019). The eICU Collaborative Research Database, a freely available multi-center database for critical care research. *Scientific Data*, 6, 180178. https://doi.org/10.1038/sdata.2018.178
- **Access:** Requires PhysioNet account + CITI training completion
- **Licence:** PhysioNet Credentialed Health Data Licence — not for redistribution

### UCI Heart Disease Dataset
- **Source:** UCI Machine Learning Repository — https://archive.ics.uci.edu/dataset/45/heart+disease
- **DOI:** 10.24432/C52P4X
- **Citation:** Detrano, R., Janosi, A., Steinbrunn, W., Pfisterer, M., Schmid, J., Sandhu, S., Guppy, K., Lee, S., & Froelicher, V. (1989). International application of a new probability algorithm for the diagnosis of coronary artery disease. *American Journal of Cardiology*, 64(5), 304–310. https://doi.org/10.1016/0002-9149(89)90524-9
- **Sites:** Cleveland Clinic Foundation · Hungarian Institute of Cardiology · University Hospital Zurich · VA Medical Center Long Beach
- **Licence:** Creative Commons Attribution 4.0 International (CC BY 4.0)

### Bank Customer Churn Prediction Dataset
- **Source:** Kaggle — https://www.kaggle.com/datasets/saurabhbadole/bank-customer-churn-prediction-dataset
- **Published by:** Saurabh Badole (Kaggle)
- **Licence:** See Kaggle dataset page for licence terms
- **Note:** Based on a publicly available banking simulation dataset widely used for churn modelling benchmarks

---

## References

- McMahan, B., Moore, E., Ramage, D., Hampson, S., & Agüera y Arcas, B. (2017). Communication-Efficient Learning of Deep Networks from Decentralized Data. *AISTATS*. https://arxiv.org/abs/1602.05629
- Li, T., Sahu, A.K., Zaheer, M., Sanjabi, M., Smola, A., & Smith, V. (2020). Federated Optimization in Heterogeneous Networks. *MLSys*. https://arxiv.org/abs/1812.06127
- Pollard, T.J. et al. (2019). The eICU Collaborative Research Database. *Scientific Data*, 6, 180178.
- Goldberger, A. et al. (2000). PhysioBank, PhysioToolkit, and PhysioNet. *Circulation*, 101(23), e215–e220.
- Detrano, R. et al. (1989). International application of a new probability algorithm for coronary artery disease. *American Journal of Cardiology*, 64(5), 304–310.

---

## License

This project is submitted as part of an M.Tech dissertation. Code is provided for academic evaluation. The eICU dataset is subject to PhysioNet's data-use agreement and must not be redistributed.

---

*FedICU v3 · Nilesh J*
