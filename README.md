# \# FedICU — Federated Learning Dashboard v3

# 

# > \*\*M.Tech Dissertation · BITS Pilani · 2026\*\*  

# > Adaptive Federated Learning Architecture for Clinical Data  

# > \*Nilesh Maruti Jagdale · 2024DA04322\*

# 

# \---

# 

# \## What This Is

# 

# FedICU is a browser-based federated learning (FL) research dashboard that demonstrates privacy-preserving predictive modelling across multiple institutions without sharing raw patient records. It simulates federated learning across geographic regions and hospital sites, compares multiple FL strategies against a centralized oracle, and visualises the results in an interactive web UI.

# 

# The system supports \*\*three datasets across two domains\*\*, proving that the same federated architecture generalises beyond a single dataset:

# 

# | Dataset | Domain | Records | Clients | Partition |

# |---|---|---|---|---|

# | eICU CRD Demo | Healthcare — ICU readmission | 2,119 | 5 | US geographic region |

# | UCI Heart Disease | Healthcare — cardiac diagnosis | 920 | 4 | Hospital site (Cleveland / Hungarian / Switzerland / VA) |

# | Bank Customer Churn | Finance — churn prediction | 10,000 | 3 | Geography (France / Germany / Spain) |

# 

# \*\*Key result:\*\* FL global AUROC is within \*\*1.5% of centralized\*\* across all 3 datasets and both algorithms (LR + MLP), without sharing any raw records between clients.

# 

# \---

# 

# \## Quick Start

# 

# ```bash

# \# 1. Clone or extract the project

# cd fedicu\_updated

# 

# \# 2. Install dependencies

# pip install flask pandas numpy scikit-learn

# 

# \# 3. Run

# python run.py

# ```

# 

# The browser opens automatically at \*\*http://localhost:5050\*\*

# 

# \---

# 

# \## Prerequisites

# 

# \- Python \*\*3.9 – 3.12\*\* (64-bit)

# \- pip (bundled with Python)

# \- 4 GB RAM minimum · 8 GB recommended for MLP on Bank Churn

# \- Any modern browser (Chrome 110+, Firefox 110+, Edge 110+, Safari 16+)

# 

# \---

# 

# \## Dataset Setup

# 

# \### UCI Heart Disease + Bank Churn (bundled — nothing to do)

# 

# Both are pre-packaged inside the `datasets/` subfolder:

# 

# ```

# datasets/

# ├── heart\_disease/

# │   ├── processed.cleveland.data

# │   ├── processed.hungarian.data

# │   ├── processed.switzerland.data

# │   └── processed.va.data

# └── bank\_churn/

# &#x20;   └── Churn\_Modelling.csv

# ```

# 

# \### eICU Collaborative Research Database Demo (user download)

# 

# The eICU dataset requires a free PhysioNet account and a data-use agreement.

# 

# 1\. Register at \*\*https://physionet.org/register/\*\*

# 2\. Complete the required CITI training (\~30 min)

# 3\. Download from \*\*https://physionet.org/content/eicu-crd-demo/2.0/\*\*

# 4\. Extract so the folder name is exactly:

# 

# ```

# datasets/eicu-collaborative-research-database-demo-2.0/

# ```

# 

# The folder must contain: `patient.csv.gz`, `hospital.csv.gz`, `apachePatientResult.csv.gz`, `diagnosis.csv.gz`, `medication.csv.gz`, `pastHistory.csv.gz`

# 

# > \*\*Note:\*\* The dashboard fully works for Heart Disease and Bank Churn while the eICU folder is absent. A warning (not an error) is printed at startup.

# 

# \---

# 

# \## File Structure

# 

# ```

# fedicu\_updated/

# ├── run.py                        ← startup launcher — run this

# ├── app.py                        ← Flask web server + SSE job queue

# ├── templates/

# │   └── index.html                ← single-page dashboard UI (Chart.js)

# ├── federated\_icu/

# │   ├── config.py                 ← experiment configuration + Config dataclass

# │   ├── data.py                   ← eICU feature engineering + data split

# │   ├── data\_loaders.py           ← Heart Disease + Bank Churn loaders

# │   ├── engine.py                 ← run\_all() — orchestrates all experiments

# │   ├── client.py                 ← FLClient — local training per site

# │   ├── server.py                 ← FLServer — FedAvg aggregation

# │   ├── strategies.py             ← S1–S5 strategy implementations

# │   ├── evaluate.py               ← AUROC, bootstrap CI, checkpoint

# │   ├── models.py                 ← LR + MLP model factory + weight inject

# │   └── logger.py                 ← RunLogger + SSE push

# ├── datasets/

# │   ├── README.txt

# │   ├── heart\_disease/            ← bundled

# │   └── bank\_churn/               ← bundled

# ├── logs/                         ← auto-created on first run

# └── checkpoints/                  ← auto-created on first run

# ```

# 

# \---

# 

# \## Command-Line Options

# 

# ```bash

# python run.py                     # default: port 5050, opens browser

# python run.py --port 8080         # use a different port

# python run.py --no-browser        # start server without opening browser

# python run.py --skip-data-check   # skip dataset folder verification

# ```

# 

# \---

# 

# \## FL Strategies

# 

# | # | Name | Description |

# |---|---|---|

# | S1 | Centralized Baseline | All data pooled — oracle upper bound. No privacy. |

# | S2 | FedAvg Region | Standard FedAvg across geographic/site clients |

# | S3 | Quality-Gated Hospital | Only hospitals above AUROC threshold participate (eICU only) |

# | S4 | FedProx Region | FedAvg + proximal term μ‖w−w\_global‖² to limit drift |

# | S5 | Non-IID Simulation | Synthetic Dirichlet(α) label-skewed clients (eICU only) |

# 

# > Heart Disease and Bank Churn run \*\*S1 + S2 only\*\* (4 and 3 clients respectively — S3/S4/S5 not applicable).

# 

# \---

# 

# \## Results Summary

# 

# \### eICU — ICU 30-Day Readmission

# 

# | Strategy | LR AUROC | MLP AUROC |

# |---|---|---|

# | S1 Centralized (oracle) | \*\*0.7540\*\* | \*\*0.7627\*\* |

# | S2 FedAvg Region | 0.7401 | 0.7613 |

# | S3 Quality-Gated | 0.6867 | 0.6822 |

# | S4 FedProx | 0.7405 | 0.7368 |

# | S5 Non-IID | 0.7112 | 0.7667 |

# 

# FL gap vs centralized: \*\*1.39% (LR) · 0.14% (MLP)\*\*

# 

# \### UCI Heart Disease — Cardiac Diagnosis

# 

# | Strategy | LR AUROC | MLP AUROC |

# |---|---|---|

# | S1 Centralized | \*\*0.8902\*\* | \*\*0.8771\*\* |

# | S2 FedAvg | 0.8894 | 0.8746 |

# 

# FL gap: \*\*0.08% (LR) · 0.25% (MLP)\*\*

# 

# \### Bank Customer Churn

# 

# | Strategy | LR AUROC | MLP AUROC |

# |---|---|---|

# | S1 Centralized | \*\*0.7532\*\* | \*\*0.8206\*\* |

# | S2 FedAvg | 0.7483 | 0.8160 |

# 

# FL gap: \*\*0.49% (LR) · 0.46% (MLP)\*\*

# 

# > All AUROC values are bootstrap means (200 resamples) on a stratified 20% held-out test set. FL gap < 1.5% across all 3 datasets.

# 

# \---

# 

# \## Regional Comparison (eICU)

# 

# The 6-experiment regional analysis evaluates whether global knowledge transfers to individual regions:

# 

# | Region | Standalone | FL Global | Uplift | Best |

# |---|---|---|---|---|

# | Midwest | 0.7143 | 0.7948 | +0.0805 | FL Global |

# | Northeast | 0.8864 | 0.9545 | +0.0681 | Centralized W |

# | South | 0.6238 | 0.7076 | +0.0838 | FL Global |

# | Unknown ⚠ | 0.1071 | 0.9643 | +0.8572 | FL Global |

# | West | 0.8176 | 0.7790 | −0.0386 | Standalone |

# 

# > ⚠ Unknown region (n=29, 3.5% positive): the standalone AUROC is highly unstable. The observed uplift should be interpreted cautiously, not as a strong result.

# 

# \---

# 

# \## Architecture

# 

# ```

# Browser (Chart.js UI)

# &#x20;       │

# &#x20;       │  POST /run {config JSON}

# &#x20;       ▼

# Flask app.py  ──────────────────────────────────────────

# &#x20;       │                                               │

# &#x20;       │  background thread                            │  SSE /poll?job=<id>

# &#x20;       ▼                                               │  streams log + progress

# &#x20;  engine.run\_all()                                     │

# &#x20;       │                                               │

# &#x20;       ├─ load dataset (data.py / data\_loaders.py)     │

# &#x20;       ├─ build global test split (stratified 80/20)   │

# &#x20;       ├─ Strategy 1: centralized CV (sklearn)         │──▶ push progress

# &#x20;       ├─ Strategy 2: FedAvg (client.py + server.py)   │

# &#x20;       ├─ Strategy 3/4/5: …                            │

# &#x20;       ├─ regional comparison (strategies.py)          │

# &#x20;       └─ return results JSON ──────────────────────────┘

# &#x20;                                                       │

# &#x20;                                             Browser renders charts

# ```

# 

# \*\*FedAvg protocol:\*\*

# 1\. Seed global model (2-row warm-start to establish parameter shape)

# 2\. Each client fits on local training data starting from global weights

# 3\. Server aggregates: `w\_global = Σ (n\_i / N) × w\_i` (sample-weighted)

# 4\. Repeat for FL rounds

# 5\. Evaluate global model on unified held-out test set

# 

# \---

# 

# \## Configuration

# 

# All defaults live in `federated\_icu/config.py`. Key fields:

# 

# ```python

# dataset\_type:   "eicu"   # "eicu" | "heart" | "bank"

# algorithm:      "logistic\_regression"   # or "mlp"

# fl\_rounds:      3

# cv\_folds:       5

# fedprox\_mu:     0.01

# dirichlet\_alpha: 0.5

# random\_state:   42

# global\_test\_size: 0.20

# augment\_data:   False

# ```

# 

# \---

# 

# \## Reproducibility

# 

# | Parameter | Value |

# |---|---|

# | Random seed | 42 |

# | Train/test split | 80/20 stratified |

# | FL rounds | 3 |

# | Aggregation | FedAvg (sample-weighted) |

# | LR regularisation | C=0.01 L2 |

# | Feature scaling | StandardScaler fitted on training data only |

# | AUROC | Bootstrap mean (200 resamples) |

# 

# \---

# 

# \## Troubleshooting

# 

# | Problem | Fix |

# |---|---|

# | `ModuleNotFoundError: flask` | `pip install flask pandas numpy scikit-learn` |

# | Port 5050 in use | `python run.py --port 8080` |

# | eICU folder not found | Place unzipped eICU in `datasets/eicu-collaborative-research-database-demo-2.0/` |

# | Feature column error after switching datasets | Fixed in v3. Use the v3 zip. |

# | Run stuck at "Importing FL engine" | Normal on first run — wait up to 30 s for JIT compilation |

# | AUROC = 0.5000 | Too few positives in training split — verify dataset folder |

# 

# \---

# 

# \## Dependencies

# 

# ```

# flask>=2.0

# pandas>=1.5

# numpy>=1.23

# scikit-learn>=1.1

# ```

# 

# Install: `pip install flask pandas numpy scikit-learn`

# 

# \---

# 

# \## Privacy Note

# 

# FedICU simulates federated learning on a \*\*single machine\*\* for research purposes. No data leaves the local process. In a real multi-institution deployment, only model weight tensors (24 scalar values per round per client) would be transmitted — never raw patient records. However, model updates can carry indirect privacy risks; real deployments would require differential privacy, secure aggregation, and appropriate institutional governance.

# 

# \---

# 

# \## Citation

# 

# If you use this code or results in your work, please cite:

# 

# ```

# Jagdale, N.M. (2026). FedICU: Adaptive Federated Learning Architecture

# for Clinical Data. M.Tech Dissertation, BITS Pilani.

# ```

# 

# \---

# 

# \## References

# 

# \- McMahan, B. et al. (2017). Communication-Efficient Learning of Deep Networks from Decentralized Data. \*AISTATS\*. https://arxiv.org/abs/1602.05629

# \- Johnson, A. et al. (2018). MIMIC-III, a freely accessible critical care database. \*Scientific Data\*.

# \- Goldberger, A. et al. (2000). PhysioBank, PhysioToolkit, and PhysioNet. \*Circulation\*.

# \- Li, T. et al. (2020). Federated Optimization in Heterogeneous Networks (FedProx). \*MLSys\*.

# \- UCI ML Repository — Heart Disease Dataset. Detrano, R. et al. (1989). \*American Journal of Cardiology\*.

# 

# \---

# 

# \## License

# 

# This project is submitted as part of an M.Tech dissertation at BITS Pilani. Code is provided for academic evaluation purposes. The eICU dataset is subject to PhysioNet's data-use agreement and must not be redistributed.

# 

# \---

# 

# \*FedICU v3 · M.Tech Dissertation · BITS Pilani · 2026 · Nilesh Maruti Jagdale\*



