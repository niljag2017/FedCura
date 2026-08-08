datasets/ — Place all dataset folders here
==========================================

Folder structure expected by the dashboard:

  datasets/
  ├── eicu-collaborative-research-database-demo-2.0/   ← eICU demo dataset (user provides)
  │   ├── patient.csv.gz
  │   ├── hospital.csv.gz
  │   ├── apachePatientResult.csv.gz
  │   ├── diagnosis.csv.gz
  │   ├── medication.csv.gz
  │   └── pastHistory.csv.gz
  │
  ├── heart_disease/                                    ← included
  │   ├── processed.cleveland.data
  │   ├── processed.hungarian.data
  │   ├── processed.switzerland.data
  │   └── processed.va.data
  │
  └── bank_churn/                                       ← included
      └── Churn_Modelling.csv

Dashboard data folder path values:
  eICU:          datasets/eicu-collaborative-research-database-demo-2.0
  Heart Disease: datasets/heart_disease
  Bank Churn:    datasets/bank_churn
