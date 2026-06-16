# Power Outage Prediction using Deep Learning

> Predicting **why** outages happen and **how long** they last — using ANN, CNN, and LSTM models, with and without weather data.

## Author
Mayank Garg
---

## What this project does

This project solves two problems using deep learning:

| Task | Goal | Metric |
|---|---|---|
| **Cause Classification** | Predict the probable cause of a power outage | Accuracy, F1-Score |
| **Duration Regression** | Predict how long an outage will last (in minutes) | R², RMSE, MAE |

Both tasks are tested across three architectures (ANN, CNN, LSTM), and classification is further compared **with vs. without weather data** to measure weather's impact on prediction quality.

---

## Project structure

```
├── ANN.ipynb            ← ANN  | Cause classification | No weather
├── ANNW.ipynb           ← ANN  | Cause classification | With weather
├── CNN.ipynb            ← CNN  | Cause classification | No weather
├── CNNW.ipynb           ← CNN  | Cause classification | With weather
├── RNN_LSTM_.ipynb      ← LSTM | Cause classification | No weather
├── RNNW_LSTM_.ipynb     ← LSTM | Cause classification | With weather
├── ANN.ipynb            ← ANN  | Duration regression  | Combined dataset
├── CNN.ipynb            ← CNN  | Duration regression  | Combined dataset
├── RNN_LSTM.ipynb       ← LSTM | Duration regression  | Combined dataset
│
├── data_without_weather.csv
├── data_with_weather.csv
└── combined_outages_compressed.csv
```

---

## Datasets

| File | Used for | Description |
|---|---|---|
| `data_without_weather.csv` | Classification | Outage records with no weather columns |
| `data_with_weather.csv` | Classification | Same records enriched with weather conditions |
| `combined_outages_compressed.csv` | Regression | Combined outage data with `TIME_DURATION` field |

**Target columns:**
- Classification → `PROBABLE_CAUSE`
- Regression → `TIME_DURATION` (parsed from `"2H 30M"` format into minutes)

**Dropped columns (identifiers, not features):**
`DATEOFDTC`, `DTC_NO_JOB_NO`, `OBSERVATION_DURING_DTC`, `PLACE_OF_DAMAGED`, `FEEDER`, `description`

---

## Models

### ANN — Artificial Neural Network
Standard feedforward network with batch normalization and dropout.
```
Input → Dense(256, ReLU) → BatchNorm → Dropout(0.3)
      → Dense(128, ReLU) → BatchNorm → Dropout(0.3)
      → Output (Softmax for classification / Linear for regression)
```

### CNN — 1D Convolutional Neural Network
Treats the feature vector as a 1D signal.  
**Input reshape:** `(samples, features, 1)`
```
Conv1D(64) → MaxPool → BatchNorm
Conv1D(32) → MaxPool → Flatten
Dense(64) → Dropout(0.3) → Output
```

### RNN — LSTM (Long Short-Term Memory)
Recurrent model treating all features as a single timestep.  
**Input reshape:** `(samples, 1, features)`
```
LSTM(64) → BatchNorm → Dropout(0.3)
Dense(32) → BatchNorm → Dropout(0.3) → Output
```

---

## Preprocessing pipeline

All notebooks use the same sklearn pipeline:

1. **Numeric features** → Median imputation → StandardScaler
2. **Categorical features** → Fill missing with `"missing"` → OneHotEncoder
3. **Rare class filter** → Remove classes with fewer than 5 samples (classification only)
4. **Train/test split** → 80/20, `stratify=y`, `random_state=42`

---

## Training configuration

| Parameter | Value |
|---|---|
| Optimizer | Adam |
| Batch size | 64 |
| Max epochs | 30 |
| Validation split | 20% of training data |
| Early stopping | patience=5, monitors `val_loss` |
| Best weights restored | Yes |

---

## Getting started

### 1. Install dependencies

```bash
pip install tensorflow scikit-learn pandas numpy jupyter
```

### 2. Clone the repo and add datasets

```bash
git clone https://github.com/your-username/power-outage-prediction.git
cd power-outage-prediction
```

Place the three CSV files in the root project directory.

### 3. Run a notebook

```bash
jupyter notebook ANN.ipynb
```

Run all cells — results are printed at the end with accuracy/F1 (classification) or R²/RMSE/MAE (regression).

---

## Quick notebook guide

**Want to classify outage causes?**
- Start with `ANN.ipynb` (no weather) and `ANNW.ipynb` (with weather) to compare.

**Want to predict outage duration?**
- Use the regression versions: `ANN.ipynb`, `CNN.ipynb`, or `RNN_LSTM.ipynb` on `combined_outages_compressed.csv`.

**Want to compare architectures?**
- Run all three models on the same dataset and compare their printed metrics.

---

## Tech stack

- **TensorFlow / Keras** — model building and training
- **scikit-learn** — preprocessing pipelines, metrics, train/test split
- **Pandas / NumPy** — data loading and manipulation
- **Jupyter Notebook** — interactive development environment
