# 🧠 An Explainable Indian Credit Scoring Model
A full end-to-end machine learning dashboard for **loan approval prediction**, built with **Streamlit** and trained using explainable models including **Logistic Regression** and **Explainable Boosting Machine (EBM)**.

✅ Upload internal + external datasets  
✅ Automatic preprocessing & feature engineering  
✅ Train explainable models  
✅ SHAP + LIME interpretability  
✅ Score new unseen applicants  
✅ Export scored results + PDF reports


---

## 📌 Screenshots (placeholders)

| Page | Preview |
|------|---------|
| Data Upload | *(insert image)* |
| Preprocessing | *(insert image)* |
| Model Training | *(insert image)* |
| Explainability (SHAP / LIME / PDP) | *(insert image)* |
| Score New Applicants | *(insert image)* |

---

## ✅ Features

| Feature | Description |
|---------|-------------|
| 📂 Data Upload | Upload **internal** + **external** datasets (CSV) |
| 🔄 Automatic Merge | Smart merge using detected key column or index fallback |
| 🧼 Preprocessing | Imputation, scaling, encoding & target engineering |
| 🤖 Model Training | Logistic Regression + Explainable Boosting Machine |
| 🧩 Global Explainability | SHAP summary plots |
| 🔍 Local Explainability | SHAP waterfall + LIME explanation |
| 📈 PDP | Partial Dependence Plots for top features |
| ✅ Score New Data | Upload unseen applicants and predict approval |
| ⬇ Export | Save scored output CSV + downloadable PDFs |

---

## 🛠️ Tech Stack

- **Python 3.8+**
- **Streamlit**
- **Pandas / NumPy / Scikit-Learn**
- **InterpretML (EBM)**
- **SHAP & LIME**
- **ReportLab** (PDF generation)

---

## 📂 Folder Structure

```

AUCML_CreditScoring/
│
├── app/
│   ├── Home.py                   # Landing page
│   ├── 01_Data_Upload.py         # Upload datasets
│   ├── 02_Preprocessing.py       # Merge + preprocessing + feature engg.
│   ├── 03_Model_Training.py      # Train LR + EBM
│   ├── 04_Explainability.py      # SHAP, LIME, PDP
│   ├── 05_Score_New_Data.py      # Predict on new unseen CSV
│   │
│   └── components/               # Utility modules
│       ├── utils.py
│       ├── shap_plotter.py
│       ├── lime_plotter.py
│       ├── pdp_plotter.py
│       └── model_loader.py
│
├── models/                       # .pkl saved models
│   ├── logistic_regression.pkl
│   └── ebm_model.pkl
│
├── reports/                      # Generated CSVs & PDFs
│
├── requirements.txt
├── AUCML_Project_Report.pdf
└── README.md

````

---

## ⬇ Installation

### 1️⃣ Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/AUCML_CreditScoring.git
cd AUCML_CreditScoring
````

### 2️⃣ Create virtual environment

```bash
python -m venv venv
source venv/bin/activate      # Mac/Linux
venv\Scripts\activate         # Windows
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶ Run Locally

```bash
streamlit run app/Home.py
```

✅ Streamlit will open in your browser
✅ Navigate pages using the left sidebar

---

## ☁ Deploy to Streamlit Cloud

1. Push this repository to GitHub
2. Visit: [https://share.streamlit.io/](https://share.streamlit.io/)
3. **New App → Select Repo**
4. Main file to run:

```
app/Home.py
```

5. Add `requirements.txt`

✅ Streamlit auto-installs
✅ Your app goes live in minutes

---

## 📊 How the Pipeline Works

```
Upload Data → Merge → Preprocess → Train Models → Explain → Score New Data → Export
```

| Step          | Output                            |
| ------------- | --------------------------------- |
| Upload CSVs   | `internal_df`, `external_df`      |
| Preprocessing | `processed_df`                    |
| Training      | `.pkl` models saved in `/models/` |
| Explain       | SHAP, LIME, PDP visualized        |
| Score         | predictions + probabilities       |
| Export        | CSV + PDF per applicant           |

---

## ✅ Example Output

### ✅ Global SHAP Summary

* Shows most important features affecting approval

### ✅ Local SHAP / LIME

* Why an individual applicant was approved/rejected

### ✅ PDP

* Impact of changing a single feature (income, age, credit history)

### ✅ Scored CSV

* Each applicant + probability + decision

---

## 📜 PDF Generation

Every applicant can have a downloadable PDF report containing:

* Features
* Decision
* Probability
* Reason codes (via SHAP)

PDFs are stored inside:

```
/reports/applicant_XX_report.pdf
```

---
