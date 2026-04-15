# Nassau Candy Distributor — Factory Reallocation & Shipping Optimization System


> An intelligent decision-making system that predicts shipping outcomes, recommends factory reassignments, and optimizes logistics efficiency for Nassau Candy Distributor.

---

## Project Overview

Nassau Candy Distributor currently assigns products to factories using static rules, leading to suboptimal shipping distances, high lead times for certain regions, and margin erosion due to logistics inefficiencies.

This project builds a **Factory Reallocation & Shipping Optimization Recommendation System** that:
- Predicts shipping lead times using Machine Learning
- Clusters routes by performance similarity
- Simulates factory–product reassignment scenarios
- Recommends optimal factory configurations at scale

---

## 🗂️ Project Structure

```
factory-optimization-project/
│
├── app/
│   └── streamlit_app.py          # Main Streamlit dashboard
│
├── data/
│   └── Nassau Candy Distributor.xlsx   # Source dataset (10,194 orders)
│
├── notebooks/
│   └── eda.ipynb                 # Exploratory Data Analysis + Modeling
│
├── requirements.txt              # Python dependencies
├── .gitignore
└── README.md
```

---

##  Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/factory-optimization-project.git
cd factory-optimization-project
```

### 2. Create Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Dashboard
```bash
cd app
streamlit run streamlit_app.py
```

### 5. Open EDA Notebook
```bash
cd notebooks
jupyter notebook eda.ipynb
```

---

## ML Models Used

| Model | RMSE | MAE | R² |
|-------|------|-----|----|
| Linear Regression | 0.812 | ~0.65 | 0.887 |
| Random Forest | 0.850 | ~0.68 | 0.876 |
| Gradient Boosting | 0.814 | ~0.65 | 0.887 |

**Best model: Linear Regression / Gradient Boosting** (tied at R²=0.887)

---

## Dashboard Features

| Tab | Feature |
|-----|---------|
|  Overview | KPIs, model evaluation, route clustering, filtered stats |
|  Factory Optimizer | ML-predicted lead time per factory with speed/profit scoring |
|  What-If Simulator | Compare current vs proposed factory assignments |
|  Recommendations | Ranked reassignment suggestions with confidence scores |
|  Risk & Impact | Slow route alerts, low-margin warnings, treemap |

---

##  Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.10+ |
| Data Processing | pandas, numpy |
| Machine Learning | scikit-learn |
| Clustering | KMeans |
| Visualization | Plotly, Seaborn, Matplotlib |
| Dashboard | Streamlit |
| Notebook | Jupyter |

---

##  Key Results

- **88.7% prediction accuracy** (R²) for lead time forecasting
- **4 factory clusters** identified by speed and profitability
- Identified **high-risk route combinations** with both slow lead times and low margins
- Optimization slider balances speed vs profit for personalized recommendations

---
