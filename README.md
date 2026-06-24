# Health_Insurance_Premium_Prediction

An end-to-end Machine Learning web application that predicts annual health insurance premiums based on individual demographic, financial, lifestyle, and medical histories. Built with **Python**, **Scikit-Learn**, and **Streamlit**.

The core innovation of this project lies in its **Model Segmentation Strategy**, which optimizes prediction accuracy by routing data into separate, specialized models based on age cohorts instead of using a single, generalized model.

---

## 🚀 Key Features

* **Age-Segmented Modeling:** Dynamically routes input profiles into specialized models (under 25 vs. over 25) to account for distinct risk behaviors and lifestyle variations.
* **Custom Medical Risk Scoring:** Features an algorithmic preprocessing pipeline that parses combined medical histories (e.g., *Diabetes & Heart disease*) into a single normalized risk metric.
* **Interactive Responsive Grid UI:** Built using a clean, 4-row structural layout in Streamlit for seamless user input.
* **Optimized Inference Pipeline:** Implements memory caching via Streamlit’s `@st.cache_resource` to load heavy machine learning artifacts exactly once, optimizing application speed.

---

## 📁 Project Structure

```text
├── artifacts/               # Serialized ML components 
│   ├── model_young.joblib   # Trained ML model for Age <= 25
│   ├── model_rest.joblib    # Trained ML model for Age > 25
│   ├── scaler_young.joblib  # Scaler configuration dictionary for the young cohort
│   └── scaler_rest.joblib   # Scaler configuration dictionary for the rest cohort
|
├── .gitignore               # Prevents tracking of virtual environments & pycache
├── LICENSE                  # Repository open-source license
├── README.md                # Project documentation
├── main.py                  # Streamlit frontend layout and UI event management
├── prediction_helper.py     # Data preprocessing, explicit column scaling, and inference logic
└── requirements.txt         # List of dependencies required to run the application


```

---

## 🧠 Model Details & Architecture

### 1. Model Segmentation Strategy

Risk factors for health insurance scale non-linearly with age. To tackle this, the dataset is segmented into two subsets:

* **Cohort Young ($\le 25$):** Custom-tailored to younger profiles, focusing on lifestyle variables and early-career economic status.
* **Cohort Rest ($> 25$):** Tailored to capture compounding age risks, larger medical histories, and family dependencies.

### 2. Custom Risk Calculations

Medical histories are split and processed dynamically. The app computes a combined score based on industry-standard relative risk factors:

* **Heart Disease:** 8 | **Diabetes:** 6 | **High Blood Pressure:** 6 | **Thyroid:** 5 | **No Disease:** 0

The raw sum is then min-max normalized against a theoretical maximum risk score of 14:


$$\text{Normalized Risk Score} = \frac{\text{Total Score} - 0}{14 - 0}$$

---

## 📋 Input Features & Schema

The underlying models expect an 18-dimensional feature vector mapped precisely from the user interface:

| Field Name | Type | Options / Bounds | Preprocessing / Encoding |
| --- | --- | --- | --- |
| **Age** | Numeric | 18 to 100 years | Min-Max Scaled |
| **Number of Dependants** | Numeric | 0 to 20 | Passed Directly |
| **Income in Lakhs** | Numeric | 0 to 200 | Min-Max Scaled |
| **Genetical Risk** | Numeric | 0 to 5 | Passed Directly |
| **Insurance Plan** | Categorical | Bronze, Silver, Gold | Ordinal Encoded (1, 2, 3) |
| **Employment Status** | Categorical | Salaried, Self-Employed, Freelancer, Unemployed | One-Hot Encoded |
| **Gender** | Categorical | Male, Female | One-Hot Encoded (`gender_Male`) |
| **Marital Status** | Categorical | Married, Unmarried | One-Hot Encoded (`marital_status_Unmarried`) |
| **BMI Category** | Categorical | Normal, Overweight, Obesity, Underweight | One-Hot Encoded |
| **Smoking Status** | Categorical | No Smoking, Regular, Occasional | One-Hot Encoded |
| **Region** | Categorical | Northwest, Southeast, Northeast, Southwest | One-Hot Encoded |
| **Medical History** | Categorical | Multiple combined diseases | Derived into `normalized_risk_score` |

---

## 💾 Model Artifacts

The application depends on four external files saved within the `artifacts/` folder:

* **`model_young.joblib` / `model_rest.joblib`:** Pre-trained estimator files (e.g., Gradient Boosting / Random Forest regressions).
* **`scaler_young.joblib` / `scaler_rest.joblib`:** Python dictionaries containing the structural instances of the `MinMaxScaler` along with an explicit feature checklist array under the `cols_to_scale` reference.

---

## 📈 Results & Performance:

* **Under-25 Model Accuracy: 98%**

* **Over-25 Model Accuracy: 99%**

---
## 🛠️ Installation and Setup

Follow these simple steps to spin up and test the web application locally.

### Prerequisites

Make sure you have **Python 3.9** or higher installed on your computer.

### Step 1: Clone the Repository

```bash
git clone https://github.com/NIKHILESH22/Health_Insurance_Premium_Prediction.git

```

### Step 2: Create a Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate

```

### Step 3: Install Required Dependencies

```bash
pip install -r requirements.txt

```

### Step 4: Run the Streamlit Application

```bash
streamlit run main.py

```

Once executed, the terminal will output a local URL (usually `http://localhost:8501`). Open this link in any browser to interact with the model dashboard.

---

## 📄 License

Distributed under the terms of the [LICENSE](https://www.google.com/search?q=LICENSE) file included in this repository.
