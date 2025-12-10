# 📡 AI Customer Retention Dashboard

> **A Business Intelligence tool powered by Machine Learning to predict customer churn and estimate revenue loss.**

---

## 📊 Overview

Customer retention is vital for profitability. This application goes beyond simple prediction; it serves as a decision-support tool for stakeholders. 

Using a robust **XGBoost Classifier** trained on Telco customer data, this dashboard allows users to simulate customer profiles and instantly see the risk of churn along with the potential financial impact.

### Key Capabilities:
* **Predictive Modeling:** Calculates the exact probability of a customer leaving using XGBoost.
* **Financial Impact Analysis:** Estimates the "Revenue at Risk" based on customer value.
* **Interactive Simulation:** Sidebar controls allow non-technical users to test "What-if" scenarios.
* **Explainable AI:** Visualizes key churn drivers using feature importance charts.

---

## 🛠️ Technology Stack

* **Core Logic:** Python 3.x
* **Machine Learning:** XGBoost, Scikit-Learn (Imbalanced Data Handling)
* **Web Framework:** Streamlit
* **Data Processing:** Pandas, NumPy
* **Visualization:** Plotly Express, Plotly Graph Objects

---

## 💻 Installation & Usage

To run this project locally, follow these steps:

**1. Clone the repository**
git clone [https://github.com/arshiaestineh2005-tech/Telco-Customer-Churn-AI.git](https://github.com/arshiaestineh2005-tech/Telco-Customer-Churn-AI.git)
cd Telco-Customer-Churn-AI

**2. Install requirements
pip install -r requirements.txt

**3. Run the App
streamlit run app.py
The application will open in your browser at http://localhost:8501.

**📂 File Structure
├── app.py                   # Main application code (Streamlit + Model Training)
├── Telco-Customer-Churn.csv # Dataset source
├── requirements.txt         # Project dependencies
├── dashboard.png            # Screenshot of the app
└── README.md                # Documentation

📬 Contact
Arshia Estineh Machine Learning Engineer | AI Solutions

.📧 Email: arshiaestineh2005@icloud.com

.🐙 GitHub: arshiaestineh2005-tech

-----
Built with ❤️ using XGBoost & Streamlit.
