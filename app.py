import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score
import plotly.express as px
import plotly.graph_objects as go

# ---------------------------------------------------------
# 1. تنظیمات صفحه و استایل‌دهی (UI/UX)
# ---------------------------------------------------------
st.set_page_config(
    page_title="Telco AI Retention Pro",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# تزریق CSS برای زیباتر کردن کارت‌های آمار و فونت‌ها
st.markdown("""
<style>
    .big-font { font-size:20px !important; font-weight: bold; }
    .stMetric {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #e9ecef;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.05);
    }
    .css-1d391kg { padding-top: 1rem; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 2. بارگذاری دیتا و آموزش مدل (The Engine)
# ---------------------------------------------------------
@st.cache_resource
def build_model():
    # بارگذاری دیتاست آپلود شده
    data_path = 'Telco-Customer-Churn.csv'
    
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        st.error(f"File not found: {data_path}. Please make sure the CSV file is in the same folder.")
        st.stop()

    # --- پیش‌پردازش حرفه‌ای (Data Cleaning) ---
    
    # 1. تبدیل TotalCharges به عدد (جاهای خالی را هندل می‌کنیم)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(inplace=True) # حذف تعداد بسیار کم ردیف‌های خالی
    
    # 2. حذف ستون بی‌فایده CustomerID
    if 'customerID' in df.columns:
        df = df.drop('customerID', axis=1)

    # 3. تبدیل متغیرهای متنی به عدد (Label Encoding)
    # ما دیکشنری انکودرها را نگه می‌داریم تا بعداً ورودی کاربر را هم تبدیل کنیم
    encoders = {}
    for col in df.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    # 4. آماده‌سازی X و y
    X = df.drop('Churn', axis=1)
    y = df['Churn'] # 1 = Yes (Left), 0 = No (Stayed)

    # 5. آموزش مدل XGBoost قدرتمند
    # استفاده از scale_pos_weight برای بالانس کردن کلاس‌ها (چون تعداد کسانی که می‌روند کمتر است)
    scale_pos = (y == 0).sum() / (y == 1).sum()
    
    model = xgb.XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        scale_pos_weight=scale_pos, # ترفند حرفه‌ای برای دیتای نامتوازن
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    
    # محاسبه دقت مدل برای نمایش در داشبورد
    acc = accuracy_score(y_test, model.predict(X_test))
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    
    return model, encoders, df, acc, auc

# اجرای مدل (کش می‌شود تا سرعت بالا باشد)
model, encoders, df_clean, acc_score, auc_score = build_model()

# ---------------------------------------------------------
# 3. سایدبار (پنل کنترل ورودی‌ها)
# ---------------------------------------------------------
with st.sidebar:
    st.header("👤 Customer Profile Simulator")
    st.write("Modify attributes to predict retention:")
    
    # ورودی‌های اصلی که بیشترین تاثیر را دارند
    # (برای سادگی دمو، برخی موارد را پیش‌فرض می‌گیریم اما موارد مهم را اسلایدر می‌کنیم)
    
    monthly_charges = st.slider('Monthly Charges ($)', 18.0, 120.0, 70.0, step=0.5)
    tenure = st.slider('Tenure (Months)', 0, 72, 24)
    total_charges = monthly_charges * tenure # محاسبه خودکار برای راحتی
    
    st.markdown("---")
    st.subheader("Services & Contract")
    
    contract_opts = ['Month-to-month', 'One year', 'Two year']
    contract = st.selectbox('Contract Type', contract_opts)
    
    internet_opts = ['DSL', 'Fiber optic', 'No']
    internet_service = st.selectbox('Internet Service', internet_opts)
    
    tech_support = st.selectbox('Tech Support?', ['No', 'Yes', 'No internet service'])
    online_security = st.selectbox('Online Security?', ['No', 'Yes', 'No internet service'])
    payment_method = st.selectbox('Payment Method', ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
    
    # دکمه اکشن
    predict_btn = st.button('Analyze Risk 🚀', use_container_width=True, type="primary")

# ---------------------------------------------------------
# 4. منطق پیش‌بینی (Prediction Logic)
# ---------------------------------------------------------
# ساختن یک ردیف دیتا شبیه دیتای اصلی برای دادن به مدل
input_data = pd.DataFrame(index=[0])

# پر کردن مقادیر (باید دقیقا مثل دیتاست اصلی باشد)
# اینجا برای دمو، مقادیر پیش‌فرض ستون‌های دیگر را از میانگین یا مد دیتاست اصلی پر می‌کنیم
# تا کلاینت مجبور نباشد ۲۰ تا فرم پر کند.
for col in df_clean.drop('Churn', axis=1).columns:
    input_data[col] = df_clean[col].mode()[0] # پیش‌فرض: رایج‌ترین مقدار

# حالا مقادیر انتخابی کاربر را جایگزین می‌کنیم
# نکته مهم: باید مقادیر متنی را با همان انکودرها تبدیل به عدد کنیم
def safe_encode(encoder, value):
    try:
        return encoder.transform([value])[0]
    except:
        return 0 # هندل کردن خطای احتمالی

input_data['MonthlyCharges'] = monthly_charges
input_data['TotalCharges'] = total_charges
input_data['tenure'] = tenure
input_data['Contract'] = safe_encode(encoders['Contract'], contract)
input_data['InternetService'] = safe_encode(encoders['InternetService'], internet_service)
input_data['TechSupport'] = safe_encode(encoders['TechSupport'], tech_support)
input_data['OnlineSecurity'] = safe_encode(encoders['OnlineSecurity'], online_security)
input_data['PaymentMethod'] = safe_encode(encoders['PaymentMethod'], payment_method)


# ---------------------------------------------------------
# 5. داشبورد اصلی (Main Dashboard)
# ---------------------------------------------------------
st.title("📡 AI Customer Retention System")
st.markdown(f"**Model Performance:** Accuracy: `{acc_score:.1%}` | ROC-AUC: `{auc_score:.2f}`")
st.divider()

col1, col2 = st.columns([2, 1.2])

with col1:
    st.subheader("🔍 Prediction Results")
    
    if predict_btn:
        # انجام پیش‌بینی
        pred_prob = model.predict_proba(input_data)[0][1] # احتمال ریزش (1)
        pred_class = int(pred_prob > 0.5)
        
        # --- نمودار عقربه‌ای (Gauge Chart) ---
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = pred_prob * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Churn Probability (%)", 'font': {'size': 24}},
            delta = {'reference': 50, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
            gauge = {
                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "#ff2b2b" if pred_prob > 0.5 else "#00cc96"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 30], 'color': 'rgba(0, 255, 0, 0.1)'},
                    {'range': [30, 70], 'color': 'rgba(255, 255, 0, 0.1)'},
                    {'range': [70, 100], 'color': 'rgba(255, 0, 0, 0.1)'}],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90}}))
        
        st.plotly_chart(fig_gauge, use_container_width=True)
        
        # --- پیام هوشمند ---
        if pred_prob > 0.5:
            st.error(f"⚠️ **HIGH RISK ALERT:** This customer is likely to cancel their subscription.")
            st.markdown("""
            **Recommended Actions:**
            - 🏷️ Offer a **15% discount** on 1-year contract renewal.
            - 📞 Schedule a support call to check for technical issues.
            """)
        else:
            st.success(f"✅ **SAFE:** This customer shows strong loyalty signals.")
            st.markdown("**Action:** No immediate intervention needed. Consider up-selling premium features.")

    else:
        st.info("👈 Adjust the customer profile in the sidebar and click 'Analyze Risk' to see the AI prediction.")

with col2:
    st.subheader("💰 Financial Impact")
    
    # محاسبه ضرر احتمالی (LTV تقریبی)
    # فرض: اگر مشتری برود، ما درآمد ۱۲ ماه آینده را از دست می‌دهیم
    potential_loss = monthly_charges * 12
    
    st.metric(
        label="Potential Annual Revenue at Risk",
        value=f"${potential_loss:,.2f}",
        delta="-Risk" if predict_btn and pred_prob > 0.5 else "Stable",
        delta_color="inverse"
    )
    
    st.markdown("---")
    st.markdown("### 📊 Key Drivers (XGBoost)")
    st.caption("Which factors are influencing this prediction the most?")
    
    # نمایش اهمیت ویژگی‌ها (Feature Importance)
    # استخراج ۵ ویژگی مهم
    importance = model.feature_importances_
    feat_names = input_data.columns
    
    # ساخت دیتای تمیز برای نمودار
    feat_df = pd.DataFrame({'Feature': feat_names, 'Importance': importance})
    feat_df = feat_df.sort_values(by='Importance', ascending=True).tail(7) # ۷ تای آخر (مهم‌ترین‌ها)
    
    fig_imp = px.bar(
        feat_df, 
        x='Importance', 
        y='Feature', 
        orientation='h',
        color='Importance',
        color_continuous_scale='Blues'
    )
    fig_imp.update_layout(xaxis_title="", yaxis_title="", showlegend=False, margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(fig_imp, use_container_width=True)

# فوتر حرفه‌ای
st.markdown("---")
st.markdown("<div style='text-align: center; color: grey;'>Built with XGBoost & Streamlit | Designed for Enterprise Analytics</div>", unsafe_allow_html=True)