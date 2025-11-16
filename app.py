import pandas as pd
import streamlit as st
import joblib
import plotly.express as px
import plotly.graph_objects as go
import time

# === PAGE CONFIG ===
st.set_page_config(page_title="Student Dropout Prediction - Nawal Rai", layout="wide")

# === CUSTOM CSS + TOASTR JS ===
st.markdown("""
<style>
    /* Full Header Background */
    .header-container {
        background: linear-gradient(rgba(0,0,0,0.6), rgba(0,0,0,0.6)), 
                    url('https://images.unsplash.com/photo-1523050854058-8df90110c9f1?ixlib=rb-4.0.3&auto=format&fit=crop&w=1350&q=80');
        background-size: cover;
        background-position: center;
        padding: 60px 20px;
        border-radius: 15px;
        margin: -80px -10px 30px -10px;
        color: white;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }
    .header-title {
        font-size: 3.5rem;
        font-weight: bold;
        text-shadow: 2px 2px 8px rgba(0,0,0,0.7);
        animation: fadeIn 1.5s ease-in-out;
    }
    .header-subtitle {
        font-size: 1.4rem;
        margin-top: 10px;
        opacity: 0.9;
    }

    /* Top Navigation Bar */
    .nav-bar {
        display: flex;
        justify-content: center;
        background: #1e3d59;
        padding: 12px 0;
        border-radius: 10px;
        margin: 20px 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    }
    .nav-button {
        margin: 0 15px;
        padding: 10px 25px;
        background: #2c5d8f;
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: bold;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    .nav-button:hover, .nav-button.active {
        background: #ffc107;
        color: #1e3d59;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }

    /* Animations */
    .fade-in { animation: fadeIn 1s ease-in-out; }
    .slide-up { animation: slideUp 0.8s ease-out; }
    @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
    @keyframes slideUp { from { transform: translateY(50px); opacity: 0; } to { transform: translateY(0); opacity: 1; } }

    /* Hide Sidebar */
    .css-1d391kg { display: none !important; }
    .css-1v0mbdj { margin-left: 0 !important; }

    /* Toastr Position */
    .toast { position: fixed; top: 20px; right: 20px; z-index: 9999; }
</style>
""", unsafe_allow_html=True)

# === TOASTR JS ===
st.markdown("""
<script src="https://cdnjs.cloudflare.com/ajax/libs/toastr.js/latest/toastr.min.js"></script>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/toastr.js/latest/toastr.min.css">
<script>
    toastr.options = {
      "closeButton": true,
      "progressBar": true,
      "positionClass": "toast-top-right",
      "timeOut": "3000",
      "showEasing": "swing",
      "hideEasing": "linear",
      "showMethod": "fadeIn",
      "hideMethod": "fadeOut"
    };
    function showPredicting() { toastr.info('Analyzing student profile...', 'Predicting'); }
    function showResult(res) { 
        if (res === 'Dropout') toastr.error('High Risk!', 'Dropout'); 
        else if (res === 'Graduate') toastr.success('Will Graduate!', 'Graduate'); 
        else toastr.warning('Enrolled', 'Enrolled'); 
    }
</script>
""", unsafe_allow_html=True)

# === HEADER WITH BACKGROUND ===
st.markdown(f"""
<div class="header-container">
    <div class="header-title fade-in">Student Dropout Prediction System</div>
    <div class="header-subtitle">Using Machine Learning Models | Nawal Rai</div>
</div>
""", unsafe_allow_html=True)

# === TOP NAVIGATION BAR ===
pages = ["Prediction", "Data Insights", "Model Performance"]

# Initialize session state safely
if "page" not in st.session_state:
    st.session_state.page = pages[0]

col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    if st.button("Prediction", key="nav1"):
        st.session_state.page = "Prediction"
        st.rerun()
with col2:
    if st.button("Data Insights", key="nav2"):
        st.session_state.page = "Data Insights"
        st.rerun()
with col3:
    if st.button("Model Performance", key="nav3"):
        st.session_state.page = "Model Performance"
        st.rerun()

# === LOAD DATA & MODEL ===
@st.cache_data
def load_data():
    df = pd.read_csv("data.csv", sep=';')
    df.columns = df.columns.str.strip().str.replace('"', '').str.replace('\t', '')
    return df

@st.cache_resource
def load_model():
    return joblib.load('model.pkl'), joblib.load('scaler.pkl'), joblib.load('label_encoder.pkl')

df = load_data()
model, scaler, le = load_model()

# === PAGE CONTENT ===
page = st.session_state.page

if page == "Prediction":
    st.markdown('<div class="slide-up">', unsafe_allow_html=True)
    st.header("Enter Student Details")

    user_input = {}
    cols = st.columns(3)
    for i, col in enumerate(df.columns[:-1]):
        with cols[i % 3]:
            min_val = float(df[col].min())
            max_val = float(df[col].max())
            default = float(df[col].median())
            user_input[col] = st.number_input(col, min_value=min_val, max_value=max_val, value=default)

    if st.button("Predict Dropout", type="primary"):
        st.markdown('<script>showPredicting();</script>', unsafe_allow_html=True)
        with st.spinner("Processing..."):
            time.sleep(2)
            input_df = pd.DataFrame([user_input])
            input_scaled = scaler.transform(input_df)
            pred = model.predict(input_scaled)[0]
            pred_proba = model.predict_proba(input_scaled)[0]
            pred_label = le.inverse_transform([pred])[0]

        st.markdown(f'<script>showResult("{pred_label}");</script>', unsafe_allow_html=True)

        fig = px.pie(values=pred_proba, names=le.classes_, title=f"Prediction: {pred_label}")
        fig.update_traces(textinfo='percent+label')
        fig.update_layout(transition_duration=800)
        st.plotly_chart(fig, use_container_width=True)

        progress = st.progress(0)
        for i in range(101):
            progress.progress(i)
            time.sleep(0.01)
        st.success(f"**{pred_label}** | Confidence: {max(pred_proba):.1%}")

        st.image("https://images.unsplash.com/photo-1551288049-bebda4e38f71?ixlib=rb-4.0.3&auto=format&fit=crop&w=1350&q=80",
                 caption="Neural Network Predicting Student Success", use_column_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

elif page == "Data Insights":
    st.header("Exploratory Data Analysis")
    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(df['Target'].value_counts(), title="Outcome Distribution",
                     color_discrete_map={'Dropout':'red','Enrolled':'orange','Graduate':'green'})
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.histogram(df, x='Age at enrollment', color='Target', title="Age vs Outcome")
        st.plotly_chart(fig, use_container_width=True)


elif page == "Model Performance":
    st.header("Model Accuracy & Insights")
    from sklearn.metrics import accuracy_score, confusion_matrix
    X = scaler.transform(df.drop('Target', axis=1))
    y = le.transform(df['Target'])
    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    
    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure(go.Indicator(mode="gauge+number", value=acc*100, title={'text': "Accuracy"}))
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        cm = confusion_matrix(y, y_pred)
        fig = px.imshow(cm, text_auto=True, title="Confusion Matrix")
        st.plotly_chart(fig, use_container_width=True)

# === FOOTER WITH NAWAL RAI'S INFO ===
st.markdown("---")
st.markdown(f"""
<div class="fade-in" style="text-align: center; padding: 20px; border-radius: 10px;">
    <h3>Project by Nawal Rai</h3>
    <p><strong>Roll No:</strong> B2361115 | <strong>Email:</strong> nawalmeghwar9@gmail.com | <strong>Department:</strong> BS Information Technology</p>
</div>
""", unsafe_allow_html=True)