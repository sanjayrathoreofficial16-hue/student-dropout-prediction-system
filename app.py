import pandas as pd
import streamlit as st
import joblib
import plotly.express as px
import plotly.graph_objects as go
import time

st.set_page_config(page_title="Student Dropout Prediction System By Nawal Rai - A Project of Data Mining & Data Warehouse!", layout="wide")

# === CUSTOM CSS + TOASTR JS ===
st.markdown("""
<style>
    .fade-in { animation: fadeIn 1s ease-in-out; }
    .slide-up { animation: slideUp 0.8s ease-out; }
    @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
    @keyframes slideUp { from { transform: translateY(50px); opacity: 0; } to { transform: translateY(0); opacity: 1; } }
    .toast { position: fixed; top: 20px; right: 20px; z-index: 9999; }
</style>
""", unsafe_allow_html=True)

# === TOASTR JS INJECTION ===
toastr_js = """
<script src="https://cdnjs.cloudflare.com/ajax/libs/toastr.js/latest/toastr.min.js"></script>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/toastr.js/latest/toastr.min.css">
<script>
    toastr.options = {
      "closeButton": true,
      "progressBar": true,
      "positionClass": "toast-top-right",
      "timeOut": "3000",
      "extendedTimeOut": "1000",
      "showEasing": "swing",
      "hideEasing": "linear",
      "showMethod": "fadeIn",
      "hideMethod": "fadeOut"
    };
    function showPredicting() {
        toastr.info('Analyzing student profile...', 'Predicting Your Data');
    }
    function showResult(result) {
        if (result === 'Dropout') {
            toastr.error('High risk of dropout!', 'Prediction: Dropout');
        } else if (result === 'Graduate') {
            toastr.success('Likely to graduate!', 'Prediction: Graduate');
        } else {
            toastr.warning('Currently enrolled', 'Prediction: Enrolled');
        }
    }
</script>
"""
st.markdown(toastr_js, unsafe_allow_html=True)

# === HEADER WITH AI IMAGE ===
st.image(
    "https://images.unsplash.com/photo-1523050854058-8df90110c9f1?ixlib=rb-4.0.3&auto=format&fit=crop&w=1350&q=80",
    use_column_width=True,
    caption="Student Analyzing Academic Data with ML"
)
st.markdown('<div class="fade-in"><h1>Student Dropout Prediction System</h1></div>', unsafe_allow_html=True)

# === LOAD DATA & MODEL ===
with st.spinner("Loading model and data..."):
    time.sleep(1)
    df = pd.read_csv("data.csv", sep=';')
    df.columns = df.columns.str.strip().str.replace('"', '').str.replace('\t', '')
    
    model = joblib.load('model.pkl')
    scaler = joblib.load('scaler.pkl')
    le = joblib.load('label_encoder.pkl')

# === SIDEBAR NAVIGATION ===
page = st.sidebar.selectbox("Navigate", ["Prediction", "Data Insights", "Model Performance"])

# === PREDICTION PAGE ===
if page == "Prediction":
    st.markdown('<div class="slide-up">', unsafe_allow_html=True)
    st.header("Enter Student Details")

    feature_columns = df.columns[:-1]
    user_input = {}
    cols = st.columns(3)
    for i, col in enumerate(feature_columns):
        with cols[i % 3]:
            min_val = float(df[col].min())
            max_val = float(df[col].max())
            default = float(df[col].median())
            user_input[col] = st.number_input(col, min_value=min_val, max_value=max_val, value=default)

    if st.button("Predict Dropout", type="primary", key="predict_btn"):
        # Trigger Toastr: Predicting...
        st.markdown('<script>showPredicting();</script>', unsafe_allow_html=True)
        
        with st.spinner("Processing..."):
            time.sleep(2)  # Simulate processing
            input_df = pd.DataFrame([user_input])[feature_columns]
            input_scaled = scaler.transform(input_df)
            pred = model.predict(input_scaled)[0]
            pred_proba = model.predict_proba(input_scaled)[0]
            pred_label = le.inverse_transform([pred])[0]

        # Show Result Toast
        st.markdown(f'<script>showResult("{pred_label}");</script>', unsafe_allow_html=True)

        # Animated Pie Chart
        fig = px.pie(values=pred_proba, names=le.classes_, title=f"Prediction: {pred_label}",
                     color_discrete_sequence=px.colors.qualitative.Bold)
        fig.update_traces(textinfo='percent+label', transition_duration=1000)
        st.plotly_chart(fig, use_container_width=True)

        # Confidence Progress
        progress = st.progress(0)
        for i in range(101):
            progress.progress(i)
            time.sleep(0.01)
        st.success(f"**Prediction: {pred_label}** | Confidence: {max(pred_proba):.1%}")

        st.image("https://images.unsplash.com/photo-1551288049-bebda4e38f71?ixlib=rb-4.0.3&auto=format&fit=crop&w=1350&q=80",
                 caption="Neural Network Predicting Student Success", use_column_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

# === DATA INSIGHTS PAGE ===
elif page == "Data Insights":
    st.header("Exploratory Data Analysis")
    col1, col2 = st.columns(2)
    with col1:
        fig1 = px.bar(df['Target'].value_counts(), title="Student Outcome Distribution",
                      color=['Dropout', 'Enrolled', 'Graduate'], color_discrete_map={'Dropout':'red','Enrolled':'orange','Graduate':'green'})
        st.plotly_chart(fig1, use_container_width=True)
    with col2:
        fig2 = px.histogram(df, x='Age at enrollment', color='Target', title="Age vs Outcome")
        st.plotly_chart(fig2, use_container_width=True)

    st.image("https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?ixlib=rb-4.0.3&auto=format&fit=crop&w=1350&q=80",
             caption="Diverse Students in ML-Enhanced Learning", use_column_width=True)

# === MODEL PERFORMANCE ===
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

    st.image("https://images.unsplash.com/photo-1555949963-aa79dcee981c?ixlib=rb-4.0.3&auto=format&fit=crop&w=1350&q=80",
             caption="ML Model Visualizing Academic Success", use_column_width=True)

# === FOOTER WITH YOUR INFO ===
st.markdown("---")
st.markdown(f"""
<div class="fade-in">
<h3>Project By Nawal Rai </h3>
<p><strong>Enrollment:</strong> B2361115 | <strong>Email:</strong> nawalmeghwar9@gmail.com | <strong>Department:</strong> BS Information Technology</p>
</div>
""", unsafe_allow_html=True)