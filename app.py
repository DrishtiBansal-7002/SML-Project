
import streamlit as st
import pandas as pd
import joblib
import json
import plotly.express as px

st.set_page_config(
    page_title="Shopper Intelligence",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_models():
    return (
        joblib.load("purchase_model.pkl"),
        joblib.load("clustering_model.pkl"),
        joblib.load("cluster_preprocessor.pkl")
    )

@st.cache_data
def load_data():
    return pd.read_csv("customer_data.csv")

model, kmeans, cluster_preprocessor = load_models()
df = load_data()

with open("model_metrics.json") as f:
    metrics = json.load(f)

# PROFESSIONAL LIGHT THEME
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: #F7F8FC;
    color: #20243A;
}

[data-testid="stSidebar"] {
    background: #FFFFFF;
    border-right: 1px solid #E8EAF2;
}

[data-testid="stSidebar"] h1 {
    color: #5145CD;
    font-weight: 800;
}

h1, h2, h3 {
    color: #20243A !important;
    font-weight: 750 !important;
}

.hero h1 {
    color: #FFFFFF !important;
}

.hero p {
    color: #FFFFFF !important;
}

p, label, span {
    color: #454B63;
}

.hero {
    background: linear-gradient(120deg, #5145CD, #8175F5);
    padding: 38px;
    border-radius: 20px;
    color: white;
    margin-bottom: 25px;
}

.hero h1, .hero p {
    color: white !important;
}

.metric-card {
    background: white;
    border: 1px solid #E8EAF2;
    padding: 22px;
    border-radius: 20px;
    box-shadow: 0 8px 24px rgba(80,65,180,0.09);
    min-height: 115px;
}

.metric-label {
    color: #737991;
    font-size: 14px;
    font-weight: 600;
}

.metric-value {
    color: #5145CD;
    font-size: 29px;
    font-weight: 800;
    margin-top: 8px;
}

.brand-card {
    background: white;
    border: 1px solid #E8EAF2;
    border-radius: 13px;
    padding: 16px 8px;
    text-align: center;
    font-size: 17px;
    font-weight: 800;
}

.section-heading {
    font-size: 21px;
    font-weight: 750;
    color: #20243A;
    margin-top: 24px;
    margin-bottom: 15px;
}

div[data-testid="stForm"] {
    background: white;
    padding: 22px;
    border-radius: 20px;
    border: 1px solid #E8EAF2;
}

.stButton button, 
.stFormSubmitButton button {
    background: #5145CD;
    color: white;
    border-radius: 10px;
    border: none;
    font-weight: 700;
}

.stButton button:hover,
.stFormSubmitButton button:hover {
    background: #3930A6;
    color: white;
}

[data-testid="stMetric"] {
    background: white;
    border: 1px solid #E8EAF2;
    padding: 18px;
    border-radius: 14px;
}

[data-testid="stMetricLabel"] {
    color: #626A82 !important;
}

[data-testid="stMetricValue"] {
    color: #5145CD !important;
}


.hero h1 {
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
}

.hero p {
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
}


.hero-layout {
    display: flex;
    align-items: center;
    justify-content: space-between;
    min-height: 210px;
}

.hero-content {
    max-width: 75%;
}

.hero-art {
    font-size: 43px;
    display: flex;
    flex-direction: column;
    gap: 8px;
    align-items: center;
}

.hero-tag {
    display: inline-block;
    background: white;
    color: #6246D9;
    padding: 10px 18px;
    border-radius: 25px;
    font-weight: 700;
    margin-top: 12px;
}

</style>
""", unsafe_allow_html=True)


# SIDEBAR
st.sidebar.markdown("# 🛍️ Shopper IQ")
st.sidebar.caption("CUSTOMER ANALYTICS PLATFORM")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "EXPLORE",
    [
        "Dashboard",
        "Customer Segmentation",
        "Purchase Prediction",
        "Model Performance",
        "About Project"
    ]
)

st.sidebar.markdown("---")
st.sidebar.caption("Powered by Machine Learning")


# HELPER FOR METRIC CARDS
def metric_card(label, value):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


# DASHBOARD
if page == "Dashboard":

    st.markdown("""
    <div class="hero">
        <h1 style="color:white !important;">Shopper Intelligence Dashboard</h1>
        <p>Discover shopping patterns. Understand customers.
        Predict purchase intentions.</p>
    </div>
    """, unsafe_allow_html=True)

    total = len(df)
    purchases = int(df["Revenue"].sum())
    no_purchases = total - purchases
    conversion = purchases / total * 100

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric_card("🛒 Total Sessions", f"{total:,}")
    with c2:
        metric_card("💜 Purchases", f"{purchases:,}")
    with c3:
        metric_card("👥 No Purchases", f"{no_purchases:,}")
    with c4:
        metric_card("📈 Conversion Rate", f"{conversion:.2f}%")

    st.markdown('<div class="section-heading">Shopping Platform Inspiration</div>',
                unsafe_allow_html=True)

    brands = st.columns(4)
    brand_names = [
        ("amazon", "Amazon", "#232F3E"),
        ("flipkart", "Flipkart", "#2874F0"),
        ("nykaa", "NYKAA", "#E91E63"),
        ("myntra", "Myntra", "#E83E8C")
    ]

    for col, (icon, name, color) in zip(brands, brand_names):
        with col:
            st.markdown(
                f'<div class="brand-card" style="color:{color}">'
                f'🛍️ {name}</div>',
                unsafe_allow_html=True
            )

    left, right = st.columns(2)

    with left:
        st.markdown('<div class="section-heading">Purchase Distribution</div>',
                    unsafe_allow_html=True)

        fig = px.pie(
            names=["Purchase", "No Purchase"],
            values=[purchases, no_purchases],
            hole=0.62,
            color_discrete_sequence=["#7568E8", "#C9C5F5"]
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#343A55"
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown('<div class="section-heading">Session Overview</div>',
                    unsafe_allow_html=True)

        fig = px.bar(
            x=["Purchase", "No Purchase"],
            y=[purchases, no_purchases],
            color=["Purchase", "No Purchase"],
            color_discrete_sequence=["#7568E8", "#B7B1EF"],
            labels={"x": "Session Type", "y": "Number of Sessions"}
        )
        fig.update_layout(
            showlegend=False,
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#343A55"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.info("This dashboard analyses online shopping sessions using Machine Learning.")


# CUSTOMER SEGMENTATION
elif page == "Customer Segmentation":

    st.markdown('<div class="hero"><h1>Customer Segmentation</h1>'
                '<p>Explore groups of similar online shopping sessions.</p></div>',
                unsafe_allow_html=True)

    counts = df["Cluster"].value_counts().sort_index()

    c1, c2 = st.columns(2)
    with c1:
        metric_card("🧩 Total Segments", "2")
    with c2:
        metric_card("👥 Sessions Analysed", f"{len(df):,}")

    fig = px.bar(
        x=counts.index.astype(str),
        y=counts.values,
        color=counts.index.astype(str),
        color_discrete_sequence=["#7568E8", "#B7B1EF"],
        labels={"x": "Cluster", "y": "Sessions"},
        title="Shopping Sessions by Cluster"
    )
    st.plotly_chart(fig, use_container_width=True)

    profile = df.groupby("Cluster").agg(
        Total_Sessions=("Revenue", "count"),
        Purchase_Rate=("Revenue", "mean"),
        Average_Page_Values=("PageValues", "mean"),
        Average_Product_Related=("ProductRelated", "mean")
    ).reset_index()

    profile["Purchase_Rate"] *= 100

    st.markdown('<div class="section-heading">Cluster Profiles</div>',
                unsafe_allow_html=True)
    st.dataframe(profile.round(2), use_container_width=True, hide_index=True)

    fig = px.bar(
        profile,
        x="Cluster",
        y="Purchase_Rate",
        color="Cluster",
        color_discrete_sequence=["#7568E8", "#B7B1EF"],
        title="Purchase Rate by Segment"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption("Clusters are groups of shopping sessions, not identified individual customers.")


# PURCHASE PREDICTION
elif page == "Purchase Prediction":

    st.markdown('<div class="hero"><h1>Purchase Prediction</h1>'
                '<p>Enter shopping session details to estimate purchase intention.</p></div>',
                unsafe_allow_html=True)

    input_data = {}

    with st.form("prediction_form"):

        st.subheader("Shopping Session Details")

        columns = st.columns(2)
        feature_names = model.named_steps["preprocessor"].feature_names_in_

        for i, column in enumerate(feature_names):
            if column not in df.columns:
                continue

            with columns[i % 2]:
                if pd.api.types.is_numeric_dtype(df[column]):
                    median = float(df[column].median())

                    if pd.api.types.is_integer_dtype(df[column]):
                        input_data[column] = st.number_input(
                            column, value=int(median)
                        )
                    else:
                        input_data[column] = st.number_input(
                            column, value=median
                        )
                else:
                    options = df[column].dropna().unique().tolist()
                    input_data[column] = st.selectbox(column, options)

        submitted = st.form_submit_button("🔮 Predict Purchase", use_container_width=True)

    if submitted:
        input_df = pd.DataFrame([input_data])
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        st.markdown('<div class="section-heading">Prediction Result</div>',
                    unsafe_allow_html=True)

        if prediction == 1:
            st.success("🛍️ The model predicts that this session may result in a purchase.")
        else:
            st.warning("The model predicts that this session may not result in a purchase.")

        st.metric("Purchase Probability", f"{probability * 100:.2f}%")
        st.progress(float(probability))


# MODEL PERFORMANCE
elif page == "Model Performance":

    st.markdown('<div class="hero"><h1>Model Performance</h1>'
                '<p>Evaluate how well the purchase prediction model performs.</p></div>',
                unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    values = list(metrics.items())

    for col, (name, value) in zip([c1, c2, c3], values[:3]):
        with col:
            metric_card(name, f"{value * 100:.2f}%")

    c4, c5 = st.columns(2)

    for col, (name, value) in zip([c4, c5], values[3:]):
        with col:
            metric_card(name, f"{value * 100:.2f}%")

    fig = px.bar(
        x=list(metrics.keys()),
        y=[v * 100 for v in metrics.values()],
        color=list(metrics.keys()),
        color_discrete_sequence=["#7568E8", "#A59CF0", "#C3BDF7"],
        labels={"x": "Evaluation Metric", "y": "Score (%)"},
        title="Model Evaluation Results",
        range_y=[0, 100]
    )
    st.plotly_chart(fig, use_container_width=True)


# ABOUT PROJECT
elif page == "About Project":

    st.markdown('<div class="hero"><h1>About Shopper Intelligence</h1>'
                '<p>Machine Learning for understanding online shopper behaviour.</p></div>',
                unsafe_allow_html=True)

    st.subheader("Project Objective")

    st.write("""
    Customer Segmentation and Purchase Prediction uses the Online Shoppers
    Purchasing Intention dataset to analyse shopping sessions and predict
    whether a session will result in a purchase.
    """)

    st.subheader("Technologies Used")

    st.markdown("""
    - Python
    - Pandas and NumPy
    - Scikit-learn
    - Streamlit
    - Plotly
    """)

    st.subheader("Machine Learning Algorithms")

    st.markdown("""
    **Logistic Regression:** Predicts purchase intention.

    **K-Means Clustering:** Groups similar shopping sessions based on browsing behaviour.
    """)

    st.subheader("Project Workflow")

    st.markdown("""
    1. Data Collection
    2. Data Cleaning
    3. Exploratory Data Analysis
    4. Data Preprocessing
    5. Purchase Prediction
    6. Model Evaluation
    7. Customer Segmentation
    8. Cluster Profiling
    """)
