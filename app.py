
import os
import io
import zipfile
import urllib.request
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# ============================================================
# RETAIL AI INTELLIGENCE
# Real-world retail analytics + demand prediction + AI chatbot
# ============================================================

st.set_page_config(
    page_title="Retail AI Intelligence",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------- THEME --------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}
.stApp {
    background: #080b12;
    color: #f4f7fb;
}
[data-testid="stSidebar"] {
    background: #0d111b;
    border-right: 1px solid #20283a;
}
[data-testid="stSidebar"] * {
    color: #eef2f7 !important;
}
h1, h2, h3, h4 {
    color: #ffffff !important;
}
p, label, .stMarkdown, .stCaption {
    color: #c9d2df !important;
}
.hero {
    background: linear-gradient(135deg, #111827 0%, #111b2f 55%, #101522 100%);
    border: 1px solid #26344e;
    border-radius: 22px;
    padding: 28px 30px;
    margin-bottom: 22px;
    box-shadow: 0 10px 40px rgba(0,0,0,.22);
}
.hero h1 {
    font-size: 38px;
    margin: 0 0 8px 0;
}
.hero p {
    font-size: 16px;
    margin: 0;
}
.kpi {
    background: #111722;
    border: 1px solid #263044;
    border-radius: 18px;
    padding: 20px;
    min-height: 125px;
}
.kpi .label {
    color: #93a4ba;
    font-size: 13px;
    font-weight: 600;
}
.kpi .value {
    color: #ffffff;
    font-size: 28px;
    font-weight: 800;
    margin-top: 8px;
}
.kpi .sub {
    color: #7f91aa;
    font-size: 12px;
    margin-top: 5px;
}
.section {
    background: #0e141f;
    border: 1px solid #202b3e;
    border-radius: 18px;
    padding: 20px;
    margin: 10px 0 18px 0;
}
.badge {
    display:inline-block;
    padding: 6px 11px;
    border-radius: 999px;
    background:#16243b;
    border:1px solid #2d456b;
    color:#9fc2ff;
    font-size:12px;
    font-weight:700;
}
div[data-baseweb="select"] > div {
    background: #111722 !important;
    color: white !important;
    border-color: #334155 !important;
}
div[data-baseweb="popover"] * {
    background: #111722 !important;
    color: white !important;
}
.stButton > button {
    background: #2563eb !important;
    color: white !important;
    border: 0 !important;
    border-radius: 12px !important;
    font-weight: 700 !important;
    padding: 0.65rem 1rem !important;
}
.stButton > button:hover {
    background: #3b82f6 !important;
}
div[data-testid="stMetric"] {
    background: #111722;
    border: 1px solid #263044;
    border-radius: 15px;
    padding: 14px;
}
div[data-testid="stMetricLabel"] { color: #93a4ba !important; }
div[data-testid="stMetricValue"] { color: white !important; }
.chatbox {
    background:#111722;
    border:1px solid #263044;
    border-radius:16px;
    padding:16px;
}
.small {
    color:#8ea0b7;
    font-size:12px;
}
</style>
""", unsafe_allow_html=True)

# ------------------------- DATA ACCESS -------------------------
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
DATA_FILE = DATA_DIR / "online_retail.csv"
UCI_URL = "https://archive.ics.uci.edu/static/public/352/online+retail.zip"

@st.cache_data(show_spinner=False)
def load_data():
    if not DATA_FILE.exists():
        try:
            raw = urllib.request.urlopen(UCI_URL, timeout=45).read()
            with zipfile.ZipFile(io.BytesIO(raw)) as z:
                excel_name = next(
                    (n for n in z.namelist() if n.lower().endswith(".xlsx")),
                    None
                )
                if excel_name is None:
                    raise FileNotFoundError("Excel dataset was not found in the downloaded archive.")
                excel_bytes = z.read(excel_name)
            df = pd.read_excel(io.BytesIO(excel_bytes), engine="openpyxl")
            df.to_csv(DATA_FILE, index=False)
        except Exception as e:
            raise RuntimeError(
                "The real-world dataset could not be downloaded automatically. "
                f"Details: {e}"
            )

    df = pd.read_csv(DATA_FILE)
    required = ["InvoiceNo", "StockCode", "Description", "Quantity",
                "InvoiceDate", "UnitPrice", "CustomerID", "Country"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing columns: {missing}")

    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
    df["UnitPrice"] = pd.to_numeric(df["UnitPrice"], errors="coerce")
    df["CustomerID"] = df["CustomerID"].astype("string")
    df["Country"] = df["Country"].astype("string")

    # Remove cancelled invoices and invalid transactions.
    invoice_text = df["InvoiceNo"].astype(str)
    df = df[~invoice_text.str.startswith("C", na=False)]
    df = df.dropna(subset=["InvoiceDate", "Quantity", "UnitPrice", "Country"])
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]
    df["Revenue"] = df["Quantity"] * df["UnitPrice"]
    df["Year"] = df["InvoiceDate"].dt.year
    df["Month"] = df["InvoiceDate"].dt.month
    df["MonthName"] = df["InvoiceDate"].dt.strftime("%b")
    df["Day"] = df["InvoiceDate"].dt.day
    df["Hour"] = df["InvoiceDate"].dt.hour
    df["Weekday"] = df["InvoiceDate"].dt.day_name()
    df["OrderDate"] = df["InvoiceDate"].dt.date
    return df.reset_index(drop=True)

try:
    df = load_data()
except Exception as e:
    st.error(str(e))
    st.info("Run the installation cell and restart Streamlit if this is your first run.")
    st.stop()

# ------------------------- HELPERS ----------------------------
def money(x):
    return f"£{x:,.2f}"

def kpi(label, value, sub=""):
    st.markdown(
        f'<div class="kpi"><div class="label">{label}</div>'
        f'<div class="value">{value}</div><div class="sub">{sub}</div></div>',
        unsafe_allow_html=True
    )

def project_context():
    return {
        "rows": len(df),
        "revenue": float(df["Revenue"].sum()),
        "orders": int(df["InvoiceNo"].nunique()),
        "customers": int(df["CustomerID"].nunique()),
        "countries": int(df["Country"].nunique()),
        "date_min": str(df["InvoiceDate"].min().date()),
        "date_max": str(df["InvoiceDate"].max().date()),
        "top_country": str(df.groupby("Country")["Revenue"].sum().idxmax()),
        "top_product": str(
            df.groupby("Description")["Revenue"].sum().sort_values(ascending=False).index[0]
        ),
    }

# -------------------------- SIDEBAR ---------------------------
st.sidebar.markdown("## 🛍️ Retail AI")
st.sidebar.caption("Real-world Retail Intelligence Platform")
page = st.sidebar.radio(
    "NAVIGATION",
    [
        "🏠 Executive Dashboard",
        "📊 Sales Explorer",
        "👥 Customer Intelligence",
        "🤖 Demand Prediction",
        "💬 AI Chatbot",
        "📁 Dataset & Project",
    ],
)
st.sidebar.markdown("---")
st.sidebar.markdown("### Dataset")
st.sidebar.write(f"**{len(df):,}** cleaned transactions")
st.sidebar.write(f"**{df['Country'].nunique()}** countries")
st.sidebar.write(
    f"**{df['InvoiceDate'].min().strftime('%d %b %Y')} → "
    f"{df['InvoiceDate'].max().strftime('%d %b %Y')}**"
)
st.sidebar.markdown("---")
st.sidebar.caption("Built for real-world applied data learning.")

# --------------------------- HEADER ---------------------------
st.markdown("""
<div class="hero">
    <span class="badge">REAL-WORLD DATA PROJECT</span>
    <h1>Retail AI Intelligence</h1>
    <p>Sales analytics • Customer intelligence • Demand prediction • AI assistant</p>
</div>
""", unsafe_allow_html=True)

# ---------------------- EXECUTIVE DASHBOARD -------------------
if page == "🏠 Executive Dashboard":
    total_revenue = df["Revenue"].sum()
    orders = df["InvoiceNo"].nunique()
    customers = df["CustomerID"].nunique()
    units = df["Quantity"].sum()

    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi("TOTAL REVENUE", money(total_revenue), "Cleaned transaction revenue")
    with c2: kpi("ORDERS", f"{orders:,}", "Unique invoices")
    with c3: kpi("CUSTOMERS", f"{customers:,}", "Known customer IDs")
    with c4: kpi("UNITS SOLD", f"{units:,.0f}", "Positive quantities")

    st.markdown("### 📈 Business Overview")
    left, right = st.columns(2)

    monthly = (
        df.groupby(df["InvoiceDate"].dt.to_period("M"))["Revenue"]
        .sum().reset_index()
    )
    monthly["InvoiceDate"] = monthly["InvoiceDate"].dt.to_timestamp()

    with left:
        fig = px.line(
            monthly, x="InvoiceDate", y="Revenue",
            markers=True, title="Monthly Revenue Trend"
        )
        fig.update_layout(template="plotly_dark", paper_bgcolor="#0e141f",
                          plot_bgcolor="#0e141f", font_color="#e8edf5")
        st.plotly_chart(fig, use_container_width=True)

    country = (
        df.groupby("Country")["Revenue"].sum()
        .sort_values(ascending=False).head(10).reset_index()
    )
    with right:
        fig = px.bar(country, x="Revenue", y="Country", orientation="h",
                     title="Top 10 Countries by Revenue")
        fig.update_layout(template="plotly_dark", paper_bgcolor="#0e141f",
                          plot_bgcolor="#0e141f", font_color="#e8edf5")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 🔎 Key Business Signals")
    p1, p2, p3 = st.columns(3)
    top_product = df.groupby("Description")["Revenue"].sum().sort_values(ascending=False).index[0]
    top_day = df.groupby("Weekday")["Revenue"].sum().sort_values(ascending=False).index[0]
    avg_order = total_revenue / max(orders, 1)

    with p1:
        kpi("TOP PRODUCT", str(top_product)[:28], "Highest total revenue")
    with p2:
        kpi("TOP SALES DAY", top_day, "Highest aggregated revenue")
    with p3:
        kpi("AVG ORDER VALUE", money(avg_order), "Revenue / unique invoice")

# ------------------------- SALES EXPLORER ---------------------
elif page == "📊 Sales Explorer":
    st.markdown("### 📊 Interactive Sales Explorer")
    st.caption("Filter the real-world dataset and inspect revenue, products, countries and time patterns.")

    countries = sorted(df["Country"].dropna().unique().tolist())
    selected = st.multiselect(
        "Select countries",
        countries,
        default=countries[:min(5, len(countries))]
    )
    work = df[df["Country"].isin(selected)] if selected else df.copy()

    c1, c2 = st.columns(2)
    with c1:
        metric = st.selectbox("Chart metric", ["Revenue", "Quantity", "Orders"])
    with c2:
        chart_type = st.selectbox("View", ["Monthly", "Weekday", "Country", "Product"])

    if metric == "Orders":
        work_metric = work.groupby("InvoiceDate", as_index=False)["InvoiceNo"].nunique()
        work_metric = work_metric.rename(columns={"InvoiceNo": "Orders"})
        y_col = "Orders"
    else:
        work_metric = work
        y_col = metric

    if chart_type == "Monthly":
        plot = work.groupby(work["InvoiceDate"].dt.to_period("M"))[metric if metric != "Orders" else "Revenue"].sum().reset_index()
        plot["InvoiceDate"] = plot["InvoiceDate"].dt.to_timestamp()
        y = metric if metric != "Orders" else "Revenue"
        fig = px.line(plot, x="InvoiceDate", y=y, markers=True, title=f"Monthly {metric}")
    elif chart_type == "Weekday":
        order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        plot = work.groupby("Weekday")[metric if metric != "Orders" else "Revenue"].sum().reindex(order).reset_index()
        y = metric if metric != "Orders" else "Revenue"
        fig = px.bar(plot, x="Weekday", y=y, title=f"{metric} by Weekday")
    elif chart_type == "Country":
        plot = work.groupby("Country")[metric if metric != "Orders" else "Revenue"].sum().sort_values(ascending=False).head(15).reset_index()
        y = metric if metric != "Orders" else "Revenue"
        fig = px.bar(plot, x=y, y="Country", orientation="h", title=f"Top Countries by {metric}")
    else:
        plot = work.groupby("Description")[metric if metric != "Orders" else "Revenue"].sum().sort_values(ascending=False).head(15).reset_index()
        y = metric if metric != "Orders" else "Revenue"
        fig = px.bar(plot, x=y, y="Description", orientation="h", title=f"Top Products by {metric}")

    fig.update_layout(template="plotly_dark", paper_bgcolor="#0e141f",
                      plot_bgcolor="#0e141f", font_color="#e8edf5")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 🧾 Filtered Transaction Sample")
    st.dataframe(
        work[["InvoiceNo","StockCode","Description","Quantity","InvoiceDate","UnitPrice","Revenue","Country"]]
        .sort_values("InvoiceDate", ascending=False).head(100),
        use_container_width=True,
        hide_index=True
    )

# --------------------- CUSTOMER INTELLIGENCE ------------------
elif page == "👥 Customer Intelligence":
    st.markdown("### 👥 Customer Intelligence")
    st.caption("RFM-style analysis: Recency, Frequency and Monetary value.")

    customer_df = df.dropna(subset=["CustomerID"]).copy()
    snapshot = customer_df["InvoiceDate"].max() + pd.Timedelta(days=1)

    rfm = customer_df.groupby("CustomerID").agg(
        Recency=("InvoiceDate", lambda x: (snapshot - x.max()).days),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("Revenue", "sum")
    ).reset_index()

    c1, c2, c3 = st.columns(3)
    with c1: kpi("CUSTOMERS ANALYZED", f"{len(rfm):,}", "Customers with IDs")
    with c2: kpi("MEDIAN FREQUENCY", f"{rfm['Frequency'].median():.0f}", "Orders per customer")
    with c3: kpi("MEDIAN VALUE", money(rfm["Monetary"].median()), "Customer monetary value")

    features = rfm[["Recency","Frequency","Monetary"]].copy()
    features["Monetary"] = np.log1p(features["Monetary"])
    features["Frequency"] = np.log1p(features["Frequency"])
    scaler = StandardScaler()
    X = scaler.fit_transform(features)

    n_clusters = st.slider("Customer segments", 2, 6, 4)
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    rfm["Segment"] = km.fit_predict(X) + 1

    left, right = st.columns(2)
    with left:
        fig = px.scatter(
            rfm, x="Frequency", y="Monetary", size="Monetary",
            color="Segment", hover_data=["CustomerID", "Recency"],
            title="Customer Segments"
        )
        fig.update_layout(template="plotly_dark", paper_bgcolor="#0e141f",
                          plot_bgcolor="#0e141f", font_color="#e8edf5")
        st.plotly_chart(fig, use_container_width=True)

    with right:
        seg = rfm.groupby("Segment").agg(
            Customers=("CustomerID","count"),
            AvgRecency=("Recency","mean"),
            AvgFrequency=("Frequency","mean"),
            Revenue=("Monetary","sum")
        ).reset_index()
        fig = px.bar(seg, x="Segment", y="Revenue", title="Revenue by Segment")
        fig.update_layout(template="plotly_dark", paper_bgcolor="#0e141f",
                          plot_bgcolor="#0e141f", font_color="#e8edf5")
        st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        rfm.sort_values("Monetary", ascending=False).head(50),
        use_container_width=True, hide_index=True
    )

# ---------------------- DEMAND PREDICTION ---------------------
elif page == "🤖 Demand Prediction":
    st.markdown("### 🤖 Demand Prediction")
    st.caption("Random Forest regression estimates expected unit demand from transaction context.")

    model_data = df.dropna(subset=["Country"]).copy()
    model_data = model_data[model_data["Quantity"] <= model_data["Quantity"].quantile(0.99)].copy()

    countries = sorted(model_data["Country"].unique().tolist())
    country_codes = {c:i for i,c in enumerate(countries)}
    model_data["CountryCode"] = model_data["Country"].map(country_codes)

    features = ["UnitPrice","Hour","Month","Year","CountryCode","WeekdayNum"]
    model_data["WeekdayNum"] = model_data["InvoiceDate"].dt.weekday

    X = model_data[features]
    y = model_data["Quantity"]

    # Keep training fast enough for Colab/Streamlit.
    if len(X) > 70000:
        sample_idx = np.random.RandomState(42).choice(len(X), 70000, replace=False)
        X = X.iloc[sample_idx]
        y = y.iloc[sample_idx]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    model = RandomForestRegressor(
        n_estimators=120, max_depth=18, min_samples_leaf=2,
        random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, pred)
    r2 = r2_score(y_test, pred)

    c1, c2, c3 = st.columns(3)
    with c1: kpi("MODEL", "Random Forest", "Supervised regression")
    with c2: kpi("MAE", f"{mae:.2f}", "Mean absolute error")
    with c3: kpi("R² SCORE", f"{r2:.3f}", "Test-set R²")

    st.markdown("#### Enter a transaction scenario")
    a, b, c = st.columns(3)
    with a:
        price = st.number_input("Unit price (£)", min_value=0.01, value=2.50, step=0.10)
        country = st.selectbox("Country", countries, index=0)
    with b:
        date = st.date_input("Transaction date", value=df["InvoiceDate"].max().date())
        hour = st.slider("Transaction hour", 0, 23, 12)
    with c:
        weekday = date.weekday()
        st.info(f"Weekday: **{date.strftime('%A')}**")
        st.info(f"Month: **{date.strftime('%B')}**")
        st.info(f"Country code: **{country_codes[country]}**")

    if st.button("🔮 Predict Expected Demand", use_container_width=True):
        row = pd.DataFrame([{
            "UnitPrice": price,
            "Hour": hour,
            "Month": date.month,
            "Year": date.year,
            "CountryCode": country_codes[country],
            "WeekdayNum": weekday
        }])
        expected = max(0.0, float(model.predict(row)[0]))
        st.success(f"Estimated expected demand: **{expected:.1f} units**")
        st.caption("This is a model estimate based on patterns learned from the real-world dataset; it is not a guaranteed future quantity.")

    importance = pd.DataFrame({
        "Feature": features,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=False)

    fig = px.bar(importance, x="Importance", y="Feature", orientation="h",
                 title="Demand Model Feature Importance")
    fig.update_layout(template="plotly_dark", paper_bgcolor="#0e141f",
                      plot_bgcolor="#0e141f", font_color="#e8edf5")
    st.plotly_chart(fig, use_container_width=True)

# --------------------------- CHATBOT --------------------------
elif page == "💬 AI Chatbot":
    st.markdown("### 💬 Retail AI Assistant")
    st.caption("Ask about the dataset, sales trends, customers, products, demand prediction, or general questions.")

    ctx = project_context()
    api_key = os.getenv("GEMINI_API_KEY", "").strip()

    with st.expander("How this assistant works"):
        if api_key:
            st.write("Gemini is connected through the GEMINI_API_KEY environment variable.")
        else:
            st.write("No API key is configured, so the assistant uses a built-in project-aware fallback. Add GEMINI_API_KEY in Colab for general-purpose AI answers.")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for role, message in st.session_state.chat_history:
        with st.chat_message(role):
            st.markdown(message)

    prompt = st.chat_input("Ask something like: What is the top country by revenue?")

    def fallback_answer(q):
        ql = q.lower()
        if "revenue" in ql and ("total" in ql or "overall" in ql):
            return f"Total cleaned transaction revenue is {money(ctx['revenue'])}."
        if "top" in ql and "country" in ql:
            return f"The country with the highest aggregated revenue is {ctx['top_country']}."
        if "product" in ql and ("top" in ql or "best" in ql):
            return f"The highest-revenue product description in the cleaned dataset is: {ctx['top_product']}."
        if "customer" in ql and ("how many" in ql or "number" in ql):
            return f"The dataset contains {ctx['customers']:,} unique customer IDs after cleaning."
        if "order" in ql:
            return f"There are {ctx['orders']:,} unique invoices/orders in the cleaned dataset."
        if "dataset" in ql or "data" in ql:
            return (
                f"This project uses a real-world Online Retail transaction dataset. "
                f"After cleaning, it contains {ctx['rows']:,} transactions covering "
                f"{ctx['date_min']} to {ctx['date_max']} across {ctx['countries']} countries."
            )
        if "model" in ql or "prediction" in ql or "random forest" in ql:
            return (
                "The Demand Prediction module uses Random Forest Regression. "
                "It learns from unit price, hour, month, year, country and weekday to estimate expected unit demand."
            )
        return (
            "I can answer project-specific questions about sales, products, countries, customers, "
            "the dataset, and the demand model. For unrestricted general questions, configure GEMINI_API_KEY "
            "in the Colab environment."
        )

    if prompt:
        st.session_state.chat_history.append(("user", prompt))
        with st.chat_message("user"):
            st.markdown(prompt)

        answer = None
        if api_key:
            try:
                from google import genai
                client = genai.Client(api_key=api_key)
                system = f"""
You are Retail AI Intelligence Assistant.
Answer clearly and accurately.
Project facts:
- Cleaned transactions: {ctx['rows']:,}
- Total revenue: {ctx['revenue']:.2f}
- Orders: {ctx['orders']:,}
- Customers: {ctx['customers']:,}
- Countries: {ctx['countries']}
- Date range: {ctx['date_min']} to {ctx['date_max']}
- Top revenue country: {ctx['top_country']}
- Top revenue product: {ctx['top_product']}
Do not invent dataset statistics. If a question needs a statistic not supplied above, explain that it should be calculated from the dataset.
"""
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=system + "\n\nUser question: " + prompt
                )
                answer = response.text
            except Exception as e:
                answer = fallback_answer(prompt) + f"\n\n_AI connection note: {str(e)[:180]}_"
        else:
            answer = fallback_answer(prompt)

        st.session_state.chat_history.append(("assistant", answer))
        with st.chat_message("assistant"):
            st.markdown(answer)

    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

# ---------------------- DATASET & PROJECT ---------------------
else:
    st.markdown("### 📁 Dataset & Project Information")
    st.markdown("""
    <div class="section">
    <h3>Project objective</h3>
    <p>
    Retail AI Intelligence is an applied data science project that transforms
    real-world retail transactions into actionable business insights.
    It combines exploratory analysis, interactive visualization, customer
    segmentation, machine-learning demand prediction and an AI assistant.
    </p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1: kpi("RAW / CLEANED VIEW", f"{len(df):,}", "Transactions after cleaning")
    with c2: kpi("FEATURES", f"{len(df.columns)}", "Columns available in app")
    with c3: kpi("COUNTRIES", f"{df['Country'].nunique()}", "Geographical coverage")

    st.markdown("### 🧹 Cleaning performed")
    st.write("""
    - Removed cancelled invoices.
    - Removed missing dates, quantities, prices and countries.
    - Removed non-positive quantities and prices.
    - Created Revenue = Quantity × UnitPrice.
    - Engineered year, month, hour, weekday and order-date features.
    """)

    st.markdown("### 🧪 Dataset preview")
    st.dataframe(df.head(100), use_container_width=True, hide_index=True)

    st.markdown("### 🎯 Skills demonstrated")
    st.write("""
    **Python • Pandas • NumPy • Data Cleaning • Exploratory Data Analysis •
    Plotly • Streamlit • Scikit-learn • Random Forest Regression • RFM Analysis •
    K-Means Clustering • Prompt-based AI Assistant • Dashboard Development**
    """)

    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download Cleaned Dataset",
        data=csv_bytes,
        file_name="online_retail_cleaned.csv",
        mime="text/csv"
    )

st.markdown("---")
st.caption("Retail AI Intelligence • Applied real-world data project • Built with Python, Streamlit and Machine Learning")
