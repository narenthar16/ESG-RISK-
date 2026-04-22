import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
import requests
import re

st.set_page_config(
    page_title="ESG Risk Monitor",
    page_icon="🌱",
    layout="wide",
)

GDRIVE_GOLD_URL    = "https://drive.google.com/file/d/1BsABOumGzIDEawnQsUsJGsVToZHMO0Wj/view?usp=sharing"
GDRIVE_SECTOR_URL  = "https://drive.google.com/file/d/1f5ubf3GnANojoSI0M14Llmd0bCNyYRd_/view?usp=sharing"
GDRIVE_METRICS_URL = "https://drive.google.com/file/d/1cHAepmIYGdHs7vkzgZ4-H1KJ3DzWlcHl/view?usp=sharing"

def get_file_id(url):
    match = re.search(r'/file/d/([a-zA-Z0-9_-]+)', url)
    return match.group(1) if match else None

def download_gdrive(url, filename):
    file_id      = get_file_id(url)
    direct_url   = f"https://drive.google.com/uc?export=download&id={file_id}"
    session      = requests.Session()
    response     = session.get(direct_url, stream=True, timeout=30)
    for key, value in response.cookies.items():
        if 'download_warning' in key:
            response = session.get(
                direct_url,
                params  = {'confirm': value},
                stream  = True,
                timeout = 30
            )
    with open(filename, 'wb') as f:
        for chunk in response.iter_content(chunk_size=32768):
            if chunk:
                f.write(chunk)

@st.cache_data(ttl=3600)
def load_data():
    try:
        download_gdrive(GDRIVE_GOLD_URL,    "gold.csv")
        download_gdrive(GDRIVE_SECTOR_URL,  "sector.csv")
        download_gdrive(GDRIVE_METRICS_URL, "metrics.json")
        gold   = pd.read_csv("gold.csv")
        sector = pd.read_csv("sector.csv")
        with open("metrics.json") as f:
            metrics = json.load(f)
        return gold, sector, metrics
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None

gold, sector, metrics = load_data()

st.title("🌱 Real-Time ESG Risk Monitor")
st.caption("Cloud Lakehouse Architecture · GNN + Random Forest · 364 Companies · USA, UK, India")

if gold is not None:
    RISK_COLORS = {"Low":"#27ae60","Medium":"#f39c12","High":"#e74c3c"}
    vc      = gold["risk_label"].value_counts()
    n_low   = int(vc.get("Low",   0))
    n_med   = int(vc.get("Medium",0))
    n_high  = int(vc.get("High",  0))
    avg_esg = gold["total_esg_score"].mean()

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Companies", len(gold))
    col2.metric("Avg ESG Score",   f"{avg_esg:.1f}")
    col3.metric("Low Risk",        n_low,  f"{n_low/len(gold)*100:.0f}%")
    col4.metric("Medium Risk",     n_med,  f"{n_med/len(gold)*100:.0f}%")
    col5.metric("High Risk",       n_high, f"{n_high/len(gold)*100:.0f}%")

    st.markdown("---")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Overview", "Company Explorer",
        "Sector Analysis", "Model Metrics", "Recommendations"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            fig = go.Figure(go.Pie(
                labels=vc.index,
                values=vc.values,
                hole=0.55,
                marker_colors=[RISK_COLORS.get(l,"#999") for l in vc.index]))
            fig.update_layout(
                title="ESG Risk Distribution",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor ="rgba(0,0,0,0)",
                font=dict(color="white"))
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.scatter(
                gold, x="total_esg_score", y="risk_score",
                color="risk_label",
                color_discrete_map=RISK_COLORS,
                hover_data=["ticker","sector","country"],
                title="ESG Score vs Risk Score")
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor ="rgba(0,0,0,0)",
                font=dict(color="white"))
            st.plotly_chart(fig, use_container_width=True)

        col3, col4 = st.columns(2)
        with col3:
            country_risk = gold.groupby(["country","risk_label"]).size().unstack(fill_value=0)
            fig = px.bar(country_risk, barmode="stack",
                         color_discrete_map=RISK_COLORS,
                         title="Risk Distribution by Country")
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor ="rgba(0,0,0,0)",
                font=dict(color="white"))
            st.plotly_chart(fig, use_container_width=True)
        with col4:
            fig = px.histogram(
                gold, x="risk_score", color="risk_label",
                color_discrete_map=RISK_COLORS,
                nbins=30, title="Risk Score Distribution")
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor ="rgba(0,0,0,0)",
                font=dict(color="white"))
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            country_filter = st.selectbox("Country", ["All"] + sorted(gold["country"].unique()))
        with col2:
            sector_filter  = st.selectbox("Sector",  ["All"] + sorted(gold["sector"].unique()))

        filtered = gold.copy()
        if country_filter != "All":
            filtered = filtered[filtered["country"]==country_filter]
        if sector_filter != "All":
            filtered = filtered[filtered["sector"]==sector_filter]

        ticker = st.selectbox("Select Company", sorted(filtered["ticker"].unique()))
        row    = filtered[filtered["ticker"]==ticker].iloc[0]

        col1, col2, col3 = st.columns(3)
        col1.metric("Risk Score",    f"{row['risk_score']:.2f}")
        col2.metric("Risk Label",    row['risk_label'])
        col3.metric("ESG Score",     f"{row['total_esg_score']:.1f}")

        col4, col5, col6 = st.columns(3)
        col4.metric("Environmental", f"{row['environmental_score']:.1f}")
        col5.metric("Social",        f"{row['social_score']:.1f}")
        col6.metric("Governance",    f"{row['governance_score']:.1f}")

        fig = go.Figure(go.Indicator(
            mode  = "gauge+number",
            value = float(row["risk_score"]),
            title = {"text": f"{ticker} Risk Score"},
            gauge = {
                "axis":  {"range":[0,100]},
                "bar":   {"color":"#2c3e50"},
                "steps": [
                    {"range":[0,  35], "color":"#27ae60"},
                    {"range":[35, 65], "color":"#f39c12"},
                    {"range":[65,100], "color":"#e74c3c"},
                ],
            }))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"))
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("All Companies")
        st.dataframe(
            filtered[["ticker","country","sector","risk_label",
                       "risk_score","total_esg_score","esg_rating",
                       "action","alert"]].sort_values("risk_score"),
            use_container_width=True)

    with tab3:
        if sector is not None:
            fig = px.bar(
                sector.sort_values("avg_esg_score"),
                x="avg_esg_score", y="sector",
                orientation="h",
                color="avg_risk_score",
                color_continuous_scale="RdYlGn_r",
                title="Average ESG Score by Sector")
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor ="rgba(0,0,0,0)",
                font=dict(color="white"))
            st.plotly_chart(fig, use_container_width=True)

            col1, col2 = st.columns(2)
            with col1:
                fig = px.bar(
                    sector.sort_values("pct_high_risk", ascending=False),
                    x="sector", y="pct_high_risk",
                    color="pct_high_risk",
                    color_continuous_scale="Reds",
                    title="% High Risk Companies by Sector")
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor ="rgba(0,0,0,0)",
                    font=dict(color="white"),
                    xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                fig = px.scatter(
                    sector,
                    x="avg_esg_score", y="avg_risk_score",
                    size="n_companies",
                    color="avg_carbon",
                    hover_data=["sector","n_companies"],
                    color_continuous_scale="Reds",
                    title="Sector ESG vs Risk vs Carbon")
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor ="rgba(0,0,0,0)",
                    font=dict(color="white"))
                st.plotly_chart(fig, use_container_width=True)

            st.subheader("Sector Summary Table")
            st.dataframe(sector, use_container_width=True)

    with tab4:
        if metrics is not None:
            reg = metrics.get("regression",{})
            cls = metrics.get("classification",{})
            cv  = metrics.get("cross_validation",{})
            acc = metrics.get("accuracy",{})

            st.subheader("Dataset Split")
            col1, col2 = st.columns(2)
            col1.metric("Training Set", f"80%")
            col2.metric("Testing Set",  f"20%")

            st.subheader("Accuracy")
            col1, col2 = st.columns(2)
            col1.metric("Training Accuracy", f"{float(acc.get('train_accuracy', 0))*100:.2f}%")
            col2.metric("Testing Accuracy",  f"{float(acc.get('test_accuracy',  0))*100:.2f}%")

            st.subheader("Regression Metrics")
            col1, col2, col3 = st.columns(3)
            col1.metric("MSE",  reg.get("mse",  "N/A"))
            col2.metric("RMSE", reg.get("rmse", "N/A"))
            col3.metric("MAE",  reg.get("mae",  "N/A"))

            st.subheader("Classification Metrics")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Precision", cls.get("precision_macro", "N/A"))
            col2.metric("Recall",    cls.get("recall_macro",    "N/A"))
            col3.metric("F1-Score",  cls.get("test_f1_macro",   "N/A"))
            col4.metric("CV F1",     cv.get("cv_f1_macro_mean", "N/A"))

            st.subheader("Confusion Matrix")
            cm   = np.array(cls.get("confusion_matrix", []))
            labs = ["Low","Medium","High"]
            if len(cm) > 0:
                fig = px.imshow(cm, text_auto=True,
                                x=labs, y=labs,
                                color_continuous_scale="Blues",
                                title="Confusion Matrix")
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="white"))
                st.plotly_chart(fig, use_container_width=True)

    with tab5:
        st.subheader("Portfolio Recommendations")

        col1, col2 = st.columns(2)
        with col1:
            st.success("BUY / HOLD — Low Risk Companies")
            low_risk = gold[gold["risk_label"]=="Low"].nsmallest(10,"risk_score")
            for _, row in low_risk.iterrows():
                st.write(f"✅ **{row['ticker']}** — Score: {row['risk_score']:.2f} — {row['country']}")

        with col2:
            st.error("SELL / DIVEST — High Risk Companies")
            high_risk = gold[gold["risk_label"]=="High"].nlargest(10,"risk_score")
            if len(high_risk) > 0:
                for _, row in high_risk.iterrows():
                    st.write(f"🚨 **{row['ticker']}** — Score: {row['risk_score']:.2f} — {row['country']}")
            else:
                st.write("No High Risk companies today")

        st.warning("MONITOR — Top 10 Medium Risk")
        med_risk = gold[gold["risk_label"]=="Medium"].nlargest(10,"risk_score")
        for _, row in med_risk.iterrows():
            st.write(f"⚠️ **{row['ticker']}** — Score: {row['risk_score']:.2f} — {row['sector']} — {row['country']}")

else:
    st.warning("Data not loaded. Please check Google Drive links.")

st.markdown("---")
st.caption(f"Last updated: {datetime.today().strftime('%Y-%m-%d %H:%M')} | "
           f"Real-Time ESG Risk Scoring System | MIT M.Tech Project")
