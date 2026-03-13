import streamlit as st
import subprocess
import os
import sys
import pandas as pd

# -----------------------------
# CREATE OUTPUT FOLDERS (FIX)
# -----------------------------
os.makedirs("output", exist_ok=True)
os.makedirs("output/eda", exist_ok=True)

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="DataAutoPilot", layout="wide")

# -----------------------------
# HEADER
# -----------------------------
st.title("🚀 DataAutoPilot")
st.write("Automated Data Science Pipeline")

st.caption("⚡ Built with Python, Scikit-learn, SHAP & Streamlit")
st.caption("👨‍💻 Created by Shagun Vishwakarma")

st.markdown("""
🔗 [GitHub](https://github.com/Shagunvishwakarma1003)  
💼 [LinkedIn](https://www.linkedin.com/in/shagun1003)
""")

st.divider()

# -----------------------------
# DATASET UPLOAD
# -----------------------------
uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])

if uploaded_file is not None:

    # Save uploaded dataset
    with open("dataset.csv", "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success("Dataset uploaded successfully!")

    # Dataset preview
    df = pd.read_csv(uploaded_file)

    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Rows", df.shape[0])

    with col2:
        st.metric("Columns", df.shape[1])

    st.divider()

    # -----------------------------
    # RUN PIPELINE BUTTON
    # -----------------------------
    if st.button("Run DataAutoPilot"):

        with st.spinner("Running DataAutoPilot Pipeline... ⏳"):

            result = subprocess.run(
                [sys.executable, "src/main.py", "--data", "dataset.csv", "--cv", "2"],
                capture_output=True,
                text=True
            )

        st.success("Pipeline executed successfully!")

        # -----------------------------
        # PIPELINE LOGS
        # -----------------------------
        st.subheader("📜 Pipeline Logs")
        st.text(result.stdout)

        if result.stderr:
            st.subheader("⚠ Errors")
            st.text(result.stderr)

        st.divider()

        # -----------------------------
        # DATA INSIGHTS
        # -----------------------------
        if os.path.exists("output/insights.txt"):

            st.subheader("🧠 Data Insights")

            with open("output/insights.txt") as f:
                for line in f:
                    st.write("•", line.strip())

        # -----------------------------
        # VISUALIZATIONS
        # -----------------------------
        st.subheader("📈 Visualizations")

        if os.path.exists("output/missing_values.png"):
            st.image("output/missing_values.png", caption="Missing Values")

        if os.path.exists("output/correlation_heatmap.png"):
            st.image("output/correlation_heatmap.png", caption="Correlation Heatmap")

        if os.path.exists("output/model_leaderboard.png"):
            st.image("output/model_leaderboard.png", caption="Model Leaderboard")

        st.divider()

        # -----------------------------
        # REPORT DOWNLOAD
        # -----------------------------
        report_path = "output/report.pdf"

        if os.path.exists(report_path):

            with open(report_path, "rb") as file:
                st.download_button(
                    label="📥 Download Full Report",
                    data=file,
                    file_name="DataAutoPilot_Report.pdf",
                    mime="application/pdf"
                )

        else:
            st.warning("Report not generated yet.")

        st.divider()

        # -----------------------------
        # DEBUG INFO
        # -----------------------------
        with st.expander("Debug Info"):

            st.write("Current Directory Files:")
            st.write(os.listdir())

            if os.path.exists("output"):
                st.write("Output Folder Files:")
                st.write(os.listdir("output"))