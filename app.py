import streamlit as st
import subprocess
import os
import sys
import pandas as pd
os.makedirs("output", exist_ok=True)
os.makedirs("output/eda", exist_ok=True)
uploaded_file = st.file_uploader("Upload CSV dataset", type=["csv"])

if uploaded_file is not None:
    with open("dataset.csv", "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success("Dataset uploaded successfully!")

    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.write(f"Rows: {df.shape[0]} | Columns: {df.shape[1]}")

    if "run_pipeline" not in st.session_state:
        st.session_state.run_pipeline = False

    if st.button("Run DataAutoPilot"):
        st.session_state.run_pipeline = True

    if st.session_state.run_pipeline:

        script = os.path.join("src", "main.py")

        with st.spinner("Running DataAutoPilot Pipeline..."):
            
            result = subprocess.run(
                [sys.executable, script, "--data", "dataset.csv", "--cv", "2"],
                capture_output=True,
                text=True,
                timeout=600
            )   

        st.success("Pipeline executed successfully!")

        st.subheader("Pipeline Logs")
        st.code(result.stdout)

        if result.stderr:
            st.subheader("Errors")
            st.code(result.stderr)
        
        # Show insights:
        if os.path.exists("output/insights.txt"):
            st.subheader("Data Insights")
            with open("output/insights.txt") as f:
                for line in f:
                    st.write("*", line.strip())

        # Show model leaderboard:
        if os.path.exists("output/model_leaderboard.png"):
            st.image("output/model_leaderboard.png", caption= "Model Leaderboard")            

        # Show correlation heatmap:
        if os.path.exists("output/eda/correlation_heatmap.png"):
            st.image("output/eda/correlation_heatmap.png", caption="Correlation Heatmap")

        # Download report:
        report_path = "output/report.pdf"
        
        if os.path.exists(report_path):
            with open(report_path, "rb") as file:
                st.download_button(
                    label="Download Report",
                    data=file,
                    file_name="DataAutoPilot_Report.pdf",
                    mime="application/pdf"
                )

        else:
            st.warning("Report not generated yet")