import streamlit as st
import subprocess
import os

st.title("🚀 DataAutoPilot")
st.write("Automated Data Science Pipeline")


st.caption("⚡ Built with Python, Scikit-learn, SHAP & Streamlit")


st.caption("👨‍💻 Created by Shagun Vishwakarma")

st.markdown("""
🔗 [GitHub](https://github.com/Shagunvishwakarma1003) | 
💼 [LinkedIn](https://www.linkedin.com/in/shagun1003)
""")

uploaded_file = st.file_uploader("Upload CSV dataset", type=["csv"])

if uploaded_file is not None:

    with open("dataset.csv", "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success("Dataset uploaded successfully!")

    if st.button("Run DataAutoPilot"):

        import sys
        import shutil

        if os.path.exists("output"):
            shutil.rmtree("output")

        subprocess.run(
            [sys.executable, "src/main.py", "--data",
             "dataset.csv", "--cv", "2"]
        )

        st.success("Pipeline executed successfully!")

        st.subheader("Data Insights")
        if os.path.exists("output/insights.txt"):
            with open("output/insights.txt") as f:
                for line in f:
                    st.write("-", line.strip())

        report_path = "output/report.pdf"

        if os.path.exists(report_path):
            with open(report_path, "rb") as file:
                st.download_button(
                    label="⬇ Download Report",
                    data=file,
                    file_name="DataAutoPilot_Report.pdf",
                    mime="application/pdf",
                )
        else:
            st.warning("Report not found. Please check pipeline output.")
