import streamlit as st
import subprocess
import os
import sys
import shutil

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

        # first create folder:
        os.makedirs("output", exist_ok=True)
        os.makedirs("output/eda", exist_ok=True)

        subprocess.run([sys.executable, "src/main.py", "--data", "dataset.csv", "--cv", "2"])

        st.success("Pipeline executed successfully!")
        
        # Debug Code
        st.write("Current directory files:")
        st.write(os.listdir())

        st.write("Output folder files:")
        st.write(os.listdir("output"))

        st.subheader("Data Insights")
        if os.path.exists("output/insights.txt"):
            with open("output/insights.txt") as f:
                for line in f:
                    st.write("-", line.strip())

        report_path = os.path.join(os.getcwd(), "output/report.pdf")

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
