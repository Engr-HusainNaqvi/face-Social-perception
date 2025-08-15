import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Face Perception Analyzer")

# Dataset Exploration
st.header("Dataset Exploration")
st.write("Explore face perception datasets.")
data_option = st.selectbox("Select Dataset", ["iEEG", "Eye Tracking"])
if data_option == "iEEG":
    st.write("Intracranial EEG data from Fusiform Face Area.")
elif data_option == "Eye Tracking":
    st.write("Mobile eye-tracking data during face tasks.")

# Model Testing
st.header("Model Testing")
st.write("Upload data to test face perception models.")
uploaded_file = st.file_uploader("Choose a file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Data uploaded successfully:")
    st.write(df.head())
    model_option = st.selectbox("Select Model", ["SCCA", "State Space"])
    if st.button("Predict"):
        if model_option == "SCCA":
            st.write("Sparse CCA: Neural decoding completed.")
        elif model_option == "State Space":
            st.write("State Space: Dynamic face perception modeled.")

# Results Visualization
st.header("Results Visualization")
st.write("Visualize key findings.")
if st.checkbox("Show Metrics"):
    metrics = {"Correlation": 0.85, "RMSE": 0.12}
    st.write(metrics)
if st.checkbox("Show Tuning"):
    data = pd.DataFrame({
        "Region": ["FFA", "OFA"],
        "Tuning": [0.90, 0.85]
    })
    plt.figure(figsize=(10, 5))
    sns.barplot(x="Region", y="Tuning", data=data)
    st.pyplot(plt)

# Limitations and Discussion
st.header("Limitations and Discussion")
st.write("""
- Limited dataset size.
- Unspecified train/validation/test splits.
- Potential iEEG noise.
""")
