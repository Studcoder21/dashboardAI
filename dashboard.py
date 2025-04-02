
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

try:
    data = pd.read_csv('CW_Dataset_4300402.csv', encoding='utf-8')
except UnicodeDecodeError:
    data = pd.read_csv('CW_Dataset_4300402.csv', encoding='latin1')

data['pressure_ratio'] = data['APVs - Specific injection pressure peak value'] / data['APSs - Specific back pressure peak value']
X = data.drop('quality', axis=1)
y = data['quality']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

rf_model = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42)
rf_model.fit(X_scaled, y - 1)

st.title('Plastic Injection Moulding Quality Predictor')

st.header('Single Prediction')
st.subheader('Input Features for Single Prediction')
melt_temp = st.slider('Melt Temperature (°C)', float(X['Melt temperature'].min()), float(X['Melt temperature'].max()), 106.0)
mold_temp = st.slider('Mold Temperature (°C)', float(X['Mold temperature'].min()), float(X['Mold temperature'].max()), 81.0)
time_to_fill = st.slider('Time to Fill (s)', float(X['time_to_fill'].min()), float(X['time_to_fill'].max()), 6.5)
shot_volume = st.slider('Shot Volume (cm³)', float(X['SVo - Shot volume'].min()), float(X['SVo - Shot volume'].max()), 18.7)

input_data = pd.DataFrame([[
    melt_temp, mold_temp, time_to_fill, 3.0, 75.0, 900.0, 920.0, 116.0, 104.0, 146.0, 900.0, 8.8, shot_volume, 900/146
]], columns=X.columns)
input_scaled = scaler.transform(input_data)

pred = rf_model.predict(input_scaled)[0] + 1
probs = rf_model.predict_proba(input_scaled)[0]
quality_map = {1: 'Waste', 2: 'Acceptable', 3: 'Target', 4: 'Inefficient'}
st.write(f'Predicted Quality: **{quality_map[pred]}**')

st.subheader('Prediction Probabilities')
fig, ax = plt.subplots(figsize=(5, 3))
ax.bar(quality_map.values(), probs)
ax.set_ylim(0, 1)
plt.xticks(rotation=45)
st.pyplot(fig)

st.header('Training Results (Best Model: LightGBM)')
col1, col2 = st.columns(2)

with col1:
    st.subheader('Feature Importance (LightGBM)')
    st.image('feature_importance.png', use_container_width=True)  # Updated parameter

with col2:
    st.subheader('Confusion Matrix (LightGBM)')
    st.image('confusion_matrix.png', use_container_width=True)  # Updated parameter

st.header('Batch Prediction')
st.subheader('Input a Small Dataset for Batch Prediction')
with st.form(key='batch_prediction_form'):
    st.write('Enter data for prediction (one row per line, comma-separated values in the order of features below):')
    st.write(X.columns.tolist())
    user_input = st.text_area('Example format', '''106.0,81.0,6.5,3.0,75.0,900.0,920.0,116.0,104.0,146.0,900.0,8.8,18.7,6.16
105.0,82.0,7.0,3.2,74.8,910.0,930.0,118.0,105.0,145.0,910.0,8.9,18.8,6.27
106.316,81.402,6.968,3.21,74.81,914.5,930,117.2,105.5,146.3,913.6,8.8,18.77,6.2
105.734,81.007,6.343999999999999,3.66,75.74,894.2,914.9,115.1,101.6,146.9,837.2,8.54,19.03,6.2
106.49273717804589,81.362,6.864,3.82,74.82,901.6,902.061429,104.6,106.4450570889321,148.1,926.5,8.82,18.72,6.2
106.028,82.13,10.972,2.96,75.62,901.1,916.6,112.1,103.4,148.4,878.3,8.79,18.78,6.2''')
    submit_button = st.form_submit_button(label='Predict')

if submit_button and user_input:
    try:
        input_lines = user_input.strip().split('\n')
        batch_data = []
        for line in input_lines:
            values = [float(x) for x in line.split(',')]
            if len(values) != len(X.columns):
                st.error(f"Each row must have {len(X.columns)} values, but got {len(values)}.")
                st.stop()
            batch_data.append(values)
        batch_df = pd.DataFrame(batch_data, columns=X.columns)

        batch_scaled = scaler.transform(batch_df)
        batch_preds = rf_model.predict(batch_scaled) + 1
        batch_probs = rf_model.predict_proba(batch_scaled)

        st.subheader('Batch Prediction Results')
        batch_results = pd.DataFrame({
            'Predicted Quality': [quality_map[pred] for pred in batch_preds],
            'Waste Probability': batch_probs[:, 0],
            'Acceptable Probability': batch_probs[:, 1],
            'Target Probability': batch_probs[:, 2],
            'Inefficient Probability': batch_probs[:, 3]
        })
        st.write(batch_results)

        st.subheader('Prediction Statistics')
        quality_counts = batch_results['Predicted Quality'].value_counts()
        st.write('Count of Each Predicted Quality:')
        st.write(quality_counts)

        fig, ax = plt.subplots(figsize=(6, 4))
        quality_counts.plot(kind='bar', ax=ax, color='skyblue')
        ax.set_title('Distribution of Predicted Qualities')
        ax.set_xlabel('Quality')
        ax.set_ylabel('Count')
        plt.xticks(rotation=45)
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Error processing input: {e}")
        st.write("Please ensure your input is correctly formatted (comma-separated numerical values).")
