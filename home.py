import streamlit as st
import pandas as pd
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
import tensorflow as tf

from PIL import Image

def run():
    # Main content
    st.title("Welcome!")
    img_url = "home.jpg"
    image = Image.open(img_url)
    st.image(image)

    # Dashboard Introduction
    st.markdown("""
    ### What is this Dashboard?

    This interactive dashboard explains:
    - Insights extracted through **Exploratory Data Analysis** to understand the characteristics of the dataset and the formation of sign language gestures.
    - Prediction of **sign language** from uploaded images.

    ---
    ### Background Problem
    The objectives of this analysis are:
    - To understand the characteristics of the sign language data being processed.
    - To develop a **model with high accuracy, lightweight size, and optimal efficiency** in predicting **sign language**.
    - To perform **sign language classification** and evaluate its correctness.

    ---
    ### Model
    The machine learning algorithm used in this modeling is **CNN within ANN** to predict **American Sign Language**. This model has undergone **hyperparameter tuning** to improve performance. The model achieved an **accuracy of 82%**.
    
    ---
    """)

    # Dataset
    st.markdown("""
    ### Dataset
    The dataset used is sourced from **Kaggle**, titled **[ASL Sign Language Alphabet Pictures [Minus J, Z]](https://www.kaggle.com/datasets/signnteam/asl-sign-language-pictures-minus-j-z)**.
    
    ---
    ### Dataset Preview
                
    """)

    # How to use Dashboard
    st.markdown("""
    ---
    ### How to use the Dashboard
    - Click on the options in the **sidebar** to explore EDA or make predictions.
    - Try uploading various data samples to get different results.

    Thank you for visiting this dashboard. See you again next time!
                
    ---
    """)