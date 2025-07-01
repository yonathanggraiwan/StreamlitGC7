import streamlit as st
import eda, predict, home

st.set_page_config(page_title = "ASL Hand Sign Classification (A–Y)",
                   layout = 'centered',
                   initial_sidebar_state = 'expanded')
with st.sidebar:
    st.write('# Navigation Sidebar')
    navigation = st.radio('Page', ['Home', 
                                   'Exploratory Data Analysis (EDA) Section', 
                                   'ASL Hand Sign Classification (A–Y) Section'])

if navigation == 'Exploratory Data Analysis (EDA) Section':
    eda.run()

if navigation == 'ASL Hand Sign Classification (A–Y) Section':
    predict.run()

if navigation == 'Home':
    home.run()
    
else:
    home.run()