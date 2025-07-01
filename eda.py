# import libraries
import streamlit as st
import pandas as pd
from PIL import Image

# isi API
def run():
    # judul
    # Title
    st.title("ASL Hand Sign Classification (A–Y) (EDA Section)")

    # Centered image using columns
    img = Image.open("asl.jpg")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(img, caption=" ", use_container_width=True)

    # deskripsi
    st.markdown('''
    This session presents the Exploratory Data Analysis (EDA) stage in the data processing workflow using a machine learning model.
    The dataset used consists of sign language images that are utilized for training the model.


    # Problem Statement
    The objectives of this analysis are:
    - To understand the characteristics of the sign language data being processed.
    - To develop a **model with high accuracy, lightweight size, and optimal efficiency** in predicting **sign language**.
    - To perform **sign language classification** and evaluate its correctness.
    ''')

    # EDA
    st.markdown("# Dataset")
    st.markdown("The dataset used is sourced from **Kaggle**, titled **[ASL Sign Language Alphabet Pictures [Minus J, Z]](https://www.kaggle.com/datasets/signnteam/asl-sign-language-pictures-minus-j-z)**.")

    # EDA 1
    st.markdown ("## 1. How is the distribution of image sizes across each label?")
    st.markdown ("""
                
The distribution of image sizes was detected using the following barplot visualization:""")
    
    img_url = "eda1.png"
    image = Image.open(img_url)
    st.image(image)

    st.markdown("""
### **Analysis Results**
As seen in the barplot above, the distribution of image sizes is not uniform, indicating the need for resizing so that all images can be processed at the same dimensions. 
This resizing process will be carried out during the model definition and model training stages.
</div>
""",unsafe_allow_html=True)
    st.markdown("---")

    # EDA 2
    st.markdown ("""## 2. How is the distribution of image colors across each label?""")
    
    st.markdown("""
The distribution of image colors was detected using the following barplot visualization:""")

    img_url = "outputwarna.png"
    image = Image.open(img_url)
    st.image(image)

    st.markdown("""
### **Analysis Result**
<style>
    .justified-text {
        text-align: justify;
    }
</style>
<div class="justified-text">
From the barplot above, it can be concluded that the average color characteristics of the images for each label are uniquely distinct. 
However, the overall color proportions across all images are similar, with red being the most dominant, followed by green, and then blue as the least. 
This occurs because the natural color patterns of human hands do not vary significantly; they typically range from lighter to darker skin tones. 
There's no such thing as human skin being green like the animated character in the movie 'Hulk', or blue like the character in the movie 'Avatar'.
</div>
""", unsafe_allow_html=True)
    st.markdown("---")

# EDA 3
    st.markdown ("""
## 3. What Are The Characteristics or Unique Features Of The Sign Language Gestures For Each Label?
""")
    
    img = Image.open("eda3.jpg")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(img, caption=" ", use_container_width=True)

    st.markdown("""
<style>
    .justified-text {
        text-align: justify;
    }
</style>
<div class="justified-text">
To answer this question, I will describe each label’s unique characteristics below:
                
- Letter A
<br> The hand is clenched into a fist, with the thumb positioned alongside the index finger.
                
- Letter B
<br> All fingers except the thumb are extended and held together, while the thumb is bent toward the center of the palm.
                
- Letter C
<br> All fingers (including the thumb) are spread apart and curved to form the shape of the letter ‘C’.

- Letter D
<br> Similar to the ‘C’ sign, but the tip of the thumb touches the tip of the middle finger, and the index finger points upward.
                
- Letter E
<br> Similar to the ‘A’ sign, but all four fingers (except the thumb) are bent upward at a 90-degree angle.
                
- Letter F
<br>All fingers are extended, with the tip of the index finger touching the tip of the thumb.
                
- Letter G
<br>The hand points sideways with the thumb placed next to the index finger.
                
- Letter H
<br>Similar to the ‘G’ sign, but the middle finger is also extended next to the index finger.
                
- Letter I
<br>Similar to the ‘A’ sign, but the pinky finger is raised upward.
                
- Letter K
<br>The hand forms a ‘2’ shape with the index and middle fingers, and the thumb rests between their bases.
                
- Letter L
<br>The hand starts in a fist, then the index finger and thumb are extended to form the shape of the letter ‘L’.
                
- Letter M
<br>Similar to the ‘A’ sign, but the thumb is inserted under the index through to the ring finger, with the tip peeking between the pinky and ring fingers.
                
- Letter N
<br>Similar to the ‘M’ sign, but the thumb peeks between the ring and middle fingers instead.
                
- Letter O
<br>Similar to the ‘C’ sign, but the tip of the thumb meets the tip of the index finger to form an ‘O’ shape.
                
- Letter P
<br>Similar to the ‘G’ sign, but the thumb and index finger form a circle while the palm is rotated downward.
                
- Letter Q
<br>Similar to the ‘C’ sign, but oriented downward.
                
- Letter R
<br>The index and middle fingers form a ‘2’, with the index finger overlapping or crossing the middle finger.
                
- Letter S
<br>Similar to the ‘A’ sign, but the thumb is shifted forward near the middle finger’s nail joint, and the palm faces outward (like preparing a punch).
                
- Letter T
<br>Similar to the ‘N’ sign, but the thumb is inserted between the index and middle fingers.
                
- Letter U
<br>Similar to forming the number ‘2’, but the index and middle fingers are held together.
                
- Letter V
<br>Shaped like the number ‘2’, with the index and middle fingers spread.
                
- Letter W
<br>Shaped like the number ‘3’, with three fingers extended.
                
- Letter X
<br>Looks like the number ‘1’, but the index finger is bent slightly toward the center.
                
- Letter Y
<br>All fingers are curled into a fist, except for the pinky and thumb which are extended outward.

<br>The EDA section is now complete. You may proceed to the prediction section by clicking on the 'ASL Hand Sign Classification (A–Y) Section' in the navigation bar.   
""", unsafe_allow_html=True)
    
    st.markdown("---")
    
if __name__ == '__main__':
    run()
