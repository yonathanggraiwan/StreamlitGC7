import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image
import random

# EDA 1
def eda1(base_path='./data/'):
    """
    Visualizes the average image area (width × height) per label as a barplot.
    """
    filepaths, labels = [], []

    # Collect file paths and labels
    for label in os.listdir(base_path):
        label_path = os.path.join(base_path, label)
        if os.path.isdir(label_path):
            for fname in os.listdir(label_path):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                    filepaths.append(os.path.join(label_path, fname))
                    labels.append(label)

    df = pd.DataFrame({'filepath': filepaths, 'label': labels})

    # Compute image area
    def get_area(path):
        try:
            with Image.open(path) as img:
                return img.width * img.height
        except:
            return None

    df['area'] = df['filepath'].apply(get_area)
    df.dropna(subset=['area'], inplace=True)

    if df.empty:
        st.warning("No valid images found in the dataset directory.")
        return

    # Group mean area per label
    mean_area = df.groupby('label')['area'].mean().reset_index()

    # Turn off matplotlib’s interactive mode to prevent automatic plotting
    plt.ioff()

    # Generate the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='label', y='area', data=mean_area, ax=ax)
    ax.set_title('Average Image Area per Label')
    ax.set_xlabel('Label')
    ax.set_ylabel('Average Area (pixels)')

    return(fig)

def vishome(base_path='./data/'):
    label_folders = sorted([label for label in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, label))])

    num_labels = len(label_folders)
    cols = 6
    rows = (num_labels + cols - 1) // cols  # ceiling division

    plt.figure(figsize=(15, 2.5 * rows))

    for idx, label in enumerate(label_folders):
        folder_path = os.path.join(base_path, label)
        image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        if image_files:
            sample_file = random.choice(image_files)
            image_path = os.path.join(folder_path, sample_file)
            try:
                img = Image.open(image_path)
                plt.subplot(rows, cols, idx + 1)
                plt.imshow(img)
                plt.title(f'Label: {label}')
                plt.axis('off')
            except Exception as e:
                print(f"Could not load {image_path}: {e}")

    plt.tight_layout()
    plt.show()

# Run in Streamlit
if __name__ == "__main__":
    run