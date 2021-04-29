import streamlit as st
from image_classification import teachable_machine_classification
from PIL import Image, ImageOps
import numpy as np



st.title("Rice Leaf Diseases Classification")
st.header("Rice Leaf Disease?")
st.text("Upload a Rice Leaf Image for disease classification")


uploaded_file = st.file_uploader("Choose a leaf image ...", type="jpg")

if uploaded_file is not None:
	image = Image.open(uploaded_file)
	st.image(image, caption='Uploaded leaf image', use_column_width=True)
	st.write("")
	st.write("Classifying...")
	label = teachable_machine_classification(image, 'rice_leaf_classification.h5')
	if label == 0:
		st.write("The leaf has Bacterial Blight ")
	elif label == 1:
		st.write("The leaf has Brown Spot")
	else:
		st.write("The leaf has Leaf Smut")