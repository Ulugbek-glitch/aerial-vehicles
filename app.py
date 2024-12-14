#import streamlit as st
#from fastai.vision.all import *
#import platform
#import pathlib
#from pathlib import Path
#plt=platform.system()
#if plt =='Linux': pathlib.WindowsPath=pathlib.PosixPath
#st.title("Havo yo'llarida harakatlanuvchi vositalarni klassifikatsiya qiluvchi model")
#file=st.file_uploader('Rasm yuklash', type=['png','jpeg','gif','svg'])
#if file:
   #st.image(file)
   #img=PILImage.create(file)
   #model=load_learner('aerial-vehicles.pkl') 
   #pred, pred_id, probs=model.predict(img)
   #st.success(f"Bashorat: {pred}")
   #st.info(f"Ehtimollik: {probs[pred_id]*100:.1f}%")
import streamlit as st
from fastai.vision.all import *
import os

# Title of the Streamlit app
st.title("Test Model Loading")

# Check if the model exists
model_path = 'aerial-vehicles.pkl'

# Print the current working directory
st.write(f"Current working directory: {os.getcwd()}")

if os.path.exists(model_path):
    try:
        # Try loading the model
        model = load_learner(model_path)
        st.success("Model loaded successfully!")

    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
else:
    st.error(f"Model file not found at {model_path}")
