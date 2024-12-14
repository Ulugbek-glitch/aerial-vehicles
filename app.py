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
from pathlib import Path
import os

# Title of the Streamlit app
st.title("Test Model Loading")

# Print current working directory to confirm where the model file should be
st.write("Current working directory:", os.getcwd())

# Test if the model can be loaded with the correct path
model_path = Path('aerial-vehicles.pkl')

# Check if the model exists
if model_path.exists():
    try:
        model = load_learner(model_path)
        st.success("Model loaded successfully!")
        
        # Once the model is loaded, let's check if prediction works
        if st.button('Predict'):
            img = PILImage.create(st.file_uploader('Upload an image', type=['png', 'jpeg', 'gif', 'svg']))
            pred, pred_id, probs = model.predict(img)
            st.success(f"Bashorat: {pred}")
            st.info(f"Ehtimollik: {probs[pred_id]*100:.1f}%")
            
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
else:
    st.error(f"Model file not found at {model_path}")

