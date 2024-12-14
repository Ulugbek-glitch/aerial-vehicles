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
from pathlib import Path
from fastai.vision.all import *

# Title of the Streamlit app
st.title("Test Image Upload and Path Handling")

# File uploader for images
file = st.file_uploader('Upload an image', type=['png', 'jpeg', 'gif', 'svg'])

if file:
    st.image(file)
    
    # Try loading the image as a PIL image
    try:
        img = PILImage.create(file)
        st.success("Image loaded successfully!")
    except Exception as e:
        st.error(f"Error loading image: {str(e)}")

