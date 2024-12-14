import streamlit as st
from fastai.vision.all import *
import pathlib
plt=platform.system()
if plt == 'Linux': pathlib.WindowsPath=pathlib.PosixPath

st.title("Havo yo'llarida harakatlanuvchi vositalarni klassifikatsiya qiluvchi model")
file=st.file_uploader('Rasm yuklash', type=['png','jpeg','gif','svg'])
if file:
   st.image(file)
   img=PILImage.create(file)
   model=load_learner('aerial-vehicles.pkl') 
   pred, pred_id, probs=model.predict(img)
   st.success(f"Bashorat: {pred}")
   st.info(f"Ehtimollik: {probs[pred_id]*100:.1f}%")
