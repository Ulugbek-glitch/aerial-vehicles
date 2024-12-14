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
import platform
import pathlib
from pathlib import Path  # Directly use Path, no need for modification

# Set up Streamlit title
st.title("Havo yo'llarida harakatlanuvchi vositalarni klassifikatsiya qiluvchi model")

# File uploader for images
file = st.file_uploader('Upload an image', type=['png', 'jpeg', 'gif', 'svg'])

# Check if the model file exists
model_path = Path('aerial-vehicles.pkl')

# Check for model existence and load if available
if model_path.exists():
    try:
        model = load_learner(model_path)
        st.success("Model loaded successfully!")

        if file:
            try:
                # Load the uploaded image
                img = PILImage.create(file)
                st.image(img, caption="Uploaded Image", use_column_width=True)

                # Make a prediction using the model
                pred, pred_id, probs = model.predict(img)
                st.success(f"Bashorat: {pred}")
                st.info(f"Ehtimollik: {probs[pred_id]*100:.1f}%")

            except Exception as e:
                st.error(f"Error processing the image: {str(e)}")

    except Exception as e:
        st.error(f"Error loading model: {str(e)}")

else:
    st.error(f"Model file not found at {model_path}")
