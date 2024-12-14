#import streamlit as st
#from fastai.vision.all import *
#from pathlib import Path

#st.title("Havo yo'llarida harakatlanuvchi vositalarni klassifikatsiya qiluvchi model")
#file=st.file_uploader('Rasm yuklash', type=['png','jpeg','gif','svg'])
#if file:
   st.image(file)
   img=PILImage.create(file)
   model=load_learner('aerial-vehicles.pkl') 
   pred, pred_id, probs=model.predict(img)
   st.success(f"Bashorat: {pred}")
   st.info(f"Ehtimollik: {probs[pred_id]*100:.1f}%")

import streamlit as st
from fastai.vision.all import *
from pathlib import Path  # Use Path instead of WindowsPath or PosixPath

# Title of the Streamlit app
st.title("Havo yo'llarida harakatlanuvchi vositalarni klassifikatsiya qiluvchi model")

# File uploader for images
file = st.file_uploader('Rasm yuklash', type=['png', 'jpeg', 'gif', 'svg'])

if file:
    # Display uploaded image
    st.image(file)

    # Create a PIL image from the file using file.getvalue()
    img = PILImage.create(file.getvalue())  # Use getvalue() to convert file to bytes

    # Load the fastai learner model
    model_path = Path('aerial-vehicles.pkl')  # Use Path to make cross-platform paths
    if not model_path.exists():
        st.error(f"Model file not found at {model_path}")
    else:
        model = load_learner(model_path)

        # Make a prediction using the model
        pred, pred_id, probs = model.predict(img)

        # Display the prediction and probability
        st.success(f"Bashorat: {pred}")
        st.info(f"Ehtimollik: {probs[pred_id]*100:.1f}%")
