import streamlit as st
from fastai.vision.all import *
import pathlib
import plotly.express as px

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath


#title of the app
st.title("Image Classification with FastAI")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = PILImage.create(uploaded_file)
    st.image(img, caption="Uploaded Image", use_container_width=True)
    st.write("")
    st.write("Classifying...")
    print(type(img))
    # Or if it's a tuple or dict:
    print(img)

    model = load_learner('transport_module_2_7_19.pkl')

    # predict
    pred, pred_id, probs = model.predict(img)
    st.success(f'Bashorat: {pred}')
    st.info(f'Ehtimollik: {probs[pred_id]*100:.2f}%')

    fig = px.bar(x=probs*100, y=model.dls.vocab)
    st.plotly_chart(fig)