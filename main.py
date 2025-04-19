import streamlit as st
from model_helper import predict
import os

st.title("Fruit Damage Detection")

uploaded_file = st.file_uploader("Upload the file", type=["jpg", "png"])

if uploaded_file:
    image_path = "temp_file.jpg"
    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
        f.flush()
        os.fsync(f.fileno())
        st.image(uploaded_file, caption="Uploaded File", use_container_width=True)
        prediction = predict(image_path)
        st.info(f"Predicted Class: {prediction}")