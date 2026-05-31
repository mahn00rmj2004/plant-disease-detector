import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# 1. Page Config must be FIRST
st.set_page_config(page_title="Plant Disease Detector", page_icon="🌿")

# 2. CSS Styling
st.markdown(
    """
    <style>
    .header {
        background-color: #2e7d32 !important; 
        padding: 20px !important;
        color: white !important;
        text-align: center;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .header h1 {
        color: #ffffff !important;
        margin: 0 !important;
    }
    .footer {
        background-color: #2e7d32 !important;
        padding: 10px !important;
        color: white !important;
        text-align: center;
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        z-index: 999;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# 3. RENDER THE HEADER (This calls the CSS class)
st.markdown('<div class="header"><h1>🌿 Plant Health Scanner</h1></div>', unsafe_allow_html=True)

# Load the model with cache
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('best_model.keras')

# Initialize model
try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

CLASS_NAMES = ['Healthy', 'Powdery', 'Rust']

st.write("Upload a leaf image to detect disease.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        img = image.resize((256, 256))
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0) 
        
        with st.spinner("Analyzing..."):
            predictions = model.predict(img_array, verbose=0)
        
        predicted_class = CLASS_NAMES[np.argmax(predictions)]
        confidence = np.max(predictions) * 100
        
        if predicted_class == "Healthy":
            st.success(f"### Prediction: {predicted_class}")
        else:
            st.error(f"### Prediction: {predicted_class}")
            
        st.info(f"**Confidence:** {confidence:.2f}%")
        
        st.write("---")
        st.write("### Class Probabilities")
        for i, name in enumerate(CLASS_NAMES):
            prob = float(predictions[0][i])
            st.write(f"{name}:")
            st.progress(prob)
            
    except Exception as e:
        st.error(f"An error occurred during processing: {e}")

# 4. RENDER THE FOOTER (This calls the CSS class)
st.markdown('<div class="footer">Developed with Streamlit & TensorFlow</div>', unsafe_allow_html=True)