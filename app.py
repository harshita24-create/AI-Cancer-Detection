import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="NeuroScan AI | Brain Tumor Detection",
    page_icon="🧠",
    layout="wide"
)

# --------------------------------------------------
# Professional Healthcare UI Styling
# --------------------------------------------------
st.markdown("""
<style>
    .main {
        padding-top: 1rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }

    .main-title {
        font-size: 40px;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 5px;
    }

    .sub-title {
        font-size: 18px;
        color: #475569;
        margin-bottom: 25px;
    }

    .info-card {
        background-color: #F8FAFC;
        padding: 20px;
        border-radius: 14px;
        border: 1px solid #E2E8F0;
        margin-bottom: 15px;
    }

    .section-title {
        font-size: 24px;
        font-weight: 600;
        margin-top: 20px;
        margin-bottom: 10px;
        color: #1E293B;
    }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Load Model
# --------------------------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model/brain_tumor_model.h5")

model = load_model()
IMG_SIZE = 224

# --------------------------------------------------
# Header Section
# --------------------------------------------------
st.markdown('<div class="main-title">🧠 NeuroScan AI</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">AI-Based Early Brain Tumor Detection using Deep CNN and MRI Image Analysis</div>',
    unsafe_allow_html=True
)

# --------------------------------------------------
# Sidebar (Medical Dashboard Style)
# --------------------------------------------------
st.sidebar.title("Hospital Dashboard")
st.sidebar.markdown("### Project Overview")
st.sidebar.write("**System:** Deep Learning Diagnostic Assistant")
st.sidebar.write("**Model Used:** EfficientNetB0 + CNN")
st.sidebar.write("**Imaging Type:** MRI Brain Scan")
st.sidebar.write("**Detection Type:** Tumor / No Tumor")
st.sidebar.write("**Deployment:** Streamlit Clinical Interface")

st.sidebar.markdown("---")
st.sidebar.markdown("### Clinical Note")
st.sidebar.info(
    "This system is developed for academic research and early screening assistance. Final diagnosis must be confirmed by certified radiologists and neurologists."
)

# --------------------------------------------------
# Main Layout
# --------------------------------------------------
left_col, right_col = st.columns([1.2, 1])

with left_col:
    st.markdown('<div class="section-title">Patient MRI Upload</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload Brain MRI Scan",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        st.image(
            uploaded_file,
            caption="Uploaded MRI Brain Scan",
            use_container_width=True
        )

with right_col:
    st.markdown('<div class="section-title">Diagnostic Workflow</div>', unsafe_allow_html=True)

    st.markdown("""
<div class="info-card">
<b>Step 1:</b> Upload MRI scan image<br><br>
<b>Step 2:</b> AI preprocesses scan using grayscale normalization<br><br>
<b>Step 3:</b> Deep CNN evaluates tumor probability<br><br>
<b>Step 4:</b> System returns prediction + confidence score
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Prediction Logic
# --------------------------------------------------
if uploaded_file is not None:
    img = image.load_img(
        uploaded_file,
        target_size=(IMG_SIZE, IMG_SIZE),
        color_mode="grayscale"
    )

    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    st.markdown('<div class="section-title">AI Diagnosis</div>', unsafe_allow_html=True)

    if st.button("Run Diagnostic Prediction"):
        with st.spinner("Analyzing MRI scan using Deep CNN model..."):
            prediction = model.predict(img_array)
            confidence = float(prediction[0][0]) * 100

            st.subheader("Prediction Report")

            if prediction[0][0] > 0.5:
                st.error("🔴 Positive Finding: Brain Tumor Detected")
                st.write(f"### Confidence Score: {confidence:.2f}%")
                st.warning(
                    "Recommendation: Immediate specialist consultation and radiological validation advised."
                )
            else:
                st.success("🟢 Negative Finding: No Tumor Detected")
                st.write(f"### Confidence Score: {100 - confidence:.2f}%")
                st.info(
                    "Recommendation: No abnormal tumor pattern identified in uploaded scan."
                )

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.markdown("---")
st.caption(
    "Developed for AI/ML Internship Project | Brain Tumor Detection using Deep Learning"
)
