import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import json
import random

# --- Configuration ---
MODEL_PATH = 'wheat_disease_model.h5'
LABELS_PATH = 'labels.json'
IMAGE_SIZE = (224, 224)

# --- Helper Functions ---

# Load the AI model (with caching to prevent reloading)
@st.cache_resource
def load_agrisight_model():
    try:
        # Load the model without compiling. This can prevent some warnings.
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        # Re-compile the model just in case, to be safe.
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Load the class labels
@st.cache_data
def load_labels():
    try:
        with open(LABELS_PATH, 'r') as f:
            labels = json.load(f)
        # Convert JSON string keys to integer
        return {int(k): v for k, v in labels.items()}
    except Exception as e:
        st.error(f"Error loading labels file: {e}")
        return None

# Preprocess the image for the model
def preprocess_image(image):
    image = image.resize(IMAGE_SIZE)
    image_array = np.array(image)
    image_array = image_array / 255.0  # Rescale to [0, 1]
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array

# Layer 2: Simulate Environmental Risk
def simulate_environmental_risk(disease_name):
    # This function simulates the output of a real weather API and LSTM model
    if disease_name in ["Yellow_Rust", "Brown_Rust", "Septoria", "Mildew"]:
        # Favourable conditions for fungal diseases
        temp = f"{random.randint(12, 22)}°C"
        humidity = f"{random.randint(75, 95)}%"
        risk_score = random.uniform(0.8, 0.99)
        risk_level = "High"
    else: # Healthy
        temp = f"{random.randint(25, 35)}°C"
        humidity = f"{random.randint(40, 60)}%"
        risk_score = random.uniform(0.05, 0.25)
        risk_level = "Low"
    
    return {
        "temperature": temp,
        "humidity": humidity,
        "risk_score": f"{risk_score:.0%}",
        "risk_level": risk_level
    }

# Fusion Engine: Combine results from both layers
def get_fusion_insight(visual_result, env_result):
    disease = visual_result['disease']
    confidence = visual_result['confidence_score']
    env_risk = env_result['risk_level']

    if disease != 'Healthy' and confidence > 0.7 and env_risk == 'High':
        return "CONFIRMED HIGH-RISK", f"**Insight:** Both visual and environmental data confirm a high risk of **{disease}**. Conditions are perfect for its spread. Immediate action is recommended."
    elif disease == 'Healthy' and env_risk == 'High':
        return "PRE-SYMPTOMATIC ALERT", f"**Insight:** No symptoms are visible, but the environmental conditions are **highly favorable** for a fungal outbreak. We recommend proactive inspection of the field."
    elif disease != 'Healthy' and confidence > 0.6 and env_risk == 'Low':
        return "ISOLATED CASE DETECTED", f"**Insight:** Symptoms of **{disease}** detected, but weather is not favorable for a widespread outbreak. Monitor this specific area closely."
    else:
        return "CROP IS HEALTHY", "**Insight:** The crop appears healthy and environmental conditions are currently safe."

# --- Streamlit UI ---

st.set_page_config(page_title="AgriSight Demo", layout="wide")

# Title and header
st.title("🌾 AgriSight: Predictive Wheat Disease Diagnosis")
st.markdown("Your Digital Agronomist - Insights before the outbreak.")

# Load model and labels
model = load_agrisight_model()
labels_map = load_labels()

if model is None or labels_map is None:
    st.stop()

# File Uploader
uploaded_file = st.file_uploader("Upload an image of a wheat leaf...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the image
    image = Image.open(uploaded_file).convert("RGB")
    
    col_img, col_spacer = st.columns([1, 2])
    with col_img:
        st.image(image, caption='Uploaded Image', use_column_width=True)

    # Analyze button
    if st.button('Analyze Image', use_container_width=True, type="primary"):
        with st.spinner('Analysis in progress...'):
            
            # --- Layer 1: Visual Analysis ---
            processed_image = preprocess_image(image)
            prediction = model.predict(processed_image)
            confidence_score = np.max(prediction)
            predicted_class_index = np.argmax(prediction)
            disease_name = labels_map.get(predicted_class_index, "Unknown")

            visual_result = {
                "disease": disease_name,
                "confidence_score": confidence_score
            }

            # --- Layer 2: Environmental Analysis (Simulated) ---
            env_result = simulate_environmental_risk(disease_name)

            # --- Fusion Engine ---
            final_status, final_insight = get_fusion_insight(visual_result, env_result)
            
            # --- Display Results ---
            st.divider()
            st.header("Analysis Report")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Layer 1: Visual Diagnosis (On-Device)")
                st.metric(label="Detected Disease", value=visual_result['disease'])
                st.metric(label="Confidence Score", value=f"{visual_result['confidence_score']:.2%}")

            with col2:
                st.subheader("Layer 2: Environmental Risk (Simulated)")
                st.metric(label="Environmental Condition", value=env_result['risk_level'], help=f"Avg Temp: {env_result['temperature']}, Avg Humidity: {env_result['humidity']}")
                st.metric(label="Risk Score", value=env_result['risk_score'])
            
            st.divider()

            # Final Insight
            if "ALERT" in final_status:
                st.warning(f"### 🚨 {final_status}")
            elif "CONFIRMED" in final_status:
                st.error(f"### 🔴 {final_status}")
            else:
                st.success(f"### ✅ {final_status}")
            
            st.markdown(final_insight)
else:
    st.info("Please upload an image to begin the analysis.")

