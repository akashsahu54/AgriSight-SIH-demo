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

# Confidence thresholds
CONF_HIGH = 0.70  # 70%
CONF_MEDIUM = 0.50 # 50%

# --- Helper Functions ---

# Load the AI model (cached so it only loads once)
@st.cache_resource
def load_agrisight_model():
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
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
        # Convert JSON string keys back to integers
        return {int(k): v for k, v in labels.items()}
    except Exception as e:
        st.error(f"Error loading labels file: {e}")
        return None

# Preprocess the image for the model
def preprocess_image(image):
    image = image.resize(IMAGE_SIZE)
    image_array = np.array(image)
    image_array = image_array / 255.0  # Rescale
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array

# Layer 2: Simulate Environmental Risk (Improved for Demo)
def simulate_environmental_risk(disease_name):
    # This function simulates the environmental risk.
    
    # Let's create a 50/50 chance for High or Low risk
    if random.choice([True, False]):
        # Simulate HIGH RISK conditions
        temp = f"{random.randint(12, 22)}°C"
        humidity = f"{random.randint(75, 95)}%"
        risk_score = random.uniform(0.8, 0.99)
        risk_level = "High"
    else:
        # Simulate LOW RISK conditions
        temp = f"{random.randint(25, 35)}°C"
        humidity = f"{random.randint(40, 60)}%"
        risk_score = random.uniform(0.05, 0.25)
        risk_level = "Low"
    
    # --- DEMO-ONLY LOGIC ---
    # To demonstrate our innovation, we will FORCE a high-risk scenario
    # every time a 'healthy' leaf is uploaded.
    if disease_name == 'healthy':
        temp = f"{random.randint(12, 22)}°C"
        humidity = f"{random.randint(75, 95)}%"
        risk_score = random.uniform(0.8, 0.99)
        risk_level = "High"
    # --- END DEMO-ONLY LOGIC ---

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

    # --- Rule 1: CONFIRMED HIGH-RISK ---
    # Strong disease confidence + High environment risk
    if disease != 'healthy' and confidence >= CONF_MEDIUM and env_risk == 'High':
        return (
            "CONFIRMED HIGH-RISK",
            f"**Insight:** Both layers agree — Disease: **{disease}** (Confidence: {confidence:.1%}), "
            f"and environmental risk is **High**. Immediate preventive action recommended."
        )

    # --- Rule 2: ISOLATED CASE DETECTED ---
    # Disease found, but environment not supporting spread
    elif disease != 'healthy' and confidence >= CONF_MEDIUM and env_risk == 'Low':
        return (
            "ISOLATED CASE DETECTED",
            f"**Insight:** Disease **{disease}** detected (Confidence: {confidence:.1%}), "
            "but environmental conditions are Low. Localized infection only — monitor closely."
        )

    # --- Rule 3: UNCERTAIN DETECTION ---
    # Weak disease confidence, regardless of risk
    elif disease != 'healthy' and confidence < CONF_MEDIUM:
        return (
            "UNCERTAIN DETECTION",
            f"**Insight:** Possible **{disease}**, but confidence is only {confidence:.1%}. "
            "AI is uncertain — upload a clearer image for re-analysis."
        )

    # --- Rule 4: PRE-SYMPTOMATIC ALERT ---
    # Visually healthy but high-risk conditions
    elif disease == 'healthy' and env_risk == 'High':
        return (
            "PRE-SYMPTOMATIC ALERT",
            f"**Insight:** Crop appears healthy (Confidence: {confidence:.1%}), "
            "but environment shows High risk. Early warning — inspect proactively."
        )

    # --- Rule 5: CROP IS HEALTHY (Default Safe Case) ---
    else:
        return (
            "CROP IS HEALTHY",
            f"**Insight:** Crop looks healthy (Confidence: {confidence:.1%}), "
            "and conditions are safe (Low risk)."
        )

# --- Streamlit UI ---

st.set_page_config(page_title="AgriSight Demo", layout="wide")

# Title and header
st.title("🌾 AgriSight: Predictive Wheat Disease Diagnosis")
st.markdown("A Digital Agronomist for proactive crop protection.")

# Load Model and Labels
model = load_agrisight_model()
labels_map = load_labels()

if model is None or labels_map is None:
    st.stop()

# File Uploader
uploaded_file = st.file_uploader("Upload an image of a wheat leaf...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    
    # Create two columns for image and analysis
    col_img, col_analysis = st.columns([0.4, 0.6])
    
    with col_img:
        st.image(image, caption='Uploaded Image', use_column_width=True)

    # Analyze button
    if st.button('Analyze', use_container_width=True, type="primary"):
        with st.spinner('Running Two-Layer Analysis...'):
            
            # --- Layer 1: Visual Analysis ---
            processed_image = preprocess_image(image)
            prediction = model.predict(processed_image, verbose=0)
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
            with col_analysis:
                st.header("Analysis Report")
                st.divider()
                
                c1, c2 = st.columns(2)
                with c1:
                    st.subheader("Layer 1: Visual Diagnosis")
                    st.metric(label="Detected Disease", value=visual_result['disease'])
                    st.metric(label="Confidence Score", value=f"{visual_result['confidence_score']:.2%}")

                with c2:
                    st.subheader("Layer 2: Environmental Risk")
                    st.metric(label="Simulated Condition", value=env_result['risk_level'], 
                              help=f"Avg Temp: {env_result['temperature']}, Avg Humidity: {env_result['humidity']}")
                    st.metric(label="Risk Score", value=env_result['risk_score'])
                
                st.divider()

                # Final Insight Box
                if "ALERT" in final_status:
                    st.warning(f"### 🚨 {final_status}")
                elif "CONFIRMED" in final_status:
                    st.error(f"### 🔴 {final_status}")
                elif "ISOLATED" in final_status:
                    st.info(f"### ℹ️ {final_status}")
                elif "UNCERTAIN" in final_status:
                    st.info(f"### ⚠️ {final_status}") # Use a different icon
                else:
                    st.success(f"### ✅ {final_status}")
                
                st.markdown(final_insight)
else:
    st.info("Please upload an image to begin the demo.")

