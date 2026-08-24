import streamlit as st
import tensorflow as tf
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. PAGE SETTINGS
# ============================================================

st.set_page_config(
    page_title="Chest X-Ray Classifier",
    page_icon="🫁",
    layout="centered"
)


# ============================================================
# 2. FIND PROJECT FOLDER
# ============================================================

project_folder = (
    Path(__file__).resolve().parent
)


# ============================================================
# 3. FIND FINAL MODEL
# ============================================================

model_files = list(
    project_folder.rglob(
        "xray_classifier_mobilenet.keras"
    )
)


if len(model_files) == 0:

    st.error(
        "Could not find the final MobileNetV2 model."
    )

    st.stop()


model_path = model_files[0]


# ============================================================
# 4. LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    return tf.keras.models.load_model(
        model_path
    )


model = load_model()


# ============================================================
# 5. FIND MOBILENETV2 BASE MODEL
# ============================================================

base_model = None

for layer in model.layers:

    if isinstance(
        layer,
        tf.keras.Model
    ):

        base_model = layer

        break


if base_model is None:

    st.error(
        "Could not find MobileNetV2 base model."
    )

    st.stop()


# ============================================================
# 6. FIND LAST CONVOLUTIONAL LAYER
# ============================================================

last_conv_layer = None

for layer in base_model.layers:

    if isinstance(
        layer,
        tf.keras.layers.Conv2D
    ):

        last_conv_layer = layer


if last_conv_layer is None:

    st.error(
        "Could not find the last convolutional layer."
    )

    st.stop()


# ============================================================
# 7. CREATE GRAD-CAM MODEL
# ============================================================

grad_model = tf.keras.Model(

    inputs=base_model.input,

    outputs=[
        last_conv_layer.output,
        base_model.output
    ]
)


# ============================================================
# 8. GRAD-CAM FUNCTION
# ============================================================

def create_grad_cam(image):

    # --------------------------------------------------------
    # Convert NumPy array → Tensor
    # --------------------------------------------------------

    image_tensor = tf.convert_to_tensor(
        image,
        dtype=tf.float32
    )


    # --------------------------------------------------------
    # Add batch dimension
    # --------------------------------------------------------

    image_batch = tf.expand_dims(
        image_tensor,
        axis=0
    )


    # --------------------------------------------------------
    # Calculate gradients
    # --------------------------------------------------------

    with tf.GradientTape() as tape:

        conv_output, features = (
            grad_model(
                image_batch,
                training=False
            )
        )


        # Final classifier layer

        classifier_output = model.layers[-1](
            features
        )


        pneumonia_score = (
            classifier_output[:, 0]
        )


    gradients = tape.gradient(
        pneumonia_score,
        conv_output
    )


    # --------------------------------------------------------
    # Average gradients
    # --------------------------------------------------------

    pooled_gradients = (
        tf.reduce_mean(
            gradients,
            axis=(1, 2)
        )
    )


    # --------------------------------------------------------
    # Remove batch dimension
    # --------------------------------------------------------

    conv_output = conv_output[0]

    pooled_gradients = (
        pooled_gradients[0]
    )


    # --------------------------------------------------------
    # Weight feature maps
    # --------------------------------------------------------

    weighted_features = (
        conv_output
        * pooled_gradients
    )


    # --------------------------------------------------------
    # Combine feature maps
    # --------------------------------------------------------

    heatmap = tf.reduce_sum(
        weighted_features,
        axis=-1
    )


    # --------------------------------------------------------
    # Keep positive influence
    # --------------------------------------------------------

    heatmap = tf.maximum(
        heatmap,
        0
    )


    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    max_value = tf.reduce_max(
        heatmap
    )

    heatmap = (
        heatmap
        / (max_value + 1e-8)
    )


    # --------------------------------------------------------
    # Resize
    # --------------------------------------------------------

    heatmap = tf.image.resize(
        heatmap[..., tf.newaxis],
        (224, 224)
    )


    return (
        heatmap
        .numpy()
        .squeeze()
    )


# ============================================================
# 9. TITLE
# ============================================================

st.title(
    "🫁 Chest X-Ray Classifier"
)


st.write(
    "Upload a chest X-ray and the trained "
    "MobileNetV2 model will classify it as "
    "**NORMAL** or **PNEUMONIA**."
)


# ============================================================
# 10. MODEL STATUS
# ============================================================

st.success(
    "✅ Final MobileNetV2 model loaded"
)


# ============================================================
# 11. UPLOAD SECTION
# ============================================================

st.subheader(
    "📤 Upload an X-ray"
)


uploaded_file = st.file_uploader(

    "Choose a chest X-ray image",

    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)


# ============================================================
# 12. PROCESS UPLOADED IMAGE
# ============================================================

if uploaded_file is not None:

    # --------------------------------------------------------
    # Load image
    # --------------------------------------------------------

    image = tf.keras.utils.load_img(

        uploaded_file,

        color_mode="grayscale",

        target_size=(224, 224)
    )


    # --------------------------------------------------------
    # Display uploaded image
    # --------------------------------------------------------

    st.image(

        image,

        caption="Uploaded X-ray",

        width="stretch"
    )


    # --------------------------------------------------------
    # Convert image → NumPy array
    # --------------------------------------------------------

    image_array = (
        tf.keras.utils.img_to_array(
            image
        )
    )


    # --------------------------------------------------------
    # IMPORTANT:
    # Convert NumPy → Tensor
    # --------------------------------------------------------

    image_tensor = tf.convert_to_tensor(

        image_array,

        dtype=tf.float32
    )


    # --------------------------------------------------------
    # Convert grayscale → RGB
    # --------------------------------------------------------

    rgb_image = (
        tf.image.grayscale_to_rgb(
            image_tensor
        )
    )


    # --------------------------------------------------------
    # MobileNetV2 preprocessing
    # --------------------------------------------------------

    processed_image = (

        tf.keras.applications
        .mobilenet_v2
        .preprocess_input(
            rgb_image
        )
    )


    # --------------------------------------------------------
    # Add batch dimension
    # --------------------------------------------------------

    processed_batch = tf.expand_dims(

        processed_image,

        axis=0
    )


    # ========================================================
    # 13. PREDICTION
    # ========================================================

    prediction_result = model.predict(

        processed_batch,

        verbose=0
    )


    probability = float(
        prediction_result[0][0]
    )


    # ========================================================
    # 14. DECISION THRESHOLD
    # ========================================================

    threshold = 0.50


    if probability >= threshold:

        prediction = "PNEUMONIA"

    else:

        prediction = "NORMAL"


    # ========================================================
    # 15. DISPLAY PREDICTION
    # ========================================================

    st.divider()

    st.subheader(
        "🔍 Prediction"
    )


    if prediction == "PNEUMONIA":

        st.error(
            f"Prediction: {prediction}"
        )

    else:

        st.success(
            f"Prediction: {prediction}"
        )


    # ========================================================
    # 16. PROBABILITY
    # ========================================================

    st.write(

        f"**Pneumonia probability:** "
        f"{probability:.2%}"

    )


    # ========================================================
    # 17. PROBABILITY BAR
    # ========================================================

    st.progress(
        probability
    )


    # ========================================================
    # 18. THRESHOLD
    # ========================================================

    st.write(

        f"**Decision threshold:** "
        f"{threshold:.2f}"

    )


    # ========================================================
    # 19. GRAD-CAM
    # ========================================================

    heatmap = create_grad_cam(

        processed_image
    )


    # ========================================================
    # 20. GRAD-CAM SECTION
    # ========================================================

    st.divider()

    st.subheader(
        "🔥 Model Attention"
    )


    st.write(

        "Grad-CAM highlights regions that "
        "contributed more strongly to the "
        "model's pneumonia prediction."

    )


    # ========================================================
    # 21. CREATE GRAD-CAM FIGURE
    # ========================================================

    fig, ax = plt.subplots(

        figsize=(7, 7)
    )


    # Original X-ray

    ax.imshow(

        image_array.squeeze(),

        cmap="gray"
    )


    # Heatmap

    ax.imshow(

        heatmap,

        cmap="jet",

        alpha=0.45
    )


    ax.axis("off")


    ax.set_title(
        "Grad-CAM Heatmap"
    )


    # ========================================================
    # 22. DISPLAY GRAD-CAM
    # ========================================================

    st.pyplot(
        fig
    )


    plt.close(
        fig
    )


# ============================================================
# 23. ABOUT THE MODEL
# ============================================================

st.divider()

st.subheader(
    "🧠 About the Model"
)


st.write(

    "This application uses a pretrained "
    "MobileNetV2 convolutional neural network "
    "adapted for chest X-ray classification."

)


st.write(
    "The model classifies images into:"
)


st.write(
    "• NORMAL\n"
    "• PNEUMONIA"
)


# ============================================================
# 24. MODEL INFORMATION
# ============================================================

col1, col2, col3 = st.columns(3)


with col1:

    st.metric(
        "Architecture",
        "MobileNetV2"
    )


with col2:

    st.metric(
        "Input",
        "224 × 224 RGB"
    )


with col3:

    st.metric(
        "Threshold",
        "0.50"
    )


# ============================================================
# 25. TEST PERFORMANCE
# ============================================================

st.subheader(
    "📊 Test Performance"
)


metric_col1, metric_col2 = (
    st.columns(2)
)


with metric_col1:

    st.metric(
        "Accuracy",
        "85.58%"
    )

    st.metric(
        "Precision",
        "84.25%"
    )


with metric_col2:

    st.metric(
        "Recall",
        "94.62%"
    )

    st.metric(
        "F1-Score",
        "89.13%"
    )


# ============================================================
# 26. CONFUSION MATRIX SUMMARY
# ============================================================

st.write(
    "**Test-set error analysis:**"
)


error_col1, error_col2 = (
    st.columns(2)
)


with error_col1:

    st.metric(
        "False Positives",
        "69"
    )


with error_col2:

    st.metric(
        "False Negatives",
        "21"
    )


# ============================================================
# 27. DISCLAIMER
# ============================================================

st.divider()


st.warning(

    "⚠️ Educational/research project only. "
    "This model is not a medical diagnostic tool "
    "and should not be used for clinical decisions."

)


# ============================================================
# 28. FOOTER
# ============================================================

st.caption(

    "Medical X-Ray Classifier • "
    "MobileNetV2 • Computer Vision Project"

)


# ============================================================
# 29. STEP COMPLETE
# ============================================================

print(
    "Step 40 complete!"
)