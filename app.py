
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

MODEL_PATH = "banana_model.tflite"

# Change these labels to match the EXACT class order
# used when training your model in Colab.
CLASS_NAMES = ["Fresh", "Semi-Spoiled", "Spoiled"]

# Change this to match your notebook's preprocessing:
# "0_255"    -> raw pixel values
# "0_1"      -> pixels divided by 255
# "minus1_1" -> pixels scaled to [-1, 1]
INPUT_SCALE = "0_1"

@st.cache_resource
def load_tflite_model():
    interpreter = tf.lite.Interpreter(
        model_path=MODEL_PATH
    )
    interpreter.allocate_tensors()
    return (
        interpreter,
        interpreter.get_input_details(),
        interpreter.get_output_details()
    )

st.set_page_config(
    page_title="Banana Freshness Detector",
    page_icon="🍌"
)

st.title("🍌 Banana Freshness Detector")
st.write(
    "Upload a banana photo to classify its freshness."
)

st.info(
    "This is an experimental model. Predictions may be "
    "incorrect, especially with a small training dataset."
)

try:
    interpreter, input_details, output_details = (
        load_tflite_model()
    )
except Exception as error:
    st.error(f"Could not load the model: {error}")
    st.stop()

input_info = input_details[0]
output_info = output_details[0]

shape = input_info["shape"]

if len(shape) != 4 or shape[0] != 1 or shape[3] != 3:
    st.error(
        f"Unexpected model input shape: {shape}. "
        "Update the image preprocessing for your model."
    )
    st.stop()

height, width = int(shape[1]), int(shape[2])

uploaded_file = st.file_uploader(
    "Upload a banana image",
    type=["jpg", "jpeg", "png", "webp"]
)

if uploaded_file:
    image = Image.open(uploaded_file)
    image = ImageOps.exif_transpose(image).convert("RGB")
    st.image(
        image,
        caption="Uploaded banana",
        use_container_width=True
    )

    if st.button("Predict Freshness"):
        image = image.resize((width, height))
        pixels = np.asarray(image, dtype=np.float32)

        if INPUT_SCALE == "0_1":
            pixels = pixels / 255.0
        elif INPUT_SCALE == "minus1_1":
            pixels = pixels / 127.5 - 1.0
        elif INPUT_SCALE != "0_255":
            st.error("Unknown INPUT_SCALE setting.")
            st.stop()

        batch = np.expand_dims(pixels, axis=0)
        dtype = input_info["dtype"]

        # Support quantized integer input tensors too.
        if np.issubdtype(dtype, np.integer):
            scale, zero_point = input_info["quantization"]
            if scale > 0:
                batch = np.round(batch / scale + zero_point)
            limits = np.iinfo(dtype)
            batch = np.clip(
                batch, limits.min, limits.max
            ).astype(dtype)
        else:
            batch = batch.astype(dtype)

        interpreter.set_tensor(
            input_info["index"], batch
        )
        interpreter.invoke()

        scores = interpreter.get_tensor(
            output_info["index"]
        )[0]

        # Dequantize output if the model uses integer scores.
        out_scale, out_zero = output_info["quantization"]
        if np.issubdtype(output_info["dtype"], np.integer):
            if out_scale > 0:
                scores = (scores.astype(np.float32) - out_zero) * out_scale

        scores = np.asarray(scores).flatten()

        if len(scores) != len(CLASS_NAMES):
            st.error(
                "The number of model outputs does not match "
                "the number of class labels. Check your notebook."
            )
            st.stop()

        # Softmax outputs are probabilities. If the model already
        # returns probabilities, do not apply softmax again.
        predicted_index = int(np.argmax(scores))
        predicted_label = CLASS_NAMES[predicted_index]

        st.subheader("Prediction")
        st.success(f"Result: {predicted_label}")
        st.write(
            f"Model score: {scores[predicted_index] * 100:.2f}%"
        )

        st.subheader("Scores by category")
        for name, score in zip(CLASS_NAMES, scores):
            st.write(f"{name}: {score * 100:.2f}%")
            st.progress(float(np.clip(score, 0, 1)))