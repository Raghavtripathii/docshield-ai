from pathlib import Path

import streamlit as st
from PIL import Image

from src.inference import InferenceEngine

MODEL_DIR = Path("outputs/robust_layoutlmv3")

st.set_page_config(
    page_title="DocShield AI",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ DocShield AI")

uploaded_file = st.file_uploader(
    "Upload a document",
    type=["png", "jpg", "jpeg"],
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    left, right = st.columns(2)

    with left:
        st.image(
            image,
            use_container_width=True,
        )

    with right:

        if not MODEL_DIR.exists():
            st.error("Model not found.")
            st.stop()

        engine = InferenceEngine()

        if st.button(
            "Run Inference",
            use_container_width=True,
        ):

            with st.spinner(
                "Running AI inference..."
            ):

                result = engine.predict_image(
                    image
                )

            st.success("Inference completed.")

            st.subheader("Predictions")

            for word, label, score in zip(
                result["words"],
                result["labels"],
                result["scores"],
            ):

                st.write(
                    f"**{word}** → {label} ({score:.2%})"
                )

else:
    st.info("Upload a document image.")