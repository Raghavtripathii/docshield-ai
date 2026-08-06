from pathlib import Path

import streamlit as st
from PIL import Image

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

        if MODEL_DIR.exists():
            st.success("Model found.")
            st.button(
                "Run Inference",
                disabled=True,
                use_container_width=True,
            )
        else:
            st.error("Trained model not found.")
            st.code("outputs/robust_layoutlmv3")
else:
    st.info("Upload a document image.")