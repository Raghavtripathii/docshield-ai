from pathlib import Path
import json
import streamlit as st
from PIL import Image
from src.inference import InferenceEngine
from src.visualizer import PredictionVisualizer

MODEL_DIR = Path("outputs/robust_layoutlmv3")


@st.cache_resource
def load_engine():
    return InferenceEngine()


st.set_page_config(
    page_title="DocShield AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.sidebar.title("🛡️ DocShield AI")
st.sidebar.markdown("---")
st.sidebar.subheader("AI Engine")
st.sidebar.success("LayoutLMv3")
st.sidebar.success("EasyOCR")

if MODEL_DIR.exists():
    st.sidebar.success("Model Ready")
else:
    st.sidebar.error("Model Missing")

st.sidebar.markdown("---")
st.sidebar.caption("AI-powered document understanding")

st.title("🛡️ DocShield AI")
st.caption("Upload a document and let AI extract structured information.")
st.markdown("---")

uploaded_file = st.file_uploader(
    "Upload Document",
    type=["png", "jpg", "jpeg"],
)

if uploaded_file is None:
    st.info("Upload a PNG or JPG document to begin.")
    st.stop()

image = Image.open(uploaded_file).convert("RGB")

engine = load_engine()
visualizer = PredictionVisualizer()

if st.button("🚀 Run AI Inference", use_container_width=True, type="primary"):
    with st.spinner("Running OCR..."):
        result = engine.predict_image(image)

    annotated = visualizer.draw(
        image,
        result["words"],
        result["boxes"],
        result["labels"],
        result["scores"],
    )

    entities = result["entities"]
    average_confidence = (
        sum(entity["score"] for entity in entities) / len(entities)
        if entities
        else 0
    )

    st.success("Analysis Completed")

    left, right = st.columns(2)

    with left:
        st.image(
            image,
            caption="Original",
            use_container_width=True,
        )

    with right:
        st.image(
            annotated,
            caption="AI Prediction",
            use_container_width=True,
        )

    st.markdown("### Legend")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.success("HEADER")

    with c2:
        st.info("QUESTION")

    with c3:
        st.error("ANSWER")

    st.markdown("---")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Entities", len(entities))
    with c2:
        st.metric("Confidence", f"{average_confidence:.1%}")
    with c3:
        st.metric("Status", "Success")

    st.markdown("---")

    summary = result["confidence"]

    st.subheader("Confidence Analysis")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("Average", f"{summary['average']:.2%}")

    with c2:
        st.metric("High", summary["high"])

    with c3:
        st.metric("Medium", summary["medium"])

    with c4:
        st.metric("Low", summary["low"])

    st.markdown("---")

    st.subheader("Document Summary")

    if not entities:
        st.warning("No entities were extracted.")
    else:
        for entity in entities:
            confidence = entity["score"]

            if confidence >= 0.95:
                badge = "🟢"
            elif confidence >= 0.80:
                badge = "🟡"
            else:
                badge = "🔴"

            with st.container(border=True):
                st.markdown(f"### {badge} {entity['label']}")
                st.markdown(f"**Text**\n\n{entity['text']}")
                st.progress(confidence)
                st.caption(f"{confidence:.2%}")

    st.markdown("---")
    st.subheader("Export Results")

    json_data = json.dumps(result, indent=4)

    tab1, tab2 = st.tabs(["JSON", "CSV"])

    with tab1:
        st.code(json_data, language="json")

    with tab2:
        st.code(result["csv"], language="text")

    st.download_button(
        label="⬇ Download JSON",
        data=json_data,
        file_name="prediction.json",
        mime="application/json",
        use_container_width=True,
    )

    st.download_button(
        label="⬇ Download CSV",
        data=result["csv"],
        file_name="prediction.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.markdown("---")

st.markdown("---")
st.caption("DocShield AI • LayoutLMv3 + EasyOCR • Built with Streamlit")