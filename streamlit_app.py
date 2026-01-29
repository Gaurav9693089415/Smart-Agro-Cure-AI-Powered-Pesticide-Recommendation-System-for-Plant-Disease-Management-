import streamlit as st
import tempfile
from pathlib import Path

# ---- Backend imports (existing project code) ----
from backend.app.ml.inference import predict_image
from backend.app.rag.rag_recommender import generate_farmer_response

# -----------------------------------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Smart Agro-Cure",
    page_icon="🌾",
    layout="wide",
)

# -----------------------------------------------------------------------------
# HEADER
# -----------------------------------------------------------------------------
st.markdown(
    """
    <h1>Smart Agro-Cure 🌾</h1>
    <p style="color:#555;">
    Upload leaf photo + ask your question. AI will detect disease and give
    farmer-friendly advice.
    </p>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# LAYOUT
# -----------------------------------------------------------------------------
left, right = st.columns([1.1, 1.4])

# -----------------------------------------------------------------------------
# LEFT: INPUTS
# -----------------------------------------------------------------------------
with left:
    st.subheader("Input")

    user_query = st.text_area("Question")

    uploaded_image = st.file_uploader(
        "Leaf Image",
        type=["jpg", "jpeg", "png"],
    )

    language = st.selectbox(
        "Language",
        ["english", "hinglish", "hindi"],
        index=0,
    )

    ask_btn = st.button("Ask")

# -----------------------------------------------------------------------------
# RIGHT: OUTPUTS
# -----------------------------------------------------------------------------
with right:
    st.subheader("Prediction & Answer")

    prediction_container = st.empty()
    answer_container = st.empty()

# -----------------------------------------------------------------------------
# MAIN LOGIC
# -----------------------------------------------------------------------------
if ask_btn:
    if not uploaded_image:
        st.warning("Please upload a leaf image.")
    else:
        with st.spinner("Processing..."):

            # ---- Save uploaded image temporarily ----
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                tmp.write(uploaded_image.read())
                image_path = Path(tmp.name)

            # ---- Image prediction ----
            image_pred = predict_image(image_path)

            # ---- RAG + LLM response ----
            farmer_answer = generate_farmer_response(
                user_query=user_query,
                crop=image_pred["crop"],
                disease=image_pred["disease"],
                confidence=image_pred["confidence"],
                language=language,
            )

            # ---- Display prediction ----
            with prediction_container:
                with st.expander("Model Prediction", expanded=False):
                    st.json(image_pred)

            # ---- Display answer ----
            answer_container.markdown(farmer_answer)

            st.success("Done ✅")
