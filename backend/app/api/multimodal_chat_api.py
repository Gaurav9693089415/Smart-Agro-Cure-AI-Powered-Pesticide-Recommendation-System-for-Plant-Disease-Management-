from fastapi import APIRouter, UploadFile, File, Form
from pathlib import Path
import shutil
import uuid

# RELATIVE imports (very important)
from ..ml.inference import predict_image
from ..rag.rag_recommender import generate_farmer_response

router = APIRouter()

# -----------------------------------------------------------------------------
# Save uploaded images under <project_root>/artifacts/uploads
# File path:
# backend/app/api/multimodal_chat_api.py
# parents[0] = api
# parents[1] = app
# parents[2] = backend
# parents[3] = project root
# -----------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]
UPLOAD_DIR = PROJECT_ROOT / "artifacts" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@router.post("/multimodal-chat")
async def multimodal_chat(
    user_query: str = Form(...),
    image: UploadFile = File(...),
    language: str = Form("english"),  # "english" | "hinglish" | "hindi"
):
    """
    Farmer sends:
      - user_query (question)
      - image (leaf photo)
      - language: "english", "hinglish", or "hindi"

    System:
      1) Saves image
      2) Runs CNN (predict_image)
      3) Calls RAG + LLM (generate_farmer_response)
      4) Returns farmer-friendly answer
    """

    # 1) Save uploaded image
    suffix = Path(image.filename).suffix
    unique_name = f"{uuid.uuid4().hex}{suffix}"
    image_path = UPLOAD_DIR / unique_name

    with open(image_path, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)

    # 2) CNN Prediction
    pred = predict_image(str(image_path))
    # e.g. "rice_bacterialblight" – useful to see alongside pesticide fallback
    class_name = pred.get("pred_class")

    # 3) RAG + Farmer Response
    answer = generate_farmer_response(
        user_query=user_query,
        crop=pred["crop"],
        disease=pred["disease"],
        confidence=pred["confidence"],
        growth_stage=None,
        language=language,
        # note: current generate_farmer_response builds the mapping key
        # from crop + disease internally using pesticide_mapping.py
    )

    # 4) Final API Response
    return {
        "question": user_query,
        "language": language,
        "image_prediction": pred,
        "cnn_class_name": class_name,  # extra field for debugging / transparency
        "farmer_answer": answer,
        "image_path": str(image_path),
    }
