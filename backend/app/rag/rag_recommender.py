from pathlib import Path
import json
from typing import List, Dict

import os
from dotenv import load_dotenv
from openai import OpenAI
import faiss
from sentence_transformers import SentenceTransformer

# 👇 NEW: fallback pesticide mapping
from .pesticide_mapping import get_pesticide_info

# -----------------------------------------------------------------------------
# ENV + OPENAI CLIENT
# -----------------------------------------------------------------------------

# Load .env from project root
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError(
        "OPENAI_API_KEY not found. "
        "Create a .env in project root with: OPENAI_API_KEY=your_key_here"
    )

client = OpenAI(api_key=OPENAI_API_KEY)

# -----------------------------------------------------------------------------
# VECTORSTORE PATHS
# -----------------------------------------------------------------------------

# This file: backend/app/rag/rag_recommender.py
# parents[0] = rag, [1] = app, [2] = backend, [3] = project root
PROJECT_ROOT = Path(__file__).resolve().parents[3]

VSTORE_DIR = PROJECT_ROOT / "artifacts" / "vectorstores"
INDEX_PATH = VSTORE_DIR / "ipm_faiss.index"
METADATA_PATH = VSTORE_DIR / "ipm_metadata.json"

# ---- Load metadata ----
with open(METADATA_PATH, "r", encoding="utf-8") as f:
    META = json.load(f)

TEXTS = META["texts"]
METADATA = META["metadata"]
EMB_MODEL_NAME = META["embedding_model"]

# ---- Load embedding model + FAISS index ----
EMB_MODEL = SentenceTransformer(EMB_MODEL_NAME)   # CPU is fine for queries
INDEX = faiss.read_index(str(INDEX_PATH))


# -----------------------------------------------------------------------------
# RETRIEVAL
# -----------------------------------------------------------------------------

def build_query(crop: str, disease: str, growth_stage: str | None = None) -> str:
    base = f"Integrated pest management recommendations for {crop} {disease} in Indian conditions."
    if growth_stage:
        base += f" The crop is at {growth_stage} stage."
    base += (
        " Include economic threshold level if available, cultural practices, "
        "biological control, and chemical control with recommended pesticides, "
        "dose per hectare, and safety precautions."
    )
    return base


def retrieve_chunks(crop: str, disease: str, top_k: int = 6) -> List[Dict]:
    """
    Vector search in FAISS and return relevant chunks
    filtered by crop.
    """
    query = build_query(crop, disease)
    q_emb = EMB_MODEL.encode([query])
    D, I = INDEX.search(q_emb.astype("float32"), top_k * 3)  # oversample

    results: List[Dict] = []
    for idx in I[0]:
        if idx == -1:
            continue
        meta = METADATA[idx]
        if meta["crop"].lower() != crop.lower():  # crop filter
            continue
        results.append(
            {
                "text": TEXTS[idx],
                "metadata": meta,
            }
        )
        if len(results) >= top_k:
            break
    return results


# -----------------------------------------------------------------------------
# FARMER-FRIENDLY RAG RESPONSE
# -----------------------------------------------------------------------------

def generate_farmer_response(
    user_query: str,
    crop: str,
    disease: str,
    confidence: float,
    growth_stage: str | None = None,
    language: str = "english",
    top_k: int = 6,
    model_name: str = "gpt-3.5-turbo",
) -> str:
    """
    Use RAG (Govt IPM docs) + GPT-3.5 to generate a
    farmer-friendly answer in the selected language.

    language:
        "english"  -> English
        "hinglish" -> Hindi in Roman script
        "hindi"    -> Pure Hindi (Devanagari)
    """

    # 1) Retrieve relevant IPM chunks
    chunks = retrieve_chunks(crop, disease, top_k=top_k)

    if not chunks:
        context_text = "No relevant IPM document chunks were retrieved for this query."
    else:
        context_text = "\n\n".join(
            f"[DOC {i}] {c['text']}"
            for i, c in enumerate(chunks, start=1)
        )

    language = (language or "english").lower()

    # ---------------------------
    # Language-specific templates
    # ---------------------------
    if language == "hindi":
        style_instructions = """
नीचे दिए गए हेडिंग्स *जैसी हैं* वैसी ही रखें (शब्द न बदलें):

1. **संभावित कारण**
2. **रोग का नाम**
3. **अनुशंसित कीटनाशक और वैकल्पिक विकल्प**
4. **सावधानियाँ**
5. **अतिरिक्त सलाह**

हर हेडिंग के नीचे 2–3 छोटी, साफ बुलेट लाइने लिखें।

- "संभावित कारण" में खेत से जुड़े कारण लिखें
  (रोग, पोषक तत्व की कमी, पानी की कमी या अधिकता आदि)।

- "रोग का नाम" में:
  - रोग का नाम: <नाम>
  - विश्वास स्तर: <0.00 से 1.00 तक>

- "अनुशंसित कीटनाशक और वैकल्पिक विकल्प" में:
  - अगर इस रोग के लिए सामान्य रूप से उपयोग होने वाले मानक कीटनाशक
    (जैसे Copper oxychloride, Streptocycline आदि) और उनकी डोज़
    आपकी तकनीकी जानकारी या दिये गये टेक्स्ट से उचित रूप से समर्थन पाते हों,
    तो उन्हें स्पष्ट रूप से लिखें:
      • मुख्य कीटनाशक: <कीटनाशक का नाम> (<डोज़>)
      • यदि कोई अलग दूसरा विकल्प हो तो:
        वैकल्पिक विकल्प: <दूसरे कीटनाशक का नाम> (<डोज़>)
      • आवश्यकता हो तो 1–2 लाइनों में छिड़काव का समय और बार लिखें
        (जैसे: सुबह या शाम, कितनी बार छिड़काव करना है)।
  - कीटनाशक का नाम English में, Hindi उच्चारण के साथ,
    या दोनों के मिश्रण में दिया जा सकता है, जैसे:
    "Copper oxychloride (कॉपर ऑक्सीक्लोराइड, 100 ग्राम / 10 लीटर पानी)"।
  - यदि केवल एक ही कीटनाशक स्पष्ट हो और कोई दूसरा विकल्प न हो,
    तो "वैकल्पिक विकल्प" में वही नाम दोहराने की बजाय इस तरह की सलाह लिखें:
      वैकल्पिक विकल्प: कृपया किसी अन्य मान्य दवा के लिए
      नज़दीकी कृषि अधिकारी या प्रमाणित विक्रेता से
      सही नाम और डोज़ एक बार अवश्य पुष्टि करें।
  - यदि किसी भी तरह से किसी कीटनाशक का उचित नाम या डोज़ तय नहीं किया जा सकता,
    तब "मुख्य कीटनाशक" और "वैकल्पिक विकल्प" की जगह केवल एक बुलेट दें:
      दवा की जानकारी: कृपया नज़दीकी कृषि अधिकारी या
      मान्यता प्राप्त विक्रेता से सही दवा का नाम और डोज़
      एक बार अवश्य पुष्टि करें।

- "सावधानियाँ" में सुरक्षा उपाय लिखें
  (दस्ताने, मास्क, बच्चों/पशुओं और पानी के स्रोत से दूर रखें,
  बारिश या तेज़ हवा में छिड़काव न करें)।

- "अतिरिक्त सलाह" में सिंचाई, मिट्टी की जाँच, फसल की निगरानी
  और यह लिखें कि यदि 3–4 दिन में सुधार न हो या रोग तेज़ी से फैले
  तो किसान नज़दीकी कृषि विशेषज्ञ / कृषि विज्ञान केंद्र से संपर्क करे।

उत्तर में कहीं भी "IPM", "डॉक्यूमेंट", "कॉन्टेक्स्ट",
"मॉडल", "AI" या "fallback" जैसे शब्द न लिखें।
""".strip()




    elif language == "hinglish":
        style_instructions = """
Use EXACTLY these headings (do not change the text):

1. **Possible Reasons**
2. **Name of the Disease**
3. **Recommended Pesticide and Alternate Option**
4. **Precautions**
5. **Additional Advice**

Write 2–3 short bullets under each heading in very simple Hinglish
(Hindi words in English letters).

- In "Possible Reasons" mention field reasons only
  (disease infection, nutrient kami, paani ki problem, etc.).
- In "Name of the Disease":
  - Disease Name: <...>
  - Confidence Level: <0.00–1.00>
- In "Recommended Pesticide and Alternate Option":
  - Primary pesticide name + dose (per litre/per hectare) if clearly available.
  - One alternate option if available.
  - How and when to spray (subah/shaam, kitni baar).
  - If clear pesticide name or dose is NOT available, do NOT invent anything.
    Instead write a line like:
    "Yeh dawa ka exact naam aur dose aap local agriculture officer ya
     certified dealer se confirm karke hi use karein."
- "Precautions" → gloves, mask, bachchon/pashuon se door, no spray in barish or strong hawa.
- "Additional Advice" → irrigation, soil health, regular monitoring, and when to talk to expert.

Do NOT mention words like "IPM document", "context", "fallback pesticide",
"model", or "AI" in the answer.
""".strip()

    else:  # english
        style_instructions = """
Use EXACTLY these headings (do not change the text):

1. **Possible Reasons**
2. **Name of the Disease**
3. **Recommended Pesticide and Alternate Option**
4. **Precautions**
5. **Additional Advice**

Under each heading, write 2–3 short, clear bullet points.

- "Possible Reasons": explain field reasons only
  (disease infection, nutrient deficiency, water stress, etc.).
- "Name of the Disease":
  - Disease Name: <...>
  - Confidence Level: <0.00–1.00>
- "Recommended Pesticide and Alternate Option":
  - Give the main pesticide name and dose (per litre or per hectare)
    only if it is clearly supported by the technical information.
  - Give one alternate pesticide if available.
  - Explain when and how to spray (morning/evening, how many times).
  - If a clear pesticide name or dose is NOT available, do NOT invent one.
    Instead write a line like:
    "Please confirm the exact pesticide name and dose once with your
     local agriculture officer or a certified input dealer before use."
- "Precautions": safety (gloves, mask), keep away from children/animals,
  avoid spraying in rain or strong wind, keep away from ponds/drinking water.
- "Additional Advice": 2–3 short tips on irrigation, soil health,
  monitoring the crop, and when to contact a local expert.

Do NOT mention phrases like "IPM document", "IPM context",
"fallback pesticide", "model", or "AI" in the answer.
The answer should sound like a direct conversation with the farmer.
""".strip()

    # 2) Final prompt
    prompt = f"""
The farmer asked: "{user_query}"

Image-based diagnosis:
- Crop: {crop}
- Disease: {disease}
- Confidence: {confidence:.2f}
- Growth stage: {growth_stage or "not specified"}

Here is technical reference text (for you only, NOT to be mentioned explicitly):

{context_text}

Now give a farmer-friendly answer following these rules:

{style_instructions}
""".strip()

    # 3) Call GPT
    response = client.chat.completions.create(
        model=model_name,
        temperature=0.3,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a careful Indian agriculture expert. "
                    "You never invent pesticide names or doses that are "
                    "not clearly justified by the reference text. "
                    "You do not mention documents, context, models or AI in your answer."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    )

    return response.choices[0].message.content.strip()