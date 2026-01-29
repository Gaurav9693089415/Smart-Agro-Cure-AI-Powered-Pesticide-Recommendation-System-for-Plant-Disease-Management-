



---

# 🌱 Smart Agro-Cure

### AI-Powered Pesticide Recommendation System for Plant Disease Management

Smart Agro-Cure is an **end-to-end AI-based multimodal system** that detects plant diseases from leaf images and generates **document-verified pesticide recommendations** using a **CNN + RAG + LLM** pipeline.

The project is designed with a **real-world production mindset**, focusing on **accuracy, safety, explainability, and clean system design** rather than cloud-specific shortcuts.

---

## 📌 Problem Statement

Crop diseases significantly reduce agricultural productivity and farmer income.
Most existing solutions:

* rely on manual inspection,
* provide generic or unsafe pesticide advice,
* lack explainability and source verification.

Smart Agro-Cure addresses these issues by combining **computer vision** with **retrieval-augmented language models (RAG)** to deliver **trusted, multilingual, and explainable advisories** grounded in official agricultural documents.

---

## 🎯 Objectives

* Detect plant diseases accurately from leaf images
* Provide **verified pesticide recommendations** from official IPM documents
* Reduce pesticide misuse and environmental impact
* Support multilingual farmer interaction (English / Hindi / Hinglish)
* Design a **deployment-ready but locally demonstrable AI system**

---

## 🧠 System Overview

### 1️⃣ Vision Model (CNN)

* **EfficientNet-B0** for plant disease classification
* Trained on Indian crop disease datasets
* Outputs:

  * Crop
  * Disease
  * Confidence score

---

### 2️⃣ Retrieval-Augmented Generation (RAG)

* Official IPM and agricultural documents indexed using **FAISS**
* Disease-aware queries dynamically retrieve relevant documents
* Ensures recommendations are **document-backed**, not hallucinated

---

### 3️⃣ Large Language Model (LLM)

* Generates farmer-friendly advisories
* Strictly constrained to retrieved content
* Designed to **never invent pesticide names or doses**

---

### 4️⃣ Backend & Interface

* **FastAPI** backend for production-style inference design
* **Streamlit** and **HTML UI** used only for **local evaluation and demos**

---

## 🏗️ End-to-End Architecture

```
User (Leaf Image)
        ↓
Image Preprocessing
        ↓
CNN Disease Detection (EfficientNet-B0)
        ↓
Disease-Aware Query Builder
        ↓
FAISS Vector Search (IPM Documents)
        ↓
LLM (RAG-based, grounded generation)
        ↓
Structured Pesticide Advisory
```

---

## 📂 Project Structure

```
smart-agro-cure/
│
├── backend/
│   ├── app/
│   │   ├── api/                # FastAPI endpoints
│   │   ├── ml/                 # CNN inference logic
│   │   ├── rag/                # RAG pipeline (FAISS + LLM)
│   │   └── main.py             # FastAPI entry point
│
├── ml/
│   ├── model.py                # CNN architecture
│   ├── data_module.py          # Data loading & preprocessing
│   ├── inference.py            # Model inference
│   └── training/               # Training scripts
│
├── scripts/
│   ├── build_class_index.py
│   ├── build_knowledge_base.py
│   └── dataset preparation scripts
│
├── artifacts/
│   ├── class_index.json        # Class mapping
│   └── vectorstores/           # FAISS index + metadata
│
├── frontend/
│   └── index.html              # Voice-enabled HTML UI (local demo)
│
├── streamlit_app.py            # Streamlit-based local demo UI
├── requirements.txt
├── .gitignore
└── README.md
```

> ⚠️ **Note:**
> Trained model weights and large artifacts are intentionally excluded from the repository to keep it lightweight and reproducible.

---

## 📊 Dataset

* **Source:** Indian Crop Diseases Dataset
* **Crops:** Rice, Wheat, Corn, Cotton
* **Classes:** 15 (diseased + healthy)
* **Images:** ~10,977 after cleaning

### Preprocessing

* Image resizing (224 × 224)
* Normalization (ImageNet statistics)
* Data augmentation
* Train / validation split (80 / 20)

---

## 🧪 Model Training

* **Architecture:** EfficientNet-B0
* **Loss:** Cross-Entropy
* **Optimizer:** AdamW
* **Batch Size:** 16
* **Early Stopping:** Epoch 7

### Performance

* **Training Accuracy:** ~99%
* **Validation Accuracy:** ~98.5%

---

## 🔐 Hallucination Control (Critical Design Choice)

To ensure **safe and reliable pesticide recommendations**:

* LLM responses are **strictly grounded in retrieved IPM documents**
* No free-form or external knowledge is allowed
* Low-confidence predictions trigger conservative, safe responses
* Explicit instructions prevent hallucinated pesticide names or doses

---

## 🌍 Key Features

* Leaf image-based disease detection
* Document-verified pesticide recommendations
* Confidence score display
* Multilingual advisory:

  * English
  * Hindi (Devanagari)
  * Hinglish (Roman Hindi)
* Modular, extensible system design

---

## 🚀 How to Run (Local Only)

This project is intended for **local execution and evaluation**.

### Streamlit Demo (Text-based)

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

---

### FastAPI + HTML UI (Voice Demo)

Supports:

* 🎤 Voice input (Speech-to-Text)
* 🔊 Voice output (Text-to-Speech)

```bash
pip install -r requirements.txt
uvicorn backend.app.main:app --reload
```

> Voice features rely on browser APIs and are **meant for local demos and interviews only**.

---

## 🔬 Research Contribution

This project extends traditional CNN-based plant disease detection by:

* Replacing static pesticide databases with **RAG-based retrieval**
* Integrating **LLM-driven, explainable advisories**
* Emphasizing **safety, verification, and real-world usability**
* Designing a system that mirrors **production ML workflows**

---

## 👤 Author

**Gaurav Kumar**
M.Sc. Artificial Intelligence & Machine Learning
IIIT Lucknow

---

## 📜 License

This project is licensed under the **MIT License**.

---


If you want next:

* interview explanation script
* 2–3 line resume bullets
* “why not deployed” answer framing

Just tell me 👍
