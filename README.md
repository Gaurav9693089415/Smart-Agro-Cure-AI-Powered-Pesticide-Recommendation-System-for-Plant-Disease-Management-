

---

# 🌱 Smart Agro-Cure

### AI-Powered Pesticide Recommendation System for Plant Disease Management

Smart Agro-Cure is an **end-to-end AI-based multimodal system** that detects plant diseases from leaf images and generates **document-verified pesticide recommendations** using a **CNN + RAG + LLM** pipeline.
The system is designed for **accuracy, safety, explainability, and real-world deployment**.

---

## 📌 Problem Statement

Crop diseases significantly reduce agricultural productivity and farmer income.
Existing solutions often:

* rely on manual inspection,
* provide generic or unsafe pesticide advice,
* lack explainability and verification.

Smart Agro-Cure addresses these gaps by combining **computer vision** with **retrieval-augmented language models** to deliver **trusted, multilingual advisories**.

---

## 🎯 Objectives

* Detect plant diseases accurately from leaf images
* Provide **verified pesticide recommendations** from official IPM documents
* Reduce pesticide misuse and environmental impact
* Support multilingual farmer interaction (English / Hindi / Hinglish)
* Deliver a **deployment-ready AI system**

---

## 🧠 System Overview

### Core Components

### 1️⃣ Vision Model (CNN)

* EfficientNet-B0 for plant disease classification
* Trained on Indian crop disease datasets
* Outputs crop, disease, and confidence score

### 2️⃣ Retrieval-Augmented Generation (RAG)

* Official IPM and agricultural documents indexed using **FAISS**
* Relevant documents retrieved dynamically based on detected disease

### 3️⃣ Large Language Model (LLM)

* Generates farmer-friendly recommendations
* Strictly grounded in retrieved documents to avoid hallucinations

### 4️⃣ Backend & Deployment

* FastAPI-based backend for production-style inference
* Streamlit-based UI for easy deployment and evaluation

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
│   ├── model_efficientnet_b0.pth
│   └── vectorstores/           # FAISS index + metadata
│
├── frontend/
│   └── index.html              # HTML UI (voice-enabled)
│
├── streamlit_app.py            # Streamlit deployment app
├── requirements.txt
├── .gitignore
└── README.md
```

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
* No free-form or external knowledge allowed
* Low-confidence predictions trigger conservative, safe responses
* Explicit instructions prevent hallucinated pesticide names or doses

---

## 🌍 Features

* Leaf image-based disease detection
* Document-verified pesticide recommendations
* Confidence score display
* Multilingual advisory:

  * English
  * Hindi (Devanagari)
  * Hinglish (Roman Hindi)
* Modular and extensible design
* Deployment-ready architecture

---

## 🚀 Deployment Modes

Smart Agro-Cure is designed with **clear separation of concerns**, following real-world engineering practices.

---

### 1️⃣ Streamlit Deployment (Cloud / Hugging Face)

* Used for **public demos and evaluation**
* Lightweight, fast UI
* No browser permission dependencies
* Supports multilingual text-based interaction

Run locally:

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

This version is deployed on **Hugging Face Spaces**.

---

### 2️⃣ FastAPI + HTML UI (Local Voice Demo)

* Supports:

  * 🎤 Voice input (Speech-to-Text)
  * 🔊 Voice output (Text-to-Speech)
* Uses browser-native APIs
* Intended for **local demo and interview presentation**

Run locally:

```bash
pip install -r requirements.txt
uvicorn backend.app.main:app --reload
```

> **Note:** Voice features are intentionally not enabled in cloud deployments due to browser permission and security constraints.

---

## 🤗 Hugging Face Deployment

The Streamlit version of Smart Agro-Cure is deployed on **Hugging Face Spaces** for easy access and evaluation.

* Application entry point: `streamlit_app.py`
* Dependencies managed via `requirements.txt`
* Sensitive credentials (e.g., OpenAI API key) are securely stored using **Hugging Face Secrets**
* No secrets are exposed in the repository

---

## 🔬 Research Contribution

This project extends traditional CNN-based plant disease detection by:

* Replacing static pesticide databases with **RAG-based retrieval**
* Integrating **LLM-driven, explainable advisories**
* Focusing on **deployment safety, verification, and real-world usability**

---





## 👤 Author

**Gaurav Kumar**
M.Sc. Artificial Intelligence & Machine Learning
IIIT Lucknow

---

## 📜 License

This project is licensed under the **MIT License**.

---

