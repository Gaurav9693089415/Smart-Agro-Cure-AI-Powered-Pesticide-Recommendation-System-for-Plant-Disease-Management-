
---

# 🌱 Smart Agro-Cure

### AI-Powered Pesticide Recommendation System for Plant Disease Management

Smart Agro-Cure is an **end-to-end AI-based multimodal system** that detects plant diseases from leaf images and generates **document-verified pesticide recommendations** using a **CNN + RAG + LLM** pipeline.
The system is designed for **accuracy, safety, and real-world deployment**.

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

1. **Vision Model (CNN)**

   * EfficientNet-B0 for plant disease classification
   * Trained on Indian crop disease datasets
   * Outputs crop, disease, and confidence score

2. **Retrieval-Augmented Generation (RAG)**

   * Official IPM and agricultural documents indexed using FAISS
   * Relevant documents retrieved dynamically based on detected disease

3. **Large Language Model (LLM)**

   * Generates farmer-friendly recommendations
   * Strictly grounded in retrieved documents to avoid hallucinations

4. **Backend & Deployment**

   * FastAPI-based backend
   * Real-time inference and advisory generation

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
│   └── class_index.json        # Class mapping (lightweight config)
│
├── frontend/
│   └── index.html              # Simple UI
│
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
* Normalization (ImageNet stats)
* Data augmentation
* Train/validation split (80/20)

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

## 🔐 Hallucination Control (Important)

To ensure safe recommendations:

* LLM responses are **restricted to retrieved documents only**
* No external or free-form knowledge allowed
* Low-confidence predictions trigger safe fallback responses

---

## 🌍 Features

* Leaf image-based disease detection
* Document-verified pesticide recommendations
* Confidence score display
* Multilingual advisory (English / Hindi / Hinglish)
* Real-time FastAPI backend
* Modular, extensible design

---

## 🚀 How to Run (High-Level)

```bash
# Install dependencies
pip install -r requirements.txt

# Run FastAPI backend
uvicorn backend.app.main:app --reload
```

*(Model weights and datasets are intentionally excluded from the repo for cleanliness and reproducibility.)*

---

## 🔬 Research Contribution

This project extends existing CNN-based plant disease detection research by:

* replacing static pesticide databases with **RAG-based retrieval**
* integrating **LLM-driven, explainable advisories**
* focusing on **real-world deployment and safety**

---

## 📈 Future Enhancements

* Disease severity estimation
* Bounding-box or segmentation-based localization
* Offline / edge deployment
* Weather and soil data integration
* Mobile application support

---

## 👤 Author

**Gaurav Kumar**
M.Sc. Artificial Intelligence & Machine Learning
IIIT Lucknow

---

## 📜 License

This project is licensed under the MIT License.

---
