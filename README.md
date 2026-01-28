# Document Extraction System using LLM API and RAG

This project is a Flask-based web application designed to analyse and extract insights from cybersecurity data. It uses **Retrieval-Augmented Generation (RAG)** and the **Llama 3.2 1B** model to process the **CISA Known Exploited Vulnerabilities (KEV)** catalog and user-uploaded PDF documents. Developed as an undergraduate exchange project at the National Taipei University of Technology, the system specifically focuses on extracting and summarizing insights from CISA’s Known Exploited Vulnerabilities (KEV) catalog.

---

## Key Features

* **Vulnerability Querying**: Employs TF-IDF vectorization and Cosine Similarity to retrieve relevant CVE records from the CISA KEV dataset.
* **Context-Aware Augmentation**: Uses the Llama 3.2 1-billion parameter model to generate detailed responses based on retrieved security data.
* **PDF Processing**: Extracts text from uploaded PDFs using PDFPlumber, chunks the content, and stores it in a **Chroma DB** vector store for analysis.
* **Real-time Interaction**: Built on a Flask backend providing a user-friendly interface for security researchers and analysts.

---

## System Architecture

<div align="center">

[![predictions](https://github.com/laikhuramkorouhanba/DOCUMENT_EXTRACTION_SYSTEM_RAG/blob/main/static/assets/ARC.png?raw=true)]((https://github.com/laikhuramkorouhanba/DOCUMENT_EXTRACTION_SYSTEM_RAG))

</div>

---

* **Framework**: Flask
* **LLM**: Meta Llama 3.2-1B (via Hugging Face API)
* **Database**: Chroma DB (Vector Storage)
* **Libraries**: Scikit-learn (TF-IDF), LangChain, Pandas, PDFPlumber

---

## Installation & Setup

### Prerequisites
* Python 3.8+ 
* Hugging Face API Token 

### Step 1: Clone and Environment Setup
```bash
# Navigate to project directory
cd DOCUMENT_EXTRACTION

# Create and activate virtual environment
python3 -m venv env
source env/bin/activate  # macOS/Linux
# .\env\Scripts\activate # Windows
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Run the Application
```bash
# For KEV dataset querying:
python app.py

# For PDF processing module:
python pdf_app.py


