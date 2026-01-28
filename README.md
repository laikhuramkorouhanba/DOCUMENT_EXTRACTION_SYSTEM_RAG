# Document Extraction System using LLM API and RAG

This project is a Flask-based web application designed to analyse and extract insights from cybersecurity data. It uses **Retrieval-Augmented Generation (RAG)** and the **Llama 3.2 1B** model to process the **CISA Known Exploited Vulnerabilities (KEV)** catalog and user-uploaded PDF documents. Developed as an undergraduate exchange project at the National Taipei University of Technology, the system specifically focuses on extracting and summarizing insights from CISA’s Known Exploited Vulnerabilities (KEV) catalog.

---

## 🚀 Key Features

* [cite_start]**Vulnerability Querying**: Employs TF-IDF vectorization and Cosine Similarity to retrieve relevant CVE records from the CISA KEV dataset[cite: 30, 193, 194].
* [cite_start]**Context-Aware Augmentation**: Uses the Llama 3.2 1-billion parameter model to generate detailed responses based on retrieved security data[cite: 29, 31].
* [cite_start]**PDF Processing**: Extracts text from uploaded PDFs using PDFPlumber, chunks the content, and stores it in a **Chroma DB** vector store for analysis[cite: 131, 201, 203].
* [cite_start]**Real-time Interaction**: Built on a Flask backend providing a user-friendly interface for security researchers and analysts[cite: 30, 221].

---

## 🏗️ System Architecture

<div align="center">

[![predictions](https://github.com/laikhuramkorouhanba/DOCUMENT_EXTRACTION_SYSTEM_RAG/blob/main/static/assets/ARC.png?raw=true)]((https://github.com/laikhuramkorouhanba/DOCUMENT_EXTRACTION_SYSTEM_RAG))

</div>

The system follows a modular RAG workflow:
1.  [cite_start]**Data Handling**: Loads and preprocesses CSV/PDF data[cite: 117, 139].
2.  [cite_start]**Vector Store**: Manages high-dimensional embeddings using Chroma DB and FastEmbed[cite: 130, 282].
3.  [cite_start]**Retrieval**: Matches user queries to the knowledge base using similarity scores[cite: 141, 195].
4.  [cite_start]**Generation**: Augments the query with retrieved context and sends it to the Hugging Face Inference API[cite: 170, 229].



---

## 🛠️ Tech Stack

* [cite_start]**Framework**: Flask [cite: 30]
* [cite_start]**LLM**: Meta Llama 3.2-1B (via Hugging Face API) [cite: 29, 231]
* [cite_start]**Database**: Chroma DB (Vector Storage) [cite: 131, 282]
* [cite_start]**Libraries**: Scikit-learn (TF-IDF), LangChain, Pandas, PDFPlumber [cite: 133, 375, 378, 455]

---

## 🔧 Installation & Setup

### Prerequisites
* [cite_start]Python 3.8+ [cite: 543]
* [cite_start]Hugging Face API Token [cite: 567]

### Step 1: Clone and Environment Setup
```bash
# Navigate to project directory
cd DOCUMENT_EXTRACTION

# Create and activate virtual environment
python3 -m venv env
source env/bin/activate  # macOS/Linux
# .\env\Scripts\activate # Windows
