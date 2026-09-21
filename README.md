# 🕵️‍♂️ Universal File Explorer 📂

A **Streamlit** web application for chatting with your documents locally. It runs 100% offline using GGUF models on your CPU, meaning no data leaves your machine and no API keys are required.

---

## 🚀 Key Features

*   **🔒 Local and Private**: All processing and text generation happen entirely on your computer's CPU and RAM.
*   **📂 Hybrid Document Routing**: Automatically handles different file types using two search methods:
    *   **BM25 (Keyword Match)**: Used for spreadsheets (`CSV`, `XLSX`, `XLS`) and `TXT` files by analyzing their column structure and headers.
    *   **FAISS (Vector Search)**: Used for text-heavy documents (`PDF`, `DOCX`) by splitting them into smaller text chunks and indexing them.
*   **⚙️ Smart Hardware Settings**: Automatically adjustments the context window size based on the model size (e.g., 3072 for 1.5B models, 2048 for 3B models) and uses available CPU cores for processing.
*   **🎯 Chit-Chat Filter**: Recognizes standard greetings or small talk (like "hello" or "thank you") and replies immediately without wasting time searching through your files.
*   **🔍 Engine Insights**: Displays which search engine (BM25 or FAISS) was used to find the answer, along with performance stats like generation speed (tokens per second).

---

## 🛠️ Tech Stack & Dependencies

*   **llama-cpp-python**: Runs GGUF models locally on the CPU.
*   **FAISS (langchain-community)**: Handles vector-based similarity search for PDFs and Word docs.
*   **Rank-BM25**: Handles keyword matching for spreadsheets and text files.
*   **langchain-huggingface**: Uses the `all-MiniLM-L6-v2` model to create text embeddings.
*   **Streamlit**: Powers the user interface and chat display.
*   **Pandas & OpenPyXL**: Used to read and extract data from spreadsheets.
*   **PyPDF & Docx2Txt**: Extracts plain text from PDFs and Word documents.

---

## 📁 Repository Structure

```text
├── local_models/                            # Directory where uploaded .gguf models are saved
├── .gitignore                               # Ignores large model files and environment folders
└── app.py                                   # Main Streamlit application code
```

---

## ⚙️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com
cd universal-file-explorer
```

### 2. Set Up a Virtual Environment & Install Dependencies

**Using standard pip:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install streamlit pypdf docx2txt pandas llama-cpp-python rank-bm25 langchain-community langchain-huggingface openpyxl
```

### 3. Launch the App
```bash
streamlit run app.py
```

---

## 📖 How To Use The App

1. **Upload a Model**: Open the sidebar and upload a `.gguf` model file (like a Qwen 2.5 1.5B or Llama 3.2 3B model). It will be saved into the `local_models` folder.
2. **Select the Model**: Choose your uploaded model from the dropdown menu in the sidebar.
3. **Upload Documents**: Upload your reference files (`.pdf`, `.docx`, `.xlsx`, `.csv`, or `.txt`). The app will automatically index them.
4. **Chat**: Type a question about your files in the chat box. Expand the "Retrieval Engine Insights" section below the answer to see how the app found the information.
5. **Reset**: Use the "Clear Interface" button to erase the chat history and clear all indexed files from memory.
