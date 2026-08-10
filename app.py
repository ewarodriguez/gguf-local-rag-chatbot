import os
import streamlit as st
import pypdf
import docx2txt
import pandas as pd
from llama_cpp import Llama
from rank_bm25 import BM25Okapi  # Run: pip install rank-bm25
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import time

# Create necessary local storage directories
MODEL_DIR = "local_models"
os.makedirs(MODEL_DIR, exist_ok=True)

st.set_page_config(page_title="Portable Offline Document Analyzer (RAG Chatbot)", layout="wide", page_icon="💾")
st.title("💾 Portable Offline Document Analyzer (RAG Chatbot)")
st.caption("100% Private local execution using GGUF models on your CPU.")

# Cache embedding model cleanly to prevent massive reload lag inside Streamlit loops
@st.cache_resource
def load_embedding_model():
    # Lightweight local 30MB model ideal for keeping memory pools optimized
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

embeddings = load_embedding_model()

# -----------------------------------------------------------------------------
# 1. SIDEBAR CONFIGURATION (Models & Files)
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("🎛️ AI Engine Configuration")
    
    st.markdown("""
    **Recommended Free Showcases:**
    * 🚀 [Download Qwen 2.5 1.5B (Fastest)](https://huggingface.co)
    * 🧠 [Download Llama 3.2 3B (Smartest)](https://huggingface.co)
    """)
    
    # Drag-and-drop model file uploader
    uploaded_model = st.file_uploader("Upload a new .gguf model:", type=["gguf"])
    if uploaded_model is not None:
        target_model_path = os.path.join(MODEL_DIR, uploaded_model.name)
        if not os.path.exists(target_model_path):
            with st.spinner("Saving model weights to disk..."):
                with open(target_model_path, "wb") as f:
                    f.write(uploaded_model.getbuffer())
            st.success(f"Saved: {uploaded_model.name}")
            st.rerun()

    # Scan directory and provide a dynamic switcher dropdown menu
    available_models = [f for f in os.listdir(MODEL_DIR) if f.endswith(".gguf")]
    selected_model_name = st.selectbox(
        "Select active AI model:",
        options=available_models,
        index=0 if available_models else None,
        placeholder="Upload a model to begin"
    )

    st.write("---")
    st.header("📂 Knowledge Base Upload")
    uploaded_docs = st.file_uploader(
        "Upload reference files:", 
        type=["pdf", "docx", "xlsx", "xls", "csv", "txt"], 
        accept_multiple_files=True
    )

# -----------------------------------------------------------------------------
# 2. FILE EXTRACTION & ISOLATED RAG PIPELINE
# -----------------------------------------------------------------------------
def chunk_text(text, max_chars=1000, overlap=150):
    chunks = []
    start = 0
    while start < len(text):
        end = start + max_chars
        chunks.append(text[start:end])
        start += (max_chars - overlap)
    return chunks

def extract_text_from_file(uploaded_file):
    file_extension = uploaded_file.name.split(".")[-1].lower()
    extracted_text = ""
    try:
        if file_extension == "pdf":
            pdf_reader = pypdf.PdfReader(uploaded_file)
            for page in pdf_reader.pages:
                text = page.extract_text()
                if text:
                    extracted_text += text + "\n"
        elif file_extension in ["docx", "doc"]:
            extracted_text = docx2txt.process(uploaded_file)
        elif file_extension == "txt":
            extracted_text = uploaded_file.read().decode("utf-8", errors="ignore")
            
        elif file_extension in ["xlsx", "xls", "csv"]:
            # Programmatic Structural Analysis for structured data
            if file_extension == "csv":
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file, engine="openpyxl")
            
            row_count = len(df)
            col_count = len(df.columns)
            
            summary_lines = [
                f"--- Spreadsheet Structure Analysis: {uploaded_file.name} ---",
                f"Total Row Count: {row_count} data records",
                f"Total Column Count: {col_count} columns",
                "\nColumn Header & Data Type Identification Matrix:"
            ]
            
            for col in df.columns:
                col_type = str(df[col].dtype)
                sample_values = df[col].dropna().head(2).tolist()
                
                if "int" in col_type or "float" in col_type:
                    friendly_type = "Numeric (Integer/Decimal numbers or Financial Currency values)"
                elif "datetime" in col_type:
                    friendly_type = "Temporal (Date and Time stamps)"
                elif "bool" in col_type:
                    friendly_type = "Boolean (True/False or Yes/No data indicators)"
                else:
                    friendly_type = "Categorical / Text Data String strings"
                
                sample_str = f" [Example records: {sample_values}]" if sample_values else " [Empty column]"
                summary_lines.append(f" * Column Header Name: '{col}' -> Detected Type: {friendly_type}{sample_str}")
                
            extracted_text = "\n".join(summary_lines) + "\n\n"
            
    except Exception as e:
        st.sidebar.error(f"Error parsing {uploaded_file.name}: {str(e)}")
        return ""
        
    return extracted_text

# Initialize split structural states
if "bm25_chunks" not in st.session_state:
    st.session_state.bm25_chunks = []
if "faiss_chunks" not in st.session_state:
    st.session_state.faiss_chunks = []
if "file_names" not in st.session_state:
    st.session_state.file_names = []
if "bm25_index" not in st.session_state:
    st.session_state.bm25_index = None
if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = None

# Monitor file input changes across page renders - strictly tracks identity diffs
if uploaded_docs:
    current_files = [doc.name for doc in uploaded_docs]
    
    if current_files != st.session_state.file_names:
        bm25_pool = []
        faiss_pool = []
        
        for doc in uploaded_docs:
            extracted_data = extract_text_from_file(doc)
            file_extension = doc.name.split(".")[-1].lower()
            
            # STRICT DATA ROUTING RULES
            if file_extension in ["xlsx", "xls", "csv", "txt"]:
                bm25_pool.append(extracted_data)
            else:
                faiss_pool.extend(chunk_text(extracted_data))
                
        st.session_state.bm25_chunks = bm25_pool
        st.session_state.faiss_chunks = faiss_pool
        st.session_state.file_names = current_files
        
        # Build individual isolated routing engines cleanly based on active pools
        if bm25_pool:
            tokenized_corpus = [doc.lower().split(" ") for doc in bm25_pool]
            st.session_state.bm25_index = BM25Okapi(tokenized_corpus)
            st.sidebar.success(f"📊 Indexed {len(bm25_pool)} layout summaries into BM25.")
        else:
            st.session_state.bm25_index = None
            
        if faiss_pool:
            st.session_state.faiss_index = FAISS.from_texts(faiss_pool, embeddings)
            st.sidebar.success(f"🧠 Indexed {len(faiss_pool)} narrative chunks into FAISS Vector.")
        else:
            st.session_state.faiss_index = None

total_chunks_loaded = len(st.session_state.bm25_chunks) + len(st.session_state.faiss_chunks)
if total_chunks_loaded > 0:
    st.sidebar.success(f"✅ Loaded {total_chunks_loaded} isolated search targets to system RAM.")

# -----------------------------------------------------------------------------
# 3. HARDWARE-OPTIMIZED DYNAMIC MODEL LOADER
# -----------------------------------------------------------------------------
@st.cache_resource
def load_selected_llm(model_name):
    if not model_name:
        return None
    model_path = os.path.join(MODEL_DIR, model_name)
    
    # Context window scales safely according to weights density footprints
    if "1.5b" in model_name.lower():
        max_context = 3072  
    elif "3b" in model_name.lower():
        max_context = 2048  
    else:
        max_context = 1536   
        
    return Llama(
        model_path=model_path, 
        n_ctx=max_context,   
        n_threads=max(1, os.cpu_count() - 1), # Uses max available CPU cores safely 
        n_batch=512,                           # Higher batch allocation drastically speeds up initial ingestion processing
        n_gpu_layers=0,                        # Locks computation cleanly directly onto system RAM channels
        verbose=False
    )

# -----------------------------------------------------------------------------
# 4. ROUTED RETRIEVAL FUNCTION WITH METADATA TRACKING
# -----------------------------------------------------------------------------
def get_routed_context_with_meta(query, max_outputs=2):
    """
    Simultaneously pulls top contextual data rows and logs which engine was triggered.
    """
    retrieved_segments = []
    triggered_engines = []
    
    # Route A: Tabular Exact Token Lookups via BM25
    if st.session_state.bm25_index and st.session_state.bm25_chunks:
        tokenized_query = query.lower().split(" ")
        bm25_matches = st.session_state.bm25_index.get_top_n(tokenized_query, st.session_state.bm25_chunks, n=max_outputs)
        if bm25_matches:
            retrieved_segments.extend(bm25_matches)
            triggered_engines.append("BM25 (Keyword Engine)")
        
    # Route B: Narrative Semantic Coordinates via FAISS Vector
    if st.session_state.faiss_index and st.session_state.faiss_chunks:
        sim_results = st.session_state.faiss_index.similarity_search(query, k=max_outputs)
        faiss_matches = [doc.page_content for doc in sim_results]
        if faiss_matches:
            retrieved_segments.extend(faiss_matches)
            triggered_engines.append("FAISS (Semantic Vector Engine)")
        
    return "\n\n---\n\n".join(retrieved_segments), triggered_engines

# -----------------------------------------------------------------------------
# 5. CHAT AND STREAMING INFERENCE LOGIC
# -----------------------------------------------------------------------------
if selected_model_name:
    llm = load_selected_llm(selected_model_name)  
    st.info(f"🟢 **Active Engine:** {selected_model_name} (Running on System CPU)")
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if st.button("🗑️ Clear Interface"):
        st.session_state.chat_history = []
        st.session_state.bm25_chunks = []
        st.session_state.faiss_chunks = []
        st.session_state.file_names = []
        st.session_state.bm25_index = None
        st.session_state.faiss_index = None
        st.rerun()

    for message in st.session_state.chat_history:
        if isinstance(message, dict) and "role" in message and "content" in message:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if "speed_metric" in message:
                    st.caption(message["speed_metric"])
                
                # Render metadata expander panel from past chat histories
                if "engines_used" in message and message["engines_used"]:
                    with st.expander("🔍 Retrieval Engine Insights", expanded=False):
                        st.markdown(f"**Active Search Channels:** `{', '.join(message['engines_used'])}`")
                        st.markdown(message["engine_explanation"])
        else:
            st.session_state.chat_history = []
            st.rerun()

    # Process live message entry
    if user_query := st.chat_input("Ask a question about your uploaded documents:"):
        with st.chat_message("user"):
            st.markdown(user_query)
        st.session_state.chat_history.append({"role": "user", "content": user_query})

        with st.chat_message("assistant"):
            
            # 🎯 BULLETPROOF CHITCHAT FILTER: Clean punctuation and check word intent tokens
            punctuation_table = str.maketrans("", "", '?.!,-_')
            clean_tokens = set(user_query.lower().translate(punctuation_table).split())

            greetings = {"hi", "hello", "hey", "greetings", "yo", "there"}
            gratitude = {"thank", "thanks", "thankyou", "appreciate", "helpful", "much", "so", "for", "the", "help"}
            farewells = {"bye", "goodbye", "later", "see", "you", "quit", "exit"}

            all_chitchat_words = greetings | gratitude | farewells
            is_chitchat = len(clean_tokens) > 0 and clean_tokens.issubset(all_chitchat_words)

            messages = []
            engines_used = []
            explanation_md = ""

            # Route 1: Small Talk Bypass
            if is_chitchat:
                messages.append({
                    "role": "system",
                    "content": "You are a polite, helpful offline document assistant. Respond to the user's greeting, gratitude, or farewell concisely."
                })
                messages.append({
                    "role": "user",
                    "content": user_query
                })
                context_snippet = ""
                engines_used = ["Bypass (Conversational Mode)"]
                explanation_md = "- **RAG Pipeline Bypassed:** The query was recognized as standard small talk or politeness, so document retrieval was skipped to allow a natural conversational response."
            
            # Route 2: Regular Document RAG Route
            elif total_chunks_loaded > 0:
                context_snippet, engines_used = get_routed_context_with_meta(user_query)
                files_list_str = ", ".join(st.session_state.file_names) if st.session_state.file_names else "None"
                
                messages.append({
                    "role": "system",
                    "content": (
                        f"You are a strict offline document analyzer. The user has uploaded: [{files_list_str}]. "
                        "Answer the question using ONLY the provided document context records. "
                        "If the answer cannot be confidently derived from the provided context block, reply exactly with: "
                        "'I cannot find that information in the uploaded documents.' Do not invent facts."
                    )
                })
                messages.append({
                    "role": "user",
                    "content": f"Isolated Reference Context:\n{context_snippet}\n\nQuestion: {user_query}"
                })
                
                if "BM25 (Keyword Engine)" in engines_used:
                    explanation_md += "- **BM25 Keyword Engine triggered:** Used exclusively for your **Excel, CSV, and TXT structural summaries**. It targets absolute keyword intersections, columns, and exact metadata rules without processing geometric embeddings.\n"
                if "FAISS (Semantic Vector Engine)" in engines_used:
                    explanation_md += "- **FAISS Semantic Engine triggered:** Used exclusively for your **PDF and Word narrative layouts**. It performs dense matrix lookups in system memory to match conceptual synonyms and deep conversational phrases.\n"
            
            # Route 3: Standard Local Assistant (No documents loaded)
            else:
                messages.append({
                    "role": "system",
                    "content": "You are a helpful AI assistant running locally on a user's CPU. No files are currently uploaded."
                })
                messages.append({
                    "role": "user",
                    "content": user_query
                })
            
            text_placeholder = st.empty()
            full_response = ""
            token_count = 0
            start_time = time.time()
            
            # Execute token stream chunk rendering loop
            if llm is not None:
                response_stream = llm.create_chat_completion(
                    messages=messages,
                    max_tokens=450,
                    temperature=0.7 if is_chitchat else 0.1,  # Strict temp for data, creative for small talk
                    stream=True
                )
                
                for chunk in response_stream:
                    if "choices" in chunk and len(chunk["choices"]) > 0:
                        delta = chunk["choices"][0]["delta"]
                        if "content" in delta:
                            token_text = delta["content"]
                            full_response += token_text
                            token_count += 1
                            text_placeholder.markdown(full_response + "▌")
                
                text_placeholder.markdown(full_response)
                
                elapsed_time = time.time() - start_time
                tokens_per_second = token_count / elapsed_time if elapsed_time > 0 else 0
                speed_metric_text = f"⏱️ Streamed {token_count} tokens in {elapsed_time:.2f}s ({tokens_per_second:.2f} tok/sec)"
                st.caption(speed_metric_text)
                
                # Render engine insights dropdown right below streaming completion block
                if engines_used:
                    with st.expander("🔍 Retrieval Engine Insights", expanded=True):
                        st.markdown(f"**Active Search Channels:** `{', '.join(engines_used)}`")
                        st.markdown(explanation_md)
                
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": full_response,
                    "speed_metric": speed_metric_text,
                    "engines_used": engines_used,
                    "engine_explanation": explanation_md
                })
            else:
                st.error("Error: Local AI engine model weights configuration missing.")
else:
    st.warning("⚠️ Drop a valid .gguf model file into the sidebar uploader to launch execution matrices.")

