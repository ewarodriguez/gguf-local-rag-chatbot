import os
import streamlit as st
import pypdf
import docx2txt
import pandas as pd
from llama_cpp import Llama
import time

# Create necessary local storage directories
MODEL_DIR = "local_models"
os.makedirs(MODEL_DIR, exist_ok=True)

st.set_page_config(page_title="Portable Offline Document Analyzer (RAG Chatbot)", layout="wide", page_icon="💾")
st.title("💾 Portable Offline Document Analyzer (RAG Chatbot)")
st.caption("100% Private local execution using GGUF models on your CPU.")

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
# 2. FILE EXTRACTION & DETAILED EXCEL ANALYSIS ENGINE
# -----------------------------------------------------------------------------
def chunk_text(text, max_chars=1000, overlap=200):
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
            
        elif file_extension in ["xlsx", "xls","csv"]:
            # 📊 Programmatic Structural Analysis for Excel
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
                # Deduce clean pandas data type rules
                col_type = str(df[col].dtype)
                sample_values = df[col].dropna().head(2).tolist()
                
                # Convert abstract raw pandas type string tags to simple human terms
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

        elif file_extension == "csv":
            # 📊 Programmatic Structural Analysis for CSV files
            df = pd.read_csv(uploaded_file)
            
            row_count = len(df)
            col_count = len(df.columns)
            
            summary_lines = [
                f"--- Spreadsheet Structure Analysis (CSV): {uploaded_file.name} ---",
                f"Total Row Count: {row_count} data records",
                f"Total Column Count: {col_count} columns",
                "\nColumn Header & Data Type Identification Matrix:"
            ]
            
            for col in df.columns:
                col_type = str(df[col].dtype)
                sample_values = df[col].dropna().head(2).tolist()
                
                if "int" in col_type or "float" in col_type:
                    friendly_type = "Numeric (Numbers or Currency values)"
                elif "datetime" in col_type:
                    friendly_type = "Temporal (Date and Time stamps)"
                elif "bool" in col_type:
                    friendly_type = "Boolean (True/False)"
                else:
                    friendly_type = "Categorical / Text Data String values"
                
                sample_str = f" [Example records: {sample_values}]" if sample_values else " [Empty column]"
                summary_lines.append(f" * Column Header Name: '{col}' -> Detected Type: {friendly_type}{sample_str}")
                
            extracted_text = "\n".join(summary_lines) + "\n\n"
            
    except Exception as e:
        st.sidebar.error(f"Error parsing {uploaded_file.name}: {str(e)}")
        return ""
        
    return extracted_text

# Initialize persistent session storage state variables
if "knowledge_chunks" not in st.session_state:
    st.session_state.knowledge_chunks = []
if "file_names" not in st.session_state:
    st.session_state.file_names = []

# Monitor file input changes across page renders
if uploaded_docs:
    raw_text = ""
    current_files = []
    for doc in uploaded_docs:
        current_files.append(doc.name)
        file_extension = doc.name.split(".")[-1].lower()
        
        # Process data structural lines natively
        extracted_data = extract_text_from_file(doc)
        
        if file_extension in ["xlsx", "xls","csv","txt"]:
            # 🎯 BYPASS CHUNKER FOR EXCEL: Keep the spreadsheet structural analysis whole
            raw_text += extracted_data
        else:
            # Keep standard text-chunk processing mechanics for Word and PDF layout
            raw_text += extracted_data
            
    # Parse standard structural snippets safely
    if any(f.split(".")[-1].lower() not in ["xlsx", "xls","csv","txt"] for f in current_files):
        st.session_state.knowledge_chunks = chunk_text(raw_text)
    else:
        # If ONLY an excel file exists, save the whole summary block directly
        st.session_state.knowledge_chunks = [raw_text]
        
    st.session_state.file_names = current_files  

if st.session_state.knowledge_chunks:
    st.sidebar.success(f"✅ Loaded {len(st.session_state.knowledge_chunks)} reference windows to system RAM.")

# -----------------------------------------------------------------------------
# 3. DYNAMIC LOCAL MODEL LOADER
# -----------------------------------------------------------------------------
@st.cache_resource
def load_selected_llm(model_name):
    model_path = os.path.join(MODEL_DIR, model_name)
    if "1.5b" in model_name.lower():
        max_context = 3072  
    elif "3b" in model_name.lower():
        max_context = 2048  
    else:
        max_context = 1536   
        
    return Llama(
        model_path=model_path, 
        n_ctx=max_context,   
        n_threads=4,         
        n_batch=256,         
        n_gpu_layers=0,   
        verbose=False
    )

# -----------------------------------------------------------------------------
# 4. KEYWORD RETRIEVER ENGINE
# -----------------------------------------------------------------------------
def get_relevant_context(query, chunks, max_chunks=3):
    """
    Ranks text segments matching user keywords to minimize prompt size.
    Extracts purely string values out of sorted tracking lists safely.
    """
    if not chunks:
        return ""
    
    query_words = [word.lower() for word in query.split() if len(word) > 2]
    if not query_words:
        query_words = [query.lower()]
        
    scored_chunks = []
    for chunk in chunks:
        chunk_lower = chunk.lower()
        # Count word match occurrences
        score = sum(1 for word in query_words if word in chunk_lower)
        scored_chunks.append((score, chunk))
    
    # Sort descending explicitly by the integer score (index 0 of the tuple)
    scored_chunks.sort(key=lambda x: x[0], reverse=True)
    
    # 🎯 THE FIX: Check item[0] (the integer score) instead of the whole tuple
    top_chunks = [item[1] for item in scored_chunks[:max_chunks] if item[0] > 0]
    
    # Fallback to the top chunks if no exact keyword match is found
    if not top_chunks:
        top_chunks = [item[1] for item in scored_chunks[:max_chunks]]
        
    return "\n\n".join(top_chunks)


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
        st.session_state.knowledge_chunks = []
        st.session_state.file_names = []
        st.rerun()

    for message in st.session_state.chat_history:
        if isinstance(message, dict) and "role" in message and "content" in message:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if "speed_metric" in message:
                    st.caption(message["speed_metric"])
        else:
            st.session_state.chat_history = []
            st.rerun()

    # Process live message entry
    if user_query := st.chat_input("Ask a question about your uploaded documents:"):
        with st.chat_message("user"):
            st.markdown(user_query)
        st.session_state.chat_history.append({"role": "user", "content": user_query})

        with st.chat_message("assistant"):
            context_snippet = get_relevant_context(user_query, st.session_state.knowledge_chunks)
            files_list_str = ", ".join(st.session_state.file_names) if st.session_state.file_names else "None"
            
            messages = []
            if st.session_state.knowledge_chunks:
                messages.append({
                    "role": "system",
                    "content": (
                        f"You are a helpful file analyzer. The user has uploaded: [{files_list_str}]. "

                        f"Answer the user precisely using the structural analysis block below. "
                        f"If the question asks about data types, row metrics, or column structure, answer directly based on this analysis metadata."
                    )
                })
                messages.append({
                    "role": "user",
                    "content": f"Document Structural Blueprint:\n{context_snippet}\n\nQuestion: {user_query}"
                })
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
            response_stream = llm.create_chat_completion(
                messages=messages,
                max_tokens=350,
                temperature=0.2,
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
            
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": full_response,
                "speed_metric": speed_metric_text
            })


# import os
# import streamlit as st
# import pypdf
# import docx2txt
# import openpyxl
# import pandas as pd
# from llama_cpp import Llama
# import time

# # Create necessary local storage directories
# MODEL_DIR = "local_models"
# os.makedirs(MODEL_DIR, exist_ok=True)

# st.set_page_config(page_title="Portable Offline Document Analyzer (RAG Chatbot)", layout="wide", page_icon="💾")
# st.title("💾 Portable Offline Document Analyzer using (RAG Chatbot)")
# st.caption("100% Private local execution using GGUF models on your CPU.")

# # -----------------------------------------------------------------------------
# # 1. SIDEBAR CONFIGURATION (Models & Files)
# # -----------------------------------------------------------------------------
# with st.sidebar:
#     st.header("🎛️ AI Engine Configuration")
    
#     st.markdown("""
#     **Recommended Free Showcases:**
#     * 🚀 [Download Qwen 2.5 1.5B (Fastest)](https://huggingface.co)
#     * 🧠 [Download Llama 3.2 3B (Smartest)](https://huggingface.co)
#     """)
    
#     # Drag-and-drop model file uploader
#     uploaded_model = st.file_uploader("Upload a new .gguf model:", type=["gguf"])
#     if uploaded_model is not None:
#         target_model_path = os.path.join(MODEL_DIR, uploaded_model.name)
#         if not os.path.exists(target_model_path):
#             with st.spinner("Saving model weights to disk..."):
#                 with open(target_model_path, "wb") as f:
#                     f.write(uploaded_model.getbuffer())
#             st.success(f"Saved: {uploaded_model.name}")
#             st.rerun()

#     # Scan directory and provide a dynamic switcher dropdown menu
#     available_models = [f for f in os.listdir(MODEL_DIR) if f.endswith(".gguf")]
#     selected_model_name = st.selectbox(
#         "Select active AI model:",
#         options=available_models,
#         index=0 if available_models else None,
#         placeholder="Upload a model to begin"
#     )

#     st.write("---")
#     st.header("📂 Knowledge Base Upload")
#     uploaded_docs = st.file_uploader(
#         "Upload reference files:", 
#         type=["pdf", "docx", "xlsx", "xls"], 
#         accept_multiple_files=True
#     )

# # -----------------------------------------------------------------------------
# # 2. FILE EXTRACTION & CHUNKING ENGINE
# # -----------------------------------------------------------------------------
# def chunk_text(text, max_chars=1000, overlap=200):
#     chunks = []
#     start = 0
#     while start < len(text):
#         end = start + max_chars
#         chunks.append(text[start:end])
#         start += (max_chars - overlap)
#     return chunks

# def extract_text_from_file(uploaded_file):
#     file_extension = uploaded_file.name.split(".")[-1].lower()
#     extracted_text = ""
#     try:
#         if file_extension == "pdf":
#             pdf_reader = pypdf.PdfReader(uploaded_file)
#             for page in pdf_reader.pages:
#                 text = page.extract_text()
#                 if text:
#                     extracted_text += text + "\n"
#         elif file_extension in ["docx", "doc"]:
#             extracted_text = docx2txt.process(uploaded_file)
#         elif file_extension in ["xlsx", "xls"]:
#             df = pd.read_excel(uploaded_file)
#             extracted_text = df.to_markdown(index=False)
#     except Exception as e:
#         return f"\nError parsing file {uploaded_file.name}: {str(e)}\n"
#     return extracted_text

# # Initialize persistent session storage state variables
# if "knowledge_chunks" not in st.session_state:
#     st.session_state.knowledge_chunks = []
# if "file_names" not in st.session_state:
#     st.session_state.file_names = []

# # Monitor file input changes across page renders
# if uploaded_docs:
#     raw_text = ""
#     current_files = []
#     for doc in uploaded_docs:
#         current_files.append(doc.name)
#         raw_text += f"\n\n--- Source Document: {doc.name} ---\n"
#         raw_text += extract_text_from_file(doc)
    
#     st.session_state.knowledge_chunks = chunk_text(raw_text)
#     st.session_state.file_names = current_files  # 🧠 Store file names persistently

# if st.session_state.knowledge_chunks:
#     st.sidebar.success(f"✅ Loaded {len(st.session_state.knowledge_chunks)} document windows to RAM.")

# # -----------------------------------------------------------------------------
# # 3. DYNAMIC LOCAL MODEL LOADER
# # -----------------------------------------------------------------------------
# @st.cache_resource
# def load_selected_llm(model_name):
#     model_path = os.path.join(MODEL_DIR, model_name)
#     if "1.5b" in model_name.lower():
#         max_context = 3072  
#     elif "3b" in model_name.lower():
#         max_context = 2048  
#     else:
#         max_context = 1536   
        
#     return Llama(
#         model_path=model_path, 
#         n_ctx=max_context,   
#         n_threads=4,         
#         n_batch=256,         
#         n_gpu_layers=0,   
#         verbose=False
#     )

# # -----------------------------------------------------------------------------
# # 4. KEYWORD RETRIEVER ENGINE
# # -----------------------------------------------------------------------------
# def get_relevant_context(query, chunks, max_chunks=3):
#     if not chunks:
#         return ""
    
#     query_words = [word.lower() for word in query.split() if len(word) > 2]
#     if not query_words:
#         query_words = [query.lower()]
        
#     scored_chunks = []
#     for chunk in chunks:
#         chunk_lower = chunk.lower()
#         score = sum(1 for word in query_words if word in chunk_lower)
#         scored_chunks.append((score, chunk))
    
#     scored_chunks.sort(key=lambda x: x[0], reverse=True)
    
#     # Extract index 1 (text string) out of our container tuple structure
#     top_chunks = [item[1] for item in scored_chunks[:max_chunks] if item[0] > 0]
    
#     if not top_chunks:
#         top_chunks = [item[1] for item in scored_chunks[:max_chunks]]
        
#     return "\n\n".join(top_chunks)

# # -----------------------------------------------------------------------------
# # 5. CHAT AND STREAMING INFERENCE LOGIC
# # -----------------------------------------------------------------------------
# if selected_model_name:
#     llm = load_selected_llm(selected_model_name)  
#     st.info(f"🟢 **Active Engine:** {selected_model_name} (Running on System CPU)")
    
#     if "chat_history" not in st.session_state:
#         st.session_state.chat_history = []

#     if st.button("🗑️ Clear Interface"):
#         st.session_state.chat_history = []
#         st.session_state.knowledge_chunks = []
#         st.session_state.file_names = []
#         st.rerun()

#     # Display historical chat text structures
#     for message in st.session_state.chat_history:
#         if isinstance(message, dict) and "role" in message and "content" in message:
#             with st.chat_message(message["role"]):
#                 st.markdown(message["content"])
#                 if "speed_metric" in message:
#                     st.caption(message["speed_metric"])
#         else:
#             st.session_state.chat_history = []
#             st.rerun()

#     # Process live message entry
#     if user_query := st.chat_input("Ask a question about your uploaded documents:"):
#         with st.chat_message("user"):
#             st.markdown(user_query)
#         st.session_state.chat_history.append({"role": "user", "content": user_query})

#         with st.chat_message("assistant"):
#             context_snippet = get_relevant_context(user_query, st.session_state.knowledge_chunks)
            
#             # Format the active files list cleanly for the system prompt
#             files_list_str = ", ".join(st.session_state.file_names) if st.session_state.file_names else "None"
            
#             messages = []
#             if st.session_state.knowledge_chunks:
#                 # 🧠 CRITICAL FIX: Explicitly tell the system prompt what files are currently in memory
#                 messages.append({
#                     "role": "system",
#                     "content": (
#                         f"You are a helpful file analyzer. The user has currently uploaded the following files: [{files_list_str}]. "
#                         f"Answer the user precisely using the provided context data block below. "
#                         f"If the answer cannot be found in the context data text, but it's a direct question about the file names or file upload status, use your system knowledge to answer it. "
#                         f"Otherwise, if specific data is missing from the text, state explicitly 'Data not found in source text'."
#                     )
#                 })
#                 messages.append({
#                     "role": "user",
#                     "content": f"Context Data Snippet:\n{context_snippet}\n\nQuestion: {user_query}"
#                 })
#             else:
#                 messages.append({
#                     "role": "system",
#                     "content": "You are a helpful AI assistant running locally on a user's CPU. No files are currently uploaded."
#                 })
#                 messages.append({
#                     "role": "user",
#                     "content": user_query
#                 })
            
#             text_placeholder = st.empty()
#             full_response = ""
#             token_count = 0
#             start_time = time.time()
            
#             # Execute token stream chunk rendering loop
#             response_stream = llm.create_chat_completion(
#                 messages=messages,
#                 max_tokens=350,
#                 temperature=0.2,
#                 stream=True  
#             )
            
#             for chunk in response_stream:
#                 if "choices" in chunk and len(chunk["choices"]) > 0:
#                     delta = chunk["choices"][0]["delta"]
#                     if "content" in delta:
#                         token_text = delta["content"]
#                         full_response += token_text
#                         token_count += 1
#                         text_placeholder.markdown(full_response + "▌")
            
#             text_placeholder.markdown(full_response)
            
#             elapsed_time = time.time() - start_time

#             # Calculate token generation performance speed
#             tokens_per_second = token_count / elapsed_time if elapsed_time > 0 else 0

#             # Format the performance metrics display layout string
#             speed_metric_text = f"⏱️ Streamed {token_count} tokens in {elapsed_time:.2f}s ({tokens_per_second:.2f} tok/sec)"

#             # Render the metric directly underneath the chat text stream container
#             st.caption(speed_metric_text)

#             # Persist the final assistant response along with metrics to session state history
#             st.session_state.chat_history.append({
#                 "role": "assistant",
#                 "content": full_response,
#                 "speed_metric": speed_metric_text
#             })

