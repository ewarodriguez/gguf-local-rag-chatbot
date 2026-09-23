import os
import streamlit as st
import pypdf
import docx  # Structural python-docx parser for high-accuracy extraction
import pandas as pd
from llama_cpp import Llama
from rank_bm25 import BM25Okapi  
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import time
import webbrowser  # Native Python library to trigger external browser tabs
from ydata_profiling import ProfileReport

# Create necessary local storage directories
MODEL_DIR = "local_models"
os.makedirs(MODEL_DIR, exist_ok=True)

st.set_page_config(page_title="Universal File Explorer", layout="wide")
st.title("🕵️‍♂️ Universal File Explorer 📂")
st.caption("100% Private local execution using GGUF models on your CPU.")

# Cache embedding model cleanly to prevent reload lag inside Streamlit loops
@st.cache_resource
def load_embedding_model():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

embeddings = load_embedding_model()

# -----------------------------------------------------------------------------
# 1. SIDEBAR CONFIGURATION (Models & Files)
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Model Configuration")
    
    st.markdown("""
    **Sample Models used in this App:**
    * [Download Qwen 2.5 1.5B (Fastest)](https://huggingface.co)
    * [Download Llama 3.2 3B (Smartest)](https://huggingface.co)
    """)
    
    uploaded_model = st.file_uploader("Upload a new .gguf model:", type=["gguf"])
    if uploaded_model is not None:
        target_model_path = os.path.join(MODEL_DIR, uploaded_model.name)
        if not os.path.exists(target_model_path):
            with st.spinner("Saving model weights to disk..."):
                with open(target_model_path, "wb") as f:
                    f.write(uploaded_model.getbuffer())
            st.success(f"Saved: {uploaded_model.name}")
            st.rerun()

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
# 2. FILE EXTRACTION & RECURSIVE BLOCKS PARSER
# -----------------------------------------------------------------------------
def chunk_text(text, max_chars=1600, overlap=450):
    """
    Advanced text splitter optimized for manually spaced signature lines.
    Keeps names, stacked titles, and proximity context bundled together in memory.
    """
    paragraphs = text.split("\n\n")
    chunks = []
    current_chunk = []
    current_length = 0
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
            
        if len(para) > max_chars:
            lines = para.split("\n")
            for line in lines:
                if len(line) > max_chars:
                    words = line.split(" ")
                    for word in words:
                        if current_length + len(word) + 1 > max_chars:
                            chunks.append(" ".join(current_chunk))
                            # Deep historical tracking window for manually stacked text blocks
                            current_chunk = current_chunk[-max(1, int(overlap/15)):] if overlap > 0 else []
                            current_length = sum(len(w) + 1 for w in current_chunk)
                        current_chunk.append(word)
                        current_length += len(word) + 1
                else:
                    if current_length + len(line) + 1 > max_chars:
                        chunks.append(" ".join(current_chunk))
                        current_chunk = []
                        current_length = 0
                    current_chunk.append(line)
                    current_length += len(line) + 1
        else:
            # Safely link sequential lines together into solid vectors
            if current_length + len(para) + 2 > max_chars:
                chunks.append("\n\n".join(current_chunk))
                current_chunk = []
                current_length = 0
            current_chunk.append(para)
            current_length += len(para) + 2
            
    if current_chunk:
        chunks.append("\n\n".join(current_chunk))
        
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
            doc_obj = docx.Document(uploaded_file)
            full_text = []
            
            # Paragraph Extraction with manual whitespace collapsing
            for para in doc_obj.paragraphs:
                p_text = para.text
                if p_text:
                    # Collapses manual tabs/indents to link names directly with titles
                    p_cleaned = " ".join(p_text.split())
                    if p_cleaned:
                        full_text.append(p_cleaned)
            
            # Table Content Processing
            for table in doc_obj.tables:
                for row in table.rows:
                    row_data = [" ".join(cell.text.split()) for cell in row.cells if cell.text.strip()]
                    if row_data:
                        full_text.append(" | ".join(row_data))
                        
            # Uses single line breaks to cleanly group stacked components during chunking
            extracted_text = "\n".join(full_text)
            
        elif file_extension == "txt":
            extracted_text = uploaded_file.read().decode("utf-8", errors="ignore")
            
        elif file_extension in ["xlsx", "xls", "csv"]:
            if file_extension == "csv":
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file, engine="openpyxl")
            
            st.session_state[f"df_{uploaded_file.name}"] = df
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

# Monitor file input changes across page renders
if uploaded_docs:
    current_files = [doc.name for doc in uploaded_docs]
    if current_files != st.session_state.file_names:
        bm25_pool = []
        faiss_pool = []
        
        for doc in uploaded_docs:
            extracted_data = extract_text_from_file(doc)
            file_extension = doc.name.split(".")[-1].lower()
            
            if file_extension in ["xlsx", "xls", "csv", "txt"]:
                bm25_pool.append(extracted_data)
            else:
                faiss_pool.extend(chunk_text(extracted_data))
                
        st.session_state.bm25_chunks = bm25_pool
        st.session_state.faiss_chunks = faiss_pool
        st.session_state.file_names = current_files
        
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
    Preserves original BM25 logic while expanding FAISS chunks to catch full employee lists.
    """
    retrieved_segments = []
    triggered_engines = []
    
    # Route A: Tabular Exact Token Lookups via BM25 (UNTOUCHED ORIGINAL LOGIC)
    if st.session_state.bm25_index and st.session_state.bm25_chunks:
        tokenized_query = query.lower().split(" ")
        bm25_matches = st.session_state.bm25_index.get_top_n(tokenized_query, st.session_state.bm25_chunks, n=max_outputs)
        if bm25_matches:
            retrieved_segments.extend(bm25_matches)
            triggered_engines.append("BM25 (Keyword Engine)")
        
    # Route B: Narrative Semantic Coordinates via FAISS Vector
    if st.session_state.faiss_index and st.session_state.faiss_chunks:
        # FIX: Expanded to k=8 chunks so names scattered across several pages 
        # or manual signature layout spacing lines are never blocked out of context.
        sim_results = st.session_state.faiss_index.similarity_search(query, k=8)
        faiss_matches = [doc.page_content for doc in sim_results]
        if faiss_matches:
            retrieved_segments.extend(faiss_matches)
            triggered_engines.append("FAISS (Semantic Vector Engine)")
        
    return "\n\n---\n\n".join(retrieved_segments), triggered_engines


# -----------------------------------------------------------------------------
# 5. CHAT AND STREAMING INFERENCE LOGIC (Part 5 - Section A - Fixed Output Loop)
# -----------------------------------------------------------------------------
if selected_model_name:
    llm = load_selected_llm(selected_model_name)  
    st.info(f"🟢 **Active Engine:** {selected_model_name} (Running on System CPU)")
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display prior conversation logs dynamically
    for message in st.session_state.chat_history:
        if isinstance(message, dict) and "role" in message and "content" in message:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if "speed_metric" in message:
                    st.caption(message["speed_metric"])
                
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
            # Clean punctuation and check word tokens flexibly
            punctuation_table = str.maketrans("", "", '?.!,-_')
            clean_tokens = set(user_query.lower().translate(punctuation_table).split())

            greetings = {"hi", "hello", "hey", "greetings", "yo", "there"}
            gratitude = {"thank", "thanks", "thankyou", "appreciate", "helpful", "much", "so", "for", "the", "help"}
            farewells = {"bye", "goodbye", "later", "see", "you", "quit", "exit"}
            all_chitchat_words = greetings | gratitude | farewells

            # Flexible intersection-based chitchat bypass rule
            is_chitchat = len(clean_tokens.intersection(all_chitchat_words)) > 0

            messages = []
            engines_used = []
            explanation_md = ""

            if is_chitchat:
                messages.append({
                    "role": "system",
                    "content": "You are a friendly companion. Respond to the user's greeting, gratitude, or farewell warmly and naturally in one short sentence."
                })
                messages.append({
                    "role": "user",
                    "content": user_query
                })
                context_snippet = ""
                engines_used = ["Bypass (Conversational Mode)"]
                explanation_md = "- **RAG Pipeline Bypassed:** Small talk or politeness recognized. System responded using a friendly tone template."
            
            elif total_chunks_loaded > 0:
                context_snippet, engines_used = get_routed_context_with_meta(user_query)
                files_list_str = ", ".join(st.session_state.file_names) if st.session_state.file_names else "None"
                
                messages.append({
                    "role": "system",
                    "content": f"You are a strict offline document analyzer. User uploaded: [{files_list_str}]. Answer using ONLY the provided text blocks."
                })
                messages.append({"role": "user", "content": f"Isolated Reference Context:\n{context_snippet}\n\nQuestion: {user_query}"})
                
                if "BM25 (Keyword Engine)" in engines_used:
                    explanation_md += "- **BM25 Keyword Engine triggered:** Evaluated tabular metadata layouts.\n"
                if "FAISS (Semantic Vector Engine)" in engines_used:
                    explanation_md += "- **FAISS Semantic Engine triggered:** Scanned context chunks to gather full matching context strings.\n"
            else:
                messages.append({"role": "system", "content": "You are a helpful AI assistant running locally on a user's CPU."})
                messages.append({"role": "user", "content": user_query})
            
            text_placeholder = st.empty()
            full_response = ""
            token_count = 0
            start_time = time.time()
            
            if llm is not None:
                response_stream = llm.create_chat_completion(
                    messages=messages, 
                    max_tokens=750, 
                    temperature=0.7 if is_chitchat else 0.1, 
                    stream=True
                )
                
                for chunk in response_stream:
                    try:
                        if isinstance(chunk, dict):
                            if "choices" in chunk and len(chunk["choices"]) > 0:
                                delta = chunk["choices"][0].get("delta", {})
                                if "content" in delta:
                                    full_response += delta["content"]
                                    token_count += 1
                                    text_placeholder.markdown(full_response + "▌")
                        else:
                            if hasattr(chunk, "choices") and len(chunk.choices) > 0:
                                delta = chunk.choices[0].delta
                                if hasattr(delta, "content") and delta.content is not None:
                                    full_response += delta.content
                                    token_count += 1
                                    text_placeholder.markdown(full_response + "▌")
                    except (IndexError, AttributeError, KeyError):
                        continue
                
                text_placeholder.markdown(full_response)
                elapsed_time = time.time() - start_time
                tokens_per_second = token_count / elapsed_time if elapsed_time > 0 else 0
                speed_metric_text = f"⏱️ Streamed {token_count} tokens in {elapsed_time:.2f}s ({tokens_per_second:.2f} tok/sec)"
                st.caption(speed_metric_text)
                
                if engines_used:
                    with st.expander("🔍 Retrieval Engine Insights", expanded=True):
                        st.markdown(f"**Active Search Channels:** `{', '.join(engines_used)}`")
                        st.markdown(explanation_md)
                
                # CRUCIAL FIX: Session memory appends BEFORE layout updates
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": full_response,
                    "speed_metric": speed_metric_text,
                    "engines_used": engines_used,
                    "engine_explanation": explanation_md
                })
                # Removed the standalone st.rerun() from here to preserve the generation stream
            else:
                st.error("Error: Local AI engine model weights configuration missing.")


    # -----------------------------------------------------------------------------
    # 5. CHAT AND STREAMING INFERENCE LOGIC (Part 5 - Section B)
    # -----------------------------------------------------------------------------
    # On-demand persistent ydata-profiling block with Stage Percentage Progress Bars
    if uploaded_docs:
        for doc in uploaded_docs:
            file_extension = doc.name.split(".")[-1].lower()
            
            if file_extension in ["xlsx", "xls", "csv"]:
                st.write("---")
                st.subheader(f"📊 Dataset Structure Profiling: {doc.name}")
                
                report_file_key = f"report_path_{doc.name}"
                
                if st.button(f"Generate Profile Analysis for {doc.name}"):
                    df_to_analyze = st.session_state.get(f"df_{doc.name}")
                    if df_to_analyze is not None:
                        is_large_file = doc.size > (10 * 1024 * 1024) # 10MB Threshold
                        
                        # Progress Bar Configuration
                        progress_text = "Phase 1/3: Ingesting dataset matrix variables..."
                        progress_bar = st.progress(0, text=progress_text)
                        
                        time.sleep(0.4)
                        progress_bar.progress(25, text="Phase 2/3: Structural data type verification...")
                        
                        time.sleep(0.4)
                        progress_bar.progress(75, text="Phase 3/3: Running deep statistical profiling distribution summaries...")
                        
                        if is_large_file:
                            st.warning("⚠️ Large file detected. Processing using minimal layout to save RAM.")
                            profile = ProfileReport(df_to_analyze, title=f"Profile: {doc.name}", minimal=True, explorative=False, progress_bar=False)
                        else:
                            profile = ProfileReport(df_to_analyze, title=f"Profile: {doc.name}", explorative=True, progress_bar=False)
                        
                        # Generate clean path string
                        output_filename = f"profile_{doc.name.split('.')[-2]}.html"
                        profile.to_file(output_filename, silent=True)
                        
                        progress_bar.progress(100, text="✨ Analysis complete! Launching document externally...")
                        time.sleep(0.6)
                        progress_bar.empty()
                        
                        # Launch generated report in separate default browser window tab
                        webbrowser.open_new_tab(output_filename)
                        st.session_state[report_file_key] = output_filename
                    else:
                        st.error("Data frame state not found. Please reload the document file.")
                
                if report_file_key in st.session_state:
                    existing_path = st.session_state[report_file_key]
                    st.success(f"✅ Active profile generated successfully! Saved locally as: `{existing_path}`")
                    
                    if st.button(f"🔄 Re-open {existing_path} in New Tab"):
                        webbrowser.open_new_tab(existing_path)

    # Clear Interface Button dropped cleanly to the absolute base (Option A)
    if st.session_state.chat_history:
        st.write("") 
        if st.button("🗑️ Clear Interface", use_container_width=False):
            st.session_state.chat_history = []
            st.session_state.bm25_chunks = []
            st.session_state.faiss_chunks = []
            st.session_state.file_names = []
            st.session_state.bm25_index = None
            st.session_state.faiss_index = None
            
            # Wipe stored dataframes and paths out of state memory
            for key in list(st.session_state.keys()):
                if key.startswith("df_") or key.startswith("report_path_"):
                    del st.session_state[key]
                    
            st.rerun()
else:
    st.warning("⚠️ Drop a valid .gguf model file into the sidebar uploader to launch execution matrices.")





# import os
# import streamlit as st
# import pypdf
# import docx2txt
# import pandas as pd
# from llama_cpp import Llama
# from rank_bm25 import BM25Okapi  
# from langchain_community.vectorstores import FAISS
# from langchain_huggingface import HuggingFaceEmbeddings
# import time
# import streamlit.components.v1 as components
# from ydata_profiling import ProfileReport

# # Create necessary local storage directories
# MODEL_DIR = "local_models"
# os.makedirs(MODEL_DIR, exist_ok=True)

# st.set_page_config(page_title="Universal File Explorer", layout="wide")
# st.title("🕵️‍♂️ Universal File Explorer 📂")
# st.caption("100% Private local execution using GGUF models on your CPU.")

# # Cache embedding model cleanly to prevent massive reload lag inside Streamlit loops
# @st.cache_resource
# def load_embedding_model():
#     return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# embeddings = load_embedding_model()

# # -----------------------------------------------------------------------------
# # 1. SIDEBAR CONFIGURATION (Models & Files)
# # -----------------------------------------------------------------------------
# with st.sidebar:
#     st.header("⚙️ Model Configuration")
    
#     st.markdown("""
#     **Sample Models used in this App:**
#     * [Download Qwen 2.5 1.5B (Fastest)](https://huggingface.co)
#     * [Download Llama 3.2 3B (Smartest)](https://huggingface.co)
#     """)
    
#     uploaded_model = st.file_uploader("Upload a new .gguf model:", type=["gguf"])
#     if uploaded_model is not None:
#         target_model_path = os.path.join(MODEL_DIR, uploaded_model.name)
#         if not os.path.exists(target_model_path):
#             with st.spinner("Saving model weights to disk..."):
#                 with open(target_model_path, "wb") as f:
#                     f.write(uploaded_model.getbuffer())
#             st.success(f"Saved: {uploaded_model.name}")
#             st.rerun()

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
#         type=["pdf", "docx", "xlsx", "xls", "csv", "txt"], 
#         accept_multiple_files=True
#     )

# # -----------------------------------------------------------------------------
# # 2. FILE EXTRACTION & ISOLATED RAG PIPELINE
# # -----------------------------------------------------------------------------
# def chunk_text(text, max_chars=1000, overlap=150):
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
#         elif file_extension == "txt":
#             extracted_text = uploaded_file.read().decode("utf-8", errors="ignore")
            
#         elif file_extension in ["xlsx", "xls", "csv"]:
#             if file_extension == "csv":
#                 df = pd.read_csv(uploaded_file)
#             else:
#                 df = pd.read_excel(uploaded_file, engine="openpyxl")
            
#             # Save raw dataframe to session state so profiling can reference it later
#             st.session_state[f"df_{uploaded_file.name}"] = df
            
#             row_count = len(df)
#             col_count = len(df.columns)
            
#             summary_lines = [
#                 f"--- Spreadsheet Structure Analysis: {uploaded_file.name} ---",
#                 f"Total Row Count: {row_count} data records",
#                 f"Total Column Count: {col_count} columns",
#                 "\nColumn Header & Data Type Identification Matrix:"
#             ]
            
#             for col in df.columns:
#                 col_type = str(df[col].dtype)
#                 sample_values = df[col].dropna().head(2).tolist()
                
#                 if "int" in col_type or "float" in col_type:
#                     friendly_type = "Numeric (Integer/Decimal numbers or Financial Currency values)"
#                 elif "datetime" in col_type:
#                     friendly_type = "Temporal (Date and Time stamps)"
#                 elif "bool" in col_type:
#                     friendly_type = "Boolean (True/False or Yes/No data indicators)"
#                 else:
#                     friendly_type = "Categorical / Text Data String strings"
                
#                 sample_str = f" [Example records: {sample_values}]" if sample_values else " [Empty column]"
#                 summary_lines.append(f" * Column Header Name: '{col}' -> Detected Type: {friendly_type}{sample_str}")
                
#             extracted_text = "\n".join(summary_lines) + "\n\n"
            
#     except Exception as e:
#         st.sidebar.error(f"Error parsing {uploaded_file.name}: {str(e)}")
#         return ""
        
#     return extracted_text

# # Initialize split structural states
# if "bm25_chunks" not in st.session_state:
#     st.session_state.bm25_chunks = []
# if "faiss_chunks" not in st.session_state:
#     st.session_state.faiss_chunks = []
# if "file_names" not in st.session_state:
#     st.session_state.file_names = []
# if "bm25_index" not in st.session_state:
#     st.session_state.bm25_index = None
# if "faiss_index" not in st.session_state:
#     st.session_state.faiss_index = None

# if uploaded_docs:
#     current_files = [doc.name for doc in uploaded_docs]
#     if current_files != st.session_state.file_names:
#         bm25_pool = []
#         faiss_pool = []
        
#         for doc in uploaded_docs:
#             extracted_data = extract_text_from_file(doc)
#             file_extension = doc.name.split(".")[-1].lower()
            
#             if file_extension in ["xlsx", "xls", "csv", "txt"]:
#                 bm25_pool.append(extracted_data)
#             else:
#                 faiss_pool.extend(chunk_text(extracted_data))
                
#         st.session_state.bm25_chunks = bm25_pool
#         st.session_state.faiss_chunks = faiss_pool
#         st.session_state.file_names = current_files
        
#         if bm25_pool:
#             tokenized_corpus = [doc.lower().split(" ") for doc in bm25_pool]
#             st.session_state.bm25_index = BM25Okapi(tokenized_corpus)
#             st.sidebar.success(f"📊 Indexed {len(bm25_pool)} layout summaries into BM25.")
#         else:
#             st.session_state.bm25_index = None
            
#         if faiss_pool:
#             st.session_state.faiss_index = FAISS.from_texts(faiss_pool, embeddings)
#             st.sidebar.success(f"🧠 Indexed {len(faiss_pool)} narrative chunks into FAISS Vector.")
#         else:
#             st.session_state.faiss_index = None

# total_chunks_loaded = len(st.session_state.bm25_chunks) + len(st.session_state.faiss_chunks)
# if total_chunks_loaded > 0:
#     st.sidebar.success(f"✅ Loaded {total_chunks_loaded} isolated search targets to system RAM.")

# # -----------------------------------------------------------------------------
# # 3. HARDWARE-OPTIMIZED DYNAMIC MODEL LOADER
# # -----------------------------------------------------------------------------
# @st.cache_resource
# def load_selected_llm(model_name):
#     if not model_name:
#         return None
#     model_path = os.path.join(MODEL_DIR, model_name)
    
#     # Context window scales safely according to weights density footprints
#     if "1.5b" in model_name.lower():
#         max_context = 3072  
#     elif "3b" in model_name.lower():
#         max_context = 2048  
#     else:
#         max_context = 1536   
        
#     return Llama(
#         model_path=model_path, 
#         n_ctx=max_context,   
#         n_threads=max(1, os.cpu_count() - 1), # Uses max available CPU cores safely 
#         n_batch=512,                           # Higher batch allocation drastically speeds up initial ingestion processing
#         n_gpu_layers=0,                        # Locks computation cleanly directly onto system RAM channels
#         verbose=False
#     )

# # -----------------------------------------------------------------------------
# # 4. ROUTED RETRIEVAL FUNCTION WITH METADATA TRACKING
# # -----------------------------------------------------------------------------
# def get_routed_context_with_meta(query, max_outputs=2):
#     """
#     Simultaneously pulls top contextual data rows and logs which engine was triggered.
#     """
#     retrieved_segments = []
#     triggered_engines = []
    
#     # Route A: Tabular Exact Token Lookups via BM25
#     if st.session_state.bm25_index and st.session_state.bm25_chunks:
#         tokenized_query = query.lower().split(" ")
#         bm25_matches = st.session_state.bm25_index.get_top_n(tokenized_query, st.session_state.bm25_chunks, n=max_outputs)
#         if bm25_matches:
#             retrieved_segments.extend(bm25_matches)
#             triggered_engines.append("BM25 (Keyword Engine)")
        
#     # Route B: Narrative Semantic Coordinates via FAISS Vector
#     if st.session_state.faiss_index and st.session_state.faiss_chunks:
#         sim_results = st.session_state.faiss_index.similarity_search(query, k=max_outputs)
#         faiss_matches = [doc.page_content for doc in sim_results]
#         if faiss_matches:
#             retrieved_segments.extend(faiss_matches)
#             triggered_engines.append("FAISS (Semantic Vector Engine)")
        
#     return "\n\n---\n\n".join(retrieved_segments), triggered_engines


# # -----------------------------------------------------------------------------
# # 5. CHAT AND STREAMING INFERENCE LOGIC (Section A)
# # -----------------------------------------------------------------------------
# if selected_model_name:
#     llm = load_selected_llm(selected_model_name)  
#     st.info(f"🟢 **Active Engine:** {selected_model_name} (Running on System CPU)")
    
#     if "chat_history" not in st.session_state:
#         st.session_state.chat_history = []

#     # Display prior conversation logs dynamically
#     for message in st.session_state.chat_history:
#         if isinstance(message, dict) and "role" in message and "content" in message:
#             with st.chat_message(message["role"]):
#                 st.markdown(message["content"])
#                 if "speed_metric" in message:
#                     st.caption(message["speed_metric"])
                
#                 if "engines_used" in message and message["engines_used"]:
#                     with st.expander("🔍 Retrieval Engine Insights", expanded=False):
#                         st.markdown(f"**Active Search Channels:** `{', '.join(message['engines_used'])}`")
#                         st.markdown(message["engine_explanation"])
#         else:
#             st.session_state.chat_history = []
#             st.rerun()

#     # Process live message entry
#     if user_query := st.chat_input("Ask a question about your uploaded documents:"):
#         with st.chat_message("user"):
#             st.markdown(user_query)
#         st.session_state.chat_history.append({"role": "user", "content": user_query})

#         with st.chat_message("assistant"):
#             # Chit-chat token filter
#             punctuation_table = str.maketrans("", "", '?.!,-_')
#             clean_tokens = set(user_query.lower().translate(punctuation_table).split())

#             greetings = {"hi", "hello", "hey", "greetings", "yo", "there"}
#             gratitude = {"thank", "thanks", "thankyou", "appreciate", "helpful", "much", "so", "for", "the", "help"}
#             farewells = {"bye", "goodbye", "later", "see", "you", "quit", "exit"}

#             all_chitchat_words = greetings | gratitude | farewells
#             is_chitchat = len(clean_tokens) > 0 and clean_tokens.issubset(all_chitchat_words)

#             messages = []
#             engines_used = []
#             explanation_md = ""

#             if is_chitchat:
#                 messages.append({"role": "system", "content": "You are a polite, helpful offline document assistant. Respond concisely."})
#                 messages.append({"role": "user", "content": user_query})
#                 engines_used = ["Bypass (Conversational Mode)"]
#                 explanation_md = "- **RAG Pipeline Bypassed:** Conversational response."
            
#             elif total_chunks_loaded > 0:
#                 context_snippet, engines_used = get_routed_context_with_meta(user_query)
#                 files_list_str = ", ".join(st.session_state.file_names) if st.session_state.file_names else "None"
                
#                 messages.append({
#                     "role": "system",
#                     "content": f"You are a strict offline document analyzer. User uploaded: [{files_list_str}]. Answer using ONLY the provided text blocks."
#                 })
#                 messages.append({"role": "user", "content": f"Isolated Reference Context:\n{context_snippet}\n\nQuestion: {user_query}"})
                
#                 if "BM25 (Keyword Engine)" in engines_used:
#                     explanation_md += "- **BM25 Keyword Engine triggered:** Evaluated tabular metadata layouts.\n"
#                 if "FAISS (Semantic Vector Engine)" in engines_used:
#                     explanation_md += "- **FAISS Semantic Engine triggered:** Scanned unstructured text chunks.\n"
#             else:
#                 messages.append({"role": "system", "content": "You are a helpful AI assistant running locally on a user's CPU."})
#                 messages.append({"role": "user", "content": user_query})
            
#             text_placeholder = st.empty()
#             full_response = ""
#             token_count = 0
#             start_time = time.time()
            
#             if llm is not None:
#                 response_stream = llm.create_chat_completion(messages=messages, max_tokens=450, temperature=0.7 if is_chitchat else 0.1, stream=True)
                
#                 for chunk in response_stream:
#                     if "choices" in chunk and len(chunk["choices"]) > 0:
#                         delta = chunk["choices"][0]["delta"]
#                         if "content" in delta:
#                             full_response += delta["content"]
#                             token_count += 1
#                             text_placeholder.markdown(full_response + "▌")
                
#                 text_placeholder.markdown(full_response)
#                 elapsed_time = time.time() - start_time
#                 tokens_per_second = token_count / elapsed_time if elapsed_time > 0 else 0
#                 speed_metric_text = f"⏱️ Streamed {token_count} tokens in {elapsed_time:.2f}s ({tokens_per_second:.2f} tok/sec)"
#                 st.caption(speed_metric_text)
                
#                 if engines_used:
#                     with st.expander("🔍 Retrieval Engine Insights", expanded=True):
#                         st.markdown(f"**Active Search Channels:** `{', '.join(engines_used)}`")
#                         st.markdown(explanation_md)
                
#                 st.session_state.chat_history.append({
#                     "role": "assistant",
#                     "content": full_response,
#                     "speed_metric": speed_metric_text,
#                     "engines_used": engines_used,
#                     "engine_explanation": explanation_md
#                 })
#                 st.rerun()
#             else:
#                 st.error("Error: Local AI engine model weights configuration missing.")

#     # -----------------------------------------------------------------------------
#     # 5. CHAT AND STREAMING INFERENCE LOGIC (Section B - Standalone Export)
#     # -----------------------------------------------------------------------------
#     import webbrowser  # Native Python library to trigger external browser tabs

#     if uploaded_docs:
#         for doc in uploaded_docs:
#             file_extension = doc.name.split(".")[-1].lower()
            
#             if file_extension in ["xlsx", "xls", "csv"]:
#                 st.write("---")
#                 st.subheader(f"📊 Dataset Structure Profiling: {doc.name}")
                
#                 # Dynamic tracker to remember if a file report has been generated
#                 report_file_key = f"report_path_{doc.name}"
                
#                 if st.button(f"Generate Profile Analysis for {doc.name}"):
#                     df_to_analyze = st.session_state.get(f"df_{doc.name}")
#                     if df_to_analyze is not None:
#                         is_large_file = doc.size > (10 * 1024 * 1024) # 10MB Threshold
                        
#                         # Progress Bar Configuration
#                         progress_text = "Phase 1/3: Ingesting dataset matrix variables..."
#                         progress_bar = st.progress(0, text=progress_text)
                        
#                         time.sleep(0.4)
#                         progress_bar.progress(25, text="Phase 2/3: Structural data type verification...")
                        
#                         time.sleep(0.4)
#                         progress_bar.progress(75, text="Phase 3/3: Running deep statistical profiling distribution summaries...")
                        
#                         if is_large_file:
#                             st.warning("⚠️ Large file detected. Processing using minimal layout to save RAM.")
#                             profile = ProfileReport(df_to_analyze, title=f"Profile: {doc.name}", minimal=True, explorative=False, progress_bar=False)
#                         else:
#                             profile = ProfileReport(df_to_analyze, title=f"Profile: {doc.name}", explorative=True, progress_bar=False)
                        
#                         # Generate the dynamic path string 
#                         output_filename = f"profile_{doc.name.split('.')[0]}.html"
                        
#                         # Write the standalone document onto local storage disk (open_browser=False lets python handle it)
#                         profile.to_file(output_filename, silent=True)
                        
#                         progress_bar.progress(100, text="✨ Analysis complete! Launching document externally...")
#                         time.sleep(0.6)
#                         progress_bar.empty()
                        
#                         # Launch the generated report in the system's default native browser window
#                         webbrowser.open_new_tab(output_filename)
                        
#                         # Save path context to session state memory
#                         st.session_state[report_file_key] = output_filename
#                     else:
#                         st.error("Data frame state not found. Please reload the document file.")
                
#                 # Display structural breadcrumb logs if file already exists in directory
#                 if report_file_key in st.session_state:
#                     existing_path = st.session_state[report_file_key]
#                     st.success(f"✅ Active profile generated successfully! Saved locally as: `{existing_path}`")
                    
#                     # Provide an option to re-open the external window tab if closed by user
#                     if st.button(f"🔄 Re-open {existing_path} in New Tab"):
#                         webbrowser.open_new_tab(existing_path)

#     # Clear Interface Button dropped cleanly to the absolute base (Option A)
#     if st.session_state.chat_history:
#         st.write("") 
#         if st.button("🗑️ Clear Interface", use_container_width=False):
#             st.session_state.chat_history = []
#             st.session_state.bm25_chunks = []
#             st.session_state.faiss_chunks = []
#             st.session_state.file_names = []
#             st.session_state.bm25_index = None
#             st.session_state.faiss_index = None
            
#             # Wipe stored dataframes and generated file paths out of state memory
#             for key in list(st.session_state.keys()):
#                 if key.startswith("df_") or key.startswith("report_path_"):
#                     del st.session_state[key]
                    
#             st.rerun()
# else:
#     st.warning("⚠️ Drop a valid .gguf model file into the sidebar uploader to launch execution matrices.")

