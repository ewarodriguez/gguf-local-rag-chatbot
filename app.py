import os
import streamlit as st
import docx  
import pandas as pd
from llama_cpp import Llama
from rank_bm25 import BM25Okapi  
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import time
import webbrowser  
import re

# Create necessary local storage directories
MODEL_DIR = "local_models"
os.makedirs(MODEL_DIR, exist_ok=True)

st.set_page_config(page_title="Universal File Explorer", layout="wide")
st.title("🕵️‍♂️ Universal File Explorer 📂")
st.caption("100% Private local execution using GGUF models on your CPU.")

# OPTIMIZATION: Cached resource pinned strictly to your CPU
@st.cache_resource(show_spinner="Loading Embedding Model(all-MiniLM-L6-v2)...")
def load_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'} 
    )

embeddings = load_embedding_model()

# -----------------------------------------------------------------------------
# 1. SIDEBAR CONFIGURATION (Models & Files)
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Model Configuration")
    
    st.markdown("""
    **Sample Models used in this App:**
    * [Download Qwen 2.5 1.5B (Faster)](https://huggingface.co)
    * [Download Llama 3.2 3B (Smarter)](https://huggingface.co)
    """)
    
    uploaded_model = st.file_uploader("Upload a new .gguf model:", type=["gguf"])
    if uploaded_model is not None:
        target_model_path = os.path.join(MODEL_DIR, uploaded_model.name)
        if not os.path.exists(target_model_path):
            with st.spinner("Saving model weights to disk in chunks (Memory-Safe)..."):
                with open(target_model_path, "wb") as f:
                    while chunk := uploaded_model.read(1024 * 1024):
                        f.write(chunk)
            st.success(f"Saved: {uploaded_model.name}")
            st.rerun()

    available_models = [f for f in os.listdir(MODEL_DIR) if f.endswith(".gguf")]
    selected_model_name = st.selectbox(
        "Select active AI model:",
        options=available_models,
        index=0 if available_models else None,
        placeholder="Upload a .gguf model here"
    )

    st.write("---")
    st.header("📂 File Upload")
    uploaded_docs = st.file_uploader(
        "Upload file here:", 
        type=["pdf", "docx", "xlsx", "xls", "csv", "txt"], 
        accept_multiple_files=True
    )


# -----------------------------------------------------------------------------
# 2. ADVANCED STRUCTURAL DOCLING PIPELINE FOR HYBRID / COMPLEX PDFs
# -----------------------------------------------------------------------------
def extract_and_chunk_pdf_with_docling(uploaded_file):
    """
    Parses complex layout paths at maximum speed. Integrates strict Token-Guard 
    clipping using a native transformers tokenizer to prevent indexing overflows.
    """
    # COMPLETE FUNCTIONAL SCOPE IMPORTS MATRIX
    from docling.datamodel.base_models import InputFormat
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
    from docling.datamodel.pipeline_options import TableStructureOptions, PdfPipelineOptions
    from docling.chunking import HybridChunker
    from transformers import AutoTokenizer


    temp_path = f"temp_{uploaded_file.name}"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    try:
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_table_structure = True
        pipeline_options.table_structure_options = TableStructureOptions(mode="fast")
        pipeline_options.do_ocr = False 
        pipeline_options.generate_page_images = False  
        pipeline_options.generate_picture_images = False
        
        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options,
                    backend=PyPdfiumDocumentBackend
                )
            }
        )
        
        result = converter.convert(temp_path)
        
        chunker = HybridChunker(
            tokenizer="BAAI/bge-small-en-v1.5",
            max_tokens=350,
            merge_max_tokens=450  
        )
        
        # FIX: Instantiate the native tokenizer directly. Fast, safe, and independent.
        native_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        
        chunks = []
        for doc_chunk in chunker.chunk(result.document):
            chunk_clean = str(doc_chunk.text).strip()
            if chunk_clean and isinstance(chunk_clean, str) and len(chunk_clean) > 0:
                
                # TOKEN-GUARD SECURE SHIELDING USING NATIVE TOKENIZER
                token_indices = native_tokenizer.encode(chunk_clean, add_special_tokens=False)
                
                if len(token_indices) > 512:
                    chunk_clean = native_tokenizer.decode(token_indices[:512], skip_special_tokens=True)
                
                chunks.append(chunk_clean)
                
        return chunks
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

# -----------------------------------------------------------------------------
# HIGH-FIDELITY FALLBACK PARSERS (Optimized for DOCX, Tables, and TXT)
# -----------------------------------------------------------------------------
def chunk_text(text, max_chars=1400, overlap=350):
    """
    Splits document body text while strictly safeguarding sentence blocks,
    table row groupings, and logical structural definitions.
    """
    paragraphs = text.split("\n\n")
    chunks = []
    current_chunk = []
    current_length = 0
    overlap_step = max(1, int(overlap / 15)) if overlap > 0 else 0
    
    for para in paragraphs:
        para = para.strip()
        if not para: continue
        if len(para) > max_chars:
            lines = para.split("\n")
            for line in lines:
                if len(line) > max_chars:
                    words = line.split(" ")
                    for word in words:
                        if current_length + len(word) + 1 > max_chars:
                            chunks.append(" ".join(current_chunk))
                            current_chunk = current_chunk[-overlap_step:] if overlap_step > 0 else []
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
            if current_length + len(para) + 2 > max_chars:
                chunks.append("\n\n".join(current_chunk))
                current_chunk = []
                current_length = 0
            current_chunk.append(para)
            current_length += len(para) + 2
            
    if current_chunk:
        chunks.append("\n\n".join(current_chunk) if "\n\n" in text else " ".join(current_chunk))
    return chunks

def extract_text_from_file(uploaded_file):
    """
    Extracts text layouts from Word, TXT, and Excel. Revised to structural
    markdown table logic so it preserves grid metrics across docx layouts.
    """
    file_extension = uploaded_file.name.split(".")[-1].lower()
    extracted_text = ""
    try:
        if file_extension == "docx":
            doc_obj = docx.Document(uploaded_file)
            full_text = []
            
            # 1. Harvest native document paragraphs sequentially (retains \n\n boundaries)
            for para in doc_obj.paragraphs:
                p_text = para.text.strip()
                if p_text: 
                    full_text.append(p_text)
            
            # 2. FIX: Harvest tables structurally using clean pipe layout delimiters
            for table in doc_obj.tables:
                full_text.append("\n[--- Document Vector Table Block ---]")
                for row in table.rows:
                    row_data = []
                    for cell in row.cells:
                        cell_cleaned = " ".join(cell.text.strip().split())
                        # Keep column indicators consistent even if cells are empty
                        row_data.append(cell_cleaned if cell_cleaned else " ")
                    if any(c.strip() for c in row_data):
                        full_text.append(f"| {' | '.join(row_data)} |")
                full_text.append("[--- End Table Block ---]\n")
                
            extracted_text = "\n\n".join(full_text)
            
        elif file_extension == "txt":
            extracted_text = uploaded_file.read().decode("utf-8", errors="ignore")
            
        elif file_extension in ["xlsx", "xls", "csv"]:
            if file_extension == "csv": df = pd.read_csv(uploaded_file)
            else: df = pd.read_excel(uploaded_file, engine="xlrd" if file_extension == "xls" else "openpyxl")
            st.session_state[f"df_{uploaded_file.name}"] = df
            summary_lines = [
                f"--- Spreadsheet Structure Analysis: {uploaded_file.name} ---",
                f"Total Row Count: {len(df)} data records | Total Column Count: {len(df.columns)} columns"
            ]
            extracted_text = "\n".join(summary_lines) + "\n\n"
    except Exception as e:
        st.sidebar.error(f"Error parsing {uploaded_file.name}: {str(e)}")
        return ""
    return extracted_text

# -----------------------------------------------------------------------------
# DISPATCH ROUTING INTERFACE
# -----------------------------------------------------------------------------
if "bm25_chunks" not in st.session_state: st.session_state["bm25_chunks"] = []
if "faiss_chunks" not in st.session_state: st.session_state["faiss_chunks"] = []
if "file_names" not in st.session_state: st.session_state["file_names"] = []
if "bm25_index" not in st.session_state: st.session_state["bm25_index"] = None
if "faiss_index" not in st.session_state: st.session_state["faiss_index"] = None

if uploaded_docs:
    current_files = [doc.name for doc in uploaded_docs]
    if current_files != st.session_state["file_names"]:
        bm25_pool, faiss_pool = [], []
        
        for doc in uploaded_docs:
            file_extension = doc.name.split(".")[-1].lower()
            
            if file_extension == "pdf":
                with st.spinner(f"Running Layout Analysis Matrix on {doc.name}..."):
                    pdf_chunks = extract_and_chunk_pdf_with_docling(doc)
                    faiss_pool.extend(pdf_chunks)
                    bm25_pool.extend(pdf_chunks)
            else:
                extracted_data = extract_text_from_file(doc)
                if not extracted_data: 
                    continue
                
                if file_extension in ["xlsx", "xls", "csv", "txt"]:
                    bm25_pool.append(extracted_data)
                elif file_extension == "docx":
                    docx_chunks = chunk_text(extracted_data)
                    # CRITICAL RETRIEVAL FIX: Dual indexing prevents search mismatch and unlocks docx answers
                    faiss_pool.extend(docx_chunks)
                    bm25_pool.extend(docx_chunks)
                    
        st.session_state["bm25_chunks"] = bm25_pool
        st.session_state["faiss_chunks"] = faiss_pool
        st.session_state["file_names"] = current_files
        
        if bm25_pool:
            tokenized_corpus = [[w for w in doc.lower().split(" ") if w] for doc in bm25_pool]
            st.session_state["bm25_index"] = BM25Okapi(tokenized_corpus)
            st.sidebar.success(f"✅ Indexed {len(bm25_pool)} tabular and narrative records into BM25.")
        else:
            st.session_state["bm25_index"] = None
            
        if faiss_pool:
            # SANITIZATION SHIELD: Ensure only valid, non-empty Python strings enter your FAISS index
            clean_faiss_pool = [str(c).strip() for c in faiss_pool if c and isinstance(c, str) and len(str(c).strip()) > 0]
            if clean_faiss_pool:
                st.session_state["faiss_index"] = FAISS.from_texts(clean_faiss_pool, embeddings)


        if faiss_pool:
            # SANITIZATION SHIELD: Ensure only valid, non-empty Python strings enter your FAISS index
            clean_faiss_pool = [str(c).strip() for c in faiss_pool if c and isinstance(c, str) and len(str(c).strip()) > 0]
            if clean_faiss_pool:
                st.session_state["faiss_index"] = FAISS.from_texts(clean_faiss_pool, embeddings)
                st.sidebar.success(f"✅ Indexed {len(clean_faiss_pool)} structural chunks into FAISS.")
            else:
                st.session_state["faiss_index"] = None
        else:
            st.session_state["faiss_index"] = None

total_chunks_loaded = len(st.session_state["bm25_chunks"]) + len(st.session_state["faiss_chunks"])
if total_chunks_loaded > 0:
    st.sidebar.success(f"✅ Loaded {total_chunks_loaded} isolated search targets to system RAM.")


# -----------------------------------------------------------------------------
# 3. HARDWARE-OPTIMIZED DYNAMIC MODEL LOADER (Tailored for SSD Execution)
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Initializing Local AI Engine (This might take a moment)...")
def load_selected_llm(model_name):
    if not model_name:
        return None
    model_path = os.path.join(MODEL_DIR, model_name)
    
    # Balanced context bounds to ensure your 16GB RAM stays fluid alongside active browser tasks
    if "1.5b" in model_name.lower():
        max_context = 4096  
    elif "3b" in model_name.lower():
        max_context = 3072  # Optimal processing roof for Llama 3.2 3B on laptop architectures
    else:
        max_context = 2048   
        
    return Llama(
        model_path=model_path, 
        n_ctx=max_context,   
        # i5-1035G1 has 4 physical cores / 8 threads. 5 threads balances background execution perfectly.
        n_threads=max(1, min(5, os.cpu_count() - 2)), 
        n_batch=128,                           
        n_gpu_layers=0,                        # Kept at 0. Laptop integrated chips cannot offload modern GGUFs safely.
        use_mmap=True,                         # High-Speed SSD optimization: streams weights instantly without clogging system RAM
        use_mlock=False,                       
        verbose=False
    )

# -----------------------------------------------------------------------------
# 4. ENSEMBLE RETRIEVAL FUNCTION WITH METADATA TRACKING
# -----------------------------------------------------------------------------
def get_routed_context_with_meta(query, max_outputs=2):
    """
    Dynamic retrieval pipeline. For hybrid PDF operations, it balances exact 
    BM25 keyword matches with deep FAISS semantic context while removing duplicates.
    """
    retrieved_segments = []
    triggered_engines = []
    seen_segments = set() # Avoid feeding duplicate context walls to the local model
    
    # 1. Interrogate BM25 Lane (Triggered by CSV, XLS, XLSX, TXT, and PDF)
    if st.session_state.bm25_index and st.session_state.bm25_chunks:
        tokenized_query = [w for w in query.lower().split(" ") if w]
        if tokenized_query:
            bm25_matches = st.session_state.get("bm25_index").get_top_n(tokenized_query, st.session_state["bm25_chunks"], n=max_outputs)
            if bm25_matches:
                for match in bm25_matches:
                    if match not in seen_segments:
                        retrieved_segments.append(match)
                        seen_segments.add(match)
                triggered_engines.append("BM25 (Keyword Engine)")
        
    # 2. Interrogate FAISS Lane (Triggered by DOCX, and PDF)
    if st.session_state.faiss_index and st.session_state.faiss_chunks:
        # OPTIMIZATION: Configured to k=6. Leverages dense layout chunks 
        # safely without triggering GGUF context buffer crashes.
        sim_results = st.session_state["faiss_index"].similarity_search(query, k=6)
        faiss_matches = [doc.page_content for doc in sim_results]
        if faiss_matches:
            for match in faiss_matches:
                if match not in seen_segments:
                    retrieved_segments.append(match)
                    seen_segments.add(match)
            triggered_engines.append("FAISS (Semantic Vector Engine)")
        
    return "\n\n---\n\n".join(retrieved_segments), triggered_engines


# -----------------------------------------------------------------------------
# 5. CHAT AND STREAMING INFERENCE LOGIC (Part 5 - Section A - Restored Streaming)
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
    if user_query := st.chat_input("Ask a question about your uploaded file:"):
        with st.chat_message("user"):
            st.markdown(user_query)
        st.session_state.chat_history.append({"role": "user", "content": user_query})

        with st.chat_message("assistant"):
            # Clean punctuation and check word tokens flexibly
            punctuation_table = str.maketrans("", "", '?.!,-_')
            clean_tokens = set(user_query.lower().translate(punctuation_table).split())

            # -----------------------------------------------------------------
            # HIGH-CAPACITY EXPANDED ENGLISH CHITCHAT VOCABULARY MATRIX (PRESERVED)
            # -----------------------------------------------------------------
            greetings = {
                "hi", "hello", "hey", "greetings", "yo", "there", "sup", "morning", 
                "afternoon", "evening", "howdy", "welcome", "heya", "whatsup"
            }
            
            gratitude = {
                "thank", "thanks", "thankyou", "appreciate", "helpful", "much", 
                "so", "for", "the", "help", "grateful", "obliged", "cheers", 
                "thanking", "appreciation", "kind", "helper", "kindness"
            }
            
            praise_affirmation = {
                "good", "great", "awesome", "perfect", "excellent", "amazing", 
                "wonderful", "cool", "nice", "job", "work", "wow", "yes", "yeah", 
                "yup", "ok", "okay", "fine", "sure", "correct", "right", "brilliant",
                "fantastic", "sweet", "rock", "genius", "smart", "incredible", "love",
                "superb", "terrific", "spot", "on", "exactly", "indeed", "gotcha", "yep",
                "well", "done", "neat", "fabulous", "outstanding"  
            }
            
            farewells = {
                "bye", "goodbye", "later", "see", "you", "quit", "exit", "close", 
                "stop", "end", "leave", "done", "peace"
            }
            
            conversational_fillers = {
                "chatbot", "bot", "ai", "assistant", "computer", "machine", "system",
                "who", "are", "you", "what", "is", "your", "name", "can", "do", "how",
                "old", "creator", "made", "built", "dude", "mate", "uh", "um", "er", "hmm",
                "very"  
            }

            # Unify all expanded English categories into one massive keyword net
            all_chitchat_words = greetings | gratitude | praise_affirmation | farewells | conversational_fillers

            # SIMPLE OVERRIDE GUARD: Uses your exact dictionary without adding complex math
            has_matching_words = len(clean_tokens.intersection(all_chitchat_words)) > 0
            
            # If any word matches your net, and the sentence is 10 words or fewer, force companion bypass
            if has_matching_words and len(clean_tokens) <= 10:
                is_chitchat = True
            else:
                is_chitchat = has_matching_words and total_chunks_loaded == 0

            messages = []
            engines_used = []
            explanation_md = ""

            if is_chitchat:
                messages.append({
                    "role": "system",
                    "content": "You are a friendly companion. Respond to the user's greeting, gratitude, or praise warmly, naturally, and concisely in one short sentence."
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
                files_list_str = ", ".join(st.session_state["file_names"]) if st.session_state["file_names"] else "None"
                
                # ADAPTIVE MULTI-DOMAIN KNOWLEDGE GENERATION SCHEDULER
                messages.append({
                    "role": "system",
                    "content": (
                        "You are an expert Data Extraction Engine and Cross-Domain Document Analyst.\n"
                        f"Your mission is to isolate precise, objective, and structurally flawless insights from these files: [{files_list_str}].\n"
                        "CRITICAL STRUCTURAL PROTOCOLS:\n"
                        "- Rely exclusively on the raw text blocks, data grids, and Markdown structures provided.\n"
                        "- ADAPT TO DATA CONTENT: If the context is technical (data science code, formulas, logic), preserve all functional indents, variable names, and bracket mappings [] using clean formatting blocks.\n"
                        "- ADAPT TO ADMIN CONTENT: If the context is general layout text (office guides, policies), output straightforward factual summaries using clear lists or headers.\n"
                        "- Retain complete structural border alignments for markdown tables exactly as presented.\n"
                        "- Keep your logic clear, structured, and completely free of conversational disclaimers or procedural boilerplate."
                    )
                })
                messages.append({
                    "role": "user", 
                    "content": f"INPUT STORAGE DATA METRICS TO PARSE:\n{context_snippet}\n\nUSER EXTRACTION DIRECTIVE: {user_query}"
                })
                
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
                            choices_list = chunk.get("choices", [])
                            if choices_list and len(choices_list) > 0:
                                delta_dict = choices_list[0].get("delta", {})
                                if "content" in delta_dict:
                                    full_response += delta_dict["content"]
                                    token_count += 1
                                    text_placeholder.markdown(full_response + "▌")
                        else:
                            if hasattr(chunk, "choices") and len(chunk.choices) > 0:
                                first_choice = chunk.choices[0]
                                delta_obj = getattr(first_choice, "delta", None)
                                if delta_obj is not None:
                                    content = getattr(delta_obj, "content", "") if not isinstance(delta_obj, dict) else delta_obj.get("content", "")
                                    if content:
                                        full_response += content
                                        token_count += 1
                                        text_placeholder.markdown(full_response + "▌")
                    except Exception:
                        continue
                
                # FALLBACK DETECTOR: If streaming text failed, run a direct non-stream completion call
                if not full_response.strip():
                    fallback_res = llm.create_chat_completion(messages=messages, max_tokens=600, stream=False)
                    if isinstance(fallback_res, dict) and "choices" in fallback_res:
                        full_response = fallback_res["choices"][0]["message"]["content"]
                
                text_placeholder.markdown(full_response)
                elapsed_time = time.time() - start_time
                tokens_per_second = token_count / elapsed_time if elapsed_time > 0 else 0
                speed_metric_text = f"⏱️ Streamed {token_count} tokens in {elapsed_time:.2f}s ({tokens_per_second:.2f} tok/sec)"
                st.caption(speed_metric_text)
                
                if engines_used:
                    with st.expander("🔍 Retrieval Engine Insights", expanded=True):
                        st.markdown(explanation_md)
                
                st.session_state["chat_history"].append({
                    "role": "assistant", 
                    "content": full_response, 
                    "speed_metric": speed_metric_text,
                    "engines_used": engines_used, 
                    "engine_explanation": explanation_md
                })
            else:
                st.error("Error: Local AI engine model weights configuration missing.")


    # -----------------------------------------------------------------------------
    # DATASET PROFILING ROUTINES (Part 5 - Section B)
    # -----------------------------------------------------------------------------
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
                        # LAZY-LOADING IMPORT OPTIMIZATION: Loaded only when button is clicked
                        from ydata_profiling import ProfileReport

                        # OPTIMIZATION: 5MB threshold protects system RAM alongside the local GGUF model
                        is_large_file = doc.size > (5 * 1024 * 1024)
                        
                        progress_bar = st.progress(0, text="Phase 1/3: Ingesting dataset matrix variables...")
                        time.sleep(0.2)
                        progress_bar.progress(25, text="Phase 2/3: Structural data type verification...")
                        time.sleep(0.2)
                        progress_bar.progress(75, text="Phase 3/3: Running deep statistical profiling summaries...")
                        
                        # OPTIMIZATION: Disables heavy correlation loops on large files to save laptop CPU stress
                        if is_large_file:
                            st.warning("⚠️ Heavy file layout detected. Processing using structural minimal configuration to save system RAM.")
                            profile = ProfileReport(
                                df_to_analyze, 
                                title=f"Profile: {doc.name}", 
                                minimal=True, 
                                explorative=False, 
                                progress_bar=False, 
                                correlations=None, 
                                interactions=None
                            )
                        else:
                            profile = ProfileReport(
                                df_to_analyze, 
                                title=f"Profile: {doc.name}", 
                                explorative=True, 
                                progress_bar=False, 
                                interactions=None
                            )
                        
                        # FIX: Use os.path.splitext to safely handle files with multiple periods (e.g., sales.v2.2026.csv)
                        base_name, _ = os.path.splitext(doc.name)
                        output_filename = f"profile_{base_name}.html"
                        profile.to_file(output_filename, silent=True)
                        
                        progress_bar.progress(100, text="✨ Analysis complete!")
                        time.sleep(0.4)
                        progress_bar.empty()
                        
                        try:
                            webbrowser.open_new_tab(output_filename)
                        except Exception:
                            pass
                        st.session_state[report_file_key] = output_filename
                    else:
                        st.error("Data frame state not found. Please reload the document file.")
                
                if report_file_key in st.session_state:
                    existing_path = st.session_state[report_file_key]
                    st.success(f"✅ Active profile generated successfully!")
                    
                    if os.path.exists(existing_path):
                        with open(existing_path, "r", encoding="utf-8") as f:
                            html_bytes = f.read()
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button(f"🔄 Re-open in Local Browser Tab", key=f"open_{existing_path}"):
                                try:
                                    webbrowser.open_new_tab(existing_path)
                                except Exception:
                                    st.error("Could not trigger browser tab natively.")
                        with col2:
                            st.download_button(
                                label="💾 Download Profiling Report (.html)", 
                                data=html_bytes,
                                file_name=existing_path, 
                                mime="text/html", 
                                key=f"dl_{existing_path}"
                            )

    # -----------------------------------------------------------------------------
    # GLOBAL MEMORY PURGE
    # -----------------------------------------------------------------------------
    if st.session_state["chat_history"]:
        st.write("") 
        if st.button("🗑️ Clear Chat History", use_container_width=False):
            st.session_state["chat_history"] = []
            st.session_state["bm25_chunks"] = []
            st.session_state["faiss_chunks"] = []
            st.session_state["file_names"] = []
            st.session_state["bm25_index"] = None
            st.session_state["faiss_index"] = None
            
            # Dynamically target and wipe background dataframe caching from RAM allocation
            for key in list(st.session_state.keys()):
                if key.startswith("df_") or key.startswith("report_path_"):
                    del st.session_state[key]
            st.rerun()
else:
    st.warning("⚠️ Drop a valid .gguf model file into the sidebar uploader to launch execution matrices.")







# import os
# import streamlit as st
# # Defer heavy analytics/ML imports until needed, or keep grouped cleanly
# import pypdf
# import docx  
# import pandas as pd
# from llama_cpp import Llama
# from rank_bm25 import BM25Okapi  
# from langchain_community.vectorstores import FAISS
# from langchain_huggingface import HuggingFaceEmbeddings
# import time
# import webbrowser  
# from ydata_profiling import ProfileReport
# import re

# # Create necessary local storage directories
# MODEL_DIR = "local_models"
# os.makedirs(MODEL_DIR, exist_ok=True)

# st.set_page_config(page_title="Universal File Explorer", layout="wide")
# st.title("🕵️‍♂️ Universal File Explorer 📂")
# st.caption("100% Private local execution using GGUF models on your CPU.")

# # OPTIMIZATION: Switched to st.cache_resource with clear instructions.
# # Consider changing model_name to a faster local-first alternative if needed.
# @st.cache_resource(show_spinner="Loading Embedding Model(all-MiniLM-L6-v2)...")
# def load_embedding_model():
#     return HuggingFaceEmbeddings(
#         model_name="all-MiniLM-L6-v2",
#         model_kwargs={'device': 'cpu'} # Explicitly pin to CPU to avoid framework overhead
#     )

# embeddings = load_embedding_model()

# # -----------------------------------------------------------------------------
# # 1. SIDEBAR CONFIGURATION (Models & Files)
# # -----------------------------------------------------------------------------
# with st.sidebar:
#     st.header("⚙️ Model Configuration")
    
#     st.markdown("""
#     **Sample Models used in this App:**
#     * [Download Qwen 2.5 1.5B (Faster)](https://huggingface.co)
#     * [Download Llama 3.2 3B (Smarter)](https://huggingface.co)
#     """)
    
#     uploaded_model = st.file_uploader("Upload a new .gguf model:", type=["gguf"])
#     if uploaded_model is not None:
#         target_model_path = os.path.join(MODEL_DIR, uploaded_model.name)
#         if not os.path.exists(target_model_path):
#             with st.spinner("Saving model weights to disk in chunks (Memory-Safe)..."):
#                 # OPTIMIZATION: Read/Write in 1MB chunks instead of holding multi-GB model in RAM
#                 with open(target_model_path, "wb") as f:
#                     while chunk := uploaded_model.read(1024 * 1024):
#                         f.write(chunk)
#             st.success(f"Saved: {uploaded_model.name}")
#             st.rerun()

#     available_models = [f for f in os.listdir(MODEL_DIR) if f.endswith(".gguf")]
#     selected_model_name = st.selectbox(
#         "Select active AI model:",
#         options=available_models,
#         index=0 if available_models else None,
#         placeholder="Upload a .gguf model here"
#     )

#     st.write("---")
#     st.header("📂 File Upload")
#     uploaded_docs = st.file_uploader(
#         "Upload file here:", 
#         type=["pdf", "docx", "xlsx", "xls", "csv", "txt"], 
#         accept_multiple_files=True
#     )

# # -----------------------------------------------------------------------------
# # 2. FILE EXTRACTION & AUTOMATED HYBRID ROUTING PIPELINE
# # -----------------------------------------------------------------------------
# def chunk_text(text, max_chars=1600, overlap=450):
#     paragraphs = text.split("\n\n")
#     chunks = []
#     current_chunk = []
#     current_length = 0
#     overlap_step = max(1, int(overlap / 15)) if overlap > 0 else 0
    
#     for para in paragraphs:
#         para = para.strip()
#         if not para:
#             continue
            
#         if len(para) > max_chars:
#             lines = para.split("\n")
#             for line in lines:
#                 if len(line) > max_chars:
#                     words = line.split(" ")
#                     for word in words:
#                         if current_length + len(word) + 1 > max_chars:
#                             chunks.append(" ".join(current_chunk))
#                             current_chunk = current_chunk[-overlap_step:] if overlap_step > 0 else []
#                             current_length = sum(len(w) + 1 for w in current_chunk)
#                         current_chunk.append(word)
#                         current_length += len(word) + 1
#                 else:
#                     if current_length + len(line) + 1 > max_chars:
#                         chunks.append(" ".join(current_chunk))
#                         current_chunk = []
#                         current_length = 0
#                     current_chunk.append(line)
#                     current_length += len(line) + 1
#         else:
#             if current_length + len(para) + 2 > max_chars:
#                 chunks.append("\n\n".join(current_chunk))
#                 current_chunk = []
#                 current_length = 0
#             current_chunk.append(para)
#             current_length += len(para) + 2
            
#     if current_chunk:
#         chunks.append("\n\n".join(current_chunk) if "\n\n" in text else " ".join(current_chunk))
        
#     return chunks

# def extract_text_from_file(uploaded_file):
#     file_extension = uploaded_file.name.split(".")[-1].lower()
#     extracted_text = ""
#     try:
#         if file_extension == "pdf":
#             pdf_reader = pypdf.PdfReader(uploaded_file)
#             cleaned_text_blocks = []
            
#             for page_idx, page in enumerate(pdf_reader.pages):
#                 layout_text = page.extract_text(extraction_mode="layout")
#                 if layout_text:
#                     page_lines = layout_text.split("\n")
#                     cleaned_lines = []
                    
#                     for line in page_lines:
#                         if line.strip():
#                             cleaned_line = re.sub(r' {4,}', '  ', line).strip()
#                             cleaned_lines.append(cleaned_line)
                    
#                     if cleaned_lines:
#                         page_text_flow = " ".join(cleaned_lines)
#                         page_summary = f"[--- PDF Page {page_idx + 1} Informational Matrix ---]\n{page_text_flow}"
#                         cleaned_text_blocks.append(page_summary)
                        
#             extracted_text = "\n\n".join(cleaned_text_blocks)
                    
#         elif file_extension == "docx":
#             doc_obj = docx.Document(uploaded_file)
#             full_text = []
#             for para in doc_obj.paragraphs:
#                 if para.text.strip():
#                     full_text.append(" ".join(para.text.split()))
            
#             seen_cells = set()
#             for table in doc_obj.tables:
#                 for row in table.rows:
#                     row_data = []
#                     for cell in row.cells:
#                         if cell._tc not in seen_cells:
#                             seen_cells.add(cell._tc)
#                             cell_cleaned = " ".join(cell.text.split())
#                             if cell_cleaned:
#                                 row_data.append(cell_cleaned)
#                     if row_data:
#                         full_text.append(" | ".join(row_data))
#             extracted_text = "\n".join(full_text)
            
#         elif file_extension == "txt":
#             extracted_text = uploaded_file.read().decode("utf-8", errors="ignore")
            
#         elif file_extension in ["xlsx", "xls", "csv"]:
#             if file_extension == "csv":
#                 df = pd.read_csv(uploaded_file)
#             else:
#                 engine = "xlrd" if file_extension == "xls" else "openpyxl"
#                 df = pd.read_excel(uploaded_file, engine=engine)
            
#             st.session_state[f"df_{uploaded_file.name}"] = df
#             row_count, col_count = len(df), len(df.columns)
#             summary_lines = [
#                 f"--- Spreadsheet Structure Analysis: {uploaded_file.name} ---",
#                 f"Total Row Count: {row_count} data records | Total Column Count: {col_count} columns",
#                 "\nColumn Header & Data Type Identification Matrix:"
#             ]
#             for col in df.columns:
#                 col_type = str(df[col].dtype)
#                 sample_values = df[col].dropna().head(2).tolist()
#                 friendly_type = "Numeric" if any(t in col_type for t in ["int", "float"]) else "Categorical"
#                 sample_str = f" [Example records: {sample_values}]" if sample_values else " [Empty column]"
#                 summary_lines.append(f" * Column Header Name: '{col}' -> Detected Type: {friendly_type}{sample_str}")
#             extracted_text = "\n".join(summary_lines) + "\n\n"
            
#     except Exception as e:
#         st.sidebar.error(f"Error parsing {uploaded_file.name}: {str(e)}")
#         return ""
        
#     return extracted_text

# if "bm25_chunks" not in st.session_state: st.session_state["bm25_chunks"] = []
# if "faiss_chunks" not in st.session_state: st.session_state["faiss_chunks"] = []
# if "file_names" not in st.session_state: st.session_state["file_names"] = []
# if "bm25_index" not in st.session_state: st.session_state["bm25_index"] = None
# if "faiss_index" not in st.session_state: st.session_state["faiss_index"] = None

# if uploaded_docs:
#     current_files = [doc.name for doc in uploaded_docs]
#     if current_files != st.session_state["file_names"]:
#         bm25_pool, faiss_pool = [], []
        
#         for doc in uploaded_docs:
#             extracted_data = extract_text_from_file(doc)
#             if not extracted_data: continue
            
#             # FIXED: Extension forced to lowercase to accurately trigger routing states
#             file_extension = doc.name.split(".")[-1].lower()
            
#             if file_extension in ["xlsx", "xls", "csv", "txt"]:
#                 bm25_pool.append(extracted_data)
#             elif file_extension == "docx":
#                 faiss_pool.extend(chunk_text(extracted_data))
#             elif file_extension == "pdf":
#                 bm25_pool.append(extracted_data)
#                 faiss_pool.extend(chunk_text(extracted_data))
                
#         st.session_state["bm25_chunks"] = bm25_pool
#         st.session_state["faiss_chunks"] = faiss_pool
#         st.session_state["file_names"] = current_files
        
#         if bm25_pool:
#             tokenized_corpus = [[w for w in doc.lower().split(" ") if w] for doc in bm25_pool]
#             st.session_state["bm25_index"] = BM25Okapi(tokenized_corpus)
#             st.sidebar.success(f"✅ Indexed {len(bm25_pool)} tabular summaries into BM25.")
#         else:
#             st.session_state["bm25_index"] = None
            
#         if faiss_pool:
#             st.session_state["faiss_index"] = FAISS.from_texts(faiss_pool, embeddings)
#             st.sidebar.success(f"✅ Indexed {len(faiss_pool)} narrative chunks into FAISS Vector.")
#         else:
#             st.session_state["faiss_index"] = None

# total_chunks_loaded = len(st.session_state["bm25_chunks"]) + len(st.session_state["faiss_chunks"])
# if total_chunks_loaded > 0:
#     st.sidebar.success(f"✅ Loaded {total_chunks_loaded} isolated search targets to system RAM.")


# # -----------------------------------------------------------------------------
# # 3. HARDWARE-OPTIMIZED DYNAMIC MODEL LOADER (Tailored for Dell 3593)
# # -----------------------------------------------------------------------------
# @st.cache_resource(show_spinner="Initializing Local AI Engine (This might take a moment)...")
# def load_selected_llm(model_name):
#     if not model_name:
#         return None
#     model_path = os.path.join(MODEL_DIR, model_name)
    
#     # Balance context lengths to keep your 16GB RAM from hitting swap limits
#     if "1.5b" in model_name.lower():
#         max_context = 4096  
#     elif "3b" in model_name.lower():
#         max_context = 3072  # Trimmed slightly from 4096 to keep your 16GB RAM completely safe
#     else:
#         max_context = 2048   
        
#     return Llama(
#         model_path=model_path, 
#         n_ctx=max_context,   
#         # i5-1035G1 has 4 physical cores / 8 threads. 4 or 5 threads is the sweet spot.
#         n_threads=max(1, min(5, os.cpu_count() - 2)), 
#         # Lower batch size prevents your i5 CPU cache from getting overwhelmed
#         n_batch=128,                           
#         n_gpu_layers=0,                        # Keep at 0. 2GB VRAM is too small for modern GGUFs.
#         use_mmap=True,                         # Crucial for 16GB RAM: streams model weights from storage instead of forcing all into RAM
#         use_mlock=False,                       
#         verbose=False
#     )


# # -----------------------------------------------------------------------------
# # 4. ENSEMBLE RETRIEVAL FUNCTION WITH METADATA TRACKING
# # -----------------------------------------------------------------------------
# def get_routed_context_with_meta(query, max_outputs=2):
#     """
#     Dynamic retrieval pipeline. For hybrid PDF operations, it balances exact 
#     BM25 keyword matches with deep FAISS semantic context while removing duplicates.
#     """
#     retrieved_segments = []
#     triggered_engines = []
#     seen_segments = set() # Avoid feeding duplicate context walls to the local model
    
#     # 1. Interrogate BM25 Lane (Triggered by CSV, XLS, XLSX, TXT, and PDF)
#     if st.session_state.bm25_index and st.session_state.bm25_chunks:
#         # OPTIMIZATION: Filter empty space tokens to match the updated Part 2 tokenization
#         tokenized_query = [w for w in query.lower().split(" ") if w]
#         if tokenized_query:
#             bm25_matches = st.session_state.bm25_index.get_top_n(tokenized_query, st.session_state.bm25_chunks, n=max_outputs)
#             if bm25_matches:
#                 for match in bm25_matches:
#                     if match not in seen_segments:
#                         retrieved_segments.append(match)
#                         seen_segments.add(match)
#                 triggered_engines.append("BM25 (Keyword Engine)")
        
#     # 2. Interrogate FAISS Lane (Triggered by DOCX, and PDF)
#     if st.session_state.faiss_index and st.session_state.faiss_chunks:
#         # Kept at k=8 to guarantee deep, detailed, and non-fragmented information
#         sim_results = st.session_state.faiss_index.similarity_search(query, k=8)
#         faiss_matches = [doc.page_content for doc in sim_results]
#         if faiss_matches:
#             for match in faiss_matches:
#                 if match not in seen_segments:
#                     retrieved_segments.append(match)
#                     seen_segments.add(match)
#             triggered_engines.append("FAISS (Semantic Vector Engine)")
        
#     return "\n\n---\n\n".join(retrieved_segments), triggered_engines


# # -----------------------------------------------------------------------------
# # 5. CHAT AND STREAMING INFERENCE LOGIC (Part 5 - Section A - Restored Streaming)
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
#     if user_query := st.chat_input("Ask a question about your uploaded file:"):
#         with st.chat_message("user"):
#             st.markdown(user_query)
#         st.session_state.chat_history.append({"role": "user", "content": user_query})

#         with st.chat_message("assistant"):
#             # Clean punctuation and check word tokens flexibly
#             punctuation_table = str.maketrans("", "", '?.!,-_')
#             clean_tokens = set(user_query.lower().translate(punctuation_table).split())

#             # -----------------------------------------------------------------
#             # HIGH-CAPACITY EXPANDED ENGLISH CHITCHAT VOCABULARY MATRIX
#             # -----------------------------------------------------------------
#             greetings = {
#                 "hi", "hello", "hey", "greetings", "yo", "there", "sup", "morning", 
#                 "afternoon", "evening", "howdy", "welcome", "heya", "whatsup"
#             }
            
#             gratitude = {
#                 "thank", "thanks", "thankyou", "appreciate", "helpful", "much", 
#                 "so", "for", "the", "help", "grateful", "obliged", "cheers", 
#                 "thanking", "appreciation", "kind", "helper", "kindness"
#             }
            
#             praise_affirmation = {
#                 "good", "great", "awesome", "perfect", "excellent", "amazing", 
#                 "wonderful", "cool", "nice", "job", "work", "wow", "yes", "yeah", 
#                 "yup", "ok", "okay", "fine", "sure", "correct", "right", "brilliant",
#                 "fantastic", "sweet", "rock", "genius", "smart", "incredible", "love",
#                 "superb", "terrific", "spot", "on", "exactly", "indeed", "gotcha", "yep",
#                 "well", "done", "neat", "fabulous", "outstanding"  
#             }
            
#             farewells = {
#                 "bye", "goodbye", "later", "see", "you", "quit", "exit", "close", 
#                 "stop", "end", "leave", "done", "peace"
#             }
            
#             conversational_fillers = {
#                 "chatbot", "bot", "ai", "assistant", "computer", "machine", "system",
#                 "who", "are", "you", "what", "is", "your", "name", "can", "do", "how",
#                 "old", "creator", "made", "built", "dude", "mate", "uh", "um", "er", "hmm",
#                 "very"  
#             }

#             # Unify all expanded English categories into one massive keyword net
#             all_chitchat_words = greetings | gratitude | praise_affirmation | farewells | conversational_fillers

#             # Length shield added to prevent multi-word analytical queries from hitting a false bypass
#             # has_matching_words = len(clean_tokens.intersection(all_chitchat_words)) > 0
#             # is_pure_chitchat = len(clean_tokens) <= 3  
#             # is_chitchat = has_matching_words and (total_chunks_loaded == 0 or is_pure_chitchat)
#             is_chitchat = len(clean_tokens.intersection(all_chitchat_words)) > 0

#             messages = []
#             engines_used = []
#             explanation_md = ""

#             if is_chitchat:
#                 messages.append({
#                     "role": "system",
#                     "content": "You are a friendly companion. Respond to the user's greeting, gratitude, or praise warmly, naturally, and concisely in one short sentence."
#                 })
#                 messages.append({
#                     "role": "user",
#                     "content": user_query
#                 })
#                 context_snippet = ""
#                 engines_used = ["Bypass (Conversational Mode)"]
#                 explanation_md = "- **RAG Pipeline Bypassed:** Small talk or politeness recognized. System responded using a friendly tone template."
            
#             elif total_chunks_loaded > 0:
#                 context_snippet, engines_used = get_routed_context_with_meta(user_query)
#                 files_list_str = ", ".join(st.session_state["file_names"]) if st.session_state["file_names"] else "None"
                
#                 # Authoritative structural prompt mapping layout configured to force peak intelligence out of Llama 3.2 3B
#                 messages.append({
#                     "role": "system",
#                     "content": (
#                         "You are an analytical data extraction expert processing local administrative files. "
#                         f"Your primary target is to extract objective, comprehensive, factual summaries from these files: [{files_list_str}].\n"
#                         "CRITICAL STRUCTURAL PROTOCOLS:\n"
#                         "- Rely exclusively on the raw text blocks supplied by the search engines.\n"
#                         "- Keep your logic clear, structured, and completely free of conversational disclaimers or procedural boilerplate warnings.\n"
#                         "- Provide precise text metrics and direct structural data maps matching the user request directly."
#                     )
#                 })
#                 messages.append({
#                     "role": "user", 
#                     "content": f"INPUT STORAGE DATA METRICS TO PARSE:\n{context_snippet}\n\nUSER EXTRACTION DIRECTIVE: {user_query}"
#                 })
                
#                 if "BM25 (Keyword Engine)" in engines_used:
#                     explanation_md += "- **BM25 Keyword Engine triggered:** Evaluated tabular metadata layouts.\n"
#                 if "FAISS (Semantic Vector Engine)" in engines_used:
#                     explanation_md += "- **FAISS Semantic Engine triggered:** Scanned context chunks to gather full matching context strings.\n"
#             else:
#                 messages.append({"role": "system", "content": "You are a helpful AI assistant running locally on a user's CPU."})
#                 messages.append({"role": "user", "content": user_query})
            
#             text_placeholder = st.empty()
#             full_response = ""
#             token_count = 0
#             start_time = time.time()
            
#             if llm is not None:
#                 response_stream = llm.create_chat_completion(
#                     messages=messages, 
#                     max_tokens=750, 
#                     temperature=0.7 if is_chitchat else 0.1, 
#                     stream=True
#                 )
                
#                 for chunk in response_stream:
#                     try:
#                         if isinstance(chunk, dict):
#                             if "choices" in chunk and len(chunk["choices"]) > 0:
#                                 delta = chunk["choices"][0].get("delta", {})
#                                 if "content" in delta:
#                                     full_response += delta["content"]
#                                     token_count += 1
#                                     text_placeholder.markdown(full_response + "▌")
#                         # FIXED: Added index [0] to object data attributes to correctly resolve the chunk choices stream
#                         else:
#                             if hasattr(chunk, "choices") and len(chunk.choices) > 0:
#                                 delta = chunk.choices[0].delta  
#                                 if hasattr(delta, "content") and delta.content is not None:
#                                     full_response += delta.content
#                                     token_count += 1
#                                     text_placeholder.markdown(full_response + "▌")
#                     except (IndexError, AttributeError, KeyError):
#                         continue
                
#                 text_placeholder.markdown(full_response)
#                 elapsed_time = time.time() - start_time
#                 tokens_per_second = token_count / elapsed_time if elapsed_time > 0 else 0
#                 speed_metric_text = f"⏱️ Streamed {token_count} tokens in {elapsed_time:.2f}s ({tokens_per_second:.2f} tok/sec)"
#                 st.caption(speed_metric_text)
                
#                 if engines_used:
#                     with st.expander("🔍 Retrieval Engine Insights", expanded=True):
#                         st.markdown(f"**Active Search Channels:** `{', '.join(engines_used)}`")
#                         st.markdown(explanation_md)
                
#                 st.session_state["chat_history"].append({
#                     "role": "assistant",
#                     "content": full_response,
#                     "speed_metric": speed_metric_text,
#                     "engines_used": engines_used,
#                     "engine_explanation": explanation_md
#                 })
#             else:
#                 st.error("Error: Local AI engine model weights configuration missing.")


#     # -----------------------------------------------------------------------------
#     # 5. CHAT AND STREAMING INFERENCE LOGIC (Part 5 - Section B)
#     # -----------------------------------------------------------------------------
#     if uploaded_docs:
#         for doc in uploaded_docs:
#             file_extension = doc.name.split(".")[-1].lower()
            
#             if file_extension in ["xlsx", "xls", "csv"]:
#                 st.write("---")
#                 st.subheader(f"📊 Dataset Structure Profiling: {doc.name}")
                
#                 report_file_key = f"report_path_{doc.name}"
                
#                 if st.button(f"Generate Profile Analysis for {doc.name}"):
#                     df_to_analyze = st.session_state.get(f"df_{doc.name}")
#                     if df_to_analyze is not None:
#                         # OPTIMIZATION: Dropped limit from 10MB to 5MB for RAM safety alongside an active local GGUF model
#                         is_large_file = doc.size > (5 * 1024 * 1024)
                        
#                         progress_text = "Phase 1/3: Ingesting dataset matrix variables..."
#                         progress_bar = st.progress(0, text=progress_text)
                        
#                         time.sleep(0.2)
#                         progress_bar.progress(25, text="Phase 2/3: Structural data type verification...")
                        
#                         time.sleep(0.2)
#                         progress_bar.progress(75, text="Phase 3/3: Running deep statistical profiling distribution summaries...")
                        
#                         # OPTIMIZATION: Explicitly turn off heavy, memory-expensive multi-variable calculations 
#                         # to keep your laptop CPU cool and prevent your 16GB RAM from hitting max thresholds.
#                         if is_large_file:
#                             st.warning("⚠️ Heavy file layout detected. Processing using structural minimal configuration to save system RAM.")
#                             profile = ProfileReport(
#                                 df_to_analyze, 
#                                 title=f"Profile: {doc.name}", 
#                                 minimal=True, 
#                                 explorative=False, 
#                                 progress_bar=False,
#                                 correlations=None,     # Disables RAM-heavy correlation loops
#                                 interactions=None      # Disables CPU-heavy scatter plots
#                             )
#                         else:
#                             profile = ProfileReport(
#                                 df_to_analyze, 
#                                 title=f"Profile: {doc.name}", 
#                                 explorative=True, 
#                                 progress_bar=False,
#                                 interactions=None      # Always disable interactions to save laptop CPU stress
#                             )
                        
#                         output_filename = f"profile_{doc.name.split('.')[-2]}.html"
#                         profile.to_file(output_filename, silent=True)
                        
#                         progress_bar.progress(100, text="✨ Analysis complete! Exporting reporting arrays...")
#                         time.sleep(0.4)
#                         progress_bar.empty()
                        
#                         # Trigger local browser fallback hook safely
#                         try:
#                             webbrowser.open_new_tab(output_filename)
#                         except Exception:
#                             pass
                            
#                         st.session_state[report_file_key] = output_filename
#                     else:
#                         st.error("Data frame state not found. Please reload the document file.")
                
#                 if report_file_key in st.session_state:
#                     existing_path = st.session_state[report_file_key]
#                     st.success(f"✅ Active profile generated successfully!")
                    
#                     # OPTIMIZATION: Provide a native Streamlit file down-streamer. If your OS browser trigger blocks
#                     # a popup windows tab, you can read or save the HTML page instantly right from the UI interface.
#                     if os.path.exists(existing_path):
#                         with open(existing_path, "r", encoding="utf-8") as f:
#                             html_bytes = f.read()
                        
#                         col1, col2 = st.columns(2)
#                         with col1:
#                             if st.button(f"🔄 Re-open in Local Browser Tab", key=f"open_{existing_path}"):
#                                 try:
#                                     webbrowser.open_new_tab(existing_path)
#                                 except Exception:
#                                     st.error("Could not trigger browser tab natively.")
#                         with col2:
#                             st.download_button(
#                                 label="💾 Download Profiling Report (.html)",
#                                 data=html_bytes,
#                                 file_name=existing_path,
#                                 mime="text/html",
#                                 key=f"dl_{existing_path}"
#                             )

#     # Clean memory wipe button using absolute bracket maps to prevent tracking bugs
#     if st.session_state["chat_history"]:
#         st.write("") 
#         if st.button("🗑️ Clear Chat History", use_container_width=False):
#             st.session_state["chat_history"] = []
#             st.session_state["bm25_chunks"] = []
#             st.session_state["faiss_chunks"] = []
#             st.session_state["file_names"] = []
#             st.session_state["bm25_index"] = None
#             st.session_state["faiss_index"] = None
            
#             for key in list(st.session_state.keys()):
#                 if key.startswith("df_") or key.startswith("report_path_"):
#                     del st.session_state[key]
                    
#             st.rerun()
# else:
#     st.warning("⚠️ Drop a valid .gguf model file into the sidebar uploader to launch execution matrices.")



