import os
import streamlit as st
import pypdf
import docx2txt
import pandas as pd
from llama_cpp import Llama

# Create necessary local storage directories
MODEL_DIR = "local_models"
os.makedirs(MODEL_DIR, exist_ok=True)

st.set_page_config(page_title="Portable Offline Document Analyzer (RAG Chatbot)", layout="wide", page_icon="💾")
st.title("💾 Portable Offline Document Analyzer using (RAG Chatbot)")
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
        type=["pdf", "docx", "xlsx", "xls"], 
        accept_multiple_files=True
    )

# -----------------------------------------------------------------------------
# 2. FILE EXTRACTION ENGINE
# -----------------------------------------------------------------------------
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
            
        elif file_extension in ["xlsx", "xls"]:
            df = pd.read_excel(uploaded_file)
            # Converts spreadsheet grid structures to Markdown tables cleanly
            extracted_text = df.to_markdown(index=False)
    except Exception as e:
        return f"\nError parsing file {uploaded_file.name}: {str(e)}\n"
        
    return extracted_text

# Aggregate document text
all_context = ""
if uploaded_docs:
    for doc in uploaded_docs:
        all_context += f"\n\n--- Source Document: {doc.name} ---\n"
        all_context += extract_text_from_file(doc)
    st.sidebar.success(f"✅ Loaded {len(uploaded_docs)} document(s) to system RAM.")

# -----------------------------------------------------------------------------
# 3. DYNAMIC LOCAL MODEL LOADER
# -----------------------------------------------------------------------------
@st.cache_resource
def load_selected_llm(model_name):
    model_path = os.path.join(MODEL_DIR, model_name)
    # Streamlit unloads old selections from RAM automatically before pulling this
    return Llama(
        model_path=model_path, 
        n_ctx=2048,       # Context boundary for prompt text + answers
        n_threads=4,      # Tailored to your physical CPU cores
        n_gpu_layers=0,   # 100% isolated CPU runtime execution
        verbose=False
    )

# # -----------------------------------------------------------------------------
# # 4. CHAT AND INFERENCE LOGIC
# # -----------------------------------------------------------------------------
# if selected_model_name:
#     llm = load_selected_llm(selected_model_name)
#     st.info(f"🟢 **Active Engine:** {selected_model_name} (Running on System CPU)")
    
#     # Simple clear button for conversation hygiene
#     if st.button("🗑️ Clear Interface"):
#         st.session_state.chat_history = []
#         st.rerun()

#     # Initialize standard persistent chat dictionary states
#     if "chat_history" not in st.session_state:
#         st.session_state.chat_history = []

#     # Display historical chat text structures
#     for message in st.session_state.chat_history:
#         with st.chat_message(message["role"]):
#             st.markdown(message["content"])

#     # Process live message entry
#     if user_query := st.chat_input("Ask a question about your uploaded documents:"):
#         with st.chat_message("user"):
#             st.markdown(user_query)
#         st.session_state.chat_history.append({"role": "user", "content": user_query})

#         with st.chat_message("assistant"):
#             with st.spinner("Analyzing data on your CPU..."):
#                 # System instructions vary slightly depending on if contextual documents are uploaded
#                 if all_context.strip():
#                     prompt = (
#                         f"System: You are an advanced file analysis assistant. Answer the user's question accurately using only the provided context data text below. If you cannot find the answer, state that it is not in the documents.\n\n"
#                         f"Context Data:\n{all_context}\n\n"
#                         f"User Question: {user_query}\n"
#                         f"Assistant:"
#                     )
#                 else:
#                     prompt = f"System: You are a helpful assistant.\nUser: {user_query}\nAssistant:"
                
#                 # Execute CPU evaluation loop
#                 output = llm(prompt, max_tokens=350)
#                 response_text = output["choices"]["text"].strip()
                
#                 st.markdown(response_text)
#         st.session_state.chat_history.append({"role": "assistant", "content": response_text})
# -----------------------------------------------------------------------------
# 4. CHAT AND INFERENCE LOGIC
# -----------------------------------------------------------------------------
if selected_model_name:
    llm = load_selected_llm(selected_model_name)
    st.info(f"🟢 **Active Engine:** {selected_model_name} (Running on System CPU)")
    
    # Initialize standard persistent chat dictionary states safely
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Simple clear button for conversation hygiene
    if st.button("🗑️ Clear Interface"):
        st.session_state.chat_history = []
        st.rerun()

    # Display historical chat text structures with dynamic error checking
    for message in st.session_state.chat_history:
        # Safeguard: ensure the message entry is a dictionary before reading keys
        if isinstance(message, dict) and "role" in message and "content" in message:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        else:
            # If corruption is caught, wipe it to prevent interface lock-up
            st.session_state.chat_history = []
            st.rerun()

    # Process live message entry
    if user_query := st.chat_input("Ask a question about your uploaded documents:"):
        with st.chat_message("user"):
            st.markdown(user_query)
        st.session_state.chat_history.append({"role": "user", "content": user_query})

        with st.chat_message("assistant"):
            with st.spinner("Analyzing data on your CPU..."):
                
                # Double-check that context exists and contains actual text characters
                if all_context and all_context.strip():
                    prompt = (
                        f"System: You are an advanced file analysis assistant. Answer the user's question accurately using only the provided context data text below. If you cannot find the answer, state that it is not in the documents.\n\n"
                        f"Context Data:\n{all_context}\n\n"
                        f"User Question: {user_query}\n"
                        f"Assistant:"
                    )
                else:
                    # Clean fallback prompt if running the model standalone as a normal chatbot
                    prompt = (
                        f"System: You are a helpful AI assistant running locally on a user's CPU.\n"
                        f"User: {user_query}\n"
                        f"Assistant:"
                    )
                
                # # Execute CPU evaluation loop
                # output = llm(prompt, max_tokens=350)
                # response_text = output["choices"]["text"].strip()
                
                # --- TO THIS REVISED SYNTAX ---
                output = llm(prompt, max_tokens=350)
                response_text = output["choices"][0]["text"].strip()

                st.markdown(response_text)
        st.session_state.chat_history.append({"role": "assistant", "content": response_text})

else:
    # Onboard warning message shown before first configuration setup
    st.warning("⚠️ **System Offline:** Please upload a `.gguf` model file or choose an active engine in the left sidebar configuration panel to wake up the system.")
