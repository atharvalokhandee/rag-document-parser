"""
Streamlit front-end for the RAG system.
ChatGPT-style UI with chat memory + downloadable TTS audio.
Supports: PDF, DOCX, TXT, PPTX, PNG, JPG, JPEG, TIFF, XML
"""
import streamlit as st
from pathlib import Path
import tempfile, uuid, os, mimetypes
from dotenv import load_dotenv
from rag_engine import (
    extract_text_unstructured,
    build_vector_store,
    list_stores,
    delete_store,
    make_chat_chain
)
from gtts import gTTS
from io import BytesIO

# ------------------ Setup ------------------
mimetypes.add_type("text/xml", ".xml")
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

st.set_page_config(page_title="RAG QA (unstructured.io)", layout="wide")
st.markdown("<h1 style='text-align: center;'>📄 Document & Web QA</h1>", unsafe_allow_html=True)

# ------------------ Sidebar ------------------
with st.sidebar:
    st.header("Vector-store Manager")
    stores = list_stores()
    if stores:
        selected = st.selectbox("Existing stores", stores)
        if st.button("Delete Selected"):
            delete_store(selected)
            st.success(f"Deleted store: {selected}")
            st.rerun()
    else:
        st.info("No vector stores found. Upload a file or URL first.")

# ------------------ Tabs ------------------
tab1, tab2, tab3 = st.tabs(["Upload File", "Use URL", "Chat"])

# ------------------ Tab 1: Upload File ------------------
with tab1:
    uploaded = st.file_uploader(
        "Upload PDF, DOCX, TXT, PPTX, PNG, JPG, JPEG, TIFF, XML, XLS, XLSX",
        type=["pdf", "docx", "txt", "pptx", "png", "jpg", "jpeg", "tiff", "xml", "xls", "xlsx"]
        )

    if uploaded and st.button("Process File"):
        with st.spinner("Extracting text..."):
            suffix = Path(uploaded.name).suffix
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            tmp.write(uploaded.read())
            tmp.close()
            try:
                text = extract_text_unstructured(file_path=Path(tmp.name))
                if not text.strip():
                    st.error("No readable text found in this document.")
                else:
                    store_name = str(uuid.uuid4())
                    build_vector_store(text, store_name)
                    st.session_state["last_store"] = store_name
                    st.success(f"Vector store created! ID = {store_name}")
            finally:
                os.unlink(tmp.name)

# ------------------ Tab 2: Use URL ------------------
with tab2:
    url = st.text_input("Paste URL (Wikipedia, news, etc.)")
    if url and st.button("Process URL"):
        with st.spinner("Extracting text..."):
            try:
                text = extract_text_unstructured(url=url.strip())
                if not text.strip():
                    st.error("No readable text found at this URL.")
                else:
                    store_name = str(uuid.uuid4())
                    build_vector_store(text, store_name)
                    st.session_state["last_store"] = store_name
                    st.success(f"Vector store created! ID = {store_name}")
            except Exception as e:
                st.error(f"Error: {e}")

# ------------------- TAB 3: CHAT -------------------
with tab3:
    # Check if a vector store exists
    if "last_store" not in st.session_state:
        st.info("Please create a vector store first (via Upload File or Use URL).")
        st.stop()

    # Initialize conversational chain once per session
    if "chain" not in st.session_state:
        st.session_state.chain = make_chat_chain(st.session_state["last_store"], GROQ_API_KEY)

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Container for the entire chat (scrollable)
    chat_container = st.container()

    # Display all previous messages
    with chat_container:
        for i, msg in enumerate(st.session_state.messages):
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
                if msg["role"] == "assistant" and msg["content"].strip():
                    # Generate TTS and provide download button
                    tts = gTTS(msg["content"], lang="en")
                    audio_bytes = BytesIO()
                    tts.write_to_fp(audio_bytes)
                    audio_bytes.seek(0)
                    st.download_button(
                        label="🔊 Download Audio",
                        data=audio_bytes,
                        file_name=f"answer_{i}.mp3",
                        mime="audio/mpeg",
                        key=f"download_{i}"
                    )

    # User input at the bottom
    if question := st.chat_input("Ask a question or follow-up"):
        # Append user message to session state
        st.session_state.messages.append({"role": "user", "content": question})

        # Generate assistant response
        chain = st.session_state.chain
        response = chain.invoke({"question": question})
        ans = response.get("answer", "").strip() or "(no answer)"
        st.session_state.messages.append({"role": "assistant", "content": ans})

        # Rerun the app to display the full chat including the new messages
        st.experimental_rerun()
