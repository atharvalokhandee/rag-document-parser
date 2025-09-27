"""
Re-usable RAG helper.
Only unstructured.io is used for text extraction.
Now handles: PDF, DOCX, TXT, PPTX, PNG/JPG/TIFF (OCR), XML, Excel (.xls/.xlsx).
"""
import os, re, uuid, shutil, requests, mimetypes
from pathlib import Path
from typing import List, Optional

import pandas as pd  # NEW for Excel support

from unstructured.partition.auto import partition
from unstructured.partition.html import partition_html
from unstructured.partition.image import partition_image
from unstructured.partition.xml import partition_xml
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# ------------------------------------------------------------------
# 1-a. PNG/JPEG/TIFF → OCR
# ------------------------------------------------------------------
def extract_text_image(file_path: Path) -> str:
    elements = partition_image(
        filename=str(file_path),
        strategy="ocr_only",
        languages=["eng"]
    )
    return "\n".join([str(el).strip() for el in elements if el.text])

# ------------------------------------------------------------------
# 1-b. XML → text
# ------------------------------------------------------------------
def extract_text_xml(file_path: Path) -> str:
    elements = partition_xml(filename=str(file_path))
    return "\n".join([str(el).strip() for el in elements if el.text])

# ------------------------------------------------------------------
# 1-c. Excel → text
# ------------------------------------------------------------------
def extract_text_excel(file_path: Path) -> str:
    """Read .xls or .xlsx and convert sheets into plain text."""
    try:
        excel_data = pd.read_excel(file_path, sheet_name=None)  # all sheets
        text = ""
        for sheet, df in excel_data.items():
            text += f"\n--- Sheet: {sheet} ---\n"
            text += df.to_string(index=False)  # convert table into readable text
        return text.strip()
    except Exception as e:
        return f"Error reading Excel: {e}"

# ------------------------------------------------------------------
# 1-d. Unified router
# ------------------------------------------------------------------
def extract_text_unstructured(file_path: Optional[Path] = None,
                              url: Optional[str] = None) -> str:
    """Route to the correct unstructured partition function."""
    if url:
        headers = {"User-Agent": "Mozilla/5.0"}
        html = requests.get(url, headers=headers, timeout=15).text
        elements = partition_html(text=html)
    else:
        mime, _ = mimetypes.guess_type(str(file_path))

        # Special cases
        if mime == "application/xml" or file_path.suffix.lower() == ".xml":
            return extract_text_xml(file_path)
        if mime in {"image/png", "image/jpeg", "image/jpg", "image/tiff"}:
            return extract_text_image(file_path)
        if mime in {
            "application/vnd.ms-excel",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        } or file_path.suffix.lower() in {".xls", ".xlsx"}:
            return extract_text_excel(file_path)

        # Default route → let unstructured handle it
        elements = partition(filename=str(file_path))

    text = "\n".join([str(el).strip() for el in elements if el.text])
    # Light Wikipedia noise filter
    text = re.sub(r"==\s*See also\s*==.*", "", text, flags=re.S)
    text = re.sub(r"==\s*References\s*==.*", "", text, flags=re.S)
    return text.strip()

# ------------------------------------------------------------------
# 2. Vector-store helpers
# ------------------------------------------------------------------
FAISS_FOLDER = Path("faiss_stores")
FAISS_FOLDER.mkdir(exist_ok=True)
CHUNK_SIZE = 1024
CHUNK_OVERLAP = 200
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def build_vector_store(text: str, store_name: str) -> FAISS:
    print(f"DEBUG build_vector_store: received {len(text)} chars")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_text(text)
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vs = FAISS.from_texts(chunks, embeddings)
    vs.save_local(FAISS_FOLDER / store_name)
    return vs

def load_vector_store(store_name: str) -> FAISS:
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return FAISS.load_local(FAISS_FOLDER / store_name, embeddings,
                            allow_dangerous_deserialization=True)

def list_stores() -> List[str]:
    return [p.name for p in FAISS_FOLDER.iterdir() if p.is_dir()]

def delete_store(store_name: str):
    shutil.rmtree(FAISS_FOLDER / store_name, ignore_errors=True)

# ------------------------------------------------------------------
# 3. Chat-memory QA chain
# ------------------------------------------------------------------
def make_chat_chain(store_name: str, groq_api_key: str):
    vs = load_vector_store(store_name)
    llm = ChatGroq(model="openai/gpt-oss-20b",
                   groq_api_key=groq_api_key,
                   temperature=0.2,
                   max_tokens=512)

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True)

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vs.as_retriever(search_kwargs={"k": 5}),
        memory=memory,
        verbose=False)
