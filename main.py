import os
import shutil
import logging
import pytesseract
import psutil
from pdf2image import convert_from_path
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI  # Corrected import for ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_openai.embeddings import OpenAIEmbeddings  # Corrected import for OpenAIEmbeddings
from chromadb.config import Settings
import chromadb

# Load environment variables
load_dotenv()

# Retrieve values from .env
TESSERACT_PATH = os.getenv("TESSERACT_PATH")
POPPLER_PATH = os.getenv("POPPLER_PATH")
CHROMA_SERVER_HOST = os.getenv("CHROMA_SERVER_HOST", "localhost")
CHROMA_SERVER_PORT = int(os.getenv("CHROMA_SERVER_PORT", 8000))
CLASS_FOLDERS = os.getenv("CLASS_FOLDERS", "data/class_folders")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Set Tesseract path
pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

def kill_chroma_process():
    """Find and terminate processes using ChromaDB files without killing Streamlit."""
    chroma_files = ["chroma.sqlite3", "chroma_lock.sqlite3"]
    current_pid = os.getpid()

    for proc in psutil.process_iter(["pid", "name", "open_files"]):
        try:
            if proc.info["open_files"]:
                for file in proc.info["open_files"]:
                    if any(chroma_file in file.path for chroma_file in chroma_files):
                        if proc.info["pid"] != current_pid:
                            logging.warning(f"Killing process {proc.info['name']} (PID: {proc.info['pid']}) using ChromaDB files.")
                            proc.terminate()
                        else:
                            logging.info("Skipping termination of main Python/Streamlit process.")
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue

def extract_text_with_ocr(pdf_path):
    """Extracts text from scanned PDFs using OCR."""
    logging.info(f"Using OCR to extract text from {pdf_path}...")
    images = convert_from_path(pdf_path, poppler_path=POPPLER_PATH)
    text = "\n".join(pytesseract.image_to_string(image) for image in images)
    return text

def process_all_pdfs(class_folder, role):
    """Processes all PDFs in a class folder and creates a unified knowledge base."""
    pdf_files = [f for f in os.listdir(class_folder) if f.endswith(".pdf")]

    if not pdf_files:
        logging.warning(f"⚠️ No PDFs found in {class_folder}. Please upload documents.")
        return None

    logging.info(f"Processing {len(pdf_files)} PDFs from {class_folder}...")

    all_text_chunks = []
    metadata_list = []

    for pdf_file in pdf_files:
        pdf_path = os.path.join(class_folder, pdf_file)
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()

        extracted_text = "".join([page.page_content for page in pages])

        if not extracted_text.strip():
            extracted_text = extract_text_with_ocr(pdf_path)
            if not extracted_text.strip():
                logging.error(f"OCR failed to extract any text from {pdf_file}. Skipping this file.")
                continue

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
        splits = text_splitter.split_text(extracted_text)

        for idx, chunk in enumerate(splits):
            all_text_chunks.append(chunk)
            metadata_list.append({"source": pdf_file, "chunk_index": idx})

    if not all_text_chunks:
        logging.error("No valid text chunks found in PDFs.")
        return None

    logging.info(f"Stored {len(all_text_chunks)} chunks from all PDFs.")

    
    chroma_client = chromadb.HttpClient(host=CHROMA_SERVER_HOST, port=CHROMA_SERVER_PORT)

    # Create or get a collection for the class and role
    collection_name = f"{os.path.basename(class_folder)}_{role}".replace(" ","")
    collection = chroma_client.get_or_create_collection(name=collection_name)

    # Add texts with metadata to the collection
    collection.add(
        documents=all_text_chunks,
        metadatas=metadata_list,
        ids=[f"{metadata['source']}_chunk_{metadata['chunk_index']}" for metadata in metadata_list]
    )

    logging.info(f"Collection '{collection_name}' updated with all class PDFs.")

    # Use OpenAI Embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY,model="text-embedding-3-large",dimensions=384)
    vectorstore = Chroma(client=chroma_client, collection_name=collection_name, embedding_function=embeddings)

    # Use ChatOpenAI for GPT-4
    llm = ChatOpenAI(model="gpt-4", temperature=0)

    prompt_template = """
    You are a highly accurate AI assistant that provides precise answers using ONLY the provided PDF documents.
    - If the information is found, summarize the relevant details.
    - If the information is not found in the provided PDFs, mention it and suggest checking the source documents or rephrasing the query.

    📄 **Relevant Document(s):** {document_names}  
    📜 **Extracted Context:**  
    {context}  

    🔎 **User Question:**  
    {question}  

    ✍️ **Final Answer:**  
    """

    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question", "document_names"])

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT.partial(document_names=", ".join(set(metadata["source"] for metadata in metadata_list)))}
    )

    return qa_chain

class QAAgent:
    """Wrapper class for the QA chain to provide an `ask_question` method."""
    def __init__(self, qa_chain):
        self.qa_chain = qa_chain

    def ask_question(self, question):
        try:
            response = self.qa_chain.invoke({"query": question})

            answer = response.get("result", "No answer found")
            source_documents = response.get("source_documents", [])

            if source_documents:
                source_info = "\n\nSource Documents:\n" + "\n".join(
                    [doc.metadata["source"] for doc in source_documents if "source" in doc.metadata]
                )
                return f"{answer}\n\n{source_info}"

            return answer

        except Exception as e:
            logging.error(f"An error occurred while retrieving the answer: {e}")
            return "Error: Something went wrong while processing your question. Please try again later."

def get_answer_from_pdfs(class_folder, role):
    try:
        qa_chain = process_all_pdfs(class_folder, role)

        if qa_chain is None:
            logging.error("Failed to process PDFs or create a QA chain.")
            return None

        return QAAgent(qa_chain)

    except Exception as e:
        logging.error(f"An error occurred while initializing the QA agent: {e}")
        return None