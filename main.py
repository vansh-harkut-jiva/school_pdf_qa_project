import os
import shutil
import logging
import pytesseract
import psutil
from pdf2image import convert_from_path
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from tqdm import tqdm
# Update the import for Ollama
from langchain_ollama import OllamaLLM  # Import OllamaLLM instead of Ollama


# Load environment variables
load_dotenv()

# Retrieve values from .env
TESSERACT_PATH = os.getenv("TESSERACT_PATH")
POPPLER_PATH = os.getenv("POPPLER_PATH")
PERSIST_DIRECTORY = os.getenv("PERSIST_DIRECTORY", "data/chroma_db")
CLASS_FOLDERS = os.getenv("CLASS_FOLDERS", "data/class_folders")

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Set Tesseract path
pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

def kill_chroma_process():
    """Find and terminate processes using ChromaDB files without killing Streamlit."""
    chroma_files = ["chroma.sqlite3", "chroma_lock.sqlite3"]
    current_pid = os.getpid()  # ✅ Get Streamlit's process ID

    for proc in psutil.process_iter(["pid", "name", "open_files"]):
        try:
            if proc.info["open_files"]:
                for file in proc.info["open_files"]:
                    if any(chroma_file in file.path for chroma_file in chroma_files):
                        if proc.info["pid"] != current_pid:  # ✅ Do NOT kill Streamlit's process
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

def process_all_pdfs(class_folder, model_name="mistral"):
    """Processes all PDFs in a class folder and creates a unified knowledge base."""
    pdf_files = [f for f in os.listdir(class_folder) if f.endswith(".pdf")]

    if not pdf_files:
        logging.warning(f"⚠️ No PDFs found in {class_folder}. Please upload documents.")
        return None  # Prevent further processing

    # ✅ Print & Log the list of PDFs before processing
    logging.info(f"Found {len(pdf_files)} PDFs in {class_folder}: {', '.join(pdf_files)}")
    print("\n📄 Processing the following PDFs:")
    for pdf in pdf_files:
        print(f"- {pdf}")

    logging.info(f"Processing {len(pdf_files)} PDFs from {class_folder}...")

    # **Kill ChromaDB processes to prevent file lock issues**
    kill_chroma_process()

    # ✅ Clear old embeddings only if the directory exists
    if os.path.exists(PERSIST_DIRECTORY) and os.path.isdir(PERSIST_DIRECTORY):
        try:
            shutil.rmtree(PERSIST_DIRECTORY)
            logging.info("✅ Old embeddings cleared before processing PDFs.")
        except Exception as e:
            logging.error(f"❌ Failed to clear old ChromaDB store: {e}")
            return None  # Prevent further processing if cleanup fails

    all_text_chunks = []
    metadata_list = []

    for pdf_file in pdf_files:
        pdf_path = os.path.join(class_folder, pdf_file)
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()

        extracted_text = "".join([page.page_content for page in pages])

        # If no text was extracted, use OCR
        if not extracted_text.strip():
            extracted_text = extract_text_with_ocr(pdf_path)
            if not extracted_text.strip():
                logging.error(f"OCR failed to extract any text from {pdf_file}. Skipping this file.")
                continue  # Skip this file if no text is found

        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
        splits = text_splitter.split_text(extracted_text)

        # Store metadata (source & page number)
        for idx, chunk in enumerate(splits):
            all_text_chunks.append(chunk)
            metadata_list.append({"source": pdf_file, "chunk_index": idx})

    if not all_text_chunks:
        logging.error("No valid text chunks found in PDFs.")
        return None

    logging.info(f"Stored {len(all_text_chunks)} chunks from all PDFs.")

    # **Extract document names for prompt**
    document_names = ", ".join(set(metadata["source"] for metadata in metadata_list))

    # **Use Ollama Embeddings**
    embeddings = OllamaEmbeddings(model=model_name)
    vectorstore = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)

    # Add texts with metadata
    vectorstore.add_texts(all_text_chunks, metadatas=metadata_list)

    logging.info(f"Vector store updated with all class PDFs.")

    # Create the LLM
    llm = Ollama(model=model_name)

    # **Enhanced Prompt Template - Enforces strict document-based responses**
    prompt_template = """
    You are a highly accurate AI assistant that provides precise answers using ONLY the provided PDF documents.
    - If the information is found, summarize the relevant details.
    - If the information is not found in the provided PDFs, mention it and suggest checking the source documents or rephrasing the query.
    - Do not generate responses outside the provided documents.

    📄 **Relevant Document(s):** {document_names}  
    📜 **Extracted Context:**  
    {context}  

    🔎 **User Question:**  
    {question}  

    ✍️ **Final Answer:**  
    """

    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question", "document_names"])

    # **Improved Retrieval - Only fetch highly relevant chunks**
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    # **Create Q&A Chain**
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,  # ✅ Improved retrieval
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT.partial(document_names=document_names)}  # ✅ Pass document names
    )

    return qa_chain

# Function to query the QA agent
# Function to query the QA agent
def get_answer_from_pdfs(class_folder, question):
    """Handles the process of querying the QA agent from PDFs in a specific class folder."""
    try:
        qa_chain = process_all_pdfs(class_folder)  # Use the dynamic path passed from app.py

        if qa_chain is None:
            logging.error("Failed to process PDFs or create a QA chain.")
            return "Error: No valid PDF files processed. Please try again."

        # Use invoke instead of run to properly handle multiple output keys
        response = qa_chain.invoke({"question": question})

        # Extract the relevant answer and source documents from the response
        answer = response.get("result", "No answer found")
        source_documents = response.get("source_documents", [])

        # If you want to return the answer with source documents, modify this response
        if source_documents:
            source_info = "\n\nSource Documents:\n" + "\n".join(
                [doc["metadata"]["source"] for doc in source_documents]
            )
            return f"{answer}\n\n{source_info}"

        return answer

    except Exception as e:
        logging.error(f"An error occurred while retrieving the answer: {e}")
        return "Error: Something went wrong while processing your question. Please try again later."
