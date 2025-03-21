# main.py:
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
    logging.info("Using OCR to extract text from PDF...")
    images = convert_from_path(pdf_path, poppler_path=POPPLER_PATH)
    text = "\n".join(pytesseract.image_to_string(image) for image in images)
    return text

def process_pdf(class_folder, pdf_filename, model_name="mistral"):
    """Processes a PDF and creates a Q&A system for the document."""
    pdf_path = os.path.join(class_folder, pdf_filename)
    
    if not os.path.exists(pdf_path):
        logging.error(f"File {pdf_path} does not exist.")
        return None

    # **Kill ChromaDB processes to prevent file lock issues**
    kill_chroma_process()

    # **Clear old embeddings before adding new ones**
    if os.path.exists(PERSIST_DIRECTORY):
        try:
            shutil.rmtree(PERSIST_DIRECTORY)
            logging.info("Old embeddings cleared before processing new PDF.")
        except Exception as e:
            logging.error(f"Failed to clear old ChromaDB store: {e}")
            return None  # Prevent further processing if cleanup fails

    logging.info(f"Processing PDF: {pdf_path}")

    # Load PDF
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    
    extracted_text = "".join([page.page_content for page in pages])
    
    # If no text was extracted, use OCR
    if not extracted_text.strip():
        extracted_text = extract_text_with_ocr(pdf_path)
        if not extracted_text.strip():
            logging.error("OCR failed to extract any text from the document.")
            return None

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_text(extracted_text)
    
    logging.info(f"Split the document into {len(splits)} chunks.")

    if not splits:
        logging.error("No text chunks were created. Ensure the PDF has extractable text.")
        return None

    # Create embeddings and store them
    embeddings = OllamaEmbeddings(model=model_name)
    vectorstore = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)

    for chunk in tqdm(splits, desc="Processing chunks"):
        vectorstore.add_texts([chunk])
    
    logging.info(f"Stored {len(splits)} chunks in the vectorstore.")

    # Create the LLM
    llm = Ollama(model=model_name)

    # Define prompt template
    prompt_template = """
    You are a helpful AI assistant that answers questions based on the provided PDF document.
    Use only the context provided to answer the question. If you don't know the answer, say so.

    Context: {context}

    Question: {question}

    Answer: Let me help you with that based on the PDF content.
    """

    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    # Create Q&A chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),  # **Accurate retrieval**
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )

    return qa_chain


# app.py
import streamlit as st
import os
import json
import logging
import pandas as pd
import time  # Used for shake effect
from main import process_pdf
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Streamlit UI Configuration
st.set_page_config(page_title="School PDF Q&A System", layout="wide")

# Load passwords from JSON file
PASSWORD_FILE = "data/passwords.json"
if os.path.exists(PASSWORD_FILE):
    with open(PASSWORD_FILE, "r") as f:
        CLASS_PASSWORDS = json.load(f)
else:
    st.error("🚨 Password file not found! Please create 'passwords.json' inside the 'data' folder.")
    st.stop()

# Access Logs File
ACCESS_LOG_FILE = "data/access_logs.csv"

# CSS for Better UI, including shake animation for incorrect passwords
st.markdown(
    """
    <style>
        @keyframes shake {
            0% { transform: translateX(0); }
            25% { transform: translateX(-5px); }
            50% { transform: translateX(5px); }
            75% { transform: translateX(-5px); }
            100% { transform: translateX(0); }
        }
        .shake { animation: shake 0.3s ease-in-out; }
        .chat-container { max-height: 500px; overflow-y: auto; padding: 10px; }
        .user-bubble { background-color: #DCF8C6; padding: 10px; border-radius: 10px; width: fit-content; max-width: 70%; margin-left: auto; }
        .bot-bubble { background-color: #E1E1E1; padding: 10px; border-radius: 10px; width: fit-content; max-width: 70%; }
        .bot-loading { font-size: 16px; font-style: italic; color: gray; }
        .header { text-align: center; margin-bottom: 20px; }
        .header img { width: 250px; }
        .header h1 { color: #2C3E50; }
        .header a { font-size: 18px; }
    </style>
    """,
    unsafe_allow_html=True
)

# Branding
st.markdown(
    """
    <div class="header">
        <img src="https://www.i95dev.com/wp-content/uploads/2020/08/i95dev-Logo-red.png">
        <h1>School PDF Question-Answering System</h1>
        <p><a href="https://www.i95dev.com/" target="_blank">Visit i95Dev</a></p>
    </div>
    """,
    unsafe_allow_html=True
)

# Select Class
class_options = list(CLASS_PASSWORDS.keys())  # Fetch class names from passwords.json
selected_class = st.selectbox("Select Your Class:", class_options)

# Define Class Folder
CLASS_FOLDERS = "classes"
class_folder = os.path.join(CLASS_FOLDERS, f"class_{selected_class.split()[1]}")

# Ensure the directory exists
os.makedirs(class_folder, exist_ok=True)

# Password Authentication
if "authenticated_classes" not in st.session_state:
    st.session_state.authenticated_classes = {}

if selected_class not in st.session_state.authenticated_classes:
    password_input = st.text_input("Enter Password:", type="password", key="password_input")

    if st.button("Submit Password"):
        correct_password = CLASS_PASSWORDS.get(selected_class, "")

        if password_input == correct_password:
            st.session_state.authenticated_classes[selected_class] = True
            st.success("✅ Access granted!")
            
            # Log Successful Access
            access_log = pd.DataFrame([[datetime.now(), selected_class, "Success"]])
            access_log.to_csv(ACCESS_LOG_FILE, mode="a", header=False, index=False)

            time.sleep(1)  # Small delay to make success message visible
            st.rerun()
        else:
            st.error("❌ Incorrect password. Please try again.")
            
            # Log Failed Attempt
            access_log = pd.DataFrame([[datetime.now(), selected_class, "Failed"]])
            access_log.to_csv(ACCESS_LOG_FILE, mode="a", header=False, index=False)
            
            # Shake Effect
            st.markdown('<script>document.getElementById("password_input").classList.add("shake");</script>', unsafe_allow_html=True)
            time.sleep(0.5)
            st.markdown('<script>document.getElementById("password_input").classList.remove("shake");</script>', unsafe_allow_html=True)
    
    st.stop()

# Display Available PDFs for Selected Class
st.subheader(f"Available PDFs for {selected_class}:")
pdf_files = [f for f in os.listdir(class_folder) if f.endswith(".pdf")]
if pdf_files:
    selected_pdf = st.selectbox("Select a PDF:", pdf_files)

    if st.button("Process & Ask Questions"):
        st.session_state.chat_history = []
        st.session_state.qa_agent = process_pdf(class_folder, selected_pdf)

        if st.session_state.qa_agent is None:
            st.error("❌ Failed to process PDF. Please check the file.")
        else:
            st.success("✅ PDF processed! You can now ask questions.")

# Ensure QA agent is available before allowing questions
if "qa_agent" in st.session_state and st.session_state.qa_agent:
    st.subheader("Chat with PDF")

    chat_container = st.container()
    for entry in st.session_state.chat_history:
        if entry["type"] == "user":
            chat_container.markdown(f'<div class="chat-container"><div class="user-bubble">{entry["content"]}</div></div>', unsafe_allow_html=True)
        else:
            chat_container.markdown(f'<div class="chat-container"><div class="bot-bubble">{entry["content"]}</div></div>', unsafe_allow_html=True)

    if "question" not in st.session_state:
        st.session_state.question = ""

    with st.form("question_form"):
        question = st.text_input("Ask a question about the PDF:", key="question", placeholder="Type your question here...")
        submit_button = st.form_submit_button("Ask")

    if submit_button and question:
        if "processing" in st.session_state and st.session_state.processing:
            st.warning("🔄 Please wait for the current response to complete!")
        else:
            st.session_state.processing = True
            st.session_state.chat_history.append({"type": "user", "content": question})

            with st.spinner("🤖 Generating response..."):
                result = st.session_state.qa_agent.invoke({"query": question})

            bot_response = "❌ An error occurred." if "error" in result else result["result"]
            st.session_state.chat_history.append({"type": "bot", "content": bot_response})

            st.session_state.processing = False
            
            # **✅ Clear Question Field Properly Without Error**
            st.session_state.pop("question", None)  # Safe way to clear

            # **✅ Refresh UI**
            st.rerun()
else:
    st.warning(f"Do Not Switch between PDF Documents during a chat.")


# admin.py
import streamlit as st
import os
import json
import pandas as pd
from datetime import datetime

# Define admin credentials
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "admin123"

# Paths to data files
PASSWORDS_FILE = "data/passwords.json"
LOGS_FILE = "data/access_logs.csv"
CLASS_FOLDERS = "classes"

# Set up Streamlit page
st.set_page_config(page_title="Admin Dashboard", layout="wide")

st.title("📊 Admin Dashboard - School PDF Q&A System")

# Admin Login System
if "admin_authenticated" not in st.session_state:
    st.session_state.admin_authenticated = False

if not st.session_state.admin_authenticated:
    st.subheader("🔑 Admin Login")
    username = st.text_input("Username:")
    password = st.text_input("Password:", type="password")
    
    if st.button("Login"):
        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            st.session_state.admin_authenticated = True
            st.success("✅ Login successful! Redirecting...")
            st.rerun()
        else:
            st.error("❌ Invalid credentials. Please try again.")
    st.stop()

# ✅ Admin Successfully Logged In - Show Dashboard Features
st.sidebar.header("Admin Actions")

# Section 1: Manage PDFs
st.subheader("📂 Manage Class PDFs")
selected_class = st.selectbox("Select a Class", [f"Class {i}" for i in range(1, 11)])

class_folder = os.path.join(CLASS_FOLDERS, f"class_{selected_class.split()[1]}")
os.makedirs(class_folder, exist_ok=True)

# Show existing PDFs
pdf_files = [f for f in os.listdir(class_folder) if f.endswith(".pdf")]
if pdf_files:
    st.write("**Existing PDFs:**")
    selected_pdf = st.selectbox("Select a PDF to Delete", pdf_files)
    if st.button("❌ Delete Selected PDF"):
        os.remove(os.path.join(class_folder, selected_pdf))
        st.success(f"Deleted {selected_pdf}")
        st.rerun()
else:
    st.write("No PDFs found for this class.")

# Upload a new PDF
st.subheader("📤 Upload a New PDF")
uploaded_pdf = st.file_uploader("Choose a PDF to Upload", type=["pdf"])
if uploaded_pdf is not None:
    with open(os.path.join(class_folder, uploaded_pdf.name), "wb") as f:
        f.write(uploaded_pdf.read())
    st.success(f"✅ Uploaded {uploaded_pdf.name}")
    st.rerun()

# Section 2: Update Class Passwords
st.subheader("🔑 Manage Class Passwords")
with open(PASSWORDS_FILE, "r") as f:
    passwords = json.load(f)

new_password = st.text_input(f"Set New Password for {selected_class}:", type="password")
if st.button("Update Password"):
    passwords[selected_class] = new_password
    with open(PASSWORDS_FILE, "w") as f:
        json.dump(passwords, f, indent=4)
    st.success(f"✅ Password updated for {selected_class}")

# Section 3: View Access Logs
st.subheader("📜 Access Logs")
if os.path.exists(LOGS_FILE):
    logs_df = pd.read_csv(LOGS_FILE)
    st.dataframe(logs_df)
else:
    st.write("No access logs found.")

# Logout Button
if st.button("Logout"):
    st.session_state.admin_authenticated = False
    st.rerun()
