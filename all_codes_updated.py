# main.py
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
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})  # Fetch top 3 most relevant chunks

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
class QAAgent:
    """Wrapper class for the QA chain to provide an `ask_question` method."""
    def __init__(self, qa_chain):
        self.qa_chain = qa_chain

    def ask_question(self, question):
        try:
            # Use the correct input key expected by the qa_chain
            response = self.qa_chain.invoke({"query": question})  # Changed "question" to "query"

            # Extract the relevant answer and source documents from the response
            answer = response.get("result", "No answer found")
            source_documents = response.get("source_documents", [])

            # Format the response with source documents if available
            if source_documents:
                source_info = "\n\nSource Documents:\n" + "\n".join(
                    [doc.metadata["source"] for doc in source_documents if "source" in doc.metadata]
                )
                return f"{answer}\n\n{source_info}"

            return answer

        except Exception as e:
            logging.error(f"An error occurred while retrieving the answer: {e}")
            return "Error: Something went wrong while processing your question. Please try again later."

def get_answer_from_pdfs(class_folder, question=None):
    """Handles the process of querying the QA agent from PDFs in a specific class folder."""
    try:
        qa_chain = process_all_pdfs(class_folder)  # Use the dynamic path passed from app.py

        if qa_chain is None:
            logging.error("Failed to process PDFs or create a QA chain.")
            return None  # Return None to indicate failure

        # Return the QAAgent object instead of directly returning the answer
        return QAAgent(qa_chain)

    except Exception as e:
        logging.error(f"An error occurred while initializing the QA agent: {e}")
        return None
    
    
# app.py
import streamlit as st
import os
import json
import logging
import pandas as pd
from datetime import datetime
from main import get_answer_from_pdfs  # Ensure this function can accept no question during initialization

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
    st.error("\ud83d\udea8 Password file not found! Please create 'passwords.json' inside the 'data' folder.")
    st.stop()

# Access Logs File
ACCESS_LOG_FILE = "data/access_logs.csv"

# Branding
st.markdown(
    """
    <div style="text-align: center;">
        <img src="https://www.i95dev.com/wp-content/uploads/2020/08/i95dev-Logo-red.png" width="250">
        <h1>School PDF Question-Answering System</h1>
        <p><a href="https://www.i95dev.com/" target="_blank">Visit i95Dev</a></p>
    </div>
    """,
    unsafe_allow_html=True
)

# Step 1: Select User Role
st.subheader("Select Your Role:")
role = st.selectbox("Are you a Principal, Teacher, or Student?", ["Principal", "Teacher", "Student"])

# Step 2: Select Class (or General for shared PDFs)
st.subheader("Select Your Class:")
class_options = [f"Class {i}" for i in range(1, 11)] + ["General"]
selected_class = st.selectbox("Choose a class or general access:", class_options)

# Define Folder Structure
CLASS_FOLDERS = "classes"
role_folder_map = {
    "Teacher": "Teacher",
    "Student": "Student",
    "Principal": ""
}

# Normalize the selected class to match folder names (class_1, class_2, ...)
class_key = f"class_{selected_class.split(' ')[1]}"  # Convert to match folder names like "class_1"
role_folder = role_folder_map.get(role, "")

# Construct the path to the relevant class and role folder
class_folder = os.path.join(CLASS_FOLDERS, class_key, f"Class {selected_class.split(' ')[1]}_{role_folder}")

# Debugging line: Print the constructed folder path
st.write(f"DEBUG: Constructed folder path is: {class_folder}")

# Ensure the directory exists
os.makedirs(class_folder, exist_ok=True)

# Step 3: Password Authentication
st.subheader("Enter Password to Proceed:")
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

password_input = st.text_input("Enter Password:", type="password", key="password_input")

if st.button("Submit Password"):
    if role == "Principal":
        correct_password = CLASS_PASSWORDS.get("Principal", "")
    else:
        correct_password = CLASS_PASSWORDS.get(f"class_{selected_class.split(' ')[1]}", {}).get(role, "")

    if password_input == correct_password:
        st.session_state.authenticated = True
        st.success("\u2705 Access granted!")

        # Log Successful Access
        access_log = pd.DataFrame([[datetime.now(), selected_class, role, "Success"]])
        access_log.to_csv(ACCESS_LOG_FILE, mode="a", header=False, index=False)
    else:
        st.error("\u274c Incorrect password. Please try again.")
        st.stop()

if not st.session_state.authenticated:
    st.stop()

# Step 4: Display Available PDFs in the Selected Folder
st.subheader(f"Available PDFs in {selected_class} ({role}):")
pdf_files = [f for f in os.listdir(class_folder) if f.endswith(".pdf")]

# Display the PDFs if any are found
if pdf_files:
    for pdf in pdf_files:
        # Use the "📄" emoji directly to avoid UnicodeEncodeError
        st.markdown(f"📄 {pdf}")
else:
    st.info("No PDFs available for this role and class.")

# Step 5: Start Chat Interface
st.subheader("Chat with Your PDFs")
if pdf_files:
    # Initialize chat history if not already present
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "qa_agent" not in st.session_state:
        try:
            st.session_state.qa_agent = get_answer_from_pdfs(class_folder)
            if st.session_state.qa_agent is None:
                st.error("Error: No valid PDF files processed or an error occurred during initialization.")
                st.stop()
        except Exception as e:
            st.error(f"An error occurred while initializing the QA agent: {e}")
            st.stop()

    chat_container = st.container()
    for entry in st.session_state.get("chat_history", []):
        chat_container.markdown(f"{entry['type']}: {entry['content']}")

    with st.form("question_form"):
        question = st.text_input("Ask a question about the PDFs:", key="question")
        submit_button = st.form_submit_button("Ask")

    if submit_button:
        if not question.strip():
            st.warning("⚠️ Please enter a valid question before submitting.")
        else:
            # Add the user question to the chat history
            st.session_state.chat_history.append({"type": "user", "content": question})

            with st.spinner("🤖 Generating response..."):
                try:
                    # Call the ask_question method
                    response = st.session_state.qa_agent.ask_question(question)
                except Exception as e:
                    logging.error(f"Error while generating response: {e}")
                    response = "Error: Unable to process your question. Please try again later."

            # Add the bot's response to the chat history
            st.session_state.chat_history.append({"type": "bot", "content": response})
            st.rerun()  # Rerun to update chat history and UI


# admin.py
import streamlit as st
import os
import json
import pandas as pd
from datetime import datetime
from main import process_all_pdfs  # ✅ Import function to process PDFs at upload

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
st.subheader("📂 Manage PDFs")
selected_class = st.selectbox("Select a Class", ["General"] + [f"Class {i}" for i in range(1, 11)])

# Define paths for Teacher/Student folders
class_folder = os.path.join(CLASS_FOLDERS, f"class_{selected_class.split()[1]}") if "Class" in selected_class else os.path.join(CLASS_FOLDERS, "general")
teacher_folder = os.path.join(class_folder, f"{selected_class}_Teacher")
student_folder = os.path.join(class_folder, f"{selected_class}_Student")

# Ensure folders exist
os.makedirs(teacher_folder, exist_ok=True)
os.makedirs(student_folder, exist_ok=True)

# Choose where to upload PDF
upload_destination = st.radio("Upload PDF to:", ["Teacher", "Student", "Both"])

# Prevent infinite rerun by tracking upload state
if "upload_complete" not in st.session_state:
    st.session_state.upload_complete = False

# Upload a new PDF
st.subheader("📤 Upload a New PDF")
uploaded_pdf = st.file_uploader("Choose a PDF to Upload", type=["pdf"])

if uploaded_pdf is not None and not st.session_state.upload_complete:
    destination_folders = []
    if upload_destination in ["Teacher", "Both"]:
        destination_folders.append(teacher_folder)
    if upload_destination in ["Student", "Both"]:
        destination_folders.append(student_folder)

    for folder in destination_folders:
        pdf_path = os.path.join(folder, uploaded_pdf.name)
        
        # Prevent duplicate uploads
        if os.path.exists(pdf_path):
            st.warning(f"⚠️ The file '{uploaded_pdf.name}' already exists. Skipping upload.")
        else:
            with open(pdf_path, "wb") as f:
                f.write(uploaded_pdf.read())

            st.info(f"⏳ Processing {uploaded_pdf.name}... (This may take a while)")

            # ✅ Ensure ChromaDB is properly released before storing new embeddings
            if any(fname.endswith(".pdf") for fname in os.listdir(folder)):
                process_all_pdfs(folder)
                st.success(f"✅ Uploaded & Processed {uploaded_pdf.name} in {upload_destination} folder.")
            else:
                st.warning("⚠️ No valid PDFs found to process. Upload at least one PDF.")

    # ✅ Set upload complete flag to prevent rerun
    st.session_state.upload_complete = True
    st.rerun()  # ✅ Corrected this line!

# Section 2: Delete PDFs
st.subheader("❌ Delete PDFs")
delete_folder = st.radio("Select Folder:", ["Teacher", "Student"])
delete_folder_path = teacher_folder if delete_folder == "Teacher" else student_folder

pdf_files = [f for f in os.listdir(delete_folder_path) if f.endswith(".pdf")]
if pdf_files:
    selected_pdf = st.selectbox("Select a PDF to Delete", pdf_files)
    if st.button("Delete Selected PDF"):
        os.remove(os.path.join(delete_folder_path, selected_pdf))
        st.success(f"Deleted {selected_pdf}")
        st.rerun()
else:
    st.write("No PDFs found in selected folder.")

# Section 3: Update Class Passwords
st.subheader("🔑 Manage Class Passwords")
with open(PASSWORDS_FILE, "r") as f:
    passwords = json.load(f)

new_password = st.text_input(f"Set New Password for {selected_class}_{upload_destination}", type="password")
if st.button("Update Password"):
    passwords[f"{selected_class}_{upload_destination}"] = new_password
    with open(PASSWORDS_FILE, "w") as f:
        json.dump(passwords, f, indent=4)
    st.success(f"✅ Password updated for {selected_class}_{upload_destination}")

# Section 4: View Access Logs
st.subheader("📜 Access Logs")
if os.path.exists(LOGS_FILE):
    try:
        logs_df = pd.read_csv(LOGS_FILE, on_bad_lines='skip')
        st.dataframe(logs_df)
    except pd.errors.ParserError:
        st.error("⚠️ Error reading access logs. The file may have formatting issues.")
else:
    st.write("No access logs found.")

# Logout Button
if st.button("Logout"):
    st.session_state.admin_authenticated = False
    st.session_state.upload_complete = False  # Reset upload state
    st.rerun()
