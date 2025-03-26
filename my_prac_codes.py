# main.py
import os
import logging
import pytesseract
from pdf2image import convert_from_path
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_openai.embeddings import OpenAIEmbeddings
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
    collection_name = f"{os.path.basename(class_folder)}_{role}".replace(" ", "")
    collection = chroma_client.get_or_create_collection(name=collection_name)

    # Add texts with metadata to the collection
    collection.add(
        documents=all_text_chunks,
        metadatas=metadata_list,
        ids=[f"{metadata['source']}_chunk_{metadata['chunk_index']}" for metadata in metadata_list]
    )

    logging.info(f"Collection '{collection_name}' updated with all class PDFs.")
    return collection_name

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

def get_answer_from_pdfs(collection_name):
    """
    Connects to a ChromaDB collection and initializes a QA agent.
    """
    try:
        # Connect to ChromaDB
        chroma_client = chromadb.HttpClient(host=CHROMA_SERVER_HOST, port=CHROMA_SERVER_PORT)

        # Use OpenAI embeddings
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, model="text-embedding-3-large", dimensions=384)

        # Initialize Chroma vector store
        vectorstore = Chroma(client=chroma_client, collection_name=collection_name, embedding_function=embeddings)

        # Use ChatOpenAI for GPT-4
        llm = ChatOpenAI(model="gpt-4", temperature=0)

        # Define the prompt template
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

        # Initialize retriever
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

        # Initialize QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT.partial(document_names="")}
        )

        return QAAgent(qa_chain)

    except Exception as e:
        logging.error(f"An error occurred while initializing the QA agent: {e}")
        return None
    
# admin.py
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import streamlit as st
import os
import json
import pandas as pd
from main import process_all_pdfs  

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

            # ✅ Process PDFs and store in ChromaDB collections
            role = "Teacher" if folder == teacher_folder else "Student"
            process_all_pdfs(folder, role)  # Pass the folder and role to process_all_pdfs
            st.success(f"✅ Uploaded & Processed {uploaded_pdf.name} in {upload_destination} folder.")

    # ✅ Set upload complete flag to prevent rerun
    st.session_state.upload_complete = True
    st.rerun()

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


# app.py
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import streamlit as st
import os
import json
import logging
from main import get_answer_from_pdfs  # Ensure this function interacts with ChromaDB collections

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
    st.error("⚠️ Password file not found! Please create 'passwords.json' inside the 'data' folder.")
    st.stop()

# Branding
st.markdown(
    """
    <div style="text-align: center;">
        <h1>School PDF Question-Answering System</h1>
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

# Normalize the selected class to match collection names
if selected_class == "General":
    collection_name = f"general_{role}"
else:
    collection_name = f"class_{selected_class.split(' ')[1]}_{role}"

# Step 3: Password Authentication
st.subheader("Enter Password to Proceed:")
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

password_input = st.text_input("Enter Password:", type="password", key="password_input")

if st.button("Submit Password"):
    if role == "Principal":
        correct_password = CLASS_PASSWORDS.get("Principal", "")
    elif selected_class == "General":
        correct_password = CLASS_PASSWORDS.get("general", {}).get(role, "")
    else:
        correct_password = CLASS_PASSWORDS.get(f"class_{selected_class.split(' ')[1]}", {}).get(role, "")

    if password_input == correct_password:
        st.session_state.authenticated = True
        st.success("✅ Access granted!")
    else:
        st.error("❌ Incorrect password. Please try again.")
        st.stop()

if not st.session_state.authenticated:
    st.stop()

# Step 4: Start Chat Interface
st.subheader("Chat with Your PDFs")
if "qa_agent" not in st.session_state:
    try:
        # Initialize the QA agent by connecting to the ChromaDB collection
        st.session_state.qa_agent = get_answer_from_pdfs(collection_name)
        if st.session_state.qa_agent is None:
            st.error("Error: No valid data found in the collection or an error occurred during initialization.")
            st.stop()
    except Exception as e:
        st.error(f"An error occurred while initializing the QA agent: {e}")
        st.stop()

# WhatsApp-like chat container
chat_container = st.container()
with chat_container:
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for entry in st.session_state.get("chat_history", []):
        if entry["type"] == "user":
            st.markdown(
                f"""
                <div style="text-align: right; background-color: #DCF8C6; padding: 10px; border-radius: 10px; margin: 5px;">
                    <strong>You:</strong> {entry['content']}
                </div>
                """,
                unsafe_allow_html=True,
            )
        elif entry["type"] == "bot":
            st.markdown(
                f"""
                <div style="text-align: left; background-color: #EDEDED; padding: 10px; border-radius: 10px; margin: 5px;">
                    <strong>Bot:</strong> {entry['content']}
                </div>
                """,
                unsafe_allow_html=True,
            )

# Input form for user questions
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




        # combined_app.py:
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import streamlit as st
import os
import json
import pandas as pd
import logging
from main import process_all_pdfs, get_answer_from_pdfs  # Import functions from main.py

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Paths to data files
PASSWORDS_FILE = "data/passwords.json"
LOGS_FILE = "data/access_logs.csv"
CLASS_FOLDERS = "classes"

# Streamlit UI Configuration
st.set_page_config(page_title="School PDF Q&A System", layout="wide")

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Admin", "User"])

# ----------------------------------------
# Admin Page
# ----------------------------------------
if page == "Admin":
    # Define admin credentials
    ADMIN_USERNAME = "admin"
    ADMIN_PASSWORD = "admin123"

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

                # ✅ Process PDFs and store in ChromaDB collections
                role = "Teacher" if folder == teacher_folder else "Student"
                process_all_pdfs(folder, role)  # Pass the folder and role to process_all_pdfs
                st.success(f"✅ Uploaded & Processed {uploaded_pdf.name} in {upload_destination} folder.")

        # ✅ Set upload complete flag to prevent rerun
        st.session_state.upload_complete = True
        st.rerun()

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

# ----------------------------------------
# User Page
# ----------------------------------------
elif page == "User":
    st.title("School PDF Q&A System")

    # Load passwords from JSON file
    if os.path.exists(PASSWORDS_FILE):
        with open(PASSWORDS_FILE, "r") as f:
            CLASS_PASSWORDS = json.load(f)
    else:
        st.error("⚠️ Password file not found! Please create 'passwords.json' inside the 'data' folder.")
        st.stop()

    # Step 1: Select User Role
    st.subheader("Select Your Role:")
    role = st.selectbox("Are you a Principal, Teacher, or Student?", ["Principal", "Teacher", "Student"])

    # Step 2: Select Class (or General for shared PDFs)
    st.subheader("Select Your Class:")
    class_options = [f"Class {i}" for i in range(1, 11)] + ["General"]
    selected_class = st.selectbox("Choose a class or general access:", class_options)

    # Normalize the selected class to match collection names
    if selected_class == "General":
        collection_name = f"general_{role}"
    else:
        collection_name = f"class_{selected_class.split(' ')[1]}_{role}"

    # Step 3: Password Authentication
    st.subheader("Enter Password to Proceed:")
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    password_input = st.text_input("Enter Password:", type="password", key="password_input")

    if st.button("Submit Password"):
        if role == "Principal":
            correct_password = CLASS_PASSWORDS.get("Principal", "")
        elif selected_class == "General":
            correct_password = CLASS_PASSWORDS.get("general", {}).get(role, "")
        else:
            correct_password = CLASS_PASSWORDS.get(f"class_{selected_class.split(' ')[1]}", {}).get(role, "")

        if password_input == correct_password:
            st.session_state.authenticated = True
            st.success("✅ Access granted!")
        else:
            st.error("❌ Incorrect password. Please try again.")
            st.stop()

    if not st.session_state.authenticated:
        st.stop()

    # Step 4: Start Chat Interface
    st.subheader("Chat with Your PDFs")
    if "qa_agent" not in st.session_state:
        try:
            # Initialize the QA agent by connecting to the ChromaDB collection
            st.session_state.qa_agent = get_answer_from_pdfs(collection_name)
            if st.session_state.qa_agent is None:
                st.error("Error: No valid data found in the collection or an error occurred during initialization.")
                st.stop()
        except Exception as e:
            st.error(f"An error occurred while initializing the QA agent: {e}")
            st.stop()

    # WhatsApp-like chat container
    chat_container = st.container()
    with chat_container:
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        for entry in st.session_state.get("chat_history", []):
            if entry["type"] == "user":
                st.markdown(
                    f"""
                    <div style="text-align: right; background-color: #DCF8C6; padding: 10px; border-radius: 10px; margin: 5px;">
                        <strong>You:</strong> {entry['content']}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            elif entry["type"] == "bot":
                st.markdown(
                    f"""
                    <div style="text-align: left; background-color: #EDEDED; padding: 10px; border-radius: 10px; margin: 5px;">
                        <strong>Bot:</strong> {entry['content']}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    # Input form for user questions
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