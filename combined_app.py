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