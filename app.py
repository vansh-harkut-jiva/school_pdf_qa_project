import streamlit as st
import os
import json
import logging
import pandas as pd
from datetime import datetime
from main import get_answer_from_pdfs  # Updated import to use main_2.py

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
if selected_class == "General":
    # For "General", construct the folder path as classes/general/general_<role>
    class_key = "general"
    role_folder = role_folder_map.get(role, "")
    class_folder = os.path.join(CLASS_FOLDERS, class_key, f"general_{role_folder}")
else:
    # For specific classes, construct the folder path as classes/class_<number>/Class <number>_<role>
    class_key = f"class_{selected_class.split(' ')[1]}"  # Convert to match folder names like "class_1"
    role_folder = role_folder_map.get(role, "")
    class_folder = os.path.join(CLASS_FOLDERS, class_key, f"Class {selected_class.split(' ')[1]}_{role_folder}")

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
    elif selected_class == "General":
        # For "General", retrieve the password from the "general" section
        correct_password = CLASS_PASSWORDS.get("general", {}).get(role, "")
    else:
        # For specific classes, retrieve the password using the class key
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
            # Pass the role to the get_answer_from_pdfs function
            st.session_state.qa_agent = get_answer_from_pdfs(class_folder, role)
            if st.session_state.qa_agent is None:
                st.error("Error: No valid PDF files processed or an error occurred during initialization.")
                st.stop()
        except Exception as e:
            st.error(f"An error occurred while initializing the QA agent: {e}")
            st.stop()

    # WhatsApp-like chat container
    chat_container = st.container()
    with chat_container:
        for entry in st.session_state.get("chat_history", []):
            if entry["type"] == "user":
                # User message (right-aligned, green background)
                st.markdown(
                    f"""
                    <div style="text-align: right; background-color: #DCF8C6; padding: 10px; border-radius: 10px; margin: 5px;">
                        <strong>You:</strong> {entry['content']}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            elif entry["type"] == "bot":
                # Bot message (left-aligned, gray background)
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