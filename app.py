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