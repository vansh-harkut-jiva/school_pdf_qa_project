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