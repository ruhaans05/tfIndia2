import streamlit as st
import os
import tempfile
import openai
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI

# === Load API Key ===
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("❌ OpenAI API key not found. Please set it in your environment.")
    st.stop()
openai.api_key = api_key

# === Streamlit Setup ===
st.set_page_config(page_title="🧬 Train Your Own Assistant", layout="centered")
st.title("🧬 Day 18 — Build Your Own Mini LLM Assistant")
st.caption("Upload data or type answers, and we'll build a localized assistant that understands your content.")

# === Input Options ===
input_method = st.radio("Choose how to add your data:", ["Upload a .txt file", "Type or paste content"])

user_text = ""
if input_method == "Upload a .txt file":
    uploaded_file = st.file_uploader("Upload a text file", type="txt")
    if uploaded_file:
        user_text = uploaded_file.read().decode("utf-8")
else:
    user_text = st.text_area("Enter your content below:", height=200, placeholder="Paste FAQs, textbook material, etc.")

if user_text.strip():
    st.success("✅ Content received. Ready to generate your assistant.")

    if st.button("🔧 Build My Assistant"):
        with st.spinner("Building your vector index..."):

            # Save content to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as temp_file:
                temp_file.write(user_text.encode())
                filepath = temp_file.name

            # Load and split text
            loader = TextLoader(filepath)
            documents = loader.load()
            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            docs = text_splitter.split_documents(documents)

            # Create vector store
            embeddings = OpenAIEmbeddings()
            vectorstore = FAISS.from_documents(docs, embeddings)

            # Store in session for Q&A
            st.session_state.vectorstore = vectorstore

        st.success("🎉 Your assistant is ready! Ask it anything below.")

# === Query Interface ===
if "vectorstore" in st.session_state:
    query = st.text_input("❓ Ask your Assistant a Question")
    if query:
        with st.spinner("Thinking..."):
            chain = load_qa_chain(OpenAI(temperature=0), chain_type="stuff")
            docs = st.session_state.vectorstore.similarity_search(query)
            answer = chain.run(input_documents=docs, question=query)
            st.markdown(f"**🧠 Answer:** {answer}")
