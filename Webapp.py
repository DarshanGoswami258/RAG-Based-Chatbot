import os
import google.generativeai as genai
from langchain.vectorstores import FAISS # This will be the vector database.
from langchain.embeddings import HuggingFaceEmbeddings # This is to perform word embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter # This is to split the text into chunks.
from pypdf import PdfReader # This is to read the PDF files.
import faiss
import streamlit as st
from pdfextractor_pdf import text_extractor_pdf


# Sidebar
st.sidebar.title("📂 Upload your Document (PDF Only)")
file_uploader = st.sidebar.file_uploader("Upload PDF", type=["pdf"])

# Main Title
st.title("🤖 :green[RAG Based Chatbot]")

# Instructions
with st.expander("📌 How to use this app"):
    st.markdown("""
    1. Upload your PDF document from the **sidebar**.  
    2. Type your query in the input box.  
    3. The chatbot will respond with answers from your document.  
    """)

# Configure LLM + Embeddings once file is uploaded
if file_uploader:
    file_text = text_extractor_pdf(file_uploader)

    # Step 1: Configure LLM
    key = os.environ.get("GOOGLE_API_KEY")
    genai.configure(api_key=key)
    llm_model = genai.GenerativeModel("gemini-2.5-flash-lite")

    # Step 2: Configure Embedding
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Step 3: Chunking
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
    chunks = splitter.split_text(file_text)

    # Step 4: Create Vector Store
    vectorstore = FAISS.from_texts(chunks, embedding_model)

    # Step 5: Retriever
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k":3})

    # Generate text function
    def generated_text(query):
        retrieved_docs = retriever.get_relevant_documents(query)
        context = " ".join([doc.page_content for doc in retrieved_docs])

        prompt = f"""
        You are a helpful assistant using RAG.
        Here is the context: {context}
        The query asked by the user is: {query}
        """

        response = llm_model.generate_content(prompt).text
        return response

    # Initialize chat history
    if "history" not in st.session_state:
        st.session_state.history = []

    # Chat UI
    st.subheader("💬 Chat with your Document")

    for msg in st.session_state.history:
        if msg["role"] == "user":
            st.chat_message("user").markdown(msg["text"])
        else:
            st.chat_message("assistant").markdown(msg["text"])

    # User input
    user_input = st.chat_input("Type your message here...")

    if user_input:
        # Save user message
        st.session_state.history.append({"role": "user", "text": user_input})

        # Generate model response
        model_output = generated_text(user_input)
        st.session_state.history.append({"role": "assistant", "text": model_output})

        # Show chatbot response immediately
        st.chat_message("assistant").markdown(model_output)

else:
    st.info("👈 Please upload a PDF from the sidebar to get started.")


    