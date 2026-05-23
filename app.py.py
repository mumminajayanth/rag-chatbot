
import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000,
        chunk_overlap=1000
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_answer(user_question):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = db.similarity_search(user_question)
    context = "\n\n".join([doc.page_content for doc in docs])
    prompt_template = """
    You are a helpful, friendly and intelligent AI assistant.
    Answer the question in a conversational, human like way.
    Use the context provided to answer accurately.
    If answer is not in context, use your own knowledge to help.
    Be friendly, clear and detailed in your answers.

    Context:\n{context}\n
    Question:\n{question}\n
    Answer:
    """
    model = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.3, api_key=os.environ["GROQ_API_KEY"])
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = prompt | model | StrOutputParser()
    response = chain.invoke({"context": context, "question": user_question})
    return response

def main():
    st.set_page_config(page_title="PDF Chatbot", page_icon="🤖", layout="wide")
    st.title("🤖 Chat with your PDF")
    st.markdown("Upload a PDF and ask questions from it!")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    with st.sidebar:
        st.title("📂 Upload PDF Here")
        pdf_docs = st.file_uploader("Choose PDF files", accept_multiple_files=True, type="pdf")
        if st.button("Submit & Process"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("✅ Done! Ask your question now.")
            else:
                st.error("Please upload a PDF first!")

        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.success("Chat cleared!")

    for message in st.session_state.messages:
        if message["role"] == "user":
            st.chat_message("user").write(message["content"])
        else:
            st.chat_message("assistant").write(message["content"])

    user_question = st.chat_input("Type your question here...")

    if user_question:
        st.session_state.messages.append({"role": "user", "content": user_question})
        st.chat_message("user").write(user_question)

        with st.spinner("Finding answer..."):
            answer = get_answer(user_question)

        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.chat_message("assistant").write(answer)

if __name__ == "__main__":
    main()
    
