import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
import time
import base64
from groq import Groq

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
        except Exception as e:
            st.error(f"Error reading PDF: {str(e)}")
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_answer(user_question, vector_store):
    try:
        docs = vector_store.similarity_search(user_question, k=2)
        context = "\n\n".join([doc.page_content for doc in docs])
        prompt_template = """
        You are a helpful, friendly and intelligent AI assistant.
        Answer the question in a conversational, human like way.
        Use the context provided to answer accurately.
        If answer is not in context, use your own knowledge to help.
        Be friendly, clear and detailed in your answers.
        Keep your answer concise and under 200 words.
        Context:\n{context}\n
        Question:\n{question}\n
        Answer:
        """
        model = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0.3,
            api_key=os.environ["GROQ_API_KEY"],
            max_retries=5
        )
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        chain = prompt | model | StrOutputParser()
        time.sleep(2)
        response = chain.invoke({"context": context, "question": user_question})
        return response
    except Exception as e:
        return f"⚠️ Error: {str(e)}"

def analyze_image(image_file, user_question):
    try:
        # Convert image to base64
        image_data = base64.standard_b64encode(image_file.read()).decode("utf-8")
        # Get image type
        image_type = image_file.type
        # Use Groq vision model
        client = Groq(api_key=os.environ["GROQ_API_KEY"])
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{image_type};base64,{image_data}"
                            }
                        },
                        {
                            "type": "text",
                            "text": user_question if user_question else "Describe this image in detail. Extract all text if any."
                        }
                    ]
                }
            ],
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"⚠️ Error analyzing image: {str(e)}"

def main():
    st.set_page_config(page_title="PDF & Image Chatbot", page_icon="🤖", layout="wide")
    st.title("🤖 Chat with your PDF or Image")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "pdf_processed" not in st.session_state:
        st.session_state.pdf_processed = False
    if "mode" not in st.session_state:
        st.session_state.mode = None

    with st.sidebar:
        st.title("📂 Upload Files Here")
        st.info("""
        👋 **How to use:**
        1. Choose PDF or Image mode
        2. Upload your file
        3. Click Process
        4. Ask your question!
        """)

        # Mode selection
        mode = st.radio(
            "Select Mode:",
            ["📄 PDF Mode", "🖼️ Image Mode"]
        )

        if mode == "📄 PDF Mode":
            st.session_state.mode = "pdf"
            pdf_docs = st.file_uploader(
                "Upload PDF files",
                accept_multiple_files=True,
                type="pdf"
            )
            if st.button("Submit & Process PDF"):
                if pdf_docs:
                    with st.spinner("Processing PDF..."):
                        raw_text = get_pdf_text(pdf_docs)
                        if not raw_text or len(raw_text.strip()) == 0:
                            st.error("❌ Could not read PDF! Make sure it has text!")
                        else:
                            text_chunks = get_text_chunks(raw_text)
                            if len(text_chunks) == 0:
                                st.error("❌ PDF has no readable content!")
                            else:
                                embeddings = HuggingFaceEmbeddings(
                                    model_name="all-MiniLM-L6-v2"
                                )
                                st.session_state.vector_store = FAISS.from_texts(
                                    text_chunks,
                                    embedding=embeddings
                                )
                                st.session_state.pdf_processed = True
                                st.success("✅ PDF processed! Ask your question now.")
                else:
                    st.error("Please upload a PDF first!")

        else:
            st.session_state.mode = "image"
            image_file = st.file_uploader(
                "Upload Image",
                type=["jpg", "jpeg", "png", "webp"]
            )
            if image_file:
                st.image(image_file, caption="Uploaded Image", use_column_width=True)
                st.session_state.pdf_processed = True
                st.session_state.image_file = image_file
                st.success("✅ Image uploaded! Ask your question now.")

        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.session_state.pdf_processed = False
            st.session_state.vector_store = None
            st.success("Cleared!")

    # Main area
    if not st.session_state.pdf_processed:
        st.markdown("""
        ## 👋 Welcome!

        ### 📄 PDF Mode:
        - Upload any PDF
        - Ask questions from it

        ### 🖼️ Image Mode:
        - Upload any image
        - AI reads and describes it
        - Extract text from images
        - Analyze diagrams and charts

        **Select a mode from the sidebar to get started!**
        """)
    else:
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.chat_message("user").write(message["content"])
            else:
                st.chat_message("assistant").write(message["content"])

        user_question = st.chat_input("Type your question here...")
        if user_question:
            st.session_state.messages.append(
                {"role": "user", "content": user_question}
            )
            st.chat_message("user").write(user_question)
            with st.spinner("Finding answer..."):
                if st.session_state.mode == "image":
                    st.session_state.image_file.seek(0)
                    answer = analyze_image(
                        st.session_state.image_file,
                        user_question
                    )
                else:
                    answer = get_answer(
                        user_question,
                        st.session_state.vector_store
                    )
            st.session_state.messages.append(
                {"role": "assistant", "content": answer}
            )
            st.chat_message("assistant").write(answer)

if __name__ == "__main__":
    main()
