import streamlit as st
import os
import uuid
import tempfile
from dotenv import load_dotenv

import chromadb.api
chromadb.api.client.SharedSystemClient.clear_system_cache()

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# Load env variables
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

# Streamlit SetUp
st.title("📚 Chat with your PDF notes!")
st.write("Upload your PDF notes or books and ask questions like you're talking to a tutor.")

# File uploader
uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)

# To load models only once
@st.cache_resource
def load_models():
    with st.spinner("Please wait while the models are loading.."):
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        llm = ChatGroq(model="gemma2-9b-it")
        return embeddings, llm

embeddings, llm = load_models()

# To create sessions
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "chat_history" not in st.session_state:
    st.session_state.chat_history = ChatMessageHistory()


if "processed_files" not in st.session_state:
    st.session_state.processed_files = []

current_files = [f.name for f in uploaded_files] if uploaded_files else []    

if uploaded_files: 
    if current_files != st.session_state.processed_files:
        with st.spinner(text="Processing.."):
            # Reset chat if new files are uploaded
            if not st.session_state.get("pdf_loaded", False):
                st.session_state.messages = []
                st.session_state.chat_history = ChatMessageHistory()

            all_docs = []
            temp_paths = []

            # Process all uploaded PDFs
            for file in uploaded_files:
                temp_path = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4().hex}.pdf")
                with open(temp_path, "wb") as f:
                    f.write(file.read())
                temp_paths.append(temp_path)
                loader = PyPDFLoader(temp_path)
                pages = loader.load()
                all_docs.extend(pages)

            # Split text into chunks
            splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
            split_docs = splitter.split_documents(all_docs)

            # Create Chroma DB and retriever
            db = Chroma.from_documents(split_docs, embeddings)
            retriever = db.as_retriever()

            # Save files to session
            st.session_state.processed_files = current_files
            st.session_state.retriever = retriever
            st.session_state.temp_path = temp_paths
    # Prompt to convert questions from context into standalone questions    
    sys_prompt = """
    Given a chat history and the latest user question which might reference context in the chat history,
    formulate a standalone question which can be understood without the chat history. Do not answer the question,
    just reformulate it if needed otherwise return it as it is.
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", sys_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    history_aware_retriever = create_history_aware_retriever(llm, st.session_state.retriever, prompt)

    # Prompt for final answer
    answer_prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer based on the context below. Be concise. If not sure, say you don't know.\n\n{context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    qna_chain = create_stuff_documents_chain(llm, answer_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, qna_chain)

    if "messages" not in st.session_state or st.session_state.get("pdf_loaded") is not True:
        st.session_state.messages = [{"role": "assistant", "content": "PDF uploaded ✅. What do you want to know?"}]
        st.session_state.pdf_loaded = True

    # Display the chat messages
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    # Take user input
    question = st.chat_input("Ask something from your PDF...")

    if question:
        st.chat_message("user").write(question)
        st.session_state.messages.append({"role": "user", "content": question})
        st.session_state.chat_history.add_user_message(question)

        chain_with_history = RunnableWithMessageHistory(
            rag_chain,
            lambda : st.session_state.chat_history,
            input_messages_key="input",
            history_messages_key="chat_history"
        )
        with st.spinner("Searching Documents.."):
            result = chain_with_history.invoke({"input": question}, config={
                "configurable": {"session_id": st.session_state.session_id}
            })

        answer = result["answer"]
        st.chat_message("assistant").write(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})

    # Remove temp files
    for path in st.session_state.temp_path:
        try:
            os.remove(path)
        except:
            pass
