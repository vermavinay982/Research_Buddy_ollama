import os
import streamlit as st
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.chains.conversational_retrieval.base import ConversationalRetrievalChain
import getpass

# credential_path = "F:\code\ResearchBuddy\creds.json"
# os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credential_path


# --- Configuration ---
PROJECT_NAME = "ResearchBuddy"
DATA_DIR = "data"

os.environ['GEMINI_API_KEY'] = "Your Key"
# --- Set your API Key ---
if 'GEMINI_API_KEY' not in os.environ:
    try:
        os.environ['GEMINI_API_KEY'] = st.secrets["GEMINI_API_KEY"]
    except Exception:
        os.environ['GEMINI_API_KEY'] = getpass.getpass("Enter your Gemini API Key: ")

# --- Core LangChain Functions ---

@st.cache_resource
def load_and_process_papers():
    """
    Loads all PDF documents from the 'data' directory, splits them, 
    and creates a Chroma vector store.
    """
    st.info(f"[{PROJECT_NAME}] Ingesting and processing research papers from the '{DATA_DIR}' folder...")

    pdf_files = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith('.pdf')]
    if not pdf_files:
        st.error(f"No PDF files found in the '{DATA_DIR}' folder. Please add your research papers.")
        return None, 0

    all_documents = []
    for file_path in pdf_files:
        try:
            loader = PyPDFLoader(file_path)
            all_documents.extend(loader.load())
        except Exception as e:
            st.warning(f"Could not load PDF file: {file_path}. Error: {e}")

    if not all_documents:
        st.error("No valid research papers could be loaded.")
        return None, 0

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_documents(all_documents)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    vector_store = Chroma.from_documents(chunks, embeddings)

    st.success(f"✅ Processed {len(pdf_files)} papers into {len(chunks)} searchable chunks.")
    return vector_store, len(pdf_files)

@st.cache_resource
def get_conversational_chain(vector_store):
    """
    Sets up the LangChain ConversationalRetrievalChain for Q&A and discussion.
    """
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.5)

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(search_type="mmr"),
        memory=memory,
        combine_docs_chain_kwargs={
            "prompt": """
            You are ResearchBuddy, an AI specialized in analyzing scientific and technical papers.
            Answer the user's question based *only* on the provided retrieved documents.
            If the information is not in the documents, state that clearly, then pivot to a discussion point related to the query.
            Be thorough, citing the source paper's topic when possible.

            Chat History:
            {chat_history}

            Context (Documents):
            {context}

            Question: {question}
            Answer:
            """
        }
    )
    return conversation_chain

# --- Streamlit UI ---
def main():
    st.set_page_config(page_title=PROJECT_NAME, layout="wide")
    st.title(f"🔬 {PROJECT_NAME}")
    st.markdown("Ask me questions or discuss the content of your uploaded research papers!")

    vector_store, num_papers = load_and_process_papers()

    if vector_store:
        if "conversation" not in st.session_state:
            st.session_state.conversation = get_conversational_chain(vector_store)
            st.session_state.chat_history = []

        st.subheader(f"Ready to discuss {num_papers} research paper(s).")

        user_question = st.text_input("Your question or discussion point:", key="user_input")

        if user_question:
            with st.spinner("💭 ResearchBuddy is thinking..."):
                response = st.session_state.conversation({'question': user_question})
                st.session_state.chat_history = response['chat_history']

        st.subheader("🧠 Discussion Log")
        chat_placeholder = st.empty()

        with chat_placeholder.container():
            for i, message in enumerate(st.session_state.chat_history):
                if message.type == "human":
                    st.markdown(f"**You:** {message.content}")
                else:
                    st.markdown(f"**{PROJECT_NAME}:** {message.content}")

if __name__ == '__main__':
    main()
