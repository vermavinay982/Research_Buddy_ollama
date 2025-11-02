import os
import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.chains.conversational_retrieval.base import ConversationalRetrievalChain

# Ollama / LangChain-Ollama imports
# Depending on your langchain/ollama integration version you might use:
# from langchain_ollama import OllamaEmbeddings, OllamaLLM
# or from langchain_community.chat_models import ChatOllama, and from langchain_community.embeddings import OllamaEmbeddings
# The example below uses langchain_ollama names which are common in docs.

try:
    from langchain_ollama import OllamaEmbeddings, OllamaLLM
except Exception:
    # fallback path used by some langchain builds
    from langchain_community.embeddings import OllamaEmbeddings
    from langchain_community.chat_models.ollama import ChatOllama as OllamaLLM

# --- Configuration ---
PROJECT_NAME = "ResearchBuddy"
DATA_DIR = "data"

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
# Choose models you've pulled locally with the `ollama pull` command:
# chat_model: e.g. "llama3.1" or "gpt-oss:20b" (whatever you pulled)
# embed_model: e.g. "mxbai-embed-large" or "nomic-embed-text"
CHAT_MODEL = os.environ.get("OLLAMA_CHAT_MODEL", "llama3.1")
EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "mxbai-embed-large")

# --- Core LangChain Functions ---

# @st.cache_resource
def load_and_process_papers():
    """
    Loads PDF documents from DATA_DIR, splits into chunks,
    and creates a Chroma vector store using Ollama embeddings.
    """
    st.info(f"[{PROJECT_NAME}] Ingesting and processing research papers from the '{DATA_DIR}' folder...")

    pdf_files = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.lower().endswith('.pdf')]
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

    # Initialize Ollama embeddings (talks to local Ollama server)
    embeddings = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_BASE_URL)

    # Create Chroma vector store from documents (embedding function provided)
    # Note: depending on Chroma/LangChain versions, you might pass `embedding_function=embeddings`
    try:
        vector_store = Chroma.from_documents(chunks, embeddings)
    except TypeError:
        vector_store = Chroma.from_documents(chunks, embedding_function=embeddings)

    st.success(f"✅ Processed {len(pdf_files)} papers into {len(chunks)} searchable chunks.")
    return vector_store, len(pdf_files)

# PromptTemplate import with fallback for different langchain versions

# PromptTemplate import fallback
try:
    from langchain.prompts import PromptTemplate
except Exception:
    from langchain_core.prompts import PromptTemplate

# load_qa_chain import fallback (some distributions)
try:
    from langchain_classic.chains.question_answering.chain import load_qa_chain
except Exception:
    try:
        from langchain.chains.question_answering import load_qa_chain
    except Exception:
        load_qa_chain = None  # we'll handle later

def get_conversational_chain(vector_store):
    """
    Build a ConversationalRetrievalChain with several fallbacks for LangChain version differences.
    Do NOT decorate with @st.cache_resource because vector_store is unhashable.
    """
    llm = OllamaLLM(model=CHAT_MODEL, temperature=0.5, base_url=OLLAMA_BASE_URL)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    qa_prompt = PromptTemplate(
        template="""
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
        """,
        input_variables=["chat_history", "context", "question"],
    )

    retriever = vector_store.as_retriever(search_type="mmr")

    # Try 1: preferred - use from_llm with combine_docs_chain_kwargs (some versions accept this)
    try:
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            combine_docs_chain_kwargs={"prompt": qa_prompt},
        )
        print("Built chain using from_llm with combine_docs_chain_kwargs")
        return conversation_chain
    except Exception as e1:
        err1 = e1

    # Try 2: use from_llm with chain_type_kwargs (older advice) but without extra keys first
    try:
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            chain_type="stuff",
            # do NOT pass prompt here if model forbids extras
        )
        # If this succeeded, try to attach the prompt into the underlying qa chain if possible:
        try:
            # Many implementations put the QA chain under conversation_chain.combine_documents_chain
            if hasattr(conversation_chain, "combine_documents_chain"):
                c = conversation_chain.combine_documents_chain
                # If the chain has a "prompt" attribute, set it
                if hasattr(c, "prompt"):
                    c.prompt = qa_prompt
            print("Built chain using from_llm(chain_type='stuff') and attached prompt if possible")
        except Exception:
            pass
        return conversation_chain
    except Exception as e2:
        err2 = e2

    # Try 3: build a QA chain manually via load_qa_chain (if available), then pass into constructor
    if load_qa_chain is not None:
        try:
            # load_qa_chain signature varies — try common forms
            try:
                qa_chain = load_qa_chain(llm=llm, chain_type="stuff", prompt=qa_prompt)
            except TypeError:
                qa_chain = load_qa_chain(llm, chain_type="stuff", prompt=qa_prompt)
            # Many ConversationalRetrievalChain implementations allow passing combine_documents_chain or combine_docs_chain
            try:
                conversation_chain = ConversationalRetrievalChain(
                    retriever=retriever,
                    combine_documents_chain=qa_chain,
                    memory=memory,
                )
                print("Built chain using manual qa_chain -> combine_documents_chain")
                return conversation_chain
            except TypeError:
                # Try alternate param name
                conversation_chain = ConversationalRetrievalChain(
                    retriever=retriever,
                    combine_docs_chain=qa_chain,
                    memory=memory,
                )
                print("Built chain using manual qa_chain -> combine_docs_chain")
                return conversation_chain
        except Exception as e3:
            err3 = e3
    else:
        err3 = None

    # Try 4: final fallback - call from_llm with prompt passed in chain_type_kwargs but wrapped as serializable dict
    # (some versions accept a dict for prompt instead of PromptTemplate)
    try:
        prompt_dict = {"template": qa_prompt.template, "input_variables": qa_prompt.input_variables}
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            chain_type="stuff",
            chain_type_kwargs={"prompt": prompt_dict},
        )
        print("Built chain using from_llm with prompt as dict in chain_type_kwargs")
        return conversation_chain
    except Exception as e4:
        err4 = e4

    # If all attempts failed, raise a combined, informative error so you can paste into chat for debugging.
    combined = "\n\n".join(
        [
            f"Attempt 1 (combine_docs_chain_kwargs) error:\n{repr(err1)}" if 'err1' in locals() else "",
            f"Attempt 2 (from_llm chain_type) error:\n{repr(err2)}" if 'err2' in locals() else "",
            f"Attempt 3 (manual load_qa_chain) error:\n{repr(err3)}" if 'err3' in locals() else "",
            f"Attempt 4 (prompt as dict) error:\n{repr(err4)}" if 'err4' in locals() else "",
        ]
    )
    raise RuntimeError(
        "Could not construct ConversationalRetrievalChain with available LangChain version.\n"
        "Tried multiple fallbacks. See errors below:\n\n" + combined
    )

import re

def extract_text_from_message(message):
    """
    Robustly extract the text content from a message-like object.
    Supports:
      - dicts with keys like 'content', 'text', 'answer', 'result', 'output_text'
      - LangChain BaseMessage-like objects with .content or .text
      - plain strings or reprs like "AIMessage(content='...')"
    """
    # If it's already a plain string
    if isinstance(message, str):
        return message

    # If it's a dict-like
    if isinstance(message, dict):
        for key in ("content", "text", "answer", "result", "output_text", "response"):
            if key in message and message[key]:
                # If it's list of chunks, join them
                val = message[key]
                if isinstance(val, list):
                    return "\n\n".join(str(v) for v in val)
                return str(val)

    # If it's an object with attributes (e.g., LangChain messages)
    # Try common attribute names
    for attr in ("content", "text", "answer", "result", "output_text", "response"):
        if hasattr(message, attr):
            val = getattr(message, attr)
            if callable(val):
                try:
                    val = val()
                except Exception:
                    pass
            if val is None:
                continue
            if isinstance(val, list):
                return "\n\n".join(str(v) for v in val)
            return str(val)

    # If nothing found, try to parse repr() like "AIMessage(content='...')"
    rep = repr(message)
    m = re.search(r"content=(?:'|\")(.+?)(?:'|\")", rep, flags=re.DOTALL)
    if m:
        return m.group(1)

    # Last resort: str(message)
    return str(message)

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

        # --- Replace your interactive block with this ---
        if user_question:
            with st.spinner("💭 ResearchBuddy is thinking..."):
                raw_response = st.session_state.conversation({'question': user_question})
                # Many chains return dicts with 'chat_history' or 'history'; handle both
                st.session_state.chat_history = raw_response.get('chat_history', raw_response.get('history', st.session_state.chat_history))

        st.subheader("🧠 Discussion Log")
        chat_placeholder = st.empty()

        with chat_placeholder.container():
            for i, message in enumerate(st.session_state.chat_history):
                # Determine role safely
                role = None
                if isinstance(message, dict):
                    role = message.get("type") or message.get("role")
                else:
                    role = getattr(message, "type", None) or getattr(message, "role", None)

                text = extract_text_from_message(message)

                # Normalize role names
                role_lower = (role or "").lower() if isinstance(role, str) else None
                if role_lower in ("human", "user", "system"):
                    st.markdown(f"**You:** {text}")
                else:
                    # assistant / ai / default
                    st.markdown(f"**{PROJECT_NAME}:** {text}")

if __name__ == '__main__':
    main()
