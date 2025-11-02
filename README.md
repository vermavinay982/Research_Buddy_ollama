# ResearchBuddy — Ollama-powered Research Assistant

**A local, Streamlit-based Research Assistant that ingests PDFs, builds a Chroma vector store, and uses Ollama (local) for embeddings and conversational QA.**

---

## 🚀 Quick summary
ResearchBuddy lets you ask questions about a collection of research papers (PDFs). It splits documents into chunks, creates embeddings (via Ollama), stores them in Chroma, and serves a conversational retrieval interface using LangChain + Ollama inside Streamlit.

This README documents how to set up and run the `ollama_research_buddy.py` script included in the repository.

---

## ✅ Features
- Ingest multiple PDF files from a `data/` folder
- Document chunking (configurable chunk size and overlap)
- Local embeddings using Ollama embedding models
- Conversational Retrieval (RAG) with Ollama chat models
- Streamlit UI for interactive Q&A and chat history
- Robust compatibility fallbacks for multiple LangChain versions

---

## 🧰 Requirements & Recommended Versions
- Python 3.9+ (3.10/3.11 recommended)
- Streamlit
- LangChain (your install may be `langchain`, `langchain_classic`, or `langchain_community` depending on versions)
- Ollama CLI & Python client (local Ollama server)
- ChromaDB (chromadb) and `langchain_chroma` or `chromadb` LangChain integration

Example (recommended) pip installs — adjust to your environment and package naming:
```bash
pip install streamlit chromadb langchain langchain-ollama langchain_chroma ollama langchain_community
```

> Note: package names and import paths for LangChain + Ollama integrations differ between releases. The `ollama_research_buddy.py` file includes fallback imports and robust chain construction to reduce import-time errors across versions.

---

## ⚙️ Setup (Ollama + Models)
1. Install Ollama and start the server. Follow instructions at https://ollama.com.
2. Pull the models you want locally. Example:
```bash
ollama pull llama3.1
ollama pull mxbai-embed-large
```
3. Confirm Ollama server is running (default `http://localhost:11434`). If you use a different host/port, set the `OLLAMA_BASE_URL` environment variable before running Streamlit.

---

## 📝 Environment variables
| Variable | Default | Purpose |
|---|---:|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Base URL where Ollama serves (change if different) |
| `OLLAMA_CHAT_MODEL` | `llama3.1` | Local chat model name pulled with `ollama pull` |
| `OLLAMA_EMBED_MODEL` | `mxbai-embed-large` | Local embedding model name pulled with `ollama pull` |

Set them in your shell or IDE environment, for example:
```bash
export OLLAMA_BASE_URL="http://localhost:11434"
export OLLAMA_CHAT_MODEL="llama3.1"
export OLLAMA_EMBED_MODEL="mxbai-embed-large"
```

On Windows (PowerShell):
```powershell
$env:OLLAMA_BASE_URL="http://localhost:11434"
$env:OLLAMA_CHAT_MODEL="llama3.1"
$env:OLLAMA_EMBED_MODEL="mxbai-embed-large"
```

---

## 📂 Project layout
```
ResearchBuddy/
├─ data/                      # put your research PDFs here
├─ ollama_research_buddy.py   # main Streamlit app (Ollama + LangChain)
├─ requirements.txt           # (optional) pinned dependencies
└─ README.md                  # this file
```

---

## ▶️ Running locally
1. Put your PDFs inside the `data/` folder.
2. Start the Ollama server (if not already running):
```bash
ollama serve
```
3. Run Streamlit:
```bash
streamlit run ollama_research_buddy.py
```
4. Open the URL printed by Streamlit (usually `http://localhost:8501`).

---

## 🛠 Troubleshooting & Tips
- **UnhashableParamError / st.cache_resource errors**: If you see Streamlit complaining about unhashable parameters, the app avoids caching the conversational chain and instead places it in `st.session_state`. This prevents hashing complex LangChain objects like Chroma stores.
- **Prompt/Chain validation (pydantic errors)**: LangChain versions differ in where they expect prompt templates. The script tries multiple fallback patterns; if you still see validation errors, run `pip show langchain` and `pip show langchain_classic` and open an issue with those details.
- **Embeddings empty or mismatched dims**: Try a different embedding model (e.g. `nomic-embed-text`, `mxbai-embed-large`) and ensure the Ollama model supports embeddings.
- **Model memory / RAM**: Large local models require large RAM and GPU; choose smaller models if you get out-of-memory errors.
- **Logging**: Observe the terminal output where you run `streamlit` and `ollama serve` for helpful error messages.

---

## 🧩 Customization pointers
- Adjust chunk size & overlap in the `RecursiveCharacterTextSplitter` instantiation for different document types.
- Swap the retriever settings (e.g., `search_type="mmr"` or `k` values) to tune retrieval relevance vs. breadth.
- Replace the default prompt template in the script to change the assistant’s behavior or persona.

---

## 🧾 License & Attribution
This project is provided as-is for educational and research use. You may adapt and reuse it with appropriate attribution.

---

## 🙏 Creator
**Vinay Verma**

---

## 📬 Feedback / Contributions
If you make improvements, please create a fork and open a PR. If you hit a bug, include the output of `pip show langchain` and `pip show langchain_classic` in your issue to help debug different LangChain versions.

---

**Enjoy exploring your papers!** 🎓🔬
