from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
# from langchain_community.chains import ConversationalRetrievalChain
from langchain_community.llms import ChatOpenAI
from langchain_community.memory import ConversationBufferMemory
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain

# Example setup
llm = ChatOpenAI(model="gpt-3.5-turbo")
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.load_local("faiss_index", embeddings)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    memory=memory
)

query = "What is LangChain?"
response = qa_chain.run(query)
print(response)
