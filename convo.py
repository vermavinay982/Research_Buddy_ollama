"""
A simple ConversationalRetrievalChain demo
backed by Cohere Embed v3 and Claude 3 Sonnet
via Amazon Bedrock 🏔️
"""

import os

from urllib.request import urlretrieve

from langchain.chains import ConversationalRetrievalChain

from langchain_community.chat_models import BedrockChat
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings.bedrock import BedrockEmbeddings
from langchain_community.vectorstores import Chroma

from langchain_text_splitters import CharacterTextSplitter

# Download the file
if not os.path.isfile("state_of_the_union.txt"):
    urlretrieve(
        "https://raw.githubusercontent.com/hwchase17/chat-your-data/master/state_of_the_union.txt",
        "state_of_the_union.txt"
    )

# Set up the vector store + retriever
raw_documents = TextLoader('state_of_the_union.txt', encoding="utf-8").load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
documents = text_splitter.split_documents(raw_documents)
vectorstore = Chroma.from_documents(
    documents, BedrockEmbeddings(model_id="cohere.embed-english-v3")
)
retriever = vectorstore.as_retriever()

# Select the model
chat = BedrockChat(
    model_id="anthropic.claude-3-sonnet-20240229-v1:0",
    region_name="us-east-1",
    model_kwargs={'temperature': 0.0}
)

# Create the chain
chain = ConversationalRetrievalChain.from_llm(
    llm=chat,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(),
    return_generated_question=True,
    verbose=False,
)

# Ask a question
response = chain.invoke({
    'question': "What did the president say about Ketanji Brown Jackson",
    'chat_history': []})
print(response['answer'])