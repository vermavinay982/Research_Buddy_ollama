from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os

# 1️⃣ Set your Gemini API key in environment variable
os.environ["GOOGLE_API_KEY"] = "AIzaSyDtQ7gbyyLSavzDxthjvqvxK8LyXucR9E8"

# 2️⃣ Initialize the embeddings model
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# 3️⃣ Test embedding generation
text = "This is a test sentence for embedding."
vector = embeddings.embed_query(text)
print(len(vector), "dimensions")
