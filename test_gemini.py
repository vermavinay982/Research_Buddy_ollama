import google.generativeai as genai
import os

os.environ['GEMINI_API_KEY'] = "AIzaSyDtQ7gbyyLSavzDxthjvqvxK8LyXucR9E8"

genai.configure(api_key=os.environ['GEMINI_API_KEY'])
# print([i for i in genai.list_models()])
model = genai.GenerativeModel("gemini-2.0-flash")
# Now you can use the 'model' object for interactions
response = model.generate_content("Hello, how are you?")
print(response.text)