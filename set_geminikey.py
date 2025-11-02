import os
import getpass

# Set your API Key as an environment variable
if 'GEMINI_API_KEY' not in os.environ:
    os.environ['GEMINI_API_KEY'] = getpass.getpass("Enter your Gemini API Key: ")