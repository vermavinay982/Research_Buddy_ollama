from google.auth import default
creds, project = default()
print("creds:", type(creds), creds)
print("project:", project)