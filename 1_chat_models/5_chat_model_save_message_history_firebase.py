# Example Source: https://python.langchain.com/v0.2/docs/integrations/memory/google_firestore/
import os
from dotenv import load_dotenv
from google.cloud import firestore # firestore client library
from langchain_google_firestore import FirestoreChatMessageHistory
from langchain_openai import ChatOpenAI

"""
Steps to replicate this example:

1. Create a Firebase account
2. Create a new Firebase project
   - Copy the Project ID
3. Create a Firestore database in the Firebase project
4. Enable the Firestore API in the Google Cloud Console:
   - https://console.cloud.google.com/apis/library/firestore.googleapis.com
5. Create a Service Account in Google Cloud Console:
   - Go to IAM & Admin → Service Accounts → Create Service Account
   - Give it a name (e.g., "langchain-dev")
   - Assign it a role:
       • For testing: Cloud Datastore Owner (roles/datastore.owner)
       • For production: Cloud Datastore User (roles/datastore.user)
   - Finish creation (you can skip “Principals with access”)
6. Generate a Service Account Key (JSON):
   - Go to your new service account → Keys tab → Add Key → Create new key → JSON
   - This downloads a `*.json` file
7. Point your environment to the JSON credentials:
   - inside Python:
       import os
       os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "path/to/key.json"
8. Install the Google Cloud CLI if not already:
   - https://cloud.google.com/sdk/docs/install
   - Authenticate locally:
       gcloud auth application-default login
   - Set your default project:
       gcloud config set project YOUR_PROJECT_ID
9. Run your LangChain + Firestore script
"""


os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "path/to/key.json"  # Update this path

load_dotenv()

# Setup Firebase Firestore
PROJECT_ID = "your_project_id"  # Update this with your Project ID
SESSION_ID = "user_session_id"  # This could be a username or a unique ID
COLLECTION_NAME = "chat_messages"  # Firestore collection name, update if needed

# Initialize Firestore Client
print("Initializing Firestore Client...")
client = firestore.Client(project=PROJECT_ID) # Connect to the Firestore database using the project ID

# Initialize Firestore Chat Message History
print("Initializing Firestore Chat Message History...")
chat_history = FirestoreChatMessageHistory( # Create a chat history instance that saves to Firestore
    session_id=SESSION_ID,
    collection=COLLECTION_NAME,
    client=client,
)
print("Chat History Initialized.")
print("Current Chat History:", chat_history.messages) # Print existing messages if any

# Initialize Chat Model
model = ChatOpenAI()

print("Start chatting with the AI. Type 'exit' to quit.")

while True:
    human_input = input("User: ")
    if human_input.lower() == "exit":
        break

    chat_history.add_user_message(human_input) # store user message
    ai_response = model.invoke(chat_history.messages) # invoke with the list of messages
    chat_history.add_ai_message(ai_response.content) # store AI message

    print(f"AI: {ai_response.content}")
