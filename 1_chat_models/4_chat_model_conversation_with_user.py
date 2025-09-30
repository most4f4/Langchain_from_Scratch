from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage

# Previously: from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
# The difference is langchain_core is the old package name, langchain.schema is the new package name
# langchain_core is still available for backward compatibility, but langchain.schema is the preferred import path now.

# ------------------------
# Load environment variables from .env
# ------------------------
load_dotenv()

# ------------------------
# Create a ChatOpenAI model
# ------------------------
model = ChatOpenAI(model="gpt-4o")

# ------------------------
# Create a conversation loop with user input
# ------------------------
chat_history = []  # Use a list to store messages

# Set an initial system message (optional)
system_message = SystemMessage(content="You are a helpful AI assistant.")
chat_history.append(system_message)  # Add system message to chat history

# Chat loop
while True:
    query = input("You: ") # Get user input
    # Exit loop if user types 'exit'
    if query.lower() == "exit":
        break
    chat_history.append(HumanMessage(content=query))  # Add user message

    # Get AI response using history
    result = model.invoke(chat_history)
    response = result.content
    chat_history.append(AIMessage(content=response))  # Add AI message

    print(f"AI: {response}")


print("---- Message History ----")
print(chat_history)
