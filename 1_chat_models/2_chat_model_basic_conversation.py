from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

# ------------------------
# Load environment variables from .env
# ------------------------
load_dotenv()

# ------------------------
# Create a ChatOpenAI model
# ------------------------
model = ChatOpenAI(model="gpt-4o")

# ------------------------
# Create list of messages for conversation
# ------------------------
messages = [
    SystemMessage(content="Solve the following math problems"),
    HumanMessage(content="What is 81 divided by 9?"),
]
# SystemMessage:
#   Message for priming AI behavior, usually passed in as the first of a sequence of input messages.
# HumanMessage:
#   Message from a human to the AI model.
# AIMessage:
#   Message from an AI.


# ------------------------
# Invoke the model with messages
# ------------------------
result = model.invoke(messages) # Here we pass a list of messages to the model instead of hardcoded string
print(f"Answer from AI: {result.content}")


# ------------------------
# # Continue the conversation
# ------------------------
messages = [
    SystemMessage(content="Solve the following math problems"),
    HumanMessage(content="What is 81 divided by 9?"),
    AIMessage(content="81 divided by 9 is 9."),
    HumanMessage(content="What is 10 times 5?"),
]

# Invoke the model with messages
result = model.invoke(messages)
print(f"Answer from AI: {result.content}")
