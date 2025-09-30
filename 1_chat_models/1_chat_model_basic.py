from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# ------------------------
# Load environment variables from .env
# ------------------------
load_dotenv()
# This will load the OPENAI_API_KEY from the .env file into the environment
# Make sure you have created a .env file with your OpenAI API key

# ------------------------
# Create a ChatOpenAI model
# ------------------------
model = ChatOpenAI(model="gpt-4o")

# ChatOpenAI is a wrapper around OpenAI's chat models.
# You can specify different models like "gpt-3.5-turbo", "gpt-4", etc.
# You can also set parameters like temperature, max_tokens, etc.
# For example:
# model = ChatOpenAI(model="gpt-4o", temperature=0.7, max_tokens=1024)
# API key is automatically picked up from the environment variable OPENAI_API_KEY

# ------------------------
# Invoke the model with a message
# ------------------------
result = model.invoke("What is the capital of France?")
# .invoke() is the magic method that sends a message to the model
# it returns a ChatResult object which contains the full response
# returned object has several attributes like metadata, but the most important one is .content
# .content contains just the text response from the model

# ------------------------
# Print the result
# ------------------------
print("Full result:")
print(result)
print("Content only:")
print(result.content)
