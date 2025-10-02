from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate 
from langchain.schema.output_parser import StrOutputParser
from langchain_openai import ChatOpenAI

# Load environment variables from .env
load_dotenv()

# Create a ChatOpenAI model
model = ChatOpenAI(model="gpt-4o")

# Define prompt templates (no need for separate Runnable chains)
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a comedian who tells jokes about {topic}."),
        ("human", "Tell me {joke_count} jokes."),
    ]
)

# Create the combined chain using LangChain Expression Language (LCEL)
chain = prompt_template | model | StrOutputParser()
# without StrOutputParser we receive the full AIMessage object not just the text content
# chain = prompt_template | model

# Run the chain
result = chain.invoke({"topic": "doctors", "joke_count": 3})
# Note: we don't invoke prompt_template and model separately here, we just run the combined chain
# chain.invoke() does everything in one go and it handles passing outputs to inputs automatically

# Output
print(result)
