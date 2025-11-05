# LangChain Learning Summary

## 1. Chat Models

### 1.1. Basic Chat Models

```python
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4o")
result = model.invoke("What is the capital of France?")
print(result.content)
```

### 1.2. Basic Conversations

```python
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

messages = [
    SystemMessage(content="Solve the following math problems"),
    HumanMessage(content="What is 81 divided by 9?"),
    AIMessage(content="81 divided by 9 is 9."),
    HumanMessage(content="What is 10 times 5?"),
]
result = model.invoke(messages)
```

### 1.3. Model Alternatives

```python
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic

model = ChatAnthropic(model="claude-3-opus-20240229")
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
```

### 1.4. Conversation With User

```python
chat_history = []
system_message = SystemMessage(content="You are a helpful AI assistant.")
chat_history.append(system_message)

while True:
    query = input("You: ")
    if query.lower() == "exit":
        break
    
    chat_history.append(HumanMessage(content=query))
    result = model.invoke(chat_history)
    response = result.content
    chat_history.append(AIMessage(content=response))
    
    print(f"AI: {response}")
```

### 1.5. Save Message History Firebase

```python
import os
from google.cloud import firestore
from langchain_google_firestore import FirestoreChatMessageHistory

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "path/to/key.json"

PROJECT_ID = "your_project_id"
SESSION_ID = "user_session_id"
COLLECTION_NAME = "chat_messages"

client = firestore.Client(project=PROJECT_ID)

chat_history = FirestoreChatMessageHistory( 
    session_id=SESSION_ID,
    collection=COLLECTION_NAME,
    client=client,
)

while True:
    human_input = input("User: ")
    if human_input.lower() == "exit":
        break
    
    chat_history.add_user_message(human_input)
    ai_response = model.invoke(chat_history.messages)
    chat_history.add_ai_message(ai_response.content)
    
    print(f"AI: {ai_response.content}")
```

---

## 2. Prompt Templates

```python
from langchain.prompts import ChatPromptTemplate

messages = [
    ("system", "You are a comedian who tells jokes about {topic}."),
    ("human", "Tell me {joke_count} jokes."),
]

prompt_template = ChatPromptTemplate.from_messages(messages)
prompt = prompt_template.invoke({"topic": "lawyers", "joke_count": 3})

result = model.invoke(prompt)
print(result.content)
```

**Important Notes:**
- ❌ This does NOT work: Mixing tuple format with HumanMessage objects with placeholders
- ✅ This works: Either all tuples with placeholders OR all Message objects without placeholders

---

## 3. Chains

### 3.1. Basics

```python
from langchain.schema.output_parser import StrOutputParser

prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a comedian who tells jokes about {topic}."),
    ("human", "Tell me {joke_count} jokes."),
])

chain = prompt_template | model | StrOutputParser()
result = chain.invoke({"topic": "doctors", "joke_count": 3})
```

### 3.2. Under the Hood

```python
from langchain.schema.runnable import RunnableLambda, RunnableSequence

format_prompt = RunnableLambda(lambda x: prompt_template.format_prompt(**x))
invoke_model = RunnableLambda(lambda x: model.invoke(x.to_messages()))
parse_output = RunnableLambda(lambda x: x.content)

chain = RunnableSequence(
    first=format_prompt, 
    middle=[invoke_model], 
    last=parse_output
)

response = chain.invoke({"topic": "doctors", "joke_count": 2})
```

### 3.3. Extended Chains

```python
uppercase_output = RunnableLambda(lambda x: x.upper())
count_words = RunnableLambda(lambda x: f"Word count: {len(x.split())}\n{x}")

chain = prompt_template | model | StrOutputParser() | uppercase_output | count_words
```

### 3.4. Parallel Chains

```python
from langchain.schema.runnable import RunnableParallel, RunnableLambda

# Define templates
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are an expert product reviewer."),
    ("human", "List the main features of the product {product_name}."),
])

pros_template = ChatPromptTemplate.from_messages([
    ("system", "You are an expert product reviewer."),
    ("human", "Given these features: {features}, list the pros of these features."),
])

cons_template = ChatPromptTemplate.from_messages([
    ("system", "You are an expert product reviewer."),
    ("human", "Given these features: {features}, list the cons of these features."),
])

# Build branch chains
pros_branch_chain = pros_template | model | StrOutputParser()
cons_branch_chain = cons_template | model | StrOutputParser()

# Combine function
def combine_pros_cons(pros, cons):
    return f"Pros:\n{pros}\n\nCons:\n{cons}"

# Create full chain
chain = (
    prompt_template
    | model
    | StrOutputParser()
    | RunnableParallel(branches={"pros": pros_branch_chain, "cons": cons_branch_chain})
    | RunnableLambda(lambda x: combine_pros_cons(x["branches"]["pros"], x["branches"]["cons"]))
)

result = chain.invoke({"product_name": "MacBook Air"})
```

### 3.5. Branching Chains

```python
from langchain.schema.runnable import RunnableBranch

# Define runnable branches for handling feedback
branches = RunnableBranch(
    (
        lambda x: "positive" in x, 
        positive_feedback_template | model | StrOutputParser()
    ),
    (
        lambda x: "negative" in x,
        negative_feedback_template | model | StrOutputParser()
    ),
    (
        lambda x: "neutral" in x,
        neutral_feedback_template | model | StrOutputParser()
    ),
    escalate_feedback_template | model | StrOutputParser()  # Default branch
)

# Create classification chain
classification_chain = classification_template | model | StrOutputParser()

# Combine classification and response generation
chain = classification_chain | branches

result = chain.invoke({"feedback": default_review})
```

---

## 4. RAG (Retrieval-Augmented Generation)

### 4.1.a. RAG Basic - Create Vector Store

**Steps:**
1. Define directory and paths
2. Check if vector store exists
3. Load text content
4. Split documents
5. Create embeddings
6. Create and persist vector store

```python
import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# Step 1: Define paths
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "books", "odyssey.txt")
persistent_directory = os.path.join(current_dir, "db", "chroma_db")

# Step 2: Check if exists
if not os.path.exists(persistent_directory):
    
    # Step 3: Load documents
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    
    loader = TextLoader(file_path, encoding="utf-8")
    documents = loader.load()
    
    # Step 4: Split documents
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)
    
    # Step 5: Create embeddings
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # Step 6: Create vector store
    db = Chroma.from_documents(
        docs,
        embeddings,
        persist_directory=persistent_directory
    )
```

### 4.1.b. RAG Basic - Load and Query Existing Vector Store

```python
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embeddings
)

retriever = db.as_retriever(
    search_type="similarity_score_threshold", 
    search_kwargs={"k": 3, "score_threshold": 0.4}, 
)

query = "Who is Odysseus' wife?"
relevant_docs = retriever.invoke(query)

# Display results
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")
    if doc.metadata:
        print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")
```

### 4.2. Add Custom Metadata

```python
documents = []

for book_file in book_files:
    file_path = os.path.join(books_dir, book_file)
    loader = TextLoader(file_path)
    book_docs = loader.load()
    
    for doc in book_docs:
        # Add metadata to each document
        doc.metadata = {
            "source": book_file,
            # Add more fields: chapter, author, year, etc.
        }
        documents.append(doc)
```

### 4.3. Text Splitting Deep Dive

```python
from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
    TextSplitter,
    TokenTextSplitter,
)

# 1. Character-based Splitting
# Splits by specified number of characters
char_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
char_docs = char_splitter.split_documents(documents)

# 2. Sentence-based Splitting
# Ensures chunks end at sentence boundaries
sent_splitter = SentenceTransformersTokenTextSplitter(chunk_size=1000)
sent_docs = sent_splitter.split_documents(documents)

# 3. Token-based Splitting
# Splits by tokens (words/subwords)
token_splitter = TokenTextSplitter(chunk_overlap=0, chunk_size=512)
token_docs = token_splitter.split_documents(documents)

# 4. Recursive Character-based Splitting ⭐ (RECOMMENDED)
# Splits at natural boundaries while respecting character limits
rec_char_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, 
    chunk_overlap=100
)
rec_char_docs = rec_char_splitter.split_documents(documents)

# 5. Custom Splitting
class CustomTextSplitter(TextSplitter):
    def split_text(self, text):
        return text.split("\n\n")  # Split by paragraphs

custom_splitter = CustomTextSplitter()
custom_docs = custom_splitter.split_documents(documents)
```

### 4.4. Embedding Alternatives

```python
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

openai_embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

huggingface_embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)
```

### 4.5. Retriever Deep Dive

```python
# 1. Similarity Search
# Retrieves top k most similar documents
retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3},
)

# 2. Max Marginal Relevance (MMR)
# Balances relevance and diversity
# fetch_k: initial fetch count
# lambda_mult: diversity control (1=min diversity, 0=max diversity)
retriever = db.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3, "fetch_k": 20, "lambda_mult": 0.5},
)

# 3. Similarity Score Threshold
# Only retrieves documents above threshold
retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 3, "score_threshold": 0.1},
)
```

### 4.6. RAG One-Off Question

```python
from langchain_core.messages import SystemMessage, HumanMessage

relevant_docs = retriever.invoke(query)

combined_input = (
    "Here are some documents that might help answer the question: "
    + query
    + "\n\nRelevant Documents:\n"
    + "\n\n".join([doc.page_content for doc in relevant_docs])
    + "\n\nPlease provide an answer based only on the provided documents. "
    + "If the answer is not found in the documents, respond with 'I'm not sure'."
)

model = ChatOpenAI(model="gpt-4o")

messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content=combined_input),
]

result = model.invoke(messages)
```

### 4.7. RAG Conversational

```python
from langchain_core.prompts import MessagesPlaceholder
from langchain.chains import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# Setup
db = Chroma(persist_directory=persistent_directory)
retriever = db.as_retriever(...)
llm = ChatOpenAI(...)

# Contextualize question prompt
# Helps reformulate questions based on chat history
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, just "
    "reformulate it if needed and otherwise return it as is."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# Create history-aware retriever
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

# Answer question prompt
qa_system_prompt = (
    "You are an assistant for question-answering tasks. Use "
    "the following pieces of retrieved context to answer the "
    "question. If you don't know the answer, just say that you "
    "don't know. Use three sentences maximum and keep the answer "
    "concise."
    "\n\n"
    "{context}"
)

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", qa_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# Create question-answer chain
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

# Create retrieval chain
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# Chat function
def continual_chat():
    print("Start chatting with the AI! Type 'exit' to end the conversation.")
    chat_history = []
    
    while True:
        query = input("You: ")
        if query.lower() == "exit":
            break
        
        result = rag_chain.invoke({"input": query, "chat_history": chat_history})
        print(f"AI: {result['answer']}")
        
        chat_history.append(HumanMessage(content=query))
        chat_history.append(SystemMessage(content=result["answer"]))
```

**RAG Conversational Flow:**
```
User question
   ↓
LLM reformulates question (based on chat history)
   ↓
Retriever fetches top-k similar docs from Chroma
   ↓
LLM reads retrieved context + question
   ↓
Generates short, grounded answer
   ↓
Conversation history updated
```

**Key Mapping:**

| Stage | Input keys | Output keys | Description |
|-------|-----------|-------------|-------------|
| `history_aware_retriever` | `input`, `chat_history` | `context` (list of documents) | Reformulates the question, retrieves relevant docs |
| `question_answer_chain` | `input`, `chat_history`, `context` | `answer` | Builds prompt using `{context}` = docs' text, `{input}` = user query |
| `rag_chain.invoke()` | `input`, `chat_history` | `answer` | Wraps both above stages end-to-end |

### 4.8. RAG Web Scrape Basics

```python
from langchain_community.document_loaders import WebBaseLoader

urls = ["https://www.apple.com/"]
loader = WebBaseLoader(urls)
documents = loader.load()

# Continue with text-splitting, embedding, etc.
```

### 4.9. Firecrawl Web Scraping

```python
from langchain_community.document_loaders import FireCrawlLoader
import os

api_key = os.getenv("FIRECRAWL_API_KEY")

loader = FireCrawlLoader(
    api_key=api_key, 
    url="https://apple.com", 
    mode="scrape"
)
docs = loader.load()

# Convert metadata values to strings if they are lists
for doc in docs:
    for key, value in doc.metadata.items():
        if isinstance(value, list):
            doc.metadata[key] = ", ".join(map(str, value))

# Continue with text-splitting, embedding, etc.
```

---

## 5. Agents and Tools

### 5.1. Agent and Tools Basics

```python
from langchain_core.tools import Tool
from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor
import datetime

# Tool definition
def get_current_time(*args, **kwargs):
    """Returns the current time in H:MM AM/PM format."""
    now = datetime.datetime.now()
    return now.strftime("%I:%M %p")

tools = [
    Tool(
        name="Time",
        func=get_current_time,
        description="Useful for when you need to know the current time",
    ),
]

# Pull ReAct prompt (Reason and Action)
prompt = hub.pull("hwchase17/react")

llm = ChatOpenAI(model="gpt-4o")

# Create ReAct agent
agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt,
    stop_sequence=True,
)

# Create agent executor
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    verbose=True,
)

# Note: Use "input" key for agents
response = agent_executor.invoke({"input": "What time is it?"})
```

### 5.2. React Agent Chat

```python
from langchain.memory import ConversationBufferMemory
from langchain.agents import create_structured_chat_agent

tools = [
    Tool(
        name="Time",
        func=get_current_time,
        description="Useful for when you need to know the current time.",
    ),
    Tool(
        name="Wikipedia",
        func=search_wikipedia,
        description="Useful for when you need to know information about a topic.",
    ),
]

prompt = hub.pull("hwchase17/structured-chat-agent")
llm = ChatOpenAI(model="gpt-4o")

# Add conversation memory
memory = ConversationBufferMemory(
    memory_key="chat_history", 
    return_messages=True
)

agent = create_structured_chat_agent(llm=llm, tools=tools, prompt=prompt)

agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    verbose=True,
    memory=memory,
    handle_parsing_errors=True,
)

# Initial system message
initial_message = "You are an AI assistant that can provide helpful answers using available tools.\nIf you are unable to answer, you can use the following tools: Time and Wikipedia."
memory.chat_memory.add_message(SystemMessage(content=initial_message))

while True:
    user_input = input("User: ")
    if user_input.lower() == "exit":
        break
    
    memory.chat_memory.add_message(HumanMessage(content=user_input))
    
    response = agent_executor.invoke({"input": user_input})
    print("Bot:", response["output"])
    
    memory.chat_memory.add_message(AIMessage(content=response["output"]))
```

### 5.3. React Agent Docstore

```python
# Uses document store retriever to answer questions

rag_chain = create_retrieval_chain(
    history_aware_retriever, question_answer_chain
)

react_docstore_prompt = hub.pull("hwchase17/react")

tools = [
    Tool(
        name="Answer Question",
        func=lambda input, **kwargs: rag_chain.invoke(
            {"input": input, "chat_history": kwargs.get("chat_history", [])}
        ), 
        description="useful for when you need to answer questions about the context",
    )
]

agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=react_docstore_prompt,
)

agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent, 
    tools=tools, 
    handle_parsing_errors=True, 
    verbose=True,
)

chat_history = []
while True:
    query = input("You: ")
    if query.lower() == "exit":
        break
    
    response = agent_executor.invoke(
        {"input": query, "chat_history": chat_history}
    )
    print(f"AI: {response['output']}")
    
    chat_history.append(HumanMessage(content=query))
    chat_history.append(AIMessage(content=response["output"]))
```

### 5.4. Tool Constructor

```python
from langchain.pydantic_v1 import BaseModel, Field
from langchain_core.tools import StructuredTool

# Pydantic model for tool arguments
class ConcatenateStringsArgs(BaseModel):
    a: str = Field(description="First string")
    b: str = Field(description="Second string")

tools = [
    # Simple Tool (single input)
    Tool(
        name="GreetUser",
        func=greet_user,
        description="Greets the user by name.",
    ),
    
    # StructuredTool (multiple inputs with schema)
    StructuredTool.from_function(
        func=concatenate_strings,
        name="ConcatenateStrings",
        description="Concatenates two strings.",
        args_schema=ConcatenateStringsArgs,
    ),
]

llm = ChatOpenAI(model="gpt-4o")
prompt = hub.pull("hwchase17/openai-tools-agent")

from langchain.agents import create_tool_calling_agent

agent = create_tool_calling_agent(
    llm=llm,
    tools=tools,
    prompt=prompt,
)

agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
)

response = agent_executor.invoke({"input": "Greet Alice"})
```

### 5.5. Tool Decorator

```python
from langchain_core.tools import tool
from langchain.pydantic_v1 import BaseModel, Field

# Simple tool
@tool()
def greet_user(name: str) -> str:
    """Greets the user by name."""
    return f"Hello, {name}!"

# Tool with args_schema
class ReverseStringArgs(BaseModel):
    text: str = Field(description="Text to be reversed")

@tool(args_schema=ReverseStringArgs)
def reverse_string(text: str) -> str:
    """Reverses the given string."""
    return text[::-1]

tools = [
    greet_user,
    reverse_string,
]

llm = ChatOpenAI(model="gpt-4o")
prompt = hub.pull("hwchase17/openai-tools-agent")

agent = create_tool_calling_agent(
    llm=llm,
    tools=tools,
    prompt=prompt,
)
```

### 5.6. Custom Tools - Subclassing BaseTool

```python
from langchain.tools import BaseTool
from langchain.pydantic_v1 import BaseModel, Field
from typing import Type
import os

class SimpleSearchInput(BaseModel):
    query: str = Field(description="should be a search query")

class SimpleSearchTool(BaseTool):
    name = "simple_search"
    description = "useful for when you need to answer questions about current events"
    args_schema: Type[BaseModel] = SimpleSearchInput
    
    def _run(self, query: str) -> str:
        """Use the tool."""
        from tavily import TavilyClient
        
        api_key = os.getenv("TAVILY_API_KEY")
        client = TavilyClient(api_key=api_key)
        results = client.search(query=query)
        return f"Search results for: {query}\n\n\n{results}\n"


class MultiplyNumbersArgs(BaseModel):
    x: float = Field(description="First number to multiply")
    y: float = Field(description="Second number to multiply")

class MultiplyNumbersTool(BaseTool):
    name = "multiply_numbers"
    description = "useful for multiplying two numbers"
    args_schema: Type[BaseModel] = MultiplyNumbersArgs
    
    def _run(self, x: float, y: float) -> str:
        """Use the tool."""
        result = x * y
        return f"The product of {x} and {y} is {result}"

tools = [
    SimpleSearchTool(),
    MultiplyNumbersTool(),
]
```

---

## Summary

This document covers:
- **Chat Models**: Basic usage, conversations, alternatives, and Firebase history
- **Prompt Templates**: Creating dynamic prompts with variables
- **Chains**: Basic, parallel, and branching chains using LCEL
- **RAG**: Vector stores, retrievers, conversational RAG, and web scraping
- **Agents**: ReAct agents, tools, and custom tool creation

For more details, refer to the [LangChain Documentation](https://python.langchain.com/).
