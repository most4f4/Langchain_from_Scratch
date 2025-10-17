# LangChain from Scratch

A comprehensive collection of LangChain examples and implementations, covering everything from basic chat models to advanced RAG (Retrieval-Augmented Generation) systems and agent architectures.

## 📋 Overview

This repository contains practical implementations and examples for learning LangChain, organized into progressive modules that build from foundational concepts to advanced patterns.

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or 3.11 (3.12 not supported)
- Poetry for dependency management

### Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd LANGCHAIN_FROM_SCRATCH
```

2. Install dependencies using Poetry:
```bash
poetry install
```

3. Set up environment variables:
```bash
cp .env.example .env
```

4. Add your API keys to `.env`:
```
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
GOOGLE_API_KEY=your_google_key
```

## 📚 Repository Structure

### 1. Chat Models (`1_chat_models/`)
Introduction to different chat model implementations:
- `1_chat_model_basic.py` - Basic chat model usage
- `2_chat_model_basic_conversation.py` - Simple conversations
- `3_chat_model_alternatives.py` - Alternative model providers
- `4_chat_model_conversation_with_user.py` - Interactive conversations
- `5_chat_model_save_message_history_firebase.py` - Persistent chat history

### 2. Prompt Templates (`2_prompt_templates/`)
Learn to structure and manage prompts effectively:
- `1_prompt_template_basic.py` - Basic prompt templates
- `2_prompt_template_with_chat_model.py` - Integration with chat models

### 3. Chains (`3_chains/`)
Build complex workflows by chaining components:
- `1_chains_basics.py` - Introduction to chains
- `2_chains_under_the_hood.py` - Understanding chain mechanics
- `3_chains_extended.py` - Advanced chain patterns
- `4_chains_parallel.py` - Parallel execution
- `5_chains_parallel_cleaned.py` - Optimized parallel chains
- `6_chains_branching.py` - Conditional branching logic

### 4. RAG (Retrieval-Augmented Generation) (`4_rag/`)
Implement document retrieval and question-answering systems:

#### Books Subdirectory
Sample documents for RAG examples

#### RAG Components
- `1a_rag_basics.py` - RAG fundamentals
- `1b_rag_basics.py` - Additional RAG patterns
- `2a_rag_basics_metadata.py` - Metadata handling
- `2b_rag_basics_metadata.py` - Advanced metadata usage
- `3_rag_text_splitting_deep_dive.py` - Document chunking strategies
- `4_rag_embedding_deep_dive.py` - Embedding techniques
- `5_rag_retriever_deep_dive.py` - Retrieval methods
- `6_rag_one_off_question.py` - Single-query RAG
- `7_rag_conversational.py` - Conversational RAG
- `8_rag_web_scrape_basic.py` - Web scraping integration
- `8_rag_web_scrape_firecrawl.py` - Advanced web scraping
- `rag-ascii-diagram.md` - Visual architecture reference

### 5. Agents and Tools (`5_agent_and_tools/`)
Create autonomous agents with tool usage:
- `1_agent_and_tools_basics.py` - Agent fundamentals
- `agent_deep_dive/` - Deep dive into agent architectures
- `tools_deep_dive/` - Custom tool development

## 🔧 Key Dependencies

- **langchain** - Core LangChain library
- **langchain-openai** - OpenAI integrations
- **langchain-anthropic** - Anthropic Claude integrations
- **langchain-google-genai** - Google AI integrations
- **chromadb** - Vector database for embeddings
- **sentence-transformers** - Embedding models
- **tiktoken** - Token counting
- **firecrawl-py** - Advanced web scraping
- **tavily-python** - Web search capabilities
- **wikipedia** - Wikipedia integration

## 💡 Usage Examples

### Basic Chat Model
```python
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4")
response = model.invoke("Hello, world!")
print(response.content)
```

### Simple RAG System
```python
# Load documents, create embeddings, and query
# See 1a_rag_basics.py for complete example
```

### Agent with Tools
```python
# Create an agent with custom tools
# See 1_agent_and_tools_basics.py for complete example
```

## 📖 Learning Path

1. **Start with Chat Models** - Understand basic LLM interactions
2. **Master Prompt Templates** - Learn to structure inputs effectively
3. **Explore Chains** - Build complex workflows
4. **Implement RAG** - Add document retrieval capabilities
5. **Build Agents** - Create autonomous systems with tools

## 🤝 Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## 📄 License

This project is open source and available under the MIT License.

## 🔗 Resources

- [LangChain Documentation](https://python.langchain.com/)
- [LangChain Hub](https://smith.langchain.com/hub)
- [OpenAI API](https://platform.openai.com/)
- [Anthropic API](https://www.anthropic.com/)

## 📝 Notes

- Each script is self-contained and can be run independently
- Make sure to set up your API keys before running examples
- Some examples require Firebase setup for message history persistence
- The `books/` directory contains sample texts for RAG examples