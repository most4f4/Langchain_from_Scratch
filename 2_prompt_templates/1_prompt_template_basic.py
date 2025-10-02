# Prompt Template Docs:
#   https://python.langchain.com/v0.2/docs/concepts/#prompt-templateshttps://python.langchain.com/v0.2/docs/concepts/#prompt-templates

from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage

# -------------------------------
# Uncomment examples below to run them
# -------------------------------

# # # Example 1: Create a ChatPromptTemplate using a template string
# template = "Tell me a joke about {topic}."
# # ChatPromptTemplate converts a template string into a prompt template that can be used to generate prompts with variable substitution.
# prompt_template = ChatPromptTemplate.from_template(template) 

# print("-----Prompt from Template-----")
# prompt = prompt_template.invoke({"topic": "cats"})
# print(prompt)
# # We don't want to call model.invoke(prompt) here because we are focusing on prompt templates in this file.



# # Example 2: Prompt with Multiple Placeholders
# template_multiple = """You are a helpful assistant.
# Human: Tell me a {adjective} story about a {animal}.
# Assistant:"""
# prompt_multiple = ChatPromptTemplate.from_template(template_multiple)
# prompt = prompt_multiple.invoke({"adjective": "funny", "animal": "panda"})
# print("\n----- Prompt with Multiple Placeholders -----\n")
# print(prompt)


# Example 3: Prompt with System and Human Messages (Using Tuples)
messages = [
    ("system", "You are a comedian who tells jokes about {topic}."),
    ("human", "Tell me {joke_count} jokes."),
]
prompt_template = ChatPromptTemplate.from_messages(messages)

# Note: we used from_messages() method which allows us to define different types of messages (system, human, AI) in a structured way.
# from_template() method is simpler and only supports a single template string.

prompt = prompt_template.invoke({"topic": "lawyers", "joke_count": 3})
print("\n----- Prompt with System and Human Messages (Tuple) -----\n")
print(prompt)

# # Extra Information about Example 3.
# # This does NOT work:
# messages = [
#     ("system", "You are a comedian who tells jokes about {topic}."),
#     HumanMessage(content="Tell me {joke_count} jokes."),
# ]
# prompt_template = ChatPromptTemplate.from_messages(messages)
# prompt = prompt_template.invoke({"topic": "lawyers", "joke_count": 3})
# print("\n----- Prompt with System and Human Messages (Tuple) -----\n")
# print(prompt)

# # This does work:
# messages = [
#     ("system", "You are a comedian who tells jokes about {topic}."),
#     HumanMessage(content="Tell me 3 jokes."),
# ]
# prompt_template = ChatPromptTemplate.from_messages(messages)
# prompt = prompt_template.invoke({"topic": "lawyers"})
# print("\n----- Prompt with System and Human Messages (Tuple) -----\n")
# print(prompt)
