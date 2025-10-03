from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnableSequence
from langchain_openai import ChatOpenAI

# Load environment variables from .env
load_dotenv()

# Create a ChatOpenAI model
model = ChatOpenAI(model="gpt-4") # already a Runnable

# Define prompt templates
prompt_template = ChatPromptTemplate.from_messages( # already a Runnable
    [
        ("system", "You are a comedian who tells jokes about {topic}."),
        ("human", "Tell me {joke_count} jokes."),
    ]
)

# Note: This is just to show how to build the chain under the hood
# In practice, we would use the higher-level abstractions provided by LangChain.
# ChatPromptTemplate and ChatOpenAI are already Runnable, and we don't need to wrap them in RunnableLambda.
# So this format_prompt = RunnableLambda(lambda x: prompt_template.format_prompt(**x))
# could be replaced with just: format_prompt = prompt_template
# and this step: model.invoke(x.to_messages())
# is equivalent to: model.invoke(x)
# That means your whole chain can become:
# chain = RunnableSequence(
#     first=prompt_template,  # no need for format_prompt(**x)
#     middle=[model],         # no need for .to_messages()
#     last=RunnableLambda(lambda x: x.content)
# )


# Create individual runnables (steps in the chain)
format_prompt = RunnableLambda(lambda x: prompt_template.format_prompt(**x))
invoke_model = RunnableLambda(lambda x: model.invoke(x.to_messages())) # type: ignore
parse_output = RunnableLambda(lambda x: x.content) # type: ignore


# Create the RunnableSequence (equivalent to the LCEL chain)
chain = RunnableSequence(first=format_prompt, middle=[invoke_model], last=parse_output)

# Run the chain
response = chain.invoke({"topic": "doctors", "joke_count": 2})

# Output
print(response)

# Note on RunnableSequence
# RunnableSequence allows us to chain multiple runnables together.
# It takes three arguments:
# - first: the first runnable to execute (format_prompt)
# - middle: a list of runnables to execute in order (invoke_model)
# - last: the last runnable to execute (parse_output)
# The output of each runnable is automatically passed as input to the next runnable.
# This is similar to the LangChain Expression Language (LCEL) syntax:
# chain = prompt_template | model | StrOutputParser()

# Note on RunnableLambda(lambda x: prompt_template.format_prompt(**x))
# Here x is a dictionary {"topic": "doctors", "joke_count": 2}
# We use **x to unpack the dictionary and pass the values to format_prompt:
#     format_prompt(topic="doctors", joke_count=2)
# format_prompt returns a ChatPromptValue object, which is then passed to the next step.
# ChatPromptValue here would be something like:
# ChatPromptValue(messages=[
#     SystemMessage(content="You are a comedian who tells jokes about doctors."),
#     HumanMessage(content="Tell me 2 jokes.")
# ])

# Note on RunnableLambda(lambda x: model.invoke(x.to_messages()))
# Here x is the ChatPromptValue object returned by the previous step.
# We convert it to a list of messages using x.to_messages() and pass it to model.invoke().
# x.to_messages() returns a list of messages like:
# [
#     SystemMessage(content="You are a comedian who tells jokes about doctors."),
#     HumanMessage(content="Tell me 2 jokes.")
# ]


