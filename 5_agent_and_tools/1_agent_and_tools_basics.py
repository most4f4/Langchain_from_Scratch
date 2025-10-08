from dotenv import load_dotenv
from langchain import hub # hub is used to pull prompt templates from the LangChain Hub
from langchain.agents import (
    AgentExecutor, # AgentExecutor is used to create an executor that runs the agent with the provided tools
    create_react_agent, # create_react_agent is a function to create a ReAct agent
)
from langchain_core.tools import Tool # Tool is used to define tools that the agent can use
from langchain_openai import ChatOpenAI

# ---------------------------------------------------------------
# This script demonstrates how to create a simple ReAct agent
# that can use a tool to get the current time.
# ---------------------------------------------------------------

# Load environment variables from .env file
load_dotenv()


# Define a very simple tool function that returns the current time
def get_current_time(*args, **kwargs):
    """Returns the current time in H:MM AM/PM format."""
    import datetime  # Import datetime module to get current time

    now = datetime.datetime.now()  # Get current time
    return now.strftime("%I:%M %p")  # Format time in H:MM AM/PM format


# List of tools available to the agent
tools = [
    Tool(
        name="Time",  # Name of the tool
        func=get_current_time,  # Function that the tool will execute
        # Description of the tool
        description="Useful for when you need to know the current time",
    ),
]

# Pull the prompt template from the hub
# ReAct = Reason and Action
# https://smith.langchain.com/hub/hwchase17/react
prompt = hub.pull("hwchase17/react")

# Initialize a ChatOpenAI model
llm = ChatOpenAI(
    model="gpt-4o", temperature=0
)

# Create the ReAct agent using the create_react_agent function
agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt,
    stop_sequence=True, # Stop sequence to indicate when the agent should stop generating
)

# Create an agent executor from the agent and tools
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent, # type: ignore
    tools=tools,
    verbose=True, # Verbose mode to print out the agent's thought process
)

# Run the agent with a test query
response = agent_executor.invoke({"input": "What time is it?"})

# Print the response from the agent
print("response:", response)
