import os
import apikeys as ak
from langchain.llms import OpenAI
from langchain.agents import Tool
from langchain.agents import AgentType
from langchain.memory import ConversationBufferMemory
from langchain.utilities import SerpAPIWrapper
from langchain.agents import initialize_agent

# API keys
os.environ['OPENAI_API_KEY'] = ak.openai_key
os.environ['SERPAPI_API_KEY'] = ak.serpai_api_key

# Tools for agent
search = SerpAPIWrapper()
tools = [
    Tool('Current Search', search.run, 
         "useful for when you need to answer questions requiring a lookup")
]

# Simple conversation buffer for recollections
memory = ConversationBufferMemory(memory_key="chat_history")

# Model and agent creation
llm = OpenAI(temperature=0.7)
agent_chain = initialize_agent(tools, llm,
                               AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
                               memory=memory)

# Cheap conversational loop
user_input = ''
while user_input != 'quit':
    user_input = input('Interact: ')
    if user_input == 'quit': continue
    print(agent_chain.run(user_input))
    
    
