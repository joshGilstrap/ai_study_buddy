import os
import apikeys as ak
from langchain.llms import OpenAI
from langchain.agents import Tool, initialize_agent, AgentType
from langchain.utilities import SerpAPIWrapper, WikipediaAPIWrapper
from langchain.tools import YouTubeSearchTool, tool
from langchain.memory import ConversationBufferMemory
from langchain.tools.python.tool import PythonREPLTool
from langchain.python import PythonREPL

# API keys
os.environ['OPENAI_API_KEY'] = ak.openai_key
os.environ['SERPAPI_API_KEY'] = ak.serpai_api_key

# Model creation
llm = OpenAI(temperature=0.9)

# Tools for agent, specifically google search API and math capabilities
search = SerpAPIWrapper()
wiki = WikipediaAPIWrapper()
yt = YouTubeSearchTool()
python_tool = PythonREPLTool()
tools = [
    # Use SerpAPI to use Google results
    Tool('Current Search', search.run, 
         "useful for when you need to answer questions requiring a lookup"),
    # Good way to find wordy explanations
    Tool('Wikipedia', wiki.run,
         "useful when you need to provide summarized information on a subject"),
    # Returns youtube URLs
    Tool('YouTube Search', yt,
         "useful for finding videos"),
    # Specifically for asking about small implementations (merge sort, binary search, etc.)
    Tool('Python', python_tool,
         "use for reading and writing python code"),
]

@tool
def read_the_code(code):
    '''Open source code and return it'''
    with open('C:/Users/jgils/programming/pathfinding_vis/pathfinding.py', encoding='UTF-8') as f:
        code = f.read()
    return code

# Simple conversation buffer for short-term memory
memory = ConversationBufferMemory(memory_key="chat_history")

# Plan-and-Execute Model: Use this for robust answers and no conversation
agent = initialize_agent(tools, llm, AgentType.ZERO_SHOT_REACT_DESCRIPTION, memory=memory, verbose=True)

# Conversation Modal: Use this for a conversational bot
# agent = initialize_agent(tools, llm, AgentType.CONVERSATIONAL_REACT_DESCRIPTION, memory=memory, verbose=True)


# Conversational loop
user_input = ''
while user_input != 'quit':
    user_input = input('\nINTERACT: ')
    if user_input == 'quit': continue
    if user_input == 'python run code':
        with open('C:/Users/jgils/programming/pathfinding_vis/pathfinding.py', encoding='UTF-8') as f:
            code = f.read()
        print(agent.run(read_the_code(code)))
    else:
        print(agent.run(user_input))
    
    
