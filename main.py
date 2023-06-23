import os
import apikeys as ak
from langchain.llms import OpenAI
from langchain.agents import Tool, initialize_agent, AgentType, load_tools
from langchain.utilities import SerpAPIWrapper, WikipediaAPIWrapper
from langchain.tools import YouTubeSearchTool
from langchain.memory import ConversationBufferMemory
from langchain.tools.python.tool import PythonREPLTool

# API keys
os.environ['OPENAI_API_KEY'] = ak.openai_key
os.environ['SERPAPI_API_KEY'] = ak.serpai_api_key
os.environ['WOLFRAM_ALPHA_APPID'] = ak.wolfram_alpha_key

# Model creation
llm = OpenAI(temperature=0.9)

# Tools for agent, specifically google search API and math capabilities
search = SerpAPIWrapper()
wiki = WikipediaAPIWrapper()
yt = YouTubeSearchTool()
python_tool = PythonREPLTool()
# Conversational bot just needs internet access
convo_tools = [
    # Use SerpAPI to use Google results
    Tool('Current Search', search.run, 
         "useful for when you need to answer questions requiring a lookup"),
]
# Research bot needs access to all kinds of information
plan_tools = [
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
# Math bot needs all the math knowledge
math_tools = load_tools(["serpapi", "llm-math", "wolfram-alpha"], llm=llm)

# Simple conversation buffer for short-term memory
memory = ConversationBufferMemory(memory_key="chat_history")

# Plan-and-Execute Model: Use this for research like programming and math
plan_agent = initialize_agent(plan_tools, llm, AgentType.ZERO_SHOT_REACT_DESCRIPTION, memory=memory, verbose=True)

# Conversation Modal: Use this for a conversational bot
convo_agent = initialize_agent(convo_tools, llm, AgentType.CONVERSATIONAL_REACT_DESCRIPTION, memory=memory, verbose=True)

# Search Model: Use this for asking questions
math_agent = initialize_agent(math_tools, llm, AgentType.CONVERSATIONAL_REACT_DESCRIPTION, memory=memory, verbose=True)


# Conversational loop
user_input = ''
print("Would you like to:")
print("(1) Chat") # convo_agent
print("(2) Research") # plan_agent
print("(3) Math") # math_agent
user_input = input('\nInput choice: ')
agent = convo_agent
prompt = "INTERACT ('quit' to exit): "
if user_input == '1':
    print("Use this AI to conversate with. Ask questions and make statements in the interest of connection!")
elif user_input == '2':
    print("Use this AI for QUESTIONS ONLY. It doesn't like to connect with people...")
    print("It'll use things like Google, Wikipedia and YouTube to answer your questions succinctly.")
    agent = plan_agent
elif user_input == '3':
    print("Use this AI as a conversational math buddy. It uses all built-in math functions, Google and WolframAlpha for math decisions.")
    agent = math_agent
while user_input != 'quit':
    user_input = input('\n'+prompt)
    if user_input == 'quit': continue
    print("---------AI PROCESS BEGIN---------\b")
    output = agent.run(user_input)
    print("---------AI PROCESS END---------\n")
    print(output)
    
    
