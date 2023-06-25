import os
import apikeys as ak
import audio_components as ac
from langchain.llms import OpenAI
from langchain.agents import Tool, initialize_agent, AgentType, load_tools
from langchain.utilities import SerpAPIWrapper, WikipediaAPIWrapper, GoogleSearchAPIWrapper
from langchain.tools import YouTubeSearchTool
from langchain.memory import ConversationBufferMemory
from langchain.tools.python.tool import PythonREPLTool

# API keys
os.environ['OPENAI_API_KEY'] = ak.openai_key
os.environ['SERPAPI_API_KEY'] = ak.serpai_api_key
os.environ['WOLFRAM_ALPHA_APPID'] = ak.wolfram_alpha_key
os.environ['GOOGLE_API_KEY'] = ak.google_api_key
os.environ['GOOGLE_CSE_ID'] = ak.google_cse_id

# Model creation
llm = OpenAI(temperature=0.9)

# Tools for agents, 
search = SerpAPIWrapper()
wiki = WikipediaAPIWrapper()
yt = YouTubeSearchTool()
python_tool = PythonREPLTool()
google_search = GoogleSearchAPIWrapper()
# Conversational bot just needs internet access
convo_tools = [
    # Use SerpAPI to use Google results
    Tool('Current Search', google_search.run, 
         "useful for when you need to answer questions requiring a lookup"),
]
# Research bot needs access to all kinds of information
plan_tools = [
    # Use SerpAPI to use Google results
    Tool('Current Search', google_search.run, 
         "useful for when you need to answer questions requiring a Google lookup"),
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
math_tools = load_tools(["llm-math", "wolfram-alpha"], llm=llm)
math_tools.append(Tool('Current Search', google_search.run,
                    'use for searching Google'))
math_tools.append(Tool('Python', python_tool,
                       "use for reading and writing python code"))

# Simple conversation buffer for short-term memory
memory = ConversationBufferMemory(memory_key="chat_history")

# Plan-and-Execute Model: Use this for research like programming and math
plan_agent = initialize_agent(plan_tools, llm, AgentType.ZERO_SHOT_REACT_DESCRIPTION, memory=memory, verbose=True, max_execution_time=5)

# Conversation Modal: Use this for a conversational bot
convo_agent = initialize_agent(convo_tools, llm, AgentType.CONVERSATIONAL_REACT_DESCRIPTION, memory=memory, verbose=True, max_execution_time=5)

# Math Model: Use this for asking questions
math_agent = initialize_agent(math_tools, llm, AgentType.CONVERSATIONAL_REACT_DESCRIPTION, memory=memory, verbose=True, max_execution_time=5)


# Main loop
user_input = ''
while user_input != "exit" or user_input != 'quit':
    print("Choose your experience:")
    print("-----------------------")
    print("(1) Chat AI") # convo_agent
    print("(2) Research AI") # plan_agent
    print("(3) Math AI") # math_agent
    print("-----------------------")
    print("'exit' or 'quit' to exit\n")
    user_input = input('Choice: ')
    agent = convo_agent
    prompt = "INTERACT ('back' to menu): "
    print('\n')
    if user_input == '1':
        print("Use this AI to conversate with. Ask questions and make statements in the interest of connection!")
    elif user_input == '2':
        print("Use this AI for QUESTIONS ONLY. It doesn't like to connect with people...")
        print("It'll use things like Google, Wikipedia and YouTube to answer your questions succinctly.")
        agent = plan_agent
    elif user_input == '3':
        print("Use this AI as a conversational math buddy. It uses all built-in math functions, Google and WolframAlpha for math decisions and discussion.")
        agent = math_agent
    elif user_input == 'exit':
        exit()
    # Inner, agent specific conversation loop
    while user_input != 'back':
        user_input = input('\n'+prompt)
        print('\n')
        if user_input == 'back': continue
        if user_input == 'exit' or user_input == 'quit':
            exit()
        if user_input == 'vocal':   
            user_input = ac.listen_for_voice()
        print("\n---------AI PROCESS BEGIN---------")
        output = agent.run(user_input)
        print("\n---------AI PROCESS END---------\n")
        print(user_input+'\n')
        print(output)
        # Say output
        ac.speak_your_mind(output)
    
