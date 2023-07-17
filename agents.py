import os
from datetime import date

import langchain
from dotenv import load_dotenv
from langchain import tools
from langchain.agents import load_tools, initialize_agent, AgentType
from langchain.agents.agent_toolkits import create_python_agent
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import CSVLoader
from langchain.tools import PythonREPLTool, tool
from langchain.vectorstores import DocArrayInMemorySearch
from IPython.display import display, Markdown

import openai
import warnings

warnings.filterwarnings('ignore')
load_dotenv()
openai.api_key = os.environ["OPENAI_API_KEY"]


def agents():
    llm = ChatOpenAI(temperature=0)
    tools = load_tools(["llm-math", "wikipedia"], llm=llm)
    agent = initialize_agent(tools=tools,
                             llm=llm,
                             agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
                             handle_parsing_errors=True,
                             verbos=True)
    print(agent(
        {"input": "Tom  M. Mitchell is an American computer scientist what book did he write?", "chat_history": []}))
    aa = ""


def python_agent():
    langchain.debug = True
    llm = ChatOpenAI(temperature=0)
    agent = create_python_agent(llm=llm,
                                tool=PythonREPLTool(),
                                verbos=True)
    customer_list = [
        ["Liam", "Mokarian"],
        ["Parisa", "Moslehi"],
        ["Maysam", "Mokarian"]
    ]
    resp = agent.run(f"""
    Sort these names by last name then first name and print the output:{customer_list}""")
    print(resp)
    aa = ""


def custom_agent():
    langchain.debug = True
    llm = ChatOpenAI(temperature=0)
    tools = load_tools(["llm-math", "wikipedia"], llm=llm)
    agent = initialize_agent(tools= tools + [time],
                             llm=llm,
                             agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
                             handle_parsing_errors=True,
                             verbos=True)
    print(agent(
        {"input": "Whats the date today?", "chat_history": []}))
    aa = ""


@tool
def time(text: str) -> str:
    """
    return todays date, use this for any
    questions related to knowing todays date.
    The input should always be an empty string,
    and this functions will always return todays date - any
    date mathmatics should occur outside this function
    """
    from datetime import date
    return str(date.today())


if __name__ == "__main__":
    custom_agent()
