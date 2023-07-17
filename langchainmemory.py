import openai
import os
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv


import warnings

warnings.filterwarnings('ignore')


load_dotenv()

openai.api_key = os.environ["OPENAI_API_KEY"]


def memory_example():
    llm = ChatOpenAI(temperature=0.0, model="gpt-3.5-turbo-0301")
    memory = ConversationBufferMemory()
    conversation = ConversationChain(
        llm=llm,
        memory=memory,
        verbose=True
    )
    p = conversation.predict(input="Hi, my name is Maysam")
    p = conversation.predict(input="What is 1 +1 ")
    p = conversation.predict(input="What is my name?")


def memory_window():
    from langchain.memory import ConversationBufferWindowMemory
    memory = ConversationBufferWindowMemory(k=1)
    llm = ChatOpenAI(temperature=0.0)
    conversation = ConversationChain(
        llm=llm,
        memory=memory,
        verbose=True
    )
    p = conversation.predict(input="Hi, my name is Maysam")
    p = conversation.predict(input="What is 1 +1 ")
    p = conversation.predict(input="What is my name ")


def memory_window_token():
    from langchain.memory import ConversationTokenBufferMemory

    llm = ChatOpenAI(temperature=0.0)
    memory = ConversationTokenBufferMemory(llm=llm, max_token_limit=50)
    conversation = ConversationChain(
        llm=llm,
        memory=memory,
        verbose=True
    )
    p = conversation.predict(input="Hi, my name is Maysam Mokarian")
    p = conversation.predict(input="What is 1 +1 ")
    p = conversation.predict(input="What is my first and last name ")
    p = conversation.predict(input="What is my job? ")


def memory_summary():
    from langchain.memory import ConversationSummaryMemory

    scheduel = """
    A software engineer's day typically starts in the early morning, around 7 or 8 am. They begin by checking emails and 
    notifications to catch up with any overnight developments or urgent requests. After that, they spend some time planning 
    their tasks for the day, prioritizing and organizing their work. They might schedule meetings or consultations with
     team members or stakeholders to discuss ongoing projects or clarify requirements. From late morning until afternoon, 
     they dive into coding and debugging, developing software solutions using different programming languages and tools. 
     They might collaborate with other engineers to troubleshoot issues or review each other's code. Around mid-day, 
     a software engineer generally takes a break for lunch, where they socialize with colleagues or take a 
     walk to refresh their mind. In the afternoon, they continue coding, researching new technologies, 
     or conducting tests to ensure the quality and functionality of their software. 
     They may also attend training sessions or webinars to enhance their knowledge and skills. 
     As the workday comes to an end, they wrap up their tasks, document their progress, 
     and leave detailed notes for themselves or teammates for the next day. Before logging off, 
     they spend some time organizing their workspace and making sure they have all the necessary 
     resources available for the next day. Finally, they bid farewell to their team members and head home, 
     ready to relax and rejuvenate for the next day's challenges.
    """

    llm = ChatOpenAI(temperature=0.0)
    memory = ConversationSummaryMemory(llm=llm, max_token_limit=200)
    conversation = ConversationChain(
        llm=llm,
        memory=memory,
        verbose=True
    )
    memory.save_context({"input": "what is your schedule today?"}, {"output:": scheduel})
    p = conversation.predict(input="Hi, my name is Maysam Mokarian")
    p = conversation.predict(input="What is 1 +1 ")
    p = conversation.predict(input="What is my first and last name ")
    p = conversation.predict(input="What is my job? ")
    what = ""


if __name__ == "__main__":
    memory_summary()
