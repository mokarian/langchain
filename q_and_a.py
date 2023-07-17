import os

from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import CSVLoader
from langchain.vectorstores import DocArrayInMemorySearch
from IPython.display import display, Markdown

import openai
import warnings

warnings.filterwarnings('ignore')
load_dotenv()
openai.api_key = os.environ["OPENAI_API_KEY"]


def q_and_a():
    file = 'OutdoorClothingCatalog_1000.csv'
    loader = CSVLoader(file_path=file)
    from langchain.indexes import VectorstoreIndexCreator
    index = VectorstoreIndexCreator(
        vectorstore_cls=DocArrayInMemorySearch
    ).from_loaders([loader])
    query = "Please list all your shirts with sun protection \
    in a table in markdown and summarize each one."
    response = index.query(query)
    display(Markdown(response))


def q_and_a_complex():
    file = 'OutdoorClothingCatalog_1000.csv'
    loader = CSVLoader(file_path=file)
    docs = loader.load()
    from langchain.embeddings import OpenAIEmbeddings
    embeddings = OpenAIEmbeddings()
    embed = embeddings.embed_query("Hi my name is Harrison")
    db = DocArrayInMemorySearch.from_documents(docs, embeddings)
    docs = db.similarity_search("Please suggest a shirt with sunblocking")
    retriever = db.as_retriever()
    llm = ChatOpenAI(temperature=0.0)
    qdocs = "".join([docs[i].page_content for i in range(len(docs))])
    response = llm.call_as_llm(f"{qdocs} Question: Please list all your \
    shirts with sun protection in a table in markdown and summarize each one.")
    display(Markdown(response))


def q_and_a_chain():
    file = 'OutdoorClothingCatalog_1000.csv'
    loader = CSVLoader(file_path=file)
    docs = loader.load()
    from langchain.embeddings import OpenAIEmbeddings
    embeddings = OpenAIEmbeddings()
    embed = embeddings.embed_query("Hi my name is Harrison")
    db = DocArrayInMemorySearch.from_documents(docs, embeddings)
    retriever = db.as_retriever()
    llm = ChatOpenAI(temperature=0.0)

    qa_stuff = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        verbose=True
    )
    query = "Please list all your shirts with sun protection in a table \
    in markdown and summarize each one."
    response = qa_stuff.run(query)
    display(Markdown(response))

if __name__ == "__main__":
    q_and_a_chain()
