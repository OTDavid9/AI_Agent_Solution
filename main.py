
import os
import bs4
# import BeautifulSoupTransformer

from langchain import hub
# from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
# from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_community.document_transformers import BeautifulSoupTransformer
from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.vectorstores import Chroma
# import Chroma
# from langchain_community.vectorstores import Pinecone
# from langchain_community.embeddings import OpenAIEmbeddings


from langchain import PromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("API_KEY")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")



urls = ["https://purpleconnect.wemabank.com/support/solutions/articles/67000511710-what-is-alat-", 
        "https://purpleconnect.wemabank.com/support/solutions/articles/67000511711-who-can-use-alat-",
        "https://purpleconnect.wemabank.com/support/solutions/articles/67000535055-what-is-salary-current-account-"]




loader = AsyncHtmlLoader(urls)
docs = loader.load()

bs_transformer = BeautifulSoupTransformer()
tags_to_extract = ['h1', 'h2', 'h3',  'span']
doc = bs_transformer.transform_documents(docs, tags_to_extract=tags_to_extract)

doc

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(doc)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(doc)






gemini_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GEMINI_API_KEY)

vectorstore = Chroma.from_documents(
                     documents=splits,                 # Data
                     embedding=gemini_embeddings,    # Embedding model
                     persist_directory="./chroma_db" # Directory to save data
                     )

# # from gemini import Gemini
# # from gemini.vectorstores.pinecone import Pinecone
# # # from pinecone import PineconeIndex
# # # from gemini_openai import OpenAIEmbeddings

from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=GEMINI_API_KEY)
# print(llm.invoke("Sing a ballad of LangChain."))


# Load from disk
vectorstore_disk = Chroma(
                        persist_directory="./chroma_db",       # Directory of db
                        embedding_function=gemini_embeddings   # Embedding model
                   )
# Get the Retriever interface for the store to use later.
# When an unstructured query is given to a retriever it will return documents.
# Read more about retrievers in the following link.
# https://python.langchain.com/docs/modules/data_connection/retrievers/
#
# Since only 1 document is stored in the Chroma vector store, search_kwargs `k`
# is set to 1 to decrease the `k` value of chroma's similarity search from 4 to
# 1. If you don't pass this value, you will get a warning.
retriever = vectorstore_disk.as_retriever(search_kwargs={"k": 1})

# Check if the retriever is working by trying to fetch the relevant docs related
# to the word 'climate'. If the length is greater than zero, it means that
# the retriever is functioning well.
# print(len(retriever.get_relevant_documents("climate")))



# Prompt template to query Gemini
llm_prompt_template = """You are an assistant for question-answering tasks for Wema Bank. Be Polite and alway say thank you.
Use the following context to answer the question.
If you don't know the answer, just say that you don't know.
Use three sentences maximum and keep the answer concise.\n
Question: {question} \nContext: {context} \nAnswer:"""

llm_prompt = PromptTemplate.from_template(llm_prompt_template)

# print(llm_prompt)



# retriever = vectorstore.as_retriever()
prompt = hub.pull("rlm/rag-prompt")
# llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key = OPENAI_API_KEY)
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=GEMINI_API_KEY)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | llm_prompt
    | llm
    | StrOutputParser()
)


while True:
    query = input("Ask me any questions about Wema Bank : ")
    print(rag_chain.invoke(query))
