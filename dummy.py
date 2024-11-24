import  dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_milvus import Milvus
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import JsonOutputParser

urls = {
    'https://medium.com/data-science-in-your-pocket/recommendation-systems-using-neural-collaborative-filtering-ncf-explained-with-codes-21a97e48a2f7',

}
docs = [WebBaseLoader(url) for url in urls]
docs_list = [item for sublist in docs for item in sublist]
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=250,chunk_overlap=25)
docs_split = text_splitter.split_documents(docs_list)

vector_store = Milvus.from_documents(
    documents=docs_split,
    collection_name='rag_milvus',
    embedding=HuggingFaceEmbeddings(model_name='BAAI/bge-small-en-v1.5', model_kwargs={'device': 'cuda'}),
    connection_args={'url':'./milvus_rag.db'}
)
retriever = vector_store.as_retriever()

# Retrieval Grader
llm = ChatOllama(model='llama3.1',format='json',temperature=0)

prompt = PromptTemplate(
    template=''' You are a grader assessing relevance of a retrieved document to a user question. If the document contains keywords related to the user question, grade it as relevant. The goal is to filter out unwanted retrievals.
    Give a binary score 'Yes' or 'No' to indicate whether the document is relevant to the question. Provide the binary score in a json format with a single key 'score'. No need for explanation
    Here is the retrieved document:{document}
    
    Here is the user question: {question}''',input_variables=['document','question']
)

retrieval_grader = prompt | llm | JsonOutputParser()

question = 'agent memory'
docs = retriever.invoke(question)
docs_txt = docs[i].page_content
print(retrieval_grader.invoke({'question':question,'document':docs_txt}))