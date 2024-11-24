from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import CSVLoader
import numpy as np
import faiss
from langchain_community.llms import Ollama
from langchain.chains.question_answering import load_qa_chain
# import streamlit as st
from langchain_groq import ChatGroq
from groq import Groq
import os
from dotenv import load_dotenv
import pandas as pd

load_dotenv()


dataset = pd.read_csv("dataset.csv")
dataset.drop(columns=['Email','Phone Number','Graduation Date','Rating'],inplace=True)
dataset.to_csv("new_dataset.csv",index=False)

# st.title("Hello")
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
def generate_text_local_llama(content,query):
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature = 0)
    chain = load_qa_chain(llm=llm, chain_type="stuff")
    # print(reranked_docs)
    response = chain.invoke({"input_documents": content, "question": query})
    print(response)
    return response

# Load embeddings and data from your CSV file
def embed(query=None):
    embedding = HuggingFaceEmbeddings(model_name='BAAI/bge-small-en-v1.5', model_kwargs={'device': 'cuda'})
    
    if query:
        embed_query = embedding.embed_query(query)
        return np.array(embed_query).astype('float32').reshape(1, -1)  # Reshape here
    loader = CSVLoader(file_path='new_dataset.csv', encoding='utf-8', csv_args={'delimiter': ','})
    data = loader.load()

    # Extract text and compute embeddings
    text = [doc.page_content for doc in data]
    embed_docs = embedding.embed_documents(text)
    embedding_np = np.array(embed_docs).astype('float32') # No need to reshape for documents.
    print("Model loaded and embeddings done!")
    # Set up FAISS index
    dimension = embedding_np.shape[1]
    index = faiss.IndexFlatL2(dimension) # uses Euclidean distance (L2) for search between neighbour embeddings
    index.reset()  # Clear any existing embeddings
    index.add(embedding_np)

    return index, data
index,data = embed()
def search_db(query, top_k=100):
    query_embedding = embed(query)
    if query_embedding.size == 0 or query_embedding.shape[1] != index.d:
        print("Query embedding is empty or dimension mismatch.")
        return []
    
    distances, indices = index.search(query_embedding, top_k)
    if indices.size == 0 or indices[0][0] == -1:
        print("No results found.")
        return []
    
    results = [(data[i], distances[0][j]) for j, i in enumerate(indices[0])]
    print(results)
    print(indices)
    return results

# Sample usage with your local Llama for generation and FAISS for retrieval
query = "List the people with the tag : Architecture and Construction ?"

def template(query):
    template = ''' Return all the people who match with the requirements of the query from the document in a table format if there are many people satisfying the condition. Do not leave out any: {query}'''
    return template.format(query=query)
# query = template(query)
search_results = search_db(query)

# Generate text using local Llama
for content,_ in search_results:
    response_text = generate_text_local_llama([content],query)
    print("Generated Response:", response_text.get('output_text',"No Response"))