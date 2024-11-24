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
import pickle # For saving the embeddings
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import sigmoid_kernel

load_dotenv()

embedding_file = 'embeddings.pkl'
metadata_file = 'metadata.pkl'

dataset = pd.read_csv("dataset.csv")
dataset.drop(columns=['Email','Phone Number','Graduation Date','Rating','Credits','About'],inplace=True)
dataset.to_csv("Reduced_dataset.csv",index=False)


df = pd.read_csv("Reduced_dataset.csv")
st.title("Hello")
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
def generate_text_local_llama(content,query):

    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature = 0)
    chain = load_qa_chain(llm=llm, chain_type="stuff")
    # print(reranked_docs)
    # print("Content in fenerate_text is::",content)
    response=chain.invoke({"input_documents":content, "question": query})
    # print(response)
    return response

# Load embeddings and data from your CSV file
def embed(query=None):
    embedding = HuggingFaceEmbeddings(model_name='BAAI/bge-small-en-v1.5', model_kwargs={'device': 'cuda'})
    
    if query:
        embed_query = embedding.embed_query(query)
        return np.array(embed_query).astype('float32').reshape(1, -1)  # Reshape here
    
    if os.path.exists(embedding_file) and os.path.exists(metadata_file):
        print("Loading existing embeddings and metadatafile...")
        with open(embedding_file,'rb') as embed_file, open(metadata_file,'rb') as meta_file:
            embedding = pickle.load(embed_file)
            data = pickle.load(meta_file)
    else:
        print("NO new embeddings, creating new ones...")
        loader = CSVLoader(file_path='Reduced_dataset.csv', encoding='utf-8', csv_args={'delimiter': ','})
        data = loader.load()
    # Extract text and compute embeddings
        text = [doc.page_content for doc in data]
        embed_docs = embedding.embed_documents(text)
        embedding = np.array(embed_docs).astype('float32') # No need to reshape for documents.
        print("Model loaded and embeddings done!")
        # Set up FAISS index
        with open(embedding_file,'wb') as embed_file, open(metadata_file,'wb') as meta_file:
            pickle.dump(embedding,embed_file)
            pickle.dump(data,meta_file)
    dimension = embedding.shape[1]
    index = faiss.IndexFlatL2(dimension) # uses Euclidean distance (L2) for search between neighbour embeddings
    index.reset()  # Clear any existing embeddings
    index.add(embedding)
    score=[]

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
    
    results = [(data[i], distances[0][j]) for j, i in enumerate(indices[0])  if distances[0][j]> 0.90]
    print(results)
    return results

# Sample usage with your local Llama for generation and FAISS for retrieval
query = "List at max 5 people with the tag : Architecture and Construction loacted in Arunachal Pradesh?"

def template(query):
    template = '''Return only the people who sastisfy the condition in the given query.Return only the name and User_id.This is the query:{query} '''
    return template.format(query=query)
# query = template(query)
search_results = search_db(query)
       
# Generate text using local Llama
final_doc=[]
for content,_ in search_results:
    final_doc.append(content)

# print(final_doc)    
final_response = generate_text_local_llama(final_doc,query)
print(final_response.get('output_text','No Response'))


# tfv = TfidfVectorizer(min_df=3,max_features=None,strip_accents='unicode',analyzer='word',token_pattern=r'\w{1,}',
#                       ngram_range=(1,3),stop_words='english')

# tfv_matrix = tfv.fit_transform(df['Skills'])
# # print(tfv_matrix)
# # print(tfv_matrix.shape)
# sig = sigmoid_kernel(tfv_matrix,tfv_matrix)
# # print("Sig:",sig[0])
# indices = pd.Series(df.index,index=df['Skills']).drop_duplicates()

# def give_recommendation(skill,sig=sig):
#     try:
#         idx = indices[skill][0]
#         print(idx)
#     except KeyError:
#         print(f"Skill '{skill}' not found in the dataset.")
#         return None
#     sig_score = list(enumerate(sig[idx]))
#     sig_score = sorted(sig_score,key= lambda x:x[1],reverse=True)
#     print(sig_score)
#     sig_score = sig_score[1:11]
#     skill_indices = [i[0] for i in sig_score]

#     return (df.iloc[skill_indices],sig_score[0:11])


# print("The Recommendations are:",give_recommendation("Python, Cybersecurity, Cloud Computing"))
