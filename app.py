from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
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
# import torch
# import torch.nn as nn
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import sigmoid_kernel
from flask import *
import csv


load_dotenv()

embedding_file = 'embeddings.pkl'
metadata_file = 'metadata.pkl'

dataset = pd.read_csv("dataset.csv")
dataset.drop(columns=['Email','Phone Number','Graduation Date','Rating','Credits','About'],inplace=True)
dataset.to_csv("Reduced_dataset.csv",index=False)


df = pd.read_csv("Reduced_dataset.csv")

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def template(query):
    template = '''You are a search engine helping to find out people matching the query.Return the people who sastisfy the condition in the given query only. Do not give redundant results.Return only the name and User_id for everyone. No extra texts.This is the query:{query} '''
    return template.format(query=query)

def embed(query=None, single_user=None):
    embedding = HuggingFaceEmbeddings(model_name='BAAI/bge-small-en-v1.5', model_kwargs={'device': 'cuda'})

    if query:
        embed_query = embedding.embed_query(query)

        return np.array(embed_query).astype('float32').reshape(1, -1)

    if single_user:
        embed_user = embedding.embed_query(single_user)
        return np.array(embed_user).astype('float32').reshape(1, -1)

    if os.path.exists(embedding_file) and os.path.exists(metadata_file):
        print("Loading existing embeddings and metadata file...")
        with open(embedding_file, 'rb') as embed_file, open(metadata_file, 'rb') as meta_file:
            embedding = pickle.load(embed_file)
            data = pickle.load(meta_file)
    else:
        print("No existing embeddings, creating new ones...")
        # Load the CSV and preprocess it
        df = pd.read_csv('Reduced_dataset.csv', encoding='utf-8')
        df['combined'] = df.apply(lambda row: ', '.join(row.dropna().astype(str)), axis=1)

        # Create Documents with concatenated content
        data = []
        for idx, row in df.iterrows():
            combined_text = row['combined']
            metadata = {"source": 'Reduced_dataset.csv', "row": idx}
            data.append(Document(page_content=combined_text, metadata=metadata))

        # Extract text and compute embeddings
        text = [doc.page_content for doc in data]
        # print(text)
        embed_docs = embedding.embed_documents(text)
        embedding = np.array(embed_docs).astype('float32')

        print("Model loaded and embeddings created!")

        # Save embeddings and metadata
        with open(embedding_file, 'wb') as embed_file, open(metadata_file, 'wb') as meta_file:
            pickle.dump(embedding, embed_file)
            pickle.dump(data, meta_file)

    # Set up FAISS index
    dimension = embedding.shape[1]
    faiss.normalize_L2(embedding)
    index = faiss.IndexFlatL2(dimension)  # Uses Euclidean distance (L2) for nearest neighbor search
    index.reset()  # Clear any existing embeddings
    index.add(embedding)

    return index, data




def add_user_to_index(single_user):
    new_single_user = single_user.split(',')
    row_data  = [
    new_single_user[0],
    new_single_user[1],
    new_single_user[2],
    new_single_user[3],
    new_single_user[4],
    new_single_user[5],
    new_single_user[6],
    new_single_user[7],
    new_single_user[8],
    new_single_user[9],
    new_single_user[10],
    new_single_user[11],
    new_single_user[12],
    new_single_user[13]]
    print("ENtered")
    new_vector = embed(single_user=single_user)
    print("EMbedded")
    if os.path.exists(embedding_file) and os.path.exists(metadata_file):
        with open (embedding_file,'rb') as embed_file , open(metadata_file,'rb') as meta_file:
            exisitng_embeddings = pickle.load(embed_file)
            metadata = pickle.load(meta_file)
    else:
        print("No New embeddings found!!")
        exisitng_embeddings = np.empty((0,new_vector.shape[1]),dtype='float32')
        metadata=[]
    
    with open('Reduced_dataset.csv','a',newline='') as file:
        writer = csv.writer(file)
        writer.writerow(row_data)
    

    # np.vstack helps in stacking arrays vertically...
    new_vector = new_vector.reshape(1,-1)
    combined_embeddings = np.vstack([exisitng_embeddings,new_vector])
    
    faiss.normalize_L2(combined_embeddings)
    dimensions = combined_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimensions)
    index.reset()
    index.add(combined_embeddings)

    with open (embedding_file,'wb') as embed_file , open(metadata_file,'wb') as meta_file:
        pickle.dump(combined_embeddings,embed_file)

        df = pd.read_csv('Reduced_dataset.csv', encoding='utf-8')
        df['combined'] = df.apply(lambda row: ', '.join(row.dropna().astype(str)), axis=1)

        # Create Documents with concatenated content
        data = []
        for idx, row in df.iterrows():
            combined_text = row['combined']
            metadata = {"source": 'Reduced_dataset.csv', "row": idx}
            data.append(Document(page_content=combined_text, metadata=metadata))

        pickle.dump(data,meta_file)
        print("Saved in metadata file")
    print("Successfully added new user to csv,faiss database and local files...")

index,data = embed()
# print(data)
def search_db(query, top_k=10):
    query_embedding = embed(query)
    # print(query_embedding)
    faiss.normalize_L2(query_embedding)
    # print(query_embedding)
    if query_embedding.size == 0 or query_embedding.shape[1] != index.d:
        print("Query embedding is empty or dimension mismatch.")
        return []
    
    distances, indices = index.search(query_embedding, top_k)
    # print("Incides:",indices[0])
    if len(indices[0]) == 0:
        print("No results found.")
        return []
    
    results = [(data[i], distances[0][j]) for j, i in enumerate(indices[0])  if distances[0][j]>= 0.6]
    print("Results are:",results)
    return results

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

# Flask APP
app = Flask(__name__)
@app.route("/")
def hello():
    return render_template('adduser.html',title="HI")

@app.route("/searching")
def search_bar():
    return render_template("search.html")

@app.route('/search-engine',methods=['GET'])
def search_engine():
    query = request.args.get('query')
    if not query:
        return jsonify({
            "Error":"Query parameter required"
        }), 400
    query = template(query)
    search_results = search_db(query)
    # print(search_results)
    final_doc=[]
    for doc in search_results:
        final_doc.append(doc[0])
    if not final_doc:
        return jsonify({"query": query, "response": "No matching results found."})
    
    # print(final_doc)    
    final_response = generate_text_local_llama(final_doc,query)
    # print(final_response.get('output_text','No Response'))
    return jsonify({
        'query':query,
        'response':final_response.get('output_text','No Response')
    })
    

    # print(final_doc)
    # results = [{"content":doc.page_content,"metadata":doc.metadata} for doc in final_doc]
    # return jsonify(results)

@app.route('/add-user',methods=['GET','POST'])
def add_user():
    if request.method=="POST":
    # for adding user to dataset.csv and faiss database
        form = request.form
        single_user = ""

        # Extract each field from the form
        first_name = form.get("First Name", "")
        single_user += first_name + ','

        middle_name = form.get("Middle Name", "")
        single_user += middle_name + ','

        last_name = form.get("Last Name", "")
        single_user += last_name + ','
        
        user_id = form.get("User ID", "")
        single_user += user_id + ','
        
        location = form.get("Location", "")
        single_user += location + ','
        
        current_company = form.get("Current Work Experience (Company)", "")
        single_user += current_company + ','
        
        current_experience_years = form.get("Current Work Experience (Years)", "0")
        single_user += current_experience_years + ','
        
        past_company = form.get("Past Work Experience (Company)", "")
        single_user += past_company + ','

        past_experience_years = form.get("Past Work Experience (Years)","0")
        single_user += str(past_experience_years)
        single_user += ','
        
        institution = form.get("Institution")
        single_user += institution
        single_user += ','
        
        major = form.get("Major")
        single_user += major
        single_user += ','
        
        skills = form.get("Skills")
        single_user += skills
        single_user += ','
        
        certificates = form.get("Certificates")
        single_user += certificates
        single_user += ','
        
        tags = form.get("Tags")
        single_user += tags

    print("User details:",single_user)
    add_user_to_index(single_user)
    return render_template("adduser.html")

# @app.route('/add-credits',methods=['GET'])
# def add_credit():
      
# Sample usage with your local Llama for generation and FAISS for retrieval

# query = "List at max 5 people with the tag : Architecture and Construction loacted in Arunachal Pradesh?"


# query = template(query)

       
# Generate text using local Llama



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

if __name__ == '__main__':
    
    app.run(debug=True)