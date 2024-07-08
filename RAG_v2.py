import streamlit as st
from elasticsearch import Elasticsearch
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.documents import Document
import ollama
import faiss
import numpy as np
import functools

# Connect to Elasticsearch
es_url = "http://localhost:9200"
es_client = Elasticsearch(es_url)
index_name = "threatdata"

# Create Ollama embeddings
embeddings = OllamaEmbeddings(model="nomic-embed-text")

def create_dynamic_mapping(es_client, index_name, sample_data):
    # Get existing mapping if it exists
    if es_client.indices.exists(index=index_name):
        existing_mapping = es_client.indices.get_mapping(index=index_name)[index_name]['mappings']
    else:
        existing_mapping = {'properties': {}}

    # Analyze sample data and create mapping
    for field, value in sample_data.items():
        if field not in existing_mapping['properties']:
            if isinstance(value, str):
                field_type = 'text'
                if len(value) <= 256:  # You can adjust this threshold
                    field_type = 'keyword'
            elif isinstance(value, int):
                field_type = 'long'
            elif isinstance(value, float):
                field_type = 'double'
            elif isinstance(value, bool):
                field_type = 'boolean'
            elif isinstance(value, dict):
                field_type = 'object'
            elif isinstance(value, list):
                field_type = 'nested'
            else:
                field_type = 'keyword'  # Default to keyword for unknown types
            
            existing_mapping['properties'][field] = {'type': field_type}

    # Ensure vector field exists
    if 'vector' not in existing_mapping['properties']:
        existing_mapping['properties']['vector'] = {
            'type': 'dense_vector',
            'dims': 384,  # Adjust this based on your embedding model
            'index': True,
            'similarity': 'cosine'
        }

    # Update the mapping
    es_client.indices.put_mapping(index=index_name, body=existing_mapping)
    return existing_mapping

def get_all_documents(es_client, index_name, batch_size=1000):
    query = {"query": {"match_all": {}}}
    resp = es_client.search(index=index_name, body=query, scroll='2m', size=batch_size)
    scroll_id = resp['_scroll_id']
    documents = []

    while len(resp['hits']['hits']):
        for hit in resp['hits']['hits']:
            doc = hit['_source']
            text = ' '.join(str(v) for k, v in doc.items() if v and k != 'vector')
            documents.append(Document(page_content=text, metadata=doc))
        
        resp = es_client.scroll(scroll_id=scroll_id, scroll='2m')

    es_client.clear_scroll(scroll_id=scroll_id)
    return documents

# Retrieve a sample document and create/update mapping
sample_query = {"query": {"match_all": {}}, "size": 1}
sample_response = es_client.search(index=index_name, body=sample_query)

if sample_response['hits']['hits']:
    sample_data = sample_response['hits']['hits'][0]['_source']
    mapping = create_dynamic_mapping(es_client, index_name, sample_data)
else:
    print("No data found in the index. Please ensure the index contains data.")
    exit()

# Retrieve all documents from the index
documents = get_all_documents(es_client, index_name)

def batch_embed(docs, batch_size=10):
    embeddings_list = []
    for i in range(0, len(docs), batch_size):
        batch = docs[i:i+batch_size]
        batch_embeddings = embeddings.embed_documents([doc.page_content for doc in batch])
        embeddings_list.extend(batch_embeddings)
    return embeddings_list

doc_embeddings = batch_embed(documents)

def build_faiss_index(embeddings):
    d = len(embeddings[0])
    index = faiss.IndexFlatL2(d)
    index.add(np.array(embeddings).astype('float32'))
    return index

faiss_index = build_faiss_index(doc_embeddings)

@functools.lru_cache(maxsize=1000)
def cached_embed_query(query):
    return embeddings.embed_query(query)

def faiss_similarity_search(query, index, docs, top_k=5):
    query_embedding = cached_embed_query(query)
    D, I = index.search(np.array([query_embedding]).astype('float32'), top_k)
    return [docs[i] for i in I[0]]

def ollama_llm_stream(question, context):
    formatted_prompt = f"Question: {question}\n\nContext: {context}"
    response = ollama.chat(model='llama3', messages=[{'role': 'user', 'content': formatted_prompt}], stream=True)
    for chunk in response:
        yield chunk['message']['content']

def combine_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def rag_chain(question):
    retrieved_docs = faiss_similarity_search(question, faiss_index, documents)
    formatted_context = combine_docs(retrieved_docs)
    return ollama_llm_stream(question, formatted_context)

# Streamlit UI
st.title("ElasticRAG")

question = st.text_input("Ask me something...")
if question:
    result_placeholder = st.empty()
    full_response = ""
    for chunk in rag_chain(question):
        full_response += chunk
        result_placeholder.text(full_response)