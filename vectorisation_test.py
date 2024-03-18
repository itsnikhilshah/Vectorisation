import configparser
from pinecone import Pinecone, PodSpec
from sentence_transformers import SentenceTransformer
import torch
import spacy
from datetime import datetime
from time import time
import pickle


config=configparser.ConfigParser()
config.read('settings.ini')
secret_key=config['DEFAULT']['SECRET_KEY']
index_name="serc-index"
pc=Pinecone(api_key=secret_key)


existing_indexes = pc.list_indexes().names()
if index_name in existing_indexes:
  print(f"Index '{index_name}' already exists.")
else:
  pc.create_index(
    name=index_name,
    dimension=1024,
    metric="cosine",
    spec=PodSpec(
      environment="gcp-starter"
    )
  )

index=pc.Index(index_name)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_name = 'sentence-transformers/all-roberta-large-v1'
model = SentenceTransformer(model_name)


filename='test.txt'
txt_file = open(filename,"r", encoding='utf8') 
corpus_sentences=txt_file.readlines()
corpus_embedding = model.encode(corpus_sentences).tolist()
ids = [f'vector_{i}' for i in range(len(corpus_embedding))]

data_with_ids = [(ids[i], corpus_embedding[i]) for i in range(len(corpus_embedding))]

chunk_size = 200

for i in range(0, len(data_with_ids), chunk_size):
    data_chunk = data_with_ids[i:i + chunk_size]
    
    index.upsert(data_chunk, namespace="UnitedvLiverpool")

    print("Chunk "+str(i)+" has been upserted")