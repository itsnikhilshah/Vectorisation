from pinecone import Pinecone, ServerlessSpec, PodSpec
from sentence_transformers import SentenceTransformer
import torch
import spacy
from datetime import datetime
from time import time
import pickle
import configparser

class PineconeInteraction:
    def __init__(self):
        config=configparser.ConfigParser()
        config.read('settings.ini')
        self.secret_key=config['DEFAULT']['SECRET_KEY']
        self.environment=config['DEFAULT']['ENVIRONMENT_LOCATION']
        index_name = 'serc-index'
        try:
            pc=Pinecone(api_key=self.secret_key)
            active_indexes=pc.list_indexes()
            index_description=pc.describe_index(index_name)
            self.index=pc.Index(index_name)
        except Exception as e:
            pc=Pinecone(api_key=self.secret_key)
            print("Creating Index")
            pc.create_index(index_name, dimension=1024,metric="euclidean" ,spec=PodSpec(
    environment="us-west4-gcp",
    pod_type="p1.x1",
    pods=1
  ))
            active_indexes=pc.list_indexes()
            index_description=pc.describe_index(index_name)
            self.index=pc.Index(index_name)
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model_name = 'sentence-transformers/all-roberta-large-v1'
        self.model = SentenceTransformer(model_name)
        self.corpus_sentences = []
    
    def corpus_embeddings(self,filename):
        txt_file = open(filename,"r", encoding='utf8') 
        txt_lines=txt_file.readlines() #reading each line individually

        course_id = []
        course_descriptions = []

        for data in txt_lines: #splitting the course id and course description
            split_data = data.split(" -> ")
            if len(split_data) >= 2:
                course_id.append(split_data[0])
                course_descriptions.append(split_data[1])
            else:
                print(f"Warning: '->' not found in item: {data}")
        
        self.corpus_sentences=course_descriptions
        #calculating corpus embeddings
        corpus_embedding = self.model.encode(self.corpus_sentences).tolist()

        #chunk the data into batches of 200
        chunk_size = 200

        #storing the calculated values into pickle file to reduce need to constantly process
        with open('course_data.pkl','wb') as f:
            pickle.dump((course_id,course_descriptions),f)


        for i in range(0, len(corpus_embedding), chunk_size):
            data_chunk = list(zip(course_id[i:i+chunk_size], corpus_embedding[i:i+chunk_size])) #preparing the data chunk
            
            #uploading to vector datastore
            self.index.upsert(data_chunk,namespace="course_descriptions")


            print("Chunk "+str(i)+" has been upserted")
        
    def delete_namespace(self, namespace):
        self.index.delete(deleteAll='true', namespace=namespace)
    
def main():
    pi = PineconeInteraction()
    
    
    pi.corpus_embeddings('course_descriptions.txt')

if __name__=='__main__':
    main()