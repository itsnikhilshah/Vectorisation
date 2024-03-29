from pinecone import Pinecone, ServerlessSpec, PodSpec
from sentence_transformers import SentenceTransformer
import torch
import spacy
from datetime import datetime
from time import time
import pickle
import configparser
import os
import pandas as pd
import glob


def clean_course_desc(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    corrected_lines = []
    for line in lines:
        if line.strip() and (line[0].isalnum() and "-" in line):
            corrected_lines.append(line)
        elif corrected_lines: 
            print('Correction made')
            corrected_lines[-1] = corrected_lines[-1].strip() + " " + line

    with open(file_path, 'w', encoding='utf-8') as file:
        file.writelines(corrected_lines)


class PineconeInteraction:
    def __init__(self):
        config=configparser.ConfigParser()
        config.read('settings.ini')
        self.secret_key=config['DEFAULT']['SECRET_KEY']
        index_name = 'serc-index'
        pc=Pinecone(api_key=self.secret_key)

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
        
        self.index=pc.Index(index_name)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model_name = 'sentence-transformers/all-roberta-large-v1'
        self.model = SentenceTransformer(model_name)
        self.corpus_sentences = []
    
    def corpus_embeddings(self,filename):
        clean_course_desc(filename)
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
        with open('data\\course_data.pkl','wb') as f:
            pickle.dump((course_id,course_descriptions),f)


        for i in range(0, len(corpus_embedding), chunk_size):
            data_chunk = list(zip(course_id[i:i+chunk_size], corpus_embedding[i:i+chunk_size])) #preparing the data chunk
            
            #uploading to vector datastore
            self.index.upsert(data_chunk,namespace="course_descriptions")


            print("Chunk "+str(i)+" has been upserted")


    def embedding_query_with_score(self):
        folderpath = "data\\roles"
        if not os.path.exists("data\\excel"):
            os.mkdir('data\\excel')

        # Load course IDs and descriptions from pickle
        with open('data\\course_data.pkl', 'rb') as f:
            course_id_list, course_descriptions = pickle.load(f)

        # Load course ID to name mapping from pickle
        with open('data\\course_id_to_name.pkl', 'rb') as f:
            course_id_to_name = pickle.load(f)

        self.corpus_sentences = course_descriptions

        # Process each roles file
        for filename in glob.glob(os.path.join(folderpath, '*.xlsx')):
            df_roles = pd.read_excel(filename)
            query_sentences = df_roles['description : String'].tolist()

            query_embeddings = self.model.encode(query_sentences).tolist()

            df = pd.DataFrame(columns=['KSAB', 'Course 1', 'Course Name 1', 'Score 1', 'Course 2', 'Course Name 2', 'Score 2', 'Course 3', 'Course Name 3', 'Score 3', 'Course 4', 'Course Name 4', 'Score 4', 'Course 5', 'Course Name 5', 'Score 5'])

            for query_embedding, query_sentence in zip(query_embeddings, query_sentences):
                res = self.index.query(vector=query_embedding, namespace="course_descriptions", top_k=5, include_values=True)

                # Extract the course IDs and scores
                course_ids = [res_match.id for res_match in res.matches]
                scores = [res_match.score for res_match in res.matches]
                course_names = [course_id_to_name.get(course_id, "Unknown") for course_id in course_ids]

                # Fill in NaN if fewer than 5 matches
                while len(course_ids) < 5:
                    course_ids.append(pd.NA)
                    course_names.append(pd.NA)
                    scores.append(pd.NA)

                df.loc[len(df)] = [query_sentence] + [val for pair in zip(course_ids, course_names, scores) for val in pair]

            base = os.path.basename(filename)
            name, ext = os.path.splitext(base)
            df.to_excel(f'data\\excel\\{name}.xlsx', sheet_name=name, index=False)
            df = df.iloc[0:0]
            print(f"{name} done")


        
    def delete_namespace(self, namespace):
        self.index.delete(deleteAll='true', namespace=namespace)


def main():
    pi = PineconeInteraction()

    #pi.corpus_embeddings('course_descriptions.txt')
    pi.embedding_query_with_score()

if __name__=='__main__':
    main()