import configparser
from pinecone import Pinecone, PodSpec

config=configparser.ConfigParser()
config.read('settings.ini')
secret_key=config['DEFAULT']['SECRET_KEY']
index_name="serc-index"
pc=Pinecone(api_key=secret_key)
pc.create_index(
  name=index_name,
  dimension=1024,
  metric="cosine",
  spec=PodSpec(
    environment="gcp-starter"
  )
)
index=pc.Index(index_name)
index.upsert(
  vectors=[
    {"id": "A", "values": [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]},
    {"id": "B", "values": [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]},
    {"id": "C", "values": [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]},
    {"id": "D", "values": [0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4]}
  ],
  namespace="ns1"
)