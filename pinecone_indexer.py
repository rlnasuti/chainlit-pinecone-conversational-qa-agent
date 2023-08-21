from pinecone_datasets import load_dataset
import os
import pinecone
import time
from dotenv import load_dotenv

load_dotenv()

dataset = load_dataset('langchain-python-docs-text-embedding-ada-002')
# we drop sparse_values as they are not needed for this example
dataset.documents.drop(['metadata'], axis=1, inplace=True)
dataset.documents.rename(columns={'blob': 'metadata'}, inplace=True)

pinecone.init()
index_name = 'gpt-4-langchain-docs-fast'

# check if index already exists (it shouldn't if this is first time)
if index_name not in pinecone.list_indexes():
    # if does not exist, create index
    pinecone.create_index(
        index_name,
        dimension=1536,  # dimensionality of text-embedding-ada-002
        metric='cosine'
    )
    # wait for index to be initialized
    time.sleep(1)

# connect to index
index = pinecone.GRPCIndex(index_name)
for batch in dataset.iter_documents(batch_size=100):
    index.upsert(batch)