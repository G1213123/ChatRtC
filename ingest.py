"""This is the logic for ingesting Notion data into LangChain."""
from pathlib import Path
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import BSHTMLLoader
import openai
from langchain.vectorstores import FAISS, Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from dotenv import load_dotenv
import glob
import os
import re
import pinecone

load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')
# initialize connection to pinecone (get API key at app.pinecone.io)
pinecone.init(
    api_key=os.getenv("PINECONE_API_KEY"),  # find at app.pinecone.io
    environment=os.getenv("PINECONE_ENV")  # next to api key in console
)
index_name = os.getenv("PINECONE_INDEX")
# check if index already exists (it shouldn't if this is first time)
if index_name not in pinecone.list_indexes():
    # if does not exist, create index
    pinecone.create_index(
        index_name,
        dimension=1536,
        metric='cosine',
        metadata_config={'indexed': ['channel_id', 'published']}
    )

def ingest():
    # Here we load in the data in the format that Notion exports it in.
    ps = list(Path("static/TPDM/").rglob("*_*.htm"))
    pattern = re.compile(r'\d+_\d+.htm')
    ps = [p for p in ps if pattern.match(p.name)]

    data = []
    # sources = []
    for p in ps:
        loader = BSHTMLLoader(p, open_encoding='utf-8')
        s = loader.load()
        for t in s:
            t.page_content = t.page_content.replace('Top\n\n  Press Ctrl-F for Keyword Search on this Page', '')
            t.metadata['source'] = str(t.metadata['source'])
        data.append(s)
        # sources.append( p )

    # Here we split the documents, as needed, into smaller chunks.
    # We do this due to the context limits of the LLMs.
    text_splitter = CharacterTextSplitter(chunk_size=1500, chunk_overlap=200, separator="\n")
    docs = []
    # metadatas = []
    for i, d in enumerate(data):
        splits = text_splitter.split_documents(d)
        docs.extend(splits)
        # metadatas.extend([{"source": d.metadatas}] * len(splits))

    # Here we create a vector store from the documents and save it to disk.
    print(docs)
    Pinecone.from_documents(docs, OpenAIEmbeddings(), index_name=index_name)

if __name__ == "__main__":
    ingest()
