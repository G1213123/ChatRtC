"""This is the logic for ingesting Notion data into LangChain."""
from pathlib import Path
import time
from langchain.text_splitter import CharacterTextSplitter
import faiss
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
import pickle
from dotenv import load_dotenv
import os
import openai
import tqdm

load_dotenv()

openai.api_key = os.getenv('OPENAI_API_KEY')

# Here we load in the data in the format that Notion exports it in.
ps = list(Path("Notion_DB/TPDM/").glob("**/**/Volume*_Chapter*.htm"))

data = []
sources = []
for p in ps:
    with open(p) as f:
        data.append(f.read())
    sources.append(p)

def chunks (data):
    # Here we split the documents, as needed, into smaller chunks.
    # We do this due to the context limits of the LLMs.
    text_splitter = CharacterTextSplitter(chunk_size=1000, separator="\n")
    for i, d in enumerate(data):
        splits = text_splitter.split_text(d)
        docs = splits
        metadatas = [{"source": sources[i]}] * len(splits)
        yield (docs, metadatas)

def upload():
    # Here we create a vector store from the documents and save it to disk.
    for index, (doc, metadata) in tqdm.tqdm(enumerate(chunks(data))):
        if index == 0:
            store = FAISS.from_texts(doc, OpenAIEmbeddings(), metadatas=metadata)
        else:
            try:
                store.add_texts(doc, metadata)
            except:
                time.sleep(60)
                store.add_texts(doc, metadata)
        faiss.write_index(store.index, "docs.index")
        #store.index = None
    with open("faiss_store.pkl", "wb") as f:
        pickle.dump(store, f)

upload()