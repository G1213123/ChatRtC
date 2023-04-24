"""This is the logic for ingesting Notion data into LangChain."""
from pathlib import Path
import pickle
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import BSHTMLLoader, PyMuPDFLoader
import openai
from langchain.vectorstores import FAISS, Pinecone
from langchain.embeddings import LlamaCppEmbeddings

from dotenv import load_dotenv
import glob
import os
import re
import pinecone
from pyllamacpp.model import Model
from tqdm import tqdm

load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

# initialize connection to pinecone (get API key at app.pinecone.io)
# pinecone.init(
#     api_key=os.getenv("PINECONE_API_KEY"),  # find at app.pinecone.io
#     environment=os.getenv("PINECONE_ENV")  # next to api key in console
# )
# index_name = os.getenv("PINECONE_INDEX")
# # check if index already exists (it shouldn't if this is first time)
# if index_name not in pinecone.list_indexes():
#     # if does not exist, create index
#     pinecone.create_index(
#         index_name,
#         dimension=1536,
#         metric='cosine',
#         metadata_config={'indexed': ['channel_id', 'published']}
#     )

#model = Model(ggml_model=os.getenv("GPT4ALL_PATH"), n_ctx=512)
llama_embeddings = LlamaCppEmbeddings(model_path=os.getenv("GPT4ALL_PATH"), n_ctx=2048, n_threads=1024)

def strip_html_tags(s):
    """remove html tags from a string"""
    data=[]
    for t in s:
        t.page_content = t.page_content.replace('Top\n\n  Press Ctrl-F for Keyword Search on this Page', '')
        t.page_content = re.sub(r'\n+', '\n', t.page_content)
        t.metadata['source'] = str(t.metadata['source'])
        data.append(s)
    return data

def get_pdf_docs(path, ext, pattern=None):
        # Here we load in the data in the format that Notion exports it in.
    ps = list(Path(path).rglob(f"*.{ext}"))
    if pattern:
        pattern = re.compile(pattern)
        ps = [p for p in ps if pattern.match(p.name)]
    
    loaders = {'pdf': PyMuPDFLoader, 'htm': BSHTMLLoader}

    data = []
    # sources = []
    for p in ps:
        loader = loaders[ext](str(p))
        s = loader.load()
        data.append(s)
        # sources.append( p )
    return data

def ingest():
    #data = get_pdf_docs("static/TPDM/", "htm", r'\d+_\d+.htm')
    data = get_pdf_docs(os.getenv("DOC_PATH"), "pdf")

    # Here we split the documents, as needed, into smaller chunks.
    # We do this due to the context limits of the LLMs.
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100, separator="\n")
    docs = []
    # metadatas = []
    for i, d in enumerate(data):
        splits = text_splitter.split_documents(d)
        docs.extend(splits)
        # metadatas.extend([{"source": d.metadatas}] * len(splits))

    # Here we create a vector store from the documents and save it to disk.
    print(len(docs))
    # for doc in tqdm(docs):
    vectorstore = FAISS.from_documents(docs, llama_embeddings)
    #Pinecone.from_documents(docs, OpenAIEmbeddings(), index_name=index_name)

    # Save vectorstore
    vectorstore.save_local("faiss_index")

if __name__ == "__main__":
    ingest()
