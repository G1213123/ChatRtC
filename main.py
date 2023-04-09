"""Python file to serve as the frontend"""
import urllib
import datetime

import streamlit as st
from streamlit_chat import message
from streamlit.components.v1 import html, iframe
import pinecone
from langchain.chat_models import ChatOpenAI
from prompt import EXAMPLE_PROMPT, QUESTION_PROMPT, COMBINE_PROMPT
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from qa import RetrievalQAWithClausesSourcesChain
from google.oauth2 import service_account
from google.cloud import storage
import openai
from dotenv import load_dotenv
import os
import re

load_dotenv()

openai.api_key = os.getenv( 'OPENAI_API_KEY' )

# Load the LangChain.
pinecone.init(
    api_key=os.getenv("PINECONE_API_KEY"),  # find at app.pinecone.io
    environment=os.getenv("PINECONE_ENV")  # next to api key in console
)
index_name = os.getenv("PINECONE_INDEX")
index = pinecone.Index(index_name)
embeddings = OpenAIEmbeddings()
store = Pinecone(index, embeddings.embed_query, "text")
# Set up prompt


chain = RetrievalQAWithClausesSourcesChain.from_llm( llm=ChatOpenAI( temperature=0 ),
                                                     document_prompt=EXAMPLE_PROMPT,
                                                     question_prompt=QUESTION_PROMPT,
                                                     combine_prompt=COMBINE_PROMPT,
                                                     vectorstore=store,
                                                     k=2,
                                                     verbose=True )

# Create API client.
credentials = service_account.Credentials.from_service_account_info(
    st.secrets["gcp_service_account"]
)
client = storage.Client(credentials=credentials)

# Retrieve file contents.
# Uses st.cache_data to only rerun when the query changes or after 10 min.
@st.cache_data()
def read_file(bucket_name, file_path):
    file_path = urllib.parse.unquote( file_path ).replace( '\\', '/' ).replace('static/','')
    bucket = client.bucket(bucket_name)
    content = bucket.blob(file_path)

    url = content.generate_signed_url(
        version="v4",
        # This URL is valid for 15 minutes
        expiration=datetime.timedelta( minutes=15 ),
        # Allow GET requests using this URL.
        method="GET",
    )
    return url

# From here down is all the StreamLit UI.
st.set_page_config( page_title="ChatRTC loaded with TPDM", page_icon=":robot:", layout="wide" )

if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []


def get_text():
    input_text = st.text_input( "You: ", "", key="input" )
    return input_text


def beautify_source_name(answer):
    clauses = re.findall( r'\[(.*?)\]', answer )
    for n in clauses:
        try:
            volume = re.search( r"(?<=TPDM\\v)(\d+)(?=\\c)", n ).group( 0 )
            chapter, section = re.search( '\d+_\d+', n ).group( 0 ).split( '_' )
            clause = n.split( 'clause ' )[1].strip()
            substitute = (f"Volume {volume}\nChapter {chapter}.{section}\n{clause}")
            answer = answer.replace( n, substitute )
        except:
            pass
    return answer


def get_full_name(name):
    volume = re.search( r"(?<=TPDM\\v)(\d+)(?=\\c)", name ).group( 0 )
    chapter, section = re.search( '\d+_\d+', name ).group( 0 ).split( '_' )
    return name.replace( f'{chapter}_{section}', f'Volume{volume}_Chapter{chapter}{section}' )


results_tabs = st.tabs( ['result1', 'result2', 'result3'] )

with st.sidebar:
    st.title( "ChatRTC loaded with TPDM", )
    user_input = get_text()

    if user_input != "":
        result = chain( {"question": user_input} )
        output = f"Answer: {result['answer']}\n\nClauses: {result['clauses']}"
        print(output)

        st.session_state.past.append( user_input )
        st.session_state.generated.append( output )

        clauses = [s.strip() for s  in result['clauses'].split(',')]

        if st.session_state["generated"]:
            for i in range(len(st.session_state["generated"]) - 1, -1, -1):
                message(st.session_state["generated"][i], key=str(i))
                message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")

        for i, h in enumerate( result['sources'].split( ',' ) ):
            bucket_name = "tpdm"
            file_path = h.replace( '.md', '.htm' ).strip()
            source_code = read_file( bucket_name, file_path )
            #HtmlFile = open( h.replace( '.md', '.htm' ).strip(), encoding="utf8", errors='ignore' )
            #source_code = HtmlFile.read()

            with results_tabs[i]:
                iframe( source_code, width=None, height=600, scrolling=True )
        #except:
        #    output = f"Answer: Sorry, I am too stupid to understand your question. Could you please ask again?"

        #    st.session_state.past.append( user_input )
        #    st.session_state.generated.append( output )


