"""Python file to serve as the frontend"""
import tempfile
import urllib
import datetime

import streamlit as st
from streamlit_chat import message
from streamlit.components.v1 import iframe, html
import pinecone
from langchain.callbacks import get_openai_callback
from langchain.chat_models import ChatOpenAI
from prompt import EXAMPLE_PROMPT, QUESTION_PROMPT, COMBINE_PROMPT
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from qa import RetrievalQAWithClausesSourcesChain
from google.oauth2 import service_account
from google.cloud import storage
import openai
from dotenv import load_dotenv
import re
import os

load_dotenv()

openai.api_key = os.getenv( 'OPENAI_API_KEY' )

# Load the LangChain.
pinecone.init(
    api_key=os.getenv( "PINECONE_API_KEY" ),  # find at app.pinecone.io
    environment=os.getenv( "PINECONE_ENV" )  # next to api key in console
)
index_name = os.getenv( "PINECONE_INDEX" )
index = pinecone.Index( index_name )
embeddings = OpenAIEmbeddings()
store = Pinecone( index, embeddings.embed_query, "text" )
retriever = store.as_retriever()

chain = RetrievalQAWithClausesSourcesChain.from_llm( llm=ChatOpenAI( temperature=0 ),
                                                     document_prompt=EXAMPLE_PROMPT,
                                                     question_prompt=QUESTION_PROMPT,
                                                     combine_prompt=COMBINE_PROMPT,
                                                     retriever=retriever,
                                                     reduce_k_below_max_tokens=True,
                                                     max_tokens_limit=1000,
                                                     verbose=True )

# Create API client.
credentials = service_account.Credentials.from_service_account_info(
    st.secrets["gcp_service_account"]
)
client = storage.Client( credentials=credentials )


# Retrieve file contents.
# Uses st.cache_data to only rerun when the query changes or after 10 min.
@st.cache_data()
def read_file(bucket_name, file_path):
    file_path = urllib.parse.unquote( file_path ).replace( '\\', '/' ).replace( 'static/', '' )
    bucket = client.bucket( bucket_name )
    content = bucket.blob( file_path )

    # Read HTML file contents
    html_contents = content.download_as_bytes().decode('utf-8')

    # Find all image filenames in HTML file contents
    image_filenames = re.findall( r'<img.+?src=[\'"](?P<src>.+?)[\'"].*?>', html_contents )

    # Generate signed URLs for each image file
    signed_image_urls = {}
    for image_filename in image_filenames:
        # Construct full path to image file
        image_blob = bucket.blob( urllib.parse.urljoin(file_path,image_filename) )
        signed_image_url = image_blob.generate_signed_url(
            expiration=datetime.timedelta( hours=1 ),
            method='GET',
            version='v4',
        )
        signed_image_urls[image_filename] = signed_image_url

    # Replace image filenames in HTML file contents with signed URLs
    for image_filename, signed_image_url in signed_image_urls.items():
        html_contents = html_contents.replace( image_filename, signed_image_url )
    return html_contents

# From here down is all the StreamLit UI.
st.set_page_config( page_title="ChatRTC loaded with TPDM", page_icon=":robot:", layout="wide" )

ss_keys = ['generated', 'past', 'cost']
for sk in ss_keys:
    if sk not in st.session_state:
        st.session_state[sk] = []


def get_text():
    input_text = st.text_input( "You: ", "", key="input" )
    return input_text


results_tabs = st.tabs( ['result1', 'result2', 'result3'] )

with st.sidebar:
    st.title( "ChatRTC loaded with TPDM", )
    user_input = get_text()

    if user_input != "":
        with get_openai_callback() as cb:
            result = chain( {"question": user_input} )
        output = f"Answer: {result['answer']}\n\nClauses: {result['clauses']}"
        print( output )

        st.session_state.past.append( user_input )
        st.session_state.generated.append( output )
        st.session_state.cost.append(cb.total_cost)

        clauses = [s.strip() for s in result['clauses'].split( ',' )]

        if st.session_state["generated"]:
            for i in range( len( st.session_state["generated"] ) - 1, -1, -1 ):
                message( st.session_state["generated"][i], key=str( i ) )
                cost = "{:.2f}".format(st.session_state["cost"][i]*7.8)
                message(f'You have wasted HK${cost} for this answer, satisfied?', avatar_style='miniavs', seed=15 ,key=str(i)+"_cost")
                message( st.session_state["past"][i], is_user=True, key=str( i ) + "_user" )

        for i, h in enumerate( result['sources'].split( ',' )[:3] ):
            with results_tabs[i]:
                with st.spinner( 'Loading Documents' ):
                    bucket_name = "tpdm"
                    file_path = h.replace( '.md', '.htm' ).strip()
                    source_code = read_file( bucket_name, file_path )
                html(source_code, scrolling=True, height=800 )

