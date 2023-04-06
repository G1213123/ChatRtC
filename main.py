"""Python file to serve as the frontend"""
import streamlit as st
from streamlit_chat import message
import streamlit.components.v1 as components
from streamlit.components.v1 import html
import faiss
from langchain.chat_models import ChatOpenAI
from prompt import EXAMPLE_PROMPT, QUESTION_PROMPT, COMBINE_PROMPT
from langchain.chains import VectorDBQAWithSourcesChain
from qa import RetrievalQAWithClausesSourcesChain
import pickle
import pathlib
import platform

plt = platform.system()
if plt == 'Windows': pathlib.PosixPath = pathlib.WindowsPath
if plt == 'Linux': pathlib.WindowsPath = pathlib.PosixPath
import openai_ratelimit
from dotenv import load_dotenv
import os
import re

load_dotenv()

openai_ratelimit.api_key = os.getenv( 'OPENAI_API_KEY' )

# Load the LangChain.
index = faiss.read_index( "docs.index" )

with open( "faiss_store.pkl", "rb" ) as f:
    store = pickle.load( f )

store.index = index

# Set up prompt


chain = RetrievalQAWithClausesSourcesChain.from_llm( llm=ChatOpenAI( temperature=0 ),
                                                     document_prompt=EXAMPLE_PROMPT,
                                                     question_prompt=QUESTION_PROMPT,
                                                     combine_prompt=COMBINE_PROMPT,
                                                     vectorstore=store,
                                                     k=2,
                                                     # reduce_k_below_max_tokens=True,
                                                     # max_tokens_limit=500,
                                                     verbose=True )

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
        try:
            result = chain( {"question": user_input} )
            output = f"Answer: {result['answer']}\n\nClauses: {result['clauses']}"

            st.session_state.past.append( user_input )
            st.session_state.generated.append( output )

            clauses = [s.strip() for s  in result['clauses'].split(',')]

            for i, h in enumerate( result['sources'].split( ',' ) ):
                HtmlFile = open( h.replace( '.md', '.htm' ).strip(), encoding="utf8", errors='ignore' )
                source_code = HtmlFile.read()

                try:
                    clause = clauses[i].split('clause ')[-1].replace(']','')
                    my_script = """
                        <script>
                        document.querySelector('[title="st.iframe"]').onload = function() {
                        // Get all the div elements with the class "myClass"
                        var divElements = document.querySelectorAll('.leftpanel');
        
                        // Loop through each div element
                        for (var i = 0; i < divElements.length; i++) {
                            // Check if the div element contains the text "myText"
                            if (divElements[i].textContent.includes('{clause}')) {
                            // Found the div element, scroll to it
                            divElements[i].scrollIntoView();
                            break; // Exit the loop
                            }
                        }
                        }
                        </script>"""
                except IndexError as e:
                    my_script = ''
                with results_tabs[i]:
                    html( source_code + my_script, width=None, height=800, scrolling=True )
        except:
            output = f"Answer: Sorry, I am too stupid to understand your question. Could you please ask again?"

            st.session_state.past.append( user_input )
            st.session_state.generated.append( output )

    if st.session_state["generated"]:

        for i in range( len( st.session_state["generated"] ) - 1, -1, -1 ):
            message( st.session_state["generated"][i], key=str( i ) )
            message( st.session_state["past"][i], is_user=True, key=str( i ) + "_user" )
