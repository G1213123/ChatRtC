"""Python file to serve as the frontend"""
import streamlit as st
from streamlit_chat import message
import streamlit.components.v1 as components
import faiss
from langchain import OpenAI
from langchain.chains import VectorDBQAWithSourcesChain
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

openai_ratelimit.api_key = os.getenv('OPENAI_API_KEY')

# Load the LangChain.
index = faiss.read_index("docs.index")

with open("faiss_store.pkl", "rb") as f:
    store = pickle.load(f)

store.index = index
chain = VectorDBQAWithSourcesChain.from_llm(llm=OpenAI(temperature=0), vectorstore=store, verbose= True)


# From here down is all the StreamLit UI.
st.set_page_config(page_title="Blendle Notion QA Bot", page_icon=":robot:", layout="wide")


if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []

def get_text():
    input_text = st.text_input("You: ", "What is pcu?", key="input")
    return input_text

def beautify_source_name(name):
    out = []
    for n in name.split(','):
        volume = re.search(r"(?<=TPDM\\v)(\d+)(?=\\c)", n).group(0)
        chapter, section = re.search('\d+_\d+',n).group(0).split('_')
        out.append(f"Volume {volume}\nChapter {chapter}.{section}")
    return "\n\n".join(out)

def get_full_name(name):
    volume = re.search( r"(?<=TPDM\\v)(\d+)(?=\\c)", name ).group( 0 )
    chapter, section = re.search( '\d+_\d+', name ).group( 0 ).split( '_' )
    return name.replace(f'{chapter}_{section}', f'Volume{volume}_Chapter{chapter}{section}')

results_tabs = st.tabs( ['result1', 'result2', 'result3'] )

with st.sidebar:
    st.title("Blendle Notion QA Bot", )
    user_input = get_text()

    if user_input:
        result = chain({"question": user_input})
        output = f"Answer: {result['answer']}\nSources:\n{beautify_source_name(result['sources'])}"

        st.session_state.past.append(user_input)
        st.session_state.generated.append(output)

        for i,h in enumerate(result['sources'].split(',')):
            HtmlFile = open( h.replace('.md','.htm').strip(), encoding="utf8", errors='ignore')
            source_code = HtmlFile.read()
            highlighted = " ".join( f"<mark>{t}</mark>" if t in result['answer'] else t for t in source_code.split() )
            with results_tabs[i]:
                components.html( source_code ,width=None, height=800, scrolling=True)

    if st.session_state["generated"]:

        for i in range(len(st.session_state["generated"]) - 1, -1, -1):
            message(st.session_state["generated"][i], key=str(i))
            message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")