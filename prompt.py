# flake8: noqa
from langchain.prompts import PromptTemplate

question_prompt_template = """Use the following portion of a long document to see if any of the text is relevant to answer the question. 
Return any relevant text verbatim.
{context}
Question: {question}
Relevant text, if any:"""
QUESTION_PROMPT = PromptTemplate(
    template=question_prompt_template, input_variables=["context", "question"]
)

combine_prompt_template = """I want you to act as a professional traffic engineer and explain like I am Five.
Given the following extracted parts of a long document and a question, create a final answer with references ("SOURCES").
If you don't know the answer, just say that you don't know. Don't try to make up an answer.
Extract the relevant "CLAUSE" number from the "SOURCES" text. 
The clause number should be in the format of "X.X.X.X", where X is integer.
The clause number should be found in the beginner part of a sentence.
The "SOURCES" and "CLAUSES" part should be beautified from the path like string to a formatted string.
Example is like this:
original: "Notion_DB\TPDM\v4\c2\2_3.md, clause 2.3.1.1"
CLAUSES: "[Notion_DB\TPDM\v4\c2\2_3.md, clause 2.3.1.1]"
SOURCES: "Notion_DB\TPDM\v4\c2\2_3.md"
Seperate each source and clause with a newline character.
Append the "CLAUSES" and the "SOURCES" at the end of the answer with the format "CLAUSES:("CLAUSES")\nSOURCES:("SOURCES")".


QUESTION: {question}
=========
{summaries}
=========
FINAL ANSWER:"""
COMBINE_PROMPT = PromptTemplate(
    template=combine_prompt_template, input_variables=["summaries", "question"]
)

EXAMPLE_PROMPT = PromptTemplate(
    template="Content: {page_content}\nSource: {source}",
    input_variables=["page_content", "source"],
)
