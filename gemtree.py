import sys
import os
import streamlit as st
import google.generativeai as genai
from llama_index.llms.gemini import Gemini
#from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings,DocumentSummaryIndex,SummaryIndex,VectorStoreIndex, SimpleDirectoryReader, get_response_synthesizer,load_index_from_storage,StorageContext
from llama_index.readers.web import SimpleWebPageReader
from IPython.display import Markdown, display
import warnings
import re
import time
from llama_index.readers.web import AsyncWebPageReader, BeautifulSoupWebReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
#from llama_index import GPTTreeIndex
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core import PromptTemplate
from llama_index.core.node_parser import SentenceSplitter
# imports
from llama_index.embeddings.gemini import GeminiEmbedding




    
with SuppressStdout():

# Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role":"assistant",
                "content":"Posez vos questions sur l'actualité"
            }
        ]

# Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Process and store Query and Response
    def llm_function(query):
        #retriever = vectorstore.similarity_search(query)
        response = query_engine.query(query)
    
        # Displaying the Assistant Message
        with st.chat_message("assistant"):
            st.markdown(response)
# Storing the User Message
        st.session_state.messages.append(
            {
                "role":"user",
                "content": query
            }
        )

    # Storing the User Message
        st.session_state.messages.append(
            {
                "role":"assistant",
                "content": response
            }
        )

# Accept user input
    query = st.chat_input("Bonjour!")

# Calling the Function when Input is Provided
    if query:
    # Displaying the User Message
        with st.chat_message("user"):
            st.markdown(query)

        llm_function(query)
