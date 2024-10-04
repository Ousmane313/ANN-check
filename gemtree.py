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
warnings.filterwarnings('ignore')
class SuppressStdout:
    def __enter__(self):
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr
urls=["https://www.financialafrik.com/category/finance/","https://www.financialafrik.com/",
      "https://www.sikafinance.com/marches/actualites_bourse_brvm","https://www.sikafinance.com/marches/communiques_brvm"]
#urls=["https://finance.yahoo.com/news/","https://www.bloomberg.com/economics","https://www.wsj.com/finance?mod=nav_top_section",
      #"https://www.aljazeera.com/news/","https://edition.cnn.com/world","https://www.foxnews.com/world",
     # "https://www.reuters.com/world/africa/","https://www.cnbc.com/world/?region=world","https://www.cnbc.com/markets/","https://www.cnbc.com/investing/"],
urlss=["https://finance.yahoo.com/news/","https://www.bloomberg.com/economics","https://www.wsj.com/finance?mod=nav_top_section",
      "https://www.foxnews.com/world","https://www.cnbc.com/markets/","https://www.cnbc.com/investing/"]

documents = BeautifulSoupWebReader().load_data(urls)

os.environ['GOOGLE_API_KEY'] = "AIzaSyDlfAbclN-xZYR8dkP_Ex3shqWEM9GKuhg"
genai.configure(api_key = os.environ['GOOGLE_API_KEY'])
#index = SummaryIndex.from_documents(documents,use_async=True)
llm = Gemini()
model_name = "models/embedding-001"

embed_model = GeminiEmbedding(
    model_name=model_name, api_key="AIzaSyDlfAbclN-xZYR8dkP_Ex3shqWEM9GKuhg", title="this is a document"
)

embeddings = embed_model.get_text_embedding("Google Gemini Embeddings.")
Settings.embed_model = embed_model
Settings.llm=Gemini()
splitter = SentenceSplitter(chunk_size=1024)
#splitter = SentenceSplitter(chunk_size=512)
response_synthesizer = get_response_synthesizer(
    response_mode="tree_summarize", use_async=True
)
doc_summary_index = DocumentSummaryIndex.from_documents(
    documents,
    llm=llm,
    transformations=[splitter],
    response_synthesizer=response_synthesizer,
    show_progress=True,
)
doc_summary_index.storage_context.persist("index")
# rebuild storage context
storage_context = StorageContext.from_defaults(persist_dir="index")
doc_summary_index = load_index_from_storage(storage_context)
query_engine = doc_summary_index.as_query_engine(
    response_mode="tree_summarize", use_async=True
)
 

    
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