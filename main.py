
import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import os
import streamlit as st
import pickle
import time
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import UnstructuredURLLoader

from dotenv import load_dotenv
load_dotenv()  

# Load CSS with background image
def main():
    """Manages the Streamlit application and user interaction."""
    st.set_page_config( page_icon=":Dragon:")
    st.markdown("""
        <style>
        # .reportview-container {
        #     background: url("C:/Users/admin/OneDrive/Pictures/fin.jpg") center;
        #     background-size: cover;
        # }
        .stTextInput {
            border-radius: 10px;
            border: 2px solid #008CBA; /* Blue */
            padding: 10px;
            box-shadow: 0px 0px 10px #008CBA; /* Blue */
        }
        .stButton > button {
            background-color: #4CAF50; /* Green */
            border: none;
            color: white;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 16px;
            margin: 4px 2px;
            transition-duration: 0.4s;
            cursor: pointer;
            border-radius: 12px;
        }
        .stButton > button:hover {
            background-color: #45a049;
        }
        .dialogue-box {
            background-color: #f0f0f0; /* Light gray */
            border-radius: 10px;
            padding: 10px;
            margin-bottom: 10px;
            box-shadow: 0px 0px 10px #888888; /* Gray */
        }
        /* Change sidebar background color */
        .sidebar .sidebar-content {
        background-color: #2E3047; /* Change this color code to the desired color */
            }            /* #116466 or #2E3047 */
        }
        /* Change text color of the other side */
        .main .block-container {
            color: black;
        }
        </style>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

st.title("URL Insight Bot🤖")
st.sidebar.title("News Article URLs")
# Load CSS with background image

    # Rest of your Streamlit app code goes here
urls = []
for i in range(2):
    url = st.sidebar.text_input(f"URL {i+1}")
    print(url)
    urls.append(url)

print("URLs: ", urls)
print(type(urls[0]))

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store.pkl"
main_placeholder = st.empty()
os.environ["GOOGLE_API_KEY"] = "Your_API_key"

llm = ChatGoogleGenerativeAI(model="gemini-pro")
if process_url_clicked:
    # load data
    loader = UnstructuredURLLoader(urls=urls)
    print(urls)
    main_placeholder.text("Data Loading...Started...✅✅✅")
    data = loader.load()
    # split data
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=2000,
        chunk_overlap=300
    )
    main_placeholder.text("Text Splitter...Started...✅✅✅")
    docs = text_splitter.split_documents(data)
    print(len(data))
    main_placeholder.text("Text Splitter...finished...✅✅✅")
   
    print(docs)
    # create embeddings and save it to FAISS index
    model_name = "sentence-transformers/all-mpnet-base-v2"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    embedding = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    main_placeholder.text("Creating embeddings...Started...✅✅✅")

    

    vector_store=FAISS.from_documents(docs,embedding)
    main_placeholder.text("Embedding Vector Started Building...✅✅✅")
    time.sleep(2)

    # Save the FAISS index to a pickle file
    with open(file_path, "wb") as f:
        pickle.dump(vector_store, f)

query = main_placeholder.text_input("Question: ")
if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)
            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
            
            result = chain({"question": query}, return_only_outputs=True)
            # result will be a dictionary of this format --> {"answer": "", "sources": [] }
            st.header("Answer")
            st.write(result["answer"])

            # Display sources, if available
            sources = result.get("sources", "")
            if sources:
                st.subheader("Sources:")
                sources_list = sources.split("\n")  
                for source in sources_list:
                    st.write(source)
