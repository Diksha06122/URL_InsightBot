# URL Insight Bot🤖

## Overview
URL Insight Bot is a Streamlit-based application that processes and analyzes news articles from provided URLs. It leverages advanced language models and embeddings to answer user queries about the content of these articles. This project uses various Python libraries, including LangChain, HuggingFace, and FAISS, to create an efficient and user-friendly tool for information retrieval and question answering.

## Features
- **User-friendly Interface**: Streamlit provides an interactive interface for users to input URLs and ask questions.
- **Background Image and Custom Styling**: Custom CSS for a visually appealing UI.
- **Data Loading and Text Splitting**: Loads and splits text from the provided URLs for efficient processing.
- **Embeddings and Vector Store**: Uses HuggingFace embeddings and FAISS for creating and storing vector representations of the text.
- **Question Answering**: Utilizes the ChatGoogleGenerativeAI model for answering user queries based on the processed data.

## Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/url-insight-bot.git
    cd url-insight-bot
    ```

2. Create and activate a virtual environment:
    ```bash
    python3 -m venv env
    source env/bin/activate  # On Windows, use `env\Scripts\activate`
    ```

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. Set up environment variables:
    - Create a `.env` file in the root directory.
    - Add your Google API key:
        ```
        GOOGLE_API_KEY=your_api_key
        ```

## Usage
1. Run the Streamlit application:
    ```bash
    streamlit run app.py
    ```

2. Open your web browser and navigate to the local Streamlit server URL.

3. Enter the URLs of the news articles you want to analyze in the sidebar.

4. Click the "Process URLs" button to start processing.

5. Once the URLs are processed, you can enter your questions about the content in the main input box.

6. The application will display the answers and the sources of information.

## Code Explanation
### Main Function
The `main` function sets up the Streamlit interface and manages user interaction:
```python
def main():
    """Manages the Streamlit application and user interaction."""
    st.set_page_config(page_icon=":Dragon:")
    st.markdown("""
    <style>
    # Custom CSS for styling
    ...
    </style>
    """, unsafe_allow_html=True)

    st.title("URL Insight Bot🤖")
    st.sidebar.title("News Article URLs")

    urls = []
    for i in range(2):
        url = st.sidebar.text_input(f"URL {i+1}")
        urls.append(url)

    process_url_clicked = st.sidebar.button("Process URLs")
    file_path = "faiss_store.pkl"
    main_placeholder = st.empty()

    if process_url_clicked:
        # Load and process data
        loader = UnstructuredURLLoader(urls=urls)
        main_placeholder.text("Data Loading...Started...✅✅✅")
        data = loader.load()

        # Split data into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            separators=['\n\n', '\n', '.', ','],
            chunk_size=2000,
            chunk_overlap=300
        )
        main_placeholder.text("Text Splitter...Started...✅✅✅")
        docs = text_splitter.split_documents(data)
        main_placeholder.text("Text Splitter...finished...✅✅✅")

        # Create embeddings and save to FAISS index
        model_name = "sentence-transformers/all-mpnet-base-v2"
        model_kwargs = {'device': 'cpu'}
        encode_kwargs = {'normalize_embeddings': False}
        embedding = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        main_placeholder.text("Creating embeddings...Started...✅✅✅")
        vector_store = FAISS.from_documents(docs, embedding)
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
            st.header("Answer")
            st.write(result["answer"])

            sources = result.get("sources", "")
            if sources:
                st.subheader("Sources:")
                sources_list = sources.split("\n")
                for source in sources_list:
                    st.write(source)
```


