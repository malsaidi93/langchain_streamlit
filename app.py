import os
import streamlit as st
from langchain.llms import OpenAI
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.chains.question_answering import load_qa_chain



# Set your OpenAI API key
OPENAI_API_KEY = "key"
PDF_DIR = os.path.join(os.getcwd(),'pdfs')
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Load pdf file and read it into pages
def load_pdf(file_path):
    loader = PyPDFLoader(file_path)
    return loader.load_and_split()

# Create vector database
def create_vector_database(pages):
    embeddings = OpenAIEmbeddings()
    vdb = FAISS.from_documents(pages, embeddings)
    return vdb

# Load QA chain
def load_qa_chain_instance():
    return load_qa_chain(OpenAI(temperature=0))


def main():
    st.title("Simple Question Answering App")

    # File upload
    uploaded_file = st.file_uploader("Choose a PDF file")
    
    # st.write("Loading: ", uploaded_file)
    if uploaded_file is not None:
        file_path = os.path.join(PDF_DIR,uploaded_file.name)
        st.write('Reading PDF file: ', file_path)
        # Read PDF file into pages
        pages = load_pdf(file_path)
        
        # Create vector database
        vdb = create_vector_database(pages)
        
        # Load QA chain
        chain = load_qa_chain_instance()

        # User input
        user_question = st.text_input("Ask a question:")

        # Button to trigger question answering
        
        if st.button("Ask"):
            # Query
            # @st.cache
            s = vdb.similarity_search(user_question)
            answer = chain.run(input_documents=s, question=user_question)

            # Display the answer
            st.subheader("Answer:")
            st.write(answer)

if __name__ == "__main__":
    main()
