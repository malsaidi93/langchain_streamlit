from imports import *

# Set your OpenAI API key
OPENAI_API_KEY = "key"
# PDF_DIR = os.path.join(os.getcwd(),'pdfs')
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


def load_pdf(file_path):
    loader = PyPDFLoader(file_path)
    return loader.load_and_split()

def create_vector_database(pages):
    embeddings = OpenAIEmbeddings()
    vdb = FAISS.from_documents(pages, embeddings)
    return vdb

def load_qa_chain_instance():
    return load_qa_chain(OpenAI(temperature=0))

def main():
    parser = argparse.ArgumentParser(description="Question Answering Bot for PDF files")
    parser.add_argument("-p", "--pdf_path", help="Path to the PDF file", required=True)
    args = parser.parse_args()

    pdf_path = args.pdf_path

    # Load pdf file and read it into pages
    pages = load_pdf(pdf_path)

    # Create vector database
    vdb = create_vector_database(pages)

    # Load QA chain
    chain = load_qa_chain_instance()

    # Query
    while True:
        query = input("Ask a question: ")
        if query.lower() in ['q', 'quit']:
            break

        s = vdb.similarity_search(query)
        answer = chain.run(input_documents=s, question=query)
        # Display the answer
        print("Answer:", answer)

if __name__ == "__main__":
    main()
