# Install the required packages.
# pip install langchain-chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_chroma import Chroma
import os
import dotenv

dotenv.load_dotenv()


def main():
    # Load all text files from the 'planets/' directory
    loader = DirectoryLoader("planets", glob="*.txt", loader_cls=TextLoader)
    documents = loader.load()

    # Initialize HuggingFace embeddings model.
    embeddings_model = HuggingFaceInferenceAPIEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        api_key=os.getenv("HUGGINGFACE_API_KEY")
    )

    # Create a Chroma vector store from the loaded documents and HuggingFace embeddings.
    db = Chroma.from_documents(documents, embeddings_model)

    # Perform a similarity search on the vector store with a sample query.
    query = input()  #"What makes Saturn unique?"
    docs = db.similarity_search(query)

    # Print the content of the most similar document.
    print(docs[0].page_content)


if __name__ == "__main__":
    main()
