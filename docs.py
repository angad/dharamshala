import os
from pprint import pprint

from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

def print_color(text, color):
    print("\033[38;5;{}m{}\033[0m".format(color, text))

def load_documents(folder):
    def filetree(folder):
        return [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(folder)) for f in fn]

    texts = []
    for file in filetree(folder):
        if file.endswith(".pdf"):
            print_color("loading " + file, 196)
            loader = PyPDFLoader(file)
            documents = loader.load_and_split()
            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
            texts.extend(text_splitter.split_documents(documents))
    return texts

def main():
    embeddings = OpenAIEmbeddings()
    db = Chroma.from_documents(load_documents("docs"), embeddings, persist_directory="docs.db")
    retriever = db.as_retriever()
    qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=retriever, return_source_documents=True)
    while True:
        question = input("Question: ")
        result = qa({"query": question})
        print_color("Answer: " + result["result"], 46)
        if os.environ["DEBUG"]:
            print_color("Source documents:", 46)
            pprint(result["source_documents"])

if __name__ == '__main__':
    main()
