from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders.parsers import TesseractBlobParser
import os

file_path = os.environ["FILE_DIR"]

image_parser = TesseractBlobParser()

#### Loading PDF with multi-modal data

loader = PyMuPDFLoader(
    file_path=file_path,
    mode="page",
    images_parser= image_parser,
    images_inner_format= "html-img",
    extract_tables="markdown", # optional. takes a lot of time

)

docs = loader.load()

# Cleaning headers with RE, since the source document needs it
import re

def clean_page_text(text: str) -> str:
    text = re.sub(r'^THE ULTIMATE USED CAR BUYING GUIDE\s+\d+\s*\n?', '', text, flags=re.IGNORECASE | re.MULTILINE)
    text = re.sub(r'^©\s*2016\s*AUTO\s*CITY\s*\n?', '', text, flags=re.IGNORECASE | re.MULTILINE)
    text = re.sub(r'^WWW\.GOAUTOCITY\.COM\s*\n?', '', text, flags=re.IGNORECASE | re.MULTILINE)
    return text.strip()

for doc in docs:
    doc.page_content = clean_page_text(doc.page_content)

print("\nLoaded the data")

#### Utilizing Recursive Character Text Splitter so we achieve overlapping of texts between chunks


from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,#hyperparameter
    chunk_overlap=50 #hyperparemeter
)

print("\nChunking started")

docs = text_splitter.split_documents(docs)

print("\nChunking is done")

#### Load documents into the vector store


from memory.vector_store import flat_milvus_vector_store

print("\nUploading the chunked docs to the vector store")
flat_milvus_vector_store.add_documents(
    documents=docs
)

vector_store_retriever = flat_milvus_vector_store.as_retriever()

print("\nVector Store is ready")


#### Load documents into the Keyword Search BM25


from memory.BM25_keyword_search import BM25_retriever

print("\nUploading the chunked docs to the BM25 Keyword Search")

BM25_retriever= BM25_retriever(docs=docs)

print("\nBM25 Keyword Search is ready")

