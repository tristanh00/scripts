import os
import pandas as pd
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain import OpenAI, VectorDBQA
from langchain.chains import RetrievalQA
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import OpenAI
from langchain.vectorstores import Pinecone
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
import pinecone

os.environ["OPENAI_API_KEY"] = ''

pinecone.init(
    api_key="",
    environment="asia-southeast1-gcp-free"
)

index_name="articles"
# Read the cleaned CSV file
# cleaned_csv_file = 'sanitized_output.csv' # Replace with your cleaned CSV file name
# df = pd.read_csv(cleaned_csv_file)

# Concatenate the content of the 'article_body' column
# df['article_body'] = df['article_body'].astype(str)

# from langchain.document_loaders import TextLoader
# loader = TextLoader("sanitized_output.csv")
# documents = loader.load()
# text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
# texts = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()

# doc_store = Pinecone.from_texts(
#     [d.page_content for d in texts], 
#     embeddings, 
#     index_name=index_name
# )

# docsearch = Chroma.from_documents(texts, embeddings)
# qa = RetrievalQA.from_chain_type(llm=OpenAI(temperature=0.1,max_tokens=500), chain_type="stuff", retriever=docsearch.as_retriever())

index = Pinecone.from_existing_index(index_name=index_name, embedding=embeddings)

docs = index.similarity_search('Who is X?', include_metadata = True, k=1)

print(docs['page_content'])
