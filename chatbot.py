import os
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain import OpenAI

# load Openai api key
# os.environ["OPENAI_API_KEY"] = ""
from dotenv import load_dotenv
load_dotenv()

# Load documents
loader = TextLoader("./docs/3students-story.txt")
documents = loader.load()

# split into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
text_chunks = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()

# create chroma db
# vector_db = Chroma.from_documents(
#     documents=text_chunks,
#     embedding=embeddings,
#     collection_name="testcollection1",
#     persist_directory="./chroma_storage"
# )
# vector_db.persist()

# load chroma db
vector_db = Chroma(collection_name="testcollection1", persist_directory="./chroma_storage", embedding_function=embeddings)

qa = RetrievalQA.from_chain_type(llm=OpenAI(model_name="text-davinci-003"), retriever=vector_db.as_retriever())

question = "Who is the indian student in the story?"

result = qa({"query": question})
print("result ", result)