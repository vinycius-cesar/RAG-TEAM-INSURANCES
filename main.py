from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
import os

# Extrair textos do PDF
caminho = r"C:\Projetos\team-insurances\user_stories.pdf"

if os.path.isfile(caminho):
    loader = PyPDFLoader(caminho)
    documents = loader.load()
    print(f"Documento carregado com {len(documents)} páginas.")
else:
    print(f"Arquivo não encontrado no caminho: {caminho}")
    exit(1)

# Dividir em chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = splitter.split_documents(documents)

# Usar modelo correto para embeddings (NÃO é o gemma3)
embedding = OllamaEmbeddings(model="nomic-embed-text")
vectorstores = FAISS.from_documents(docs, embedding)
vectorstores.save_local("user_stories_index")

# Usar o gemma3:1b apenas como LLM para respostas
llm = Ollama(model="gemma3:1b")
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstores.as_retriever())

# Criar API
app = FastAPI()

class Pergunta(BaseModel):
    pergunta: str

@app.post("/perguntar")
def responder(pergunta: Pergunta):
    resposta = qa_chain.run(pergunta.pergunta)
    return {"resposta": resposta}
