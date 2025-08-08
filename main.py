from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware  # Import necessário
from pydantic import BaseModel
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
import os

# ✅ Criar o app primeiro
app = FastAPI()

# ✅ Configurar CORS depois que o app foi criado
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ou substitua por ["http://localhost:3000"] no caso do React
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Carregar PDF
caminho = r"C:\Projetos\team-insurances\processamento-batch.pdf"

if os.path.isfile(caminho):
    loader = PyPDFLoader(caminho)
    documents = loader.load()
    print(f"Documento carregado com {len(documents)} páginas.")
else:
    print(f"Arquivo não encontrado no caminho: {caminho}")
    exit(1)

# Dividir texto
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = splitter.split_documents(documents)

# Embeddings e indexação
embedding = OllamaEmbeddings(model="nomic-embed-text")
vectorstores = FAISS.from_documents(docs, embedding)
vectorstores.save_local("user_stories_index")

# LLM e QA chain
llm = Ollama(model="gemma3:1b")
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstores.as_retriever())

# Modelo da requisição
class Pergunta(BaseModel):
    pergunta: str

# Rota da API
@app.post("/perguntar")
def responder(pergunta: Pergunta):
    resposta = qa_chain.run(pergunta.pergunta)
    return {"resposta": resposta}
