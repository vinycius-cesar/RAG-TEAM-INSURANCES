from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import Ollama
from fastapi import FastAPI, Request
from pydantic import BaseModel

#Extrair textos do pdf
loader - PyPDFLoader("user_stories.pdf")
documents = loader.load()

#Dividir em chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=100)
docs = splitter.split_documents(documents)

#Criar indice vetorial FAISS
embedding - OllamaEmbeddings(model="llama2")
vectorstores = FAISS.from_documents(docs, embedding)
vectorstores.save_local("user_stories_index")

#Configurar pipeline RAG
llm = Ollama(model="llama2")
qa_chain = RetrievalQA.from_chain_type(llm = llm, retriever = vectorstores.as_retriever())

#Criar API local para o time
app = FastAPI()

class Pergunta(BaseModel):
    pergunta: str

@app.post("/perguntar")
def responder(pergunta: Pergunta):
    resposta = qa_chain.run(pergunta.pergunta)
    return {"resposta": resposta}

#digitar no prompt uvicorn main:app --reload --port 8000 para o time acessar http://<seu-ip>:8000/perguntar 


#SEGURANÇA DA APLICAÇÃO
#JWT #Para autenticação, você pode usar o seguinte comando curl para obter um token JWT
#Substitua 'user' e '123456' pelo seu nome de usuário e senha reais, e 'string' pelos valores corretos de client_id e client_secret.
curl --location 'https://tech-challenge-0xyz.onrender.com/token' \
--header 'accept: application/json' \
--header 'Content-Type: application/x-www-form-urlencoded' \
--data-urlencode 'grant_type=password' \
--data-urlencode 'username=user' \
--data-urlencode 'password=123456' \
--data-urlencode 'client_id=string' \
--data-urlencode 'client_secret=string'
