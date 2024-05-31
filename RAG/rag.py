import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.llms import Ollama
from langchain import hub
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import torch
import os

os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
os.environ['LANGCHAIN_API_KEY'] = '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'

class Agent:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size = 800, chunk_overlap = 200, length_function = len, is_separator_regex = False)
        self.embeddings = HuggingFaceEmbeddings()
        self.folder = "chroma_db"
        self.llm = Ollama(model = "llama3:8b")
        self.prompt = hub.pull("rlm/rag-prompt")
        self.vector_store = self.load_or_create_db()
        self.retriever = self.vector_store.as_retriever()
    
    def load_data(self):
        research_papers = ["papers/self-reflection.pdf",
                   "papers/squad.pdf",
                   "papers/mmlu.pdf",
                   "papers/llm-eval.pdf",
                   "papers/evaluation.pdf",
                   "papers/helm.pdf",
                   "papers/ragas.pdf",
                   "papers/inspector.pdf",
                   "papers/chainpoll.pdf",
                   "papers/bleurt.pdf",
                   "papers/bert.pdf",
                   "papers/ares.pdf",
                   "papers/meteor.pdf"]
        return [PyMuPDFLoader(paper) for paper in research_papers]
    
    def load_or_create_db(self):
        torch.cuda.empty_cache()
        if os.path.exists(self.folder) and os.listdir(self.folder):
            print("Loading Database")
            return Chroma(embedding_function=self.embeddings, persist_directory=self.folder)
        else:
            loaders = self.load_data()
            print("Populating Database")
            for loader in loaders:
                docs = loader.load()
                chunks = self.text_splitter.split_documents(docs)
                return Chroma.from_documents(documents = chunks, embedding = self.embeddings, persist_directory = self.folder)

    def format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    def ask(self, question):
        rag_chain = ({"context" : self.retriever | self.format_docs, "question" : RunnablePassthrough()}
                     | self.prompt
                     | self.llm
                     | StrOutputParser()
                     )
        answer = rag_chain.invoke(question)
        return answer
