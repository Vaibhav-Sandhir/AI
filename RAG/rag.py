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
from langsmith import traceable
from langsmith.evaluation import LangChainStringEvaluator, evaluate


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
    
    @traceable
    def retrieve_docs(self, question):
        return self.retriever.invoke(question)
    
    @traceable
    def invoke_llm(self, question, docs):
        formatted_docs = self.format_docs(docs)
        prompt = (
            "You are a helpful AI research assistant with expertise in evaluation frameworks and metrics of Large Language Models. "
            "Use the following documents to produce an answer to the user question.\n\n"
            f"## Docs\n\n{formatted_docs}\n\n"
            f"## Question\n\n{question}"
        )
        response = self.llm(prompt = prompt)
        return {"answer": response, "content": [str(doc) for doc in docs]}
    
    @traceable()
    def get_answer(self, question: str):
        docs = self.retrieve_docs(question)
        response = self.invoke_llm(question, docs)
        return response
    
    def predict_rag_answer_with_context(self, example: dict):
        """Use this for evaluation of retrieved documents and hallucinations"""
        response = self.get_answer(example["question"])
        return {"answer" : response["answer"], "content": response["content"]}
    
    def evaluation(self):
        docs_relevance_evaluator = LangChainStringEvaluator(
            "score_string",
            config = {
                "llm": self.llm,
                "criteria": {
                    "accuracy": """The Assistant's Answer is a set of documents retrieved from a vectorstore. The input is a question
                    used for retrieveal. You will score whether the Assistant's answer {retrieved docs} are releveant to the input
                    question. A score of [[0]] means that the Assistant answer contains documents that are not at all relevant to the
                    input questions. A score of [[5]] means that the Assistant answer contains some documents are relevant to the
                    input question. A score of [[10]] means that all of the Assistant answer documents are all relevant to the input
                    question"""
                },
                "normalize_by": 10,
            },
            prepare_data = lambda run, example: {
                "prediction":run.outputs["content"],
                "input": example.inputs["question"],
            },
        )
        dataset_name = "example"
        experiment_results = evaluate(
            self.predict_rag_answer_with_context,
            data = dataset_name,
            evaluators = [docs_relevance_evaluator],
        )
    

    

