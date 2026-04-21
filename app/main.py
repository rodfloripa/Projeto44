import os
import time
import pandas as pd

from neo4j import GraphDatabase

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision

# =========================
# OPENAI
# =========================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Missing OPENAI_API_KEY")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
embeddings = OpenAIEmbeddings()

# =========================
# NEO4J (WITH RETRY FIX)
# =========================
driver = None

def get_driver():
    global driver

    if driver:
        return driver

    for i in range(15):
        try:
            print(f"⏳ Conectando Neo4j... tentativa {i+1}/15")

            driver = GraphDatabase.driver(
                "bolt://neo4j:7687",
                auth=("neo4j", "test1234")
            )

            with driver.session() as session:
                session.run("RETURN 1")

            print("✅ Neo4j conectado!")
            return driver

        except Exception as e:
            print("❌ Neo4j não pronto:", e)
            time.sleep(3)

    raise Exception("Neo4j não iniciou")

# =========================
# LOAD PDF
# =========================
PDF_PATH = "app/docs/doc.pdf"

loader = PyPDFLoader(PDF_PATH)
raw_docs = loader.load()

# =========================
# SPLIT
# =========================
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

docs = splitter.split_documents(raw_docs)

# =========================
# VECTOR STORE (FAISS)
# =========================
vectorstore = FAISS.from_documents(docs, embeddings)

# =========================
# GRAPH BUILD
# =========================
def build_graph(docs):
    driver = get_driver()

    with driver.session() as session:
        for d in docs:
            words = d.page_content.split()
            entities = [w for w in words if w.istitle()]

            for e in entities:
                session.run(
                    "MERGE (n:Entity {name: $name})",
                    name=e
                )

# =========================
# GRAPH SEARCH
# =========================
def graph_search(query):
    driver = get_driver()

    with driver.session() as session:
        result = session.run(
            """
            MATCH (n:Entity)
            WHERE toLower(n.name) CONTAINS toLower($q)
            RETURN n.name LIMIT 5
            """,
            q=query
        )
        return [r["n.name"] for r in result]

# =========================
# PROMPT
# =========================
def build_prompt(ctx, q):
    return f"""
Use only the context below.

Context:
{ctx}

Question:
{q}

Answer:
"""

# =========================
# GRAPH RAG
# =========================
def rag_graph(q):
    docs = vectorstore.similarity_search(q, k=2)
    ctx = " ".join([d.page_content for d in docs])

    graph_ctx = graph_search(q)
    graph_text = " ".join(graph_ctx)

    final_ctx = ctx + "\nGraph: " + graph_text

    return llm.invoke(build_prompt(final_ctx, q)).content, final_ctx

# =========================
# QA DATASET (SEU ZENML)
# =========================
qa_pairs = [
    (
        "O que é o ZenML?",
        "É um framework MLOps de código aberto projetado para facilitar a criação e o gerenciamento de pipelines de Machine Learning de forma organizada e reproduzível."
    ),
    (
        "Qual é o papel do ZenML na infraestrutura de ML?",
        "Ele atua como a cola que conecta as diferentes partes da infraestrutura de ML, lidando com versionamento de pipelines e gerenciamento de artefatos."
    ),
    (
        "Quais são os principais benefícios de escolher o ZenML citados no texto?",
        "Ele é agnóstico a frameworks, nativo da nuvem, extensível através de plugins e focado em reprodutibilidade."
    ),
    (
        "O que significa dizer que o ZenML é Framework Agnostic?",
        "Significa que ele funciona com TensorFlow, PyTorch e scikit-learn sem acoplamento a um único framework."
    ),
    (
        "Qual é o requisito de versão do Python para utilizar o ZenML?",
        "Python 3.7 ou superior."
    ),
    (
        "Como se instala o ZenML via terminal?",
        "pip install zenml"
    ),
    (
        "Para que serve o comando zenml init?",
        "Inicializa o ZenML no projeto e cria a estrutura de configuração do pipeline."
    ),
    (
        "Qual comando é utilizado para abrir a interface visual do ZenML?",
        "zenml up"
    ),
    (
        "O que são Steps no ZenML?",
        "São funções Python decoradas com @step que executam tarefas específicas dentro do pipeline."
    ),
    (
        "O que é um Pipeline no ZenML?",
        "É uma coleção de steps que formam um fluxo de trabalho de Machine Learning."
    ),
    (
        "Qual é a função do decorador @pipeline?",
        "Define a estrutura do pipeline e conecta os steps."
    ),
    (
        "O que acontece com a saída de cada step?",
        "A saída vira um artefato versionado automaticamente pelo ZenML."
    ),
    (
        "Quais são as etapas do pipeline de exemplo?",
        "Carregamento de dados, pré-processamento, treinamento e avaliação."
    ),
    (
        "Qual dataset é usado no exemplo?",
        "Iris dataset do sklearn."
    ),
    (
        "Qual modelo é usado no exemplo?",
        "RandomForestClassifier."
    ),
    (
        "O que compõe uma Stack no ZenML?",
        "Orquestradores, storage de artefatos e tracking de experimentos."
    ),
    (
        "Como listar runs de pipeline?",
        "zenml pipeline runs list"
    ),
    (
        "É possível desativar cache no ZenML?",
        "Sim, usando enable_cache=False no decorator do pipeline."
    ),
    (
        "Boas práticas de dependências no ZenML?",
        "Usar requirements.txt ou pyproject.toml com versões fixas."
    ),
    (
        "Boas práticas sobre steps?",
        "Manter cada step com uma única responsabilidade."
    )
]

# =========================
# RUN PIPELINE
# =========================
rows = []

build_graph(docs)

for q, gt in qa_pairs:
    a, c = rag_graph(q)

    rows.append({
        "question": q,
        "answer": a,
        "contexts": [c],
        "ground_truth": gt
    })

dataset = Dataset.from_pandas(pd.DataFrame(rows))

# =========================
# RAGAS EVALUATION
# =========================
result = evaluate(
    dataset,
    metrics=[
        faithfulness,
        answer_relevancy,
        context_precision
    ]
)

print("\nRESULTADO FINAL:\n")
print(result)
