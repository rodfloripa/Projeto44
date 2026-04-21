
<p align="justify"><h1>Graph RAG Híbrido (FAISS + Neo4j + RAGAS)</h1></p>

<p align="justify">Este projeto implementa um sistema de <b>Retrieval-Augmented Generation (RAG) híbrido</b>, combinando busca vetorial, grafos de conhecimento e avaliação automática de qualidade. O objetivo é aumentar a precisão e a capacidade de raciocínio de sistemas baseados em LLM ao unir duas abordagens complementares de recuperação de informação: <b>similaridade semântica</b> e <b>relações estruturadas</b>.</p>

<p align="justify">O sistema utiliza <b>FAISS</b> para busca vetorial em embeddings de documentos, <b>Neo4j</b> para modelagem de relações entre entidades e <b>RAGAS</b> para avaliação quantitativa das respostas geradas pelo modelo.</p>

Para rodar:
Descompacte neo4j_data.tar.gz
Insira sua chave OpenAI no .env
> docker-compose up --build

<p align="justify"><h3>1. Visão geral do sistema</h3></p>

<p align="justify">O pipeline do sistema segue a seguinte estrutura:</p>

<p align="center">
PDF → Chunking → FAISS (Vector Search) + Neo4j (Graph) → LLM → Resposta → RAGAS
</p>

<p align="justify">A principal ideia é combinar contexto semântico (vetores) com contexto estrutural (grafo), permitindo respostas mais robustas e com menor taxa de alucinação.</p>

<p align="justify"><h3>2. Carregamento e preparação do documento</h3></p>

<p align="justify">O sistema inicia carregando um documento PDF e dividindo o texto em partes menores (chunks) para melhorar a eficiência da recuperação de informação.</p>

```python
loader = PyPDFLoader("app/docs/doc.pdf")
raw_docs = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

docs = splitter.split_documents(raw_docs)
````

<p align="justify">Essa etapa garante que o modelo trabalhe com unidades de informação menores e mais relevantes para a busca semântica.</p>

<p align="justify"><h3>3. Busca vetorial com FAISS</h3></p>

<p align="justify">A busca vetorial é responsável por encontrar trechos semanticamente semelhantes à pergunta do usuário, mesmo quando não há correspondência exata de palavras.</p>

```python
vectorstore = FAISS.from_documents(docs, embeddings)
```

<p align="justify">Essa abordagem é altamente eficaz para consultas em linguagem natural, mas não captura relações explícitas entre entidades.</p>

<p align="justify"><h3>4. Construção do grafo com Neo4j</h3></p>

<p align="justify">Além da busca vetorial, o sistema constrói um grafo de conhecimento simples utilizando Neo4j. As entidades são extraídas diretamente dos textos e armazenadas como nós.</p>

```python
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
```

<p align="justify">Esse grafo permite consultas baseadas em relações e melhora a capacidade do sistema de realizar raciocínio multi-hop.</p>

<p align="justify"><h3>5. Busca no grafo</h3></p>

<p align="justify">A busca no grafo complementa a busca vetorial ao recuperar entidades relacionadas à consulta do usuário.</p>

```python
def graph_search(query):
    with driver.session() as session:
        result = session.run("""
            MATCH (n:Entity)
            WHERE toLower(n.name) CONTAINS toLower($q)
            RETURN n.name LIMIT 5
        """, q=query)

        return [r["n.name"] for r in result]
```

<p align="justify">Essa etapa ajuda a trazer contexto estrutural que não seria capturado apenas por embeddings.</p>

<p align="justify"><h3>6. RAG híbrido (combinação final)</h3></p>

<p align="justify">A etapa central do sistema combina o contexto vetorial e o contexto do grafo para gerar a resposta final do modelo.</p>

```python
def rag_graph(q):
    docs = vectorstore.similarity_search(q, k=2)
    ctx = " ".join([d.page_content for d in docs])

    graph_ctx = graph_search(q)
    graph_text = " ".join(graph_ctx)

    final_ctx = ctx + "\nGraph: " + graph_text

    return llm.invoke(build_prompt(final_ctx, q)).content, final_ctx
```

<p align="justify">Essa fusão permite que o modelo utilize tanto similaridade semântica quanto relações explícitas, aumentando a qualidade das respostas.</p>

<p align="justify"><h3>7. Avaliação com RAGAS</h3></p>

<p align="justify">O sistema utiliza o RAGAS para avaliar automaticamente a qualidade das respostas geradas pelo modelo.</p>

```python
result = evaluate(
    dataset,
    metrics=[
        faithfulness,
        answer_relevancy,
        context_precision
    ]
)
```

<p align="justify">As métricas avaliam consistência, relevância da resposta e qualidade do contexto recuperado.</p>

<p align="justify"><h3>8. Comparação conceitual</h3></p>

<p align="justify"><b>Vector RAG</b> é eficiente para similaridade semântica, mas não entende relações estruturais. <b>Graph RAG</b> é eficiente para relações e raciocínio estruturado, mas limitado em linguagem natural. O modelo híbrido combina os dois para obter maior precisão e robustez.</p>

## Resultados e Conclusão

**RESULTADO FINAL:**
- Faithfulness: 0.7542
- Answer Relevancy: 0.8171
- Context Precision: 0.8500

O sistema implementa um pipeline completo de <b>RAG híbrido</b>, combinando recuperação vetorial, grafos de conhecimento e LLMs para gerações mais precisas e explicáveis.


