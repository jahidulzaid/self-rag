import os
from datasets import load_dataset
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
from typing_extensions import TypedDict
import pandas as pd
from functools import partial
from langgraph.graph import StateGraph, START, END

BASE_URL = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text:v1.5"
LLM_MODEL = "gpt-oss:20b"

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def load_squad_documents(split="train[:200]"):
    dataset = load_dataset("rajpurkar/squad_v2", split=split)
    documents = []

    for item in dataset:
        query = item["question"]
        query_id = item["id"]
        passage_id = f"{query_id}_0"
        content = item["context"].strip()

        metadata = {
            "query_id": query_id,
            "query": query,
            "passage_id": passage_id,
            "is_selected": 1
        }

        documents.append(Document(page_content=content, metadata=metadata))

    return documents, dataset

def create_df(dataset):
    records = []
    for item in dataset:
        query = item["question"]
        query_id = item["id"]
        answers = item["answers"]["text"]
        if not answers:
            continue
        joined_answers = " ||| ".join([a.strip() for a in answers])
        records.append({
            "query_id": query_id,
            "query": query,
            "ground_truth": joined_answers
        })
    return pd.DataFrame(records)

def get_split_documents(documents, chunk_size=250, chunk_overlap=0):
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(documents)

def create_retriever(documents):
    embedding = OllamaEmbeddings(base_url=BASE_URL, model=EMBED_MODEL)
    vectorstore = Chroma.from_documents(
        documents=documents,
        collection_name="selfrag_squad_chroma",
        embedding=embedding
    )
    return vectorstore.as_retriever()

def create_retrieval_grader(llm):
    retrieval_prompt = PromptTemplate(
        template="""You are a grader assessing relevance of a retrieved document to a user question. \n 
        Here is the retrieved document: \n\n {document} \n\n
        Here is the user question: {question} \n
        If the document contains keywords related to the user question, grade it as relevant. \n
        It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
        Provide the binary score as a JSON with a single key 'score' and no premable or explanation.""",
        input_variables=["question", "document"],
    )
    return retrieval_prompt | llm | JsonOutputParser()

def create_rag_chain(llm):
    prompt = hub.pull("rlm/rag-prompt")
    return prompt | llm | StrOutputParser()

def create_hallucination_grader(llm):
    hallucination_prompt = PromptTemplate(
        template="""You are a grader assessing whether an answer is grounded in / supported by a set of facts. \n 
        You must respond with ONLY valid JSON, no extra text.
        The JSON must follow exactly this format: {"score": "yes"} or {"score": "no"}.
        Here are the facts:
        \n ------- \n
        {documents} 
        \n ------- \n
        Here is the answer: {generation}
        Give a binary score 'yes' or 'no' score to indicate whether the answer is grounded in / supported by a set of facts. \n
        Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.""",
        input_variables=["generation", "documents"],
    )
    parser = JsonOutputParser()
    chain = hallucination_prompt | llm | parser
    return chain, llm

def create_answer_grader(llm):
    answer_prompt = PromptTemplate(
        template="""You are a grader assessing whether an answer is useful to resolve a question. \n 
        Here is the answer:
        \n ------- \n
        {generation} 
        \n ------- \n
        Here is the question: {question}
        Give a binary score 'yes' or 'no' to indicate whether the answer is useful to resolve a question. \n
        Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.""",
        input_variables=["generation", "question"],
    )
    return answer_prompt | llm | JsonOutputParser()

def create_question_rewriter(llm):
    rewrite_prompt = PromptTemplate(
        template="""You a question re-writer that converts an input question to a better version that is optimized \n 
        for vectorstore retrieval. Look at the initial and formulate an improved question. \n
        Here is the initial question: \n\n {question}. Improved question with no preamble: \n """,
        input_variables=["question"],
    )
    return rewrite_prompt | llm | StrOutputParser()

def build_workflow(
    retriever,
    rag_chain,
    retrieval_grader,
    question_rewriter,
    hallucination_grader,
    answer_grader,
    GraphState
):
    workflow = StateGraph(GraphState)
    workflow.add_node("retrieve", partial(retrieve, retriever=retriever))
    workflow.add_node("generate", partial(generate, rag_chain=rag_chain))
    workflow.add_node("grade_documents", partial(grade_documents, retrieval_grader=retrieval_grader))
    workflow.add_node("transform_query", partial(transform_query, question_rewriter=question_rewriter))
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges("grade_documents", decide_to_generate, {
        "transform_query": "transform_query",
        "generate": "generate"
    })
    workflow.add_edge("transform_query", "retrieve")
    workflow.add_conditional_edges("generate",
        partial(grade_generation_v_documents_and_question,
                hallucination_grader=hallucination_grader,
                answer_grader=answer_grader),
        {
            "not supported": "generate",
            "useful": END,
            "not useful": "transform_query"
        })

    return workflow.compile()

def retrieve(state, retriever):
    question = state["question"]
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}

def generate(state, rag_chain):
    question = state["question"]
    documents = state["documents"]
    generation = rag_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}

def grade_documents(state, retrieval_grader):
    question = state["question"]
    documents = state["documents"]
    filtered_docs = []
    for d in documents:
        score = retrieval_grader.invoke({"question": question, "document": d.page_content})
        if score["score"] == "yes":
            filtered_docs.append(d)
    return {"documents": filtered_docs, "question": question}

def transform_query(state, question_rewriter):
    question = state["question"]
    documents = state["documents"]
    better_question = question_rewriter.invoke({"question": question})
    return {"documents": documents, "question": better_question}

def decide_to_generate(state):
    if not state["documents"]:
        return "transform_query"
    else:
        return "generate"

from langchain_core.exceptions import OutputParserException

def safe_invoke_grader(grader_chain, raw_llm, inputs):
    try:
        return grader_chain.invoke(inputs)
    except Exception as e:
        print("JSON parse failed, retrying...")
        raw = raw_llm.invoke(inputs)  # call the LLM directly
        import json, re
        match = re.search(r"\{.*\}", raw, re.S)
        if match:
            return json.loads(match.group())
        return {"score": "no"}  # fail-safe



def grade_generation_v_documents_and_question(state, hallucination_grader, answer_grader):
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    # score = hallucination_grader.invoke({"documents": documents, "generation": generation})
    score = safe_invoke_grader(hallucination_grader, hallucination_llm, {
    "documents": documents,
    "generation": generation
})


    
    if score["score"] == "yes":
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        score = answer_grader.invoke({"question": question, "generation": generation})
        if score["score"] == "yes":
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful"
    else:
        print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        return "not supported"

def main():
    class GraphState(TypedDict):
        question: str
        generation: str
        documents: list

    documents, dataset = load_squad_documents()
    doc_splits = get_split_documents(documents)
    reference_df = create_df(dataset)

    retriever = create_retriever(doc_splits)
    llm = Ollama(base_url=BASE_URL, model=LLM_MODEL)
    llm_json = Ollama(base_url=BASE_URL, model=LLM_MODEL, format="json", temperature=0)


    hallucination_grader, hallucination_llm = create_hallucination_grader(llm_json)
    answer_grader, answer_llm = create_answer_grader(llm_json)

    app = build_workflow(
        retriever,
        create_rag_chain(llm),
        create_retrieval_grader(llm),
        create_question_rewriter(llm),
        create_hallucination_grader(llm_json),
        create_answer_grader(llm_json),
        GraphState
    )

    eval_rows, results_rows = [], []

    for idx, row in reference_df.iterrows():
        query_id = row.query_id
        query = row.query
        ground_truth = row.ground_truth

        try:
            print(f"Running query {idx+1}/1000: {query[:80]}...")
            final_state = None
            for output in app.stream({"question": query}):
                final_state = list(output.values())[0]

            final_answer = final_state.get("generation", "")
            retrieved_docs = [doc.page_content for doc in final_state.get("documents", [])]

        except Exception as e:
            print(f"Error for Query ID {query_id}: {e}")
            final_answer = "[ERROR]"
            retrieved_docs = []

        eval_rows.append({
            "query": query,
            "ground_truth": ground_truth,
            "final_answer": final_answer,
            "retrieved_docs": retrieved_docs
        })

        results_rows.append({
            "query_id": query_id,
            "query": query,
            "ground_truth": ground_truth,
            "final_answer": final_answer
        })

    eval_df = pd.DataFrame(eval_rows)
    results_df = pd.DataFrame(results_rows)

    eval_df.to_csv("squadv2_eval_results.csv", index=False, encoding="utf-8")
    results_df.to_csv("squadv2_results_summary.csv", index=False, encoding="utf-8")

    print("\nSaved evaluations and summary results")

    print("\nFirst Query Output:")
    print(f"Query         : {eval_df.iloc[0]['query']}")
    print(f"Ground Truth  : {eval_df.iloc[0]['ground_truth']}")
    print(f"Final Answer  : {eval_df.iloc[0]['final_answer']}")

if __name__ == "__main__":
    main()

