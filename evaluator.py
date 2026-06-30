import os
import warnings
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)
from rag_agent import build_rag_agent
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
load_dotenv()

def get_ragas_llm_and_embeddings():
    """
    RAGAS relies heavily on OpenAI models by default. Because we are using 
    OpenRouter and local BGE embeddings, we must pass our custom models into RAGAS.
    """
    from compressor import get_llm
    from embeddings import get_bge_embeddings
    
    llm = get_llm()
    embeddings = get_bge_embeddings()
    
    # In recent RAGAS versions, LangChain objects need to be wrapped
    try:
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        return LangchainLLMWrapper(llm), LangchainEmbeddingsWrapper(embeddings)
    except ImportError:
        # Fallback for older ragas versions
        return llm, embeddings

def run_evaluation():
    print("Setting up RAGAS Evaluation Environment...")
    rag_chain = build_rag_agent()
    
    # 1. Define our robust evaluation dataset
    questions = [
        "Where is the European office?",
        "Where is the headquarters?",
        "Can software subscriptions be refunded?",
        "Can hardware be refunded?",
        "How long do I have to return hardware?",
        "Can I cancel my subscription?",
        "Where is the North American office?",
        "Does the document mention student discounts?"
    ]
    # The gold standard answers we EXPECT the agent to match
    ground_truths = [
        "The European office is located in London, UK.",
        "The primary headquarters is located in San Francisco, California.",
        "No, software subscriptions cannot be refunded once the billing cycle has started.",
        "Yes, hardware purchases can be refunded within 30 days of the original purchase date in its original packaging.",
        "You have 30 days from the original purchase date to return hardware.",
        "Yes, you can cancel your software subscription at any time to prevent future charges.",
        "The North American office (primary headquarters) is located in San Francisco, California.",
        "No, the provided document does not mention student discounts.",
    ]
    
    answers = []
    contexts = []
    
    print("Generating answers for evaluation dataset...")
    # 2. Run our agent against the questions to generate actual answers and retrieve contexts
    for q in questions:
        print(f"  Evaluating Query: '{q}'")
        res = rag_chain.invoke({"input": q})
        answers.append(res["answer"])
        # RAGAS expects context as a list of strings
        contexts.append([doc.page_content for doc in res["context"]])
        
    data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    }
    
    # 3. Package it into a HuggingFace Dataset (required by RAGAS)
    dataset = Dataset.from_dict(data)
    
    eval_llm, eval_embeddings = get_ragas_llm_and_embeddings()
    
    metrics = [
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    ]
    
    print("\nRunning RAGAS Metrics (this mathematically scores the answers against ground truths)...")
    
    # 4. Evaluate!
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=eval_llm,
        embeddings=eval_embeddings
    )
    
    print("\n=== Evaluation Results ===")
    print(result)

if __name__ == "__main__":
    run_evaluation()
