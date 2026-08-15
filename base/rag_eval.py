import os
from ragas.run_config import RunConfig

# nan fix
run_config = RunConfig(
    max_workers=1,
    timeout=180,
    max_retries=5,
)

# avoiding parallelism collide 
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from sentence_transformers import SentenceTransformer

from ragas import (
    SingleTurnSample,
    EvaluationDataset,
    evaluate,
)

from ragas.metrics import (
    Faithfulness,
    ResponseRelevancy,
)

from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from langchain_ollama import ChatOllama
from langchain_core.embeddings import Embeddings

from rag_app import generate_answer


embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2",
    device="cpu",
)

# relevancy check, same model for data and artificial questions
class MiniLMEmbeddings(Embeddings):

    def embed_documents(self, texts):
        vectors = embedding_model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        )

        return vectors.tolist()

    def embed_query(self, text):
        vector = embedding_model.encode(
            [text],
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        )

        return vector[0].tolist()


judge_embeddings = LangchainEmbeddingsWrapper(
    MiniLMEmbeddings()
)


ollama_api_key = os.environ.get("OLLAMA_API_KEY")

# fully deterministic temp
# extract claims for faithfullness
# reverse engineer ques for relevancy
judge_model = ChatOllama(
    model="gpt-oss:20b-cloud",
    base_url="https://ollama.com",
    client_kwargs={
        "headers": {
            "Authorization": f"Bearer {ollama_api_key}"
        }
    },
    temperature=0,
)

judge_llm = LangchainLLMWrapper(judge_model)


test_questions = [
    "What are the main economic consequences of climate change discussed in the review?",

    "How does climate change affect GDP, foreign direct investment, and financial markets?",
]

def build_dataset():
    samples = []

    for question in test_questions:
        result = generate_answer(
            question,
            top_k=5,
        )

        samples.append(
            SingleTurnSample(
                user_input=result["query"],
                response=result["answer"],
                retrieved_contexts=result["contexts"],
            )
        )

    return EvaluationDataset(
        samples=samples
    )


if __name__ == "__main__":

    dataset = build_dataset()

    result = evaluate(
        dataset=dataset,
        metrics=[
            Faithfulness(),
            ResponseRelevancy(),
        ],
        llm=judge_llm,
        embeddings=judge_embeddings,
        run_config=run_config,
        raise_exceptions=True,
    )
    print(result)

    df = result.to_pandas()

    df.to_csv(
        "eval_results.csv",
        index=False,
    )

    print("\nPer-question breakdown:")
    print(
        df[
            [
                "user_input",
                "faithfulness",
                "answer_relevancy",
            ]
        ].to_string(index=False)
    )

    print("\nScores:")
    print(
        f"Faithfulness: {df['faithfulness'].mean():.4f}"
    )
    print(
        f"Answer Relevancy: {df['answer_relevancy'].mean():.4f}"
    )