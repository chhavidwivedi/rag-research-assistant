
from search import load_search_resources, search_papers


TEST_QUESTIONS = [
    {
        "question": "What is parametric memory in RAG?",
        "expected_pages": [
            ("rag-paper.pdf", 1),
            ("rag-paper.pdf", 2)
        ]
    },
    {
        "question": "What is non-parametric memory in RAG?",
        "expected_pages": [
            ("rag-paper.pdf", 1),
            ("rag-paper.pdf", 2)
        ]
    },
    {
        "question": "What is the difference between parametric and non-parametric memory?",
        "expected_pages": [
            ("rag-paper.pdf", 1),
            ("rag-paper.pdf", 2)
        ]
    },
    {
        "question": "How does RAG retrieve external knowledge?",
        "expected_pages": [
            ("rag-survey.pdf", 5),
            ("rag-paper.pdf", 1),
            ("rag-paper.pdf", 2)
        ]
    },
    {
        "question": "Why can parametric-only models hallucinate?",
        "expected_pages": [
            ("rag-paper.pdf", 1)
        ]
    }
]


def main():

    print("Loading retrieval system...")

    embedder, reranker, index, chunks = load_search_resources()

    hits = 0
    reciprocal_ranks = []

    print("\nStarting retrieval evaluation")

    for number, test in enumerate(TEST_QUESTIONS, start=1):

        question = test["question"]
        expected_pages = set(test["expected_pages"])

        print("\n" + "=" * 65)
        print(f"TEST {number}: {question}")
        print("=" * 65)

        results = search_papers(
            question,
            embedder,
            reranker,
            index,
            chunks
        )

        relevant_rank = None

        for rank, result in enumerate(results, start=1):

            chunk = result["chunk"]

            source_page = (
                chunk["source"],
                chunk["page"]
            )

            is_relevant = source_page in expected_pages

            if is_relevant and relevant_rank is None:
                relevant_rank = rank

            print(f"\nRank: {rank}")
            print("Document:", chunk["source"])
            print("Page:", chunk["page"])
            print("Relevant:", is_relevant)
            print("CrossEncoder score:", round(result["rerank_score"], 5))  
            print("Passage:", chunk["text"][:250])

        if relevant_rank is not None:
            hits += 1
            reciprocal_ranks.append(1 / relevant_rank)

            print("\nFirst relevant passage rank:", relevant_rank)

        else:
            reciprocal_ranks.append(0)
            print("\nNo labelled relevant passage found in top 5.")

    total = len(TEST_QUESTIONS)

    hit_at_5 = hits / total
    mrr = sum(reciprocal_ranks) / total

    print("\n" + "=" * 65)
    print("FINAL RETRIEVAL METRICS")
    print("=" * 65)

    print(f"Hit@5: {hit_at_5:.2%}")
    print(f"MRR: {mrr:.3f}")

    print("\nEvaluation completed.")


if __name__ == "__main__":
    main()
