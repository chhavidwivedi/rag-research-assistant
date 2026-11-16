
import re

from search import load_search_resources, search_papers


# ----------------------------------
# CONFIGURATION
# ----------------------------------

TOP_K_RESULTS = 5
MAX_ANSWER_SENTENCES = 2
MIN_KEYWORD_MATCHES = 1

INSUFFICIENT_EVIDENCE = (
    "I couldn't find sufficient evidence to answer this question "
    "in the uploaded research papers."
)


# ----------------------------------
# TEXT PROCESSING
# ----------------------------------

STOPWORDS = {
    "what", "which", "when", "where", "who", "why", "how",
    "the", "a", "an", "is", "are", "was", "were", "in",
    "on", "at", "to", "of", "for", "and", "or", "with",
    "does", "do", "did", "can", "could", "would", "should",
    "between", "difference", "explain", "describe"
}


def tokenize(text):

    words = re.findall(r"\b[a-zA-Z0-9]+\b", text.lower())

    synonyms = {
        "parameterized": "parametric",
        "parameters": "parametric",
        "hallucinations": "hallucinate",
        "hallucinated": "hallucinate",
        "retrieval": "retrieve",
        "retriever": "retrieve",
        "retrieved": "retrieve",
        "retrieving": "retrieve",
        "databases": "database",
        "documents": "document"
    }

    normalized = []

    for word in words:

        if word in STOPWORDS or len(word) <= 1:
            continue

        normalized.append(synonyms.get(word, word))

    return normalized


def split_sentences(text):

    # Normalize PDF line breaks and whitespace.
    text = re.sub(r"\s+", " ", text)

    # Split after sentence-ending punctuation.
    sentences = re.split(
        r"""(?<=[.!?])\s+(?=[A-Z0-9"'])""",
        text
    )

    cleaned = []

    for sentence in sentences:

        sentence = sentence.strip()

        # Remove common PDF section-heading prefixes.
        sentence = re.sub(
            r"^\d+\s+(Introduction|Discussion)\s+",
            "",
            sentence,
            flags=re.IGNORECASE
        )

        if len(sentence.split()) >= 8:
            cleaned.append(sentence)

    return cleaned


# ----------------------------------
# QUESTION INTENT DETECTION
# ----------------------------------

def detect_question_intent(question):

    question = question.lower().strip()

    if any(word in question for word in [
        "difference", "differentiate", "compare"
    ]):
        return "comparison"

    if question.startswith(("what is", "what are", "define")):
        return "definition"

    if question.startswith(("how does", "how do", "how can", "how is")):
        return "process"

    if question.startswith(("why does", "why do", "why can", "why is")):
        return "explanation"

    if any(word in question for word in [
        "limitations", "disadvantages", "drawbacks", "challenges"
    ]):
        return "limitations"

    return "general"


# ----------------------------------
# KEYWORD MATCHING AND SCORING
# ----------------------------------

def score_sentence(sentence, question, question_keywords, chunk_text, intent):

    sentence_lower = sentence.lower()

    sentence_keywords = set(tokenize(sentence))
    matches = question_keywords.intersection(sentence_keywords)

    score = len(matches)

    # Definition questions
    if intent == "definition":

        definition_phrases = [
            "parametric memory is",
            "non-parametric memory is",
            "parametric memory henceforth",
            "document index as the non-parametric memory",
            "parameterized implicit knowledge base",
            "is stored implicitly"
        ]

        for phrase in definition_phrases:
            if phrase in sentence_lower:
                score += 8

    # Comparison questions
    elif intent == "comparison":

        if "parametric" in sentence_lower and "non-parametric" in sentence_lower:
            score += 10

        if "parametric memory is" in sentence_lower:
            score += 5

        if "non-parametric memory is" in sentence_lower:
            score += 5

    # Process questions
    elif intent == "process":

        process_phrases = [
            "query vector",
            "similarity scores",
            "top k chunks",
            "retrieve documents",
            "dense vector index",
            "retriever",
            "embedding"
        ]

        for phrase in process_phrases:
            if phrase in sentence_lower:
                score += 3

    # Explanation questions
    elif intent == "explanation":

        if any(word in sentence_lower for word in [
            "because", "due to", "results in", "may produce"
        ]):
            score += 3

    # Limitations questions
    elif intent == "limitations":

        if any(word in sentence_lower for word in [
            "limitations", "cannot", "difficult", "challenges",
            "hallucinations", "drawbacks"
        ]):
            score += 4

    return score


# ----------------------------------
# EXTRACTIVE ANSWER GENERATION
# ----------------------------------

def generate_extractive_answer(question, search_results):

    if not search_results:
        return INSUFFICIENT_EVIDENCE

    question_keywords = set(tokenize(question))

    intent = detect_question_intent(question)

    print(f"\nDetected question intent: {intent}")

    if not question_keywords:
        return INSUFFICIENT_EVIDENCE

    candidates = []

    for result in search_results[:TOP_K_RESULTS]:

        chunk = result["chunk"]

        source = chunk["source"]
        page = chunk["page"]
        text = chunk["text"]

        sentences = split_sentences(text)

        for sentence in sentences:

            if sentence.lower().startswith(
                "retrieval-augmented generation for large language models"
            ):
                continue

            score = score_sentence(
                sentence,
                question,
                question_keywords,
                text,
                intent
            )

            if score >= MIN_KEYWORD_MATCHES:

                candidates.append({
                    "sentence": sentence,
                    "score": score,
                    "source": source,
                    "page": page
                })

    if not candidates:
        return INSUFFICIENT_EVIDENCE

    # Rank sentences by score.
    candidates.sort(
        key=lambda item: item["score"],
        reverse=True
    )

    print("\nDEBUG: TOP 10 EXTRACTED CANDIDATES")

    for item in candidates[:10]:

        print(
            f"\nScore: {item['score']}"
            f"\nSource: {item['source']}"
            f"\nPage: {item['page']}"
            f"\nSentence: {item['sentence']}"
        )

    # Select final answer sentences.
    selected = []
    seen_sentences = set()
    used_pages = set()

    answer_limit = (
        2 if intent == "comparison"
        else 1 if intent == "definition"
        else MAX_ANSWER_SENTENCES
    )
    for candidate in candidates:

        normalized = re.sub(
            r"[^a-z0-9]+",
            " ",
            candidate["sentence"].lower()
        ).strip()

        if normalized in seen_sentences:
            continue

        if intent == "comparison":

            page_key = (
                candidate["source"],
                candidate["page"]
            )

            if page_key in used_pages:
                continue

        selected.append(candidate)
        seen_sentences.add(normalized)

        if intent == "comparison":
            used_pages.add(page_key)

        if len(selected) >= answer_limit:
            break

    if not selected:
        return INSUFFICIENT_EVIDENCE

    # ----------------------------------
    # BUILD ANSWER AND ADD CITATIONS
    # ----------------------------------

    answer_parts = []
    citations = {}

    for candidate in selected:

        citation_key = (
            candidate["source"],
            candidate["page"]
        )

        if citation_key not in citations:
            citation_id = f"E{len(citations) + 1}"
            citations[citation_key] = citation_id

        citation_id = citations[citation_key]

        answer_parts.append(
            f'{candidate["sentence"]} [{citation_id}]'
        )

    # Print the supporting source details.
    print("\nSUPPORTING SOURCES")
    print("-" * 60)

    for (source, page), citation_id in citations.items():

        print(
            f"[{citation_id}] {source}, Page {page}"
        )

    # IMPORTANT: Return the final answer.
    return "\n\n".join(answer_parts)


# ----------------------------------
# MAIN APPLICATION
# ----------------------------------

def main():

    embedder, reranker, index, chunks = (
        load_search_resources()
    )

    print("\nExtractive Research Assistant is ready!")

    while True:

        question = input(
            "\nAsk a research question (or type exit): "
        ).strip()

        if question.lower() in {"exit", "quit"}:
            print("Exiting Research Assistant.")
            break

        if not question:
            print("Please enter a question.")
            continue

        results = search_papers(
            question,
            embedder,
            reranker,
            index,
            chunks
        )

        print("\nDEBUG: TOP RETRIEVED CHUNKS")

        for i, result in enumerate(results[:5], start=1):

            chunk = result["chunk"]

            print(f"\n--- CHUNK {i} ---")
            print(f"Source: {chunk['source']}")
            print(f"Page: {chunk['page']}")
            print(f"Reranker score: {result.get('rerank_score')}")
            print(f"Similarity score: {result.get('similarity_score')}")
            print(f"Text:\n{chunk['text']}")

        print("\n" + "=" * 60)
        print("EXTRACTIVE ANSWER")
        print("=" * 60)

        answer = generate_extractive_answer(
            question,
            results
        )

        print("\nANSWER")
        print("-" * 60)
        print(answer)

        print("=" * 60)


if __name__ == "__main__":
    main()
