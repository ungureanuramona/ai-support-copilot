from pathlib import Path

from ollama import chat
from sentence_transformers import SentenceTransformer, util


EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
LLM_MODEL_NAME = "gemma3:4b"


def load_documents(knowledge_base_folder):
    documents = []

    for file_path in knowledge_base_folder.glob("*.txt"):
        documents.append(
            {
                "source": file_path.name,
                "content": file_path.read_text(encoding="utf-8"),
            }
        )

    return documents


def find_relevant_documents(ticket_text, documents, model, limit=2):
    document_texts = [document["content"] for document in documents]

    ticket_embedding = model.encode(ticket_text, convert_to_tensor=True)
    document_embeddings = model.encode(document_texts, convert_to_tensor=True)

    similarity_scores = util.cos_sim(ticket_embedding, document_embeddings)[0]

    scored_documents = []

    for index, score in enumerate(similarity_scores):
        scored_documents.append(
            {
                "source": documents[index]["source"],
                "content": documents[index]["content"],
                "score": float(score),
            }
        )

    scored_documents.sort(key=lambda document: document["score"], reverse=True)

    return scored_documents[:limit]


def generate_support_plan(title, description, relevant_documents):
    context = "\n\n".join(
        [
            f"Source: {document['source']}\n{document['content']}"
            for document in relevant_documents
        ]
    )

    prompt = f"""
You are an AI support copilot.

Analyze this fictional support ticket using the support guidance provided below.

Ticket title: {title}
Ticket description: {description}

Support guidance:
{context}

Use exactly these headings:

Category:
Priority:
Recommended next steps:
Customer response:

For recommended next steps, use the support guidance and do not claim that an investigation step has already happened.

For customer response, write a concise message from the support team to the customer.

At the end, add:
Sources:

List the filenames you used.
"""

    response = chat(
        model=LLM_MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        options={"temperature": 0},
    )

    return response.message.content


def main():
    project_folder = Path(__file__).parent
    knowledge_base_folder = project_folder / "knowledge_base"
    documents = load_documents(knowledge_base_folder)

    print("Loading the semantic search model...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    print("\n--- AI Support Copilot ---")

    title = input("\nTicket title: ")
    description = input("Ticket description: ")

    if not title.strip() or not description.strip():
        print("\nPlease provide both a title and a description.")
        return

    ticket_text = f"{title}\n{description}"

    relevant_documents = find_relevant_documents(
        ticket_text,
        documents,
        embedding_model,
    )

    print("\n--- Relevant Knowledge Base Sources ---")
    for document in relevant_documents:
        print(f"- {document['source']} ({document['score']:.2f})")

    print("\nAnalyzing ticket...\n")

    support_plan = generate_support_plan(
        title,
        description,
        relevant_documents,
    )

    print(support_plan)


if __name__ == "__main__":
    main()