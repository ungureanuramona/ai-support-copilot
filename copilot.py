from ollama import chat


LLM_MODEL_NAME = "gemma3:4b"


def generate_support_plan(title, description):
    prompt = f"""
You are an AI support copilot.

Analyze the fictional support ticket below and provide a practical support response.

Ticket title: {title}
Ticket description: {description}

Use exactly these headings:

Category:
Priority:
Recommended next steps:
Customer response:

Keep the answer concise and professional.
Do not invent facts that are not present in the ticket.
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
    print("--- AI Support Copilot ---")

    title = input("\nTicket title: ")
    description = input("Ticket description: ")

    if not title.strip() or not description.strip():
        print("\nPlease provide both a title and a description.")
        return

    print("\nAnalyzing ticket...\n")

    support_plan = generate_support_plan(title, description)
    print(support_plan)


if __name__ == "__main__":
    main()