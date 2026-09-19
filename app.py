from pathlib import Path

import streamlit as st
from sentence_transformers import SentenceTransformer

from copilot import (
    EMBEDDING_MODEL_NAME,
    find_relevant_documents,
    generate_support_plan,
    load_documents,
)


@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


@st.cache_data
def load_knowledge_base():
    project_folder = Path(__file__).parent
    knowledge_base_folder = project_folder / "knowledge_base"
    return load_documents(knowledge_base_folder)


st.set_page_config(
    page_title="AI Support Copilot",
    page_icon="🛠️",
)

st.title("🛠️ AI Support Copilot")
st.write("Analyze a support ticket and generate a grounded response plan.")

documents = load_knowledge_base()
embedding_model = load_embedding_model()

title = st.text_input(
    "Ticket title",
    placeholder="For example: Users cannot log in",
)

description = st.text_area(
    "Ticket description",
    placeholder="Describe the issue, impact, and any relevant context.",
    height=150,
)

if st.button("Analyze ticket"):
    if not title.strip() or not description.strip():
        st.warning("Please provide both a ticket title and a description.")
    else:
        ticket_text = f"{title}\n{description}"

        with st.spinner("Searching the knowledge base and analyzing the ticket..."):
            relevant_documents = find_relevant_documents(
                ticket_text,
                documents,
                embedding_model,
            )

            support_plan = generate_support_plan(
                title,
                description,
                relevant_documents,
            )

        st.subheader("Support plan")
        st.markdown(support_plan)

        st.subheader("Relevant knowledge base sources")
        for document in relevant_documents:
            st.write(
                f"**{document['source']}** "
                f"— similarity: {document['score']:.2f}"
            )

            with st.expander(f"View {document['source']}"):
                st.text(document["content"])