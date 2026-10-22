
from pathlib import Path

import streamlit as st
from sentence_transformers import SentenceTransformer

from copilot import (
    EMBEDDING_MODEL_NAME,
    find_relevant_documents,
    generate_support_plan,
    load_documents,
)
from log_analyzer import analyze_logs


st.set_page_config(
    page_title="AI Support Copilot",
    page_icon="🛠️",
    layout="wide",
)


@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


@st.cache_data
def load_knowledge_base():
    project_folder = Path(__file__).parent
    knowledge_base_folder = project_folder / "knowledge_base"
    return load_documents(knowledge_base_folder)


st.title("🛠️ AI Support Copilot")
st.caption(
    "Incident investigation, knowledge base retrieval "
    "and technical log analysis."
)

ticket_tab, logs_tab = st.tabs(
    ["🎫 Ticket Investigation", "📋 Log Analyzer"]
)


with ticket_tab:
    st.header("Ticket Investigation")

    st.write(
        "Analyze a support incident using the local "
        "knowledge base and AI."
    )

    title = st.text_input(
        "Ticket title",
        placeholder="Users cannot log in",
    )

    description = st.text_area(
        "Ticket description",
        placeholder=(
            "Describe the issue, impact, "
            "and troubleshooting performed."
        ),
        height=150,
    )

    if st.button("Analyze ticket", type="primary"):
        if not title.strip() or not description.strip():
            st.warning(
                "Please provide both a ticket title "
                "and a description."
            )
        else:
            with st.spinner("Analyzing support ticket..."):
                documents = load_knowledge_base()
                embedding_model = load_embedding_model()

                relevant_documents = find_relevant_documents(
                    f"{title}\n{description}",
                    documents,
                    embedding_model,
                )

                support_plan = generate_support_plan(
                    title,
                    description,
                    relevant_documents,
                )

            st.subheader("Support Plan")
            st.markdown(support_plan)

            st.subheader("Knowledge Base Sources")

            if relevant_documents:
                for document in relevant_documents:
                    st.write(
                        f"**{document['source']}** "
                        f"— similarity: {document['score']:.2f}"
                    )

                    with st.expander(
                        f"View {document['source']}"
                    ):
                        st.text(document["content"])
            else:
                st.info(
                    "No sufficiently relevant knowledge "
                    "base documents were found."
                )


with logs_tab:
    st.header("Technical Log Analyzer")

    st.write(
        "Upload an application log file to identify "
        "errors, warnings and potential incident patterns."
    )

    uploaded_file = st.file_uploader(
        "Upload log file",
        type=["log", "txt"],
        max_upload_size=10,
    )

    if uploaded_file is not None:
        try:
            log_text = uploaded_file.getvalue().decode("utf-8")
        except UnicodeDecodeError:
            st.error(
                "Unable to read this file. "
                "Please upload a UTF-8 encoded log."
            )
        else:
            if st.button("Analyze logs", type="primary"):
                results = analyze_logs(log_text)

                st.subheader("Log Analysis Summary")

                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "Total Lines",
                    results["total_lines"],
                )

                col2.metric(
                    "Errors",
                    results["errors"],
                )

                col3.metric(
                    "Warnings",
                    results["warnings"],
                )

                st.subheader("Detected Issues")

                if results["detected_issues"]:
                    for issue, count in (
                        results["detected_issues"].items()
                    ):
                        st.write(f"- **{issue}**: {count}")
                else:
                    st.info(
                        "No known issue patterns were detected."
                    )

                st.subheader("Incident Events")

                if results["events"]:
                    st.dataframe(
                        results["events"],
                        use_container_width=True,
                        hide_index=True,
                    )
                else:
                    st.info(
                        "No ERROR or WARNING events found."
                    )

                st.download_button(
                    "Download JSON Report",
                    data=__import__("json").dumps(
                        results,
                        indent=2,
                    ),
                    file_name="log_analysis_report.json",
                    mime="application/json",
                )
