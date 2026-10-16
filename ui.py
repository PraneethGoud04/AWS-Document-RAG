
import json

import requests
import streamlit as st

st.set_page_config(
    page_title="AWS Document RAG",
    page_icon="📄",
    layout="wide"
)

st.title("📄 AWS Document RAG")
st.caption(
    "Upload PDFs, process them with AWS, and ask questions "
    "using information grounded in your documents."
)

# --------------------------------------------------
# Configuration
# --------------------------------------------------

api_url = st.sidebar.text_input(
    "FastAPI URL",
    value="http://127.0.0.1:8000"
).rstrip("/")

st.sidebar.info(
    "Pipeline: S3 → Lambda → Text extraction/OCR "
    "→ Chunks → Bedrock embeddings → Retrieval → Answer"
)

# --------------------------------------------------
# API health check
# --------------------------------------------------

with st.sidebar:
    st.subheader("API connection")

    if st.button("Check API connection", use_container_width=True):
        try:
            response = requests.get(
                f"{api_url}/health",
                timeout=5
            )
            response.raise_for_status()
            data = response.json()

            st.success(
                f"API connected: {data.get('status', 'OK')}"
            )

        except requests.RequestException as exc:
            st.error(f"API connection failed: {exc}")

        except ValueError:
            st.error("The API returned an invalid health response.")

# --------------------------------------------------
# Document upload
# --------------------------------------------------

st.header("1. Upload a document")

uploaded_file = st.file_uploader(
    "Choose a PDF document",
    type=["pdf"],
    accept_multiple_files=False
)

if uploaded_file is not None:
    st.write(f"Selected file: **{uploaded_file.name}**")

    if st.button("Upload PDF", type="primary"):
        try:
            with st.spinner("Checking for duplicates and uploading..."):
                response = requests.post(
                    f"{api_url}/upload",
                    files={
                        "file": (
                            uploaded_file.name,
                            uploaded_file.getvalue(),
                            "application/pdf"
                        )
                    },
                    timeout=120
                )
                response.raise_for_status()
                upload_result = response.json()

            st.session_state["last_upload_name"] = uploaded_file.name
            st.session_state.pop("last_answer_result", None)
            st.session_state.pop("last_document_status", None)

            if upload_result.get("duplicate"):
                # Use the ID returned by the duplicate-check endpoint.
                existing_id = upload_result.get("existing_document_id")

                st.session_state["last_upload"] = {
                    **upload_result,
                    "document_id": existing_id
                }
                st.session_state["document_ready"] = False

                st.warning(
                    "This PDF has already been uploaded. "
                    "A new S3 copy was not created."
                )

                if existing_id:
                    st.caption(f"Existing document ID: {existing_id}")

                st.info(
                    "Check the existing document's processing status below."
                )

            else:
                st.session_state["last_upload"] = upload_result
                st.session_state["document_ready"] = False

                st.success("PDF uploaded to S3 successfully.")

                document_id = upload_result.get("document_id")
                if document_id:
                    st.caption(f"Document ID: {document_id}")

                st.info(
                    "The AWS processing pipeline runs asynchronously. "
                    "Check the document status below."
                )

        except requests.RequestException as exc:
            st.error(f"Upload failed: {exc}")

            if getattr(exc, "response", None) is not None:
                st.code(exc.response.text)

        except ValueError:
            st.error("The API returned an invalid upload response.")

# --------------------------------------------------
# Processing status
# --------------------------------------------------

st.header("2. Document processing status")

upload_result = st.session_state.get("last_upload", {})
document_id = upload_result.get("document_id")

if document_id:
    st.write(
        f"Last selected file: "
        f"**{st.session_state.get('last_upload_name', 'PDF')}**"
    )

    st.caption(f"Document ID: {document_id}")

    if st.button("Check processing status", type="secondary"):
        try:
            with st.spinner("Checking AWS processing status..."):
                response = requests.get(
                    f"{api_url}/status/{document_id}",
                    timeout=10
                )
                response.raise_for_status()
                status_result = response.json()

            status = status_result.get("status", "unknown")
            st.session_state["last_document_status"] = status
            st.session_state["document_ready"] = status == "ready"

            if status == "ready":
                st.success("Ready — document embeddings are available.")

            elif status == "embedding_in_progress":
                st.info(
                    "Generating embeddings — text extraction and chunking "
                    "are complete."
                )

            elif status == "processing":
                st.warning(
                    "Processing — waiting for text extraction and chunking."
                )

            else:
                st.info(
                    status_result.get(
                        "message",
                        "The document status is not yet known."
                    )
                )

        except requests.RequestException as exc:
            st.error(f"Could not check processing status: {exc}")

            if getattr(exc, "response", None) is not None:
                st.code(exc.response.text)

        except ValueError:
            st.error("The API returned an invalid status response.")

    # Show the last known status after Streamlit reruns.
    current_status = st.session_state.get(
        "last_document_status",
        "not_checked"
    )

    if current_status == "ready":
        st.success("Last known status: Ready")
    elif current_status == "embedding_in_progress":
        st.info("Last known status: Embeddings in progress")
    elif current_status == "processing":
        st.warning("Last known status: Processing")
    elif current_status == "not_checked":
        st.caption("Click Check processing status to check progress.")

else:
    st.caption(
        "Upload a PDF to track its processing status. "
        "Already-indexed documents can still be queried below."
    )

# --------------------------------------------------
# Question answering
# --------------------------------------------------

st.header("3. Ask a question")

st.write(
    "Ask about information contained in your indexed documents."
)

question = st.text_area(
    "Your question",
    placeholder=(
        "Example: What is the status of the Atlas migration project?"
    ),
    height=100,
    key="question_input"
)

top_k = st.slider(
    "Number of passages to retrieve",
    min_value=1,
    max_value=5,
    value=3
)

if st.button("Get answer", type="primary"):
    if not question.strip():
        st.warning("Please enter a question.")

    else:
        try:
            with st.spinner(
                "Retrieving document passages and generating an answer..."
            ):
                response = requests.post(
                    f"{api_url}/ask",
                    json={
                        "question": question.strip(),
                        "top_k": top_k
                    },
                    timeout=180
                )
                response.raise_for_status()
                result = response.json()

            if "body" in result:
                body = result["body"]

                if isinstance(body, str):
                    body = json.loads(body)

                if not isinstance(body, dict):
                    raise ValueError(
                        "The API response body is not a JSON object."
                    )

                result = body

            if not isinstance(result, dict):
                raise ValueError(
                    "The API response is not a JSON object."
                )

            st.session_state["last_answer_result"] = result

        except requests.RequestException as exc:
            st.error(f"Question request failed: {exc}")

            if getattr(exc, "response", None) is not None:
                st.code(exc.response.text)

        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            st.error(f"Could not read the API response: {exc}")

# --------------------------------------------------
# Display answer and sources
# --------------------------------------------------

result = st.session_state.get("last_answer_result")

if result:
    st.divider()
    st.subheader("Answer")

    st.markdown(
        result.get("answer", "No answer was returned.")
    )

    citations = result.get("citations", [])
    unique_citations = []
    seen = set()

    for citation in citations:
        if not isinstance(citation, dict):
            continue

        document = citation.get("document_name") or "Unknown document"
        page = citation.get("page_number")
        citation_key = (document, page)

        if citation_key not in seen:
            seen.add(citation_key)
            unique_citations.append({
                "document_name": document,
                "page_number": page
            })

    st.subheader("Sources")

    if unique_citations:
        for index, citation in enumerate(unique_citations, start=1):
            document = citation["document_name"]
            page = citation["page_number"]

            page_text = (
                f"Page {page}"
                if page is not None
                else "Page unavailable"
            )

            st.markdown(
                f"{index}. **{document}** — {page_text}"
            )
    else:
        st.info("No supporting source citations were returned.")

    if "retrieved_chunks" in result:
        st.caption(
            f"Passages retrieved: {result['retrieved_chunks']}"
        )

st.divider()
st.caption(
    "AWS Document RAG | Amazon S3 • AWS Lambda • Amazon Bedrock"
)

