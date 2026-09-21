import requests
import streamlit as st

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Climate QA", layout="wide")

st.markdown("""
<style>
    html, body, [class*="css"] {
        font-family: -apple-system, "Segoe UI", Helvetica, Arial, sans-serif;
    }

    .block-container {
        padding-top: 3rem;
        max-width: 1500px;
    }

    h1 {
        font-weight: 600;
        font-size: 2rem;
        color: #37352F;
        margin-bottom: 0.2rem;
    }

    h3 {
        font-weight: 600;
        color: #37352F;
    }

    .section-label {
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        color: #9B9A97;
        margin-bottom: 0.6rem;
    }

    .note {
        color: #9B9A97;
        font-size: 0.92rem;
    }
    .note-success {
        color: #37352F;
        font-size: 0.92rem;
    }
    .note-error {
        color: #E03E3E;
        font-size: 0.92rem;
    }

    div.stButton > button {
        background-color: #FFFFFF;
        color: #37352F;
        border: 1px solid #E3E2E0;
        border-radius: 6px;
        padding: 0.4rem 1rem;
        font-weight: 500;
        box-shadow: none;
    }
    div.stButton > button:hover {
        border-color: #37352F;
        color: #37352F;
    }

    div[data-testid="stTextInput"] input {
        border: 1px solid #E3E2E0;
        border-radius: 6px;
        padding: 0.5rem;
    }

    section[data-testid="stFileUploaderDropzone"] {
        background-color: #F7F6F3;
        border: 1px dashed #E3E2E0;
        box-shadow: none;
    }

    hr {
        border-top: 1px solid #EDECEA;
    }

    .answer-box {
        border-top: 1px solid #EDECEA;
        padding-top: 1rem;
        margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

st.title("Climate AI Q&A")
st.markdown('<div class="note">Ask questions grounded in your own PDF.</div>', unsafe_allow_html=True)
st.write("")

if "doc_id" not in st.session_state:
    st.session_state.doc_id = None
if "filename" not in st.session_state:
    st.session_state.filename = None

left, right = st.columns([1, 2], gap="large")

with left:
    st.markdown('<div class="section-label">Upload</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose a PDF", type="pdf", label_visibility="collapsed")

    if uploaded_file is not None and st.button("Index this PDF"):
        with st.spinner("Building index..."):
            try:
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                resp = requests.post(f"{API_URL}/upload", files=files, timeout=300)
                resp.raise_for_status()
                data = resp.json()
                st.session_state.doc_id = data["doc_id"]
                st.session_state.filename = uploaded_file.name
                st.markdown(
                    f'<div class="note-success">Indexed {data["chunks_indexed"]} chunks from {uploaded_file.name}</div>',
                    unsafe_allow_html=True,
                )
            except requests.exceptions.RequestException as e:
                st.markdown(f'<div class="note-error">Upload failed: {e}</div>', unsafe_allow_html=True)

    if st.session_state.doc_id:
        st.write("")
        st.markdown(f'<div class="note">Active: <b>{st.session_state.filename}</b></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="note">doc_id: {st.session_state.doc_id}</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="section-label">Ask</div>', unsafe_allow_html=True)

    if not st.session_state.doc_id:
        st.markdown('<div class="note">Upload and index a PDF first.</div>', unsafe_allow_html=True)
    else:
        question = st.text_input("Your question", label_visibility="collapsed", placeholder="Ask something about the paper...")

        if st.button("Ask") and question.strip():
            with st.spinner("Thinking..."):
                try:
                    resp = requests.post(
                        f"{API_URL}/ask",
                        json={"question": question, "doc_id": st.session_state.doc_id},
                        timeout=300,
                    )
                    resp.raise_for_status()
                    result = resp.json()

                    st.markdown('<div class="answer-box">', unsafe_allow_html=True)
                    st.write(result["answer"])
                    st.markdown(
                        f'<div class="note">Faithfulness score: {result["score"]:.3f} · attempts: {result["attempts"]}</div>',
                        unsafe_allow_html=True,
                    )
                    st.markdown('</div>', unsafe_allow_html=True)

                    with st.expander("See all attempts"):
                        for h in result["history"]:
                            st.markdown(f"**Attempt {h['attempt']}** — score: {h['faithfulness_score']:.3f}")
                            st.write(h["answer"])
                            st.markdown("---")

                except requests.exceptions.RequestException as e:
                    st.markdown(f'<div class="note-error">Request failed: {e}</div>', unsafe_allow_html=True)