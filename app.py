
import base64
import io
import os
import time

import requests
import streamlit as st
from dotenv import load_dotenv
from PIL import Image, ImageOps
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer, CrossEncoder

load_dotenv()

XAI_KEY = os.getenv("XAI_API_KEY")
XAI_MODEL = os.getenv("XAI_MODEL", "grok-4.7")
QDRANT_PATH = os.getenv("QDRANT_PATH", "./qdrant_data")
COLLECTION = os.getenv(
    "COLLECTION_NAME", "palmistry_references"
)
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)
RERANKER_MODEL = os.getenv(
    "RERANKER_MODEL",
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
)
TOP_K = int(os.getenv("TOP_K", "20"))
FINAL_K = int(os.getenv("FINAL_K", "5"))
XAI_URL = "https://api.x.ai/v1/responses"

st.set_page_config(
    page_title="Palmistry RAG Explorer",
    page_icon="🔎",
    layout="wide",
)


@st.cache_resource
def get_encoder():
    return SentenceTransformer(EMBEDDING_MODEL)


@st.cache_resource
def get_reranker():
    return CrossEncoder(RERANKER_MODEL)


@st.cache_resource
def get_qdrant():
    return QdrantClient(path=QDRANT_PATH)


def extract_response_text(data):
    # Supports Responses API text output.
    if data.get("output_text"):
        return data["output_text"]

    chunks = []
    for item in data.get("output", []):
        for block in item.get("content", []):
            if block.get("type") in ("output_text", "text"):
                if block.get("text"):
                    chunks.append(block["text"])

    if not chunks:
        raise ValueError(
            "The model returned no text. Check the API response."
        )
    return "\n".join(chunks)


def call_grok(content):
    if not XAI_KEY:
        raise ValueError(
            "Set XAI_API_KEY in your .env file."
        )

    response = requests.post(
        XAI_URL,
        headers={
            "Authorization": f"Bearer {XAI_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": XAI_MODEL,
            "input": [
                {
                    "role": "user",
                    "content": content,
                }
            ],
            "store": False,
        },
        timeout=120,
    )

    if not response.ok:
        raise RuntimeError(
            f"xAI API error {response.status_code}: "
            f"{response.text[:1000]}"
        )

    return extract_response_text(response.json())


def prepare_image(upload):
    image = Image.open(upload)
    image = ImageOps.exif_transpose(image).convert("RGB")

    # Bound upload dimensions to control latency and payload size.
    image.thumbnail((1280, 1280))

    output = io.BytesIO()
    image.save(output, format="JPEG", quality=85, optimize=True)
    return output.getvalue(), image


def analyze_palm(image_bytes):
    encoded = base64.b64encode(image_bytes).decode("utf-8")
    data_url = f"data:image/jpeg;base64,{encoded}"

    instruction = """
    Describe the visible palm image for a cultural palmistry
    reference application.

    Report only visible observations: which hand is visible if
    identifiable, image clarity, orientation, and apparent major
    crease patterns. Use cautious wording when uncertain.
    Do not diagnose health, infer identity, or claim to know
    the person's personality or future from the photograph.
    Return a concise plain-text feature description.
    """

    return call_grok([
        {"type": "input_text", "text": instruction},
        {
            "type": "input_image",
            "image_url": data_url,
            "detail": "high",
        },
    ])


def retrieve_documents(query, topic, tradition, source_quality):
    encoder = get_encoder()
    client = get_qdrant()

    if not client.collection_exists(COLLECTION):
        raise RuntimeError(
            "Qdrant collection is missing. Run: python ingest.py"
        )

    query_vector = encoder.encode(
        query,
        normalize_embeddings=True,
    ).tolist()

    conditions = []

    for field, value in [
        ("topic", topic),
        ("tradition", tradition),
        ("source_quality", source_quality),
    ]:
        if value != "All":
            conditions.append(
                models.FieldCondition(
                    key=field,
                    match=models.MatchValue(value=value),
                )
            )

    query_filter = (
        models.Filter(must=conditions)
        if conditions else None
    )

    results = client.query_points(
        collection_name=COLLECTION,
        query=query_vector,
        query_filter=query_filter,
        limit=TOP_K,
        with_payload=True,
    ).points

    candidates = [
        point.payload for point in results
        if point.payload and point.payload.get("text")
    ]

    if not candidates:
        return []

    # The cross-encoder compares the query and each candidate
    # jointly; these scores are ranking signals, not probabilities.
    pairs = [
        (
            query,
            f"{doc.get('title', '')}\n{doc['text']}",
        )
        for doc in candidates
    ]

    scores = get_reranker().predict(pairs)

    ranked = sorted(
        zip(candidates, scores),
        key=lambda item: float(item[1]),
        reverse=True,
    )

    return [
        {**doc, "rerank_score": float(score)}
        for doc, score in ranked[:FINAL_K]
    ]


def generate_reading(question, features, documents):
    if not documents:
        return (
            "No matching references were found. Add reviewed "
            "sources to the knowledge base or change the filters."
        )

    context_parts = []
    for index, doc in enumerate(documents, start=1):
        context_parts.append(
            f"[S{index}]\n"
            f"Title: {doc.get('title', 'Untitled')}\n"
            f"Source: {doc.get('source', 'Not provided')}\n"
            f"Topic: {doc.get('topic', 'Unknown')}\n"
            f"Source quality: {doc.get('source_quality', 'Unknown')}\n"
            f"Text: {doc['text']}"
        )

    context = "\n\n".join(context_parts)

    prompt = f"""
    You are a careful cultural-reference assistant.

    Explain traditional palmistry interpretations as cultural
    beliefs or entertainment, not as scientifically established
    facts or reliable predictions.

    Rules:
    1. Use only the supplied references for claims about traditions.
    2. Cite supporting references with labels such as [S1].
    3. Never invent source titles, authors, URLs, or quotations.
    4. If evidence is missing, say so.
    5. Do not predict lifespan, illness, death, fertility,
       marriage outcomes, wealth, or specific future events.
    6. Do not claim that the photograph proves personality,
       intelligence, morality, or destiny.
    7. Distinguish visible image observations from traditional
       interpretations.
    8. Give practical, non-deterministic reflection prompts.

    Visible feature description:
    {features}

    User question:
    {question}

    Retrieved reference passages:
    {context}

    Write these sections:
    - Visible observations and uncertainties
    - Traditional interpretations supported by the sources
    - What cannot be inferred from the image
    - Reflection questions or practical next steps
    - Sources used, by reference label
    """

    return call_grok([
        {"type": "input_text", "text": prompt}
    ])


st.title("🔎 Palmistry RAG Explorer")
st.caption(
    "Image analysis + Qdrant retrieval + reranking + "
    "source-grounded cultural interpretation"
)

st.warning(
    "Palmistry is not a scientifically validated method of "
    "predicting the future. This tool provides cultural "
    "interpretations, not factual forecasts."
)

with st.sidebar:
    st.header("Retrieval settings")
    topic = st.selectbox(
        "Topic filter",
        [
            "All", "relationships", "personality",
            "life_line", "career", "limitations",
        ],
    )
    tradition = st.selectbox(
        "Tradition",
        ["All", "general"],
    )
    source_quality = st.selectbox(
        "Source quality",
        ["All", "reviewed", "demo"],
    )
    st.caption(
        f"Retrieve up to {TOP_K} candidates; rerank to {FINAL_K}."
    )

uploaded = st.file_uploader(
    "Upload a clear palm image",
    type=["png", "jpg", "jpeg", "webp"],
)

question = st.text_input(
    "What would you like to explore?",
    value="Explain the traditional symbolism of the visible palm lines.",
)

if uploaded is not None:
    try:
        image_bytes, preview = prepare_image(uploaded)
        st.image(
            preview,
            caption="Uploaded palm image",
            width=360,
        )

        if st.button("Analyze palm", type="primary"):
            start = time.perf_counter()

            with st.spinner("Analyzing visible features..."):
                features = analyze_palm(image_bytes)

            st.subheader("Visible feature description")
            st.write(features)

            retrieval_query = (
                f"User question: {question}\n"
                f"Visible palm features: {features}"
            )

            with st.spinner("Retrieving and reranking sources..."):
                documents = retrieve_documents(
                    retrieval_query,
                    topic,
                    tradition,
                    source_quality,
                )

            with st.spinner("Preparing source-grounded reading..."):
                answer = generate_reading(
                    question, features, documents
                )

            st.subheader("Interpretation")
            st.markdown(answer)

            st.subheader("Retrieved references")
            for index, doc in enumerate(documents, start=1):
                with st.expander(
                    f"[S{index}] {doc.get('title', 'Untitled')}"
                ):
                    st.write(doc.get("text", ""))
                    st.write(
                        f"Source: {doc.get('source', 'Not provided')}"
                    )
                    st.write(
                        f"Topic: {doc.get('topic', 'Unknown')}"
                    )
                    st.write(
                        f"Source quality: "
                        f"{doc.get('source_quality', 'Unknown')}"
                    )
                    st.write(
                        f"Reranker score: "
                        f"{doc['rerank_score']:.4f}"
                    )

            st.caption(
                f"End-to-end time: "
                f"{time.perf_counter() - start:.2f} seconds. "
                "Timing includes API latency and model warm-up."
            )

    except Exception as exc:
        st.error(f"Analysis failed: {exc}")

else:
    st.info("Upload a palm image to begin.")
