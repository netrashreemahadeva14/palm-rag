
import json
import os
import uuid
from pathlib import Path

from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

load_dotenv()

DATA_FILE = Path("knowledge_base.jsonl")
QDRANT_PATH = os.getenv("QDRANT_PATH", "./qdrant_data")
COLLECTION = os.getenv(
    "COLLECTION_NAME", "palmistry_references"
)
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)


def load_documents():
    if not DATA_FILE.exists():
        raise FileNotFoundError(DATA_FILE)

    documents = []

    with DATA_FILE.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            doc = json.loads(line)
            required = ["id", "title", "text", "source",
                        "tradition", "topic", "language",
                        "source_quality"]

            missing = [
                key for key in required
                if not doc.get(key)
            ]
            if missing:
                raise ValueError(
                    f"Line {line_number}: missing {missing}"
                )

            documents.append(doc)

    if not documents:
        raise ValueError("Knowledge base is empty.")

    return documents


def main():
    documents = load_documents()

    encoder = SentenceTransformer(EMBEDDING_MODEL)
    client = QdrantClient(path=QDRANT_PATH)

    dimension = encoder.get_sentence_embedding_dimension()

    if client.collection_exists(COLLECTION):
        info = client.get_collection(COLLECTION)
        if info.config.params.vectors.size != dimension:
            client.close()
            raise ValueError(
                "Embedding dimensions changed. Use a new "
                "collection name or rebuild the collection."
            )
    else:
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=models.VectorParams(
                size=dimension,
                distance=models.Distance.COSINE,
            ),
        )

        # Index fields that are frequently filtered.
        for field in [
            "topic", "tradition", "language", "source_quality"
        ]:
            client.create_payload_index(
                collection_name=COLLECTION,
                field_name=field,
                field_schema=models.PayloadSchemaType.KEYWORD,
            )

    # Batch embedding is more efficient than one request per text.
    texts = [
        f"{doc['title']}\n{doc['text']}"
        for doc in documents
    ]
    vectors = encoder.encode(
        texts,
        batch_size=32,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    points = []
    for doc, vector in zip(documents, vectors):
        point_id = str(
            uuid.uuid5(uuid.NAMESPACE_URL, doc["id"])
        )
        points.append(
            models.PointStruct(
                id=point_id,
                vector=vector.tolist(),
                payload=doc,
            )
        )

    # Re-running ingestion updates matching IDs instead
    # of creating duplicate copies of the same documents.
    client.upload_points(
        collection_name=COLLECTION,
        points=points,
        batch_size=64,
        parallel=1,
    )

    print(f"Indexed {len(points)} documents.")
    print(f"Collection: {COLLECTION}")
    print(f"Embedding dimensions: {dimension}")
    print(f"Qdrant path: {QDRANT_PATH}")

    client.close()


if __name__ == "__main__":
    main()
