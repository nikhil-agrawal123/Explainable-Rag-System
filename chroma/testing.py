import chromadb
from sentence_transformers import SentenceTransformer
client = chromadb.Client()

collections = client.create_collection(name="test_collection")
model = SentenceTransformer('all-MiniLM-L6-v2')


def embeddingFunction(texts):
    return model.encode(texts,normalize_embeddings=True)

def addDocumentsToCollection(collection, docs, metas, ids):
    collection.add(
        documents=docs,
        metadatas=metas,
        ids=ids,
        embeddings=embeddingFunction(docs)
    )

def basicQueryCollection(collection, query_texts, n_results):
    query_embeddings = model.encode(query_texts, normalize_embeddings=True)
    results = collection.query(
        query_embeddings=query_embeddings,
        n_results=n_results
    )
    return results