import chromadb
import persistant

client = persistant.Client()
collection = persistant.get_or_make("documents")

results = collection.query(
    query_texts=["Bernoulli contributions"],
    n_results=1
)

print("Query Results:", results)
retived_data = results['metadatas'][0][0]