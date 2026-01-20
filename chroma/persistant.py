import json
import chromadb

load_dotenv()

client = chromadb.PersistentClient(path="./persistent_chroma_db")

def Client():
    return client

def get_or_make(name: str):
    return client.get_or_create_collection(name)

def add_example_document():
    collection = get_or_make("documents")
    doc_text = "Jakob Bernoulli introduced the Law of Large Numbers in his work Ars Conjectandi."
    entities_list=["Jakob Bernoulli", "Law of Large Numbers", "Ars Conjectandi"]
    relations_data=[
        ["Jakob Bernoulli", "introduced", "Law of Large Numbers"],
        ["Jakob Bernoulli", "wrote", "Ars Conjectandi"]
    ]
    collection.add(
        documents=[doc_text],
        metadatas=[{
            "document_id": "DOC3",             
            "chunk_id": "DOC3_CH5",            
            "domain": "Probability Theory",    
            "entities": json.dumps(entities_list),  
            "relations": json.dumps(relations_data)  
        }],
        ids=["DOC3_CH5"] # Using chunk_id as the unique vector ID is a good practice
    )
    print("Example document added.")
