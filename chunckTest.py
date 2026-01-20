from langchain_core.documents import Document
from app.pipeline.chuncking import Chuncking

mock_file = """
Stage 1: Data Storage.
Most RAG systems lack transparency. This is a big problem.
We need to map entities like 'Jakob Bernoulli' and 'Probability Theory'.

Stage 2: Query Decomposition.
We will break down complex user questions.
"""

moc_doc = [Document(page_content=mock_file, metadata={"source": "test.pdf"})]

chuncker = Chuncking(chunk_size=100, chunk_overlap=10)
chuncks = chuncker.chunk_file(moc_doc)

print(f"--- Original Length: {len(mock_file)} chars ---")
print(f"--- Created {len(chuncks)} Chunks ---")

for i, chunk in enumerate(chuncks):
    print(f"\n[Chunk {i}] (Len: {len(chunk.page_content)})")
    print(f"'{chunk.page_content}'")