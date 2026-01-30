"""
Test to verify that start_time and end_time for audio chunks
are properly stored and retrieved from the database.
"""
import sys
import io

# Fix Windows console encoding for Unicode
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from app.models.schemas import ChunkRecord, ExtractedMetadata
from app.db.chroma_client import ChromaClient
from app.pipeline.stage_3_retrieval import MultiQueryRetrievalPipeline

def test_audio_metadata_storage_and_retrieval():
    """Test that audio chunk metadata with start_time and end_time is preserved through the full pipeline"""
    
    # Get collection
    collection = ChromaClient.get_collection(name="test_audio_metadata")
    
    try:
        # Create test audio chunk with start_time and end_time
        test_chunk = ChunkRecord(
            chunk_id="test_audio_CH0",
            document_id="test_audio",
            text="This is a test audio segment about machine learning and artificial intelligence.",
            source="test_audio.mp3",
            page_number=0,
            start_time=0.5,
            end_time=5.2,
            metadata=ExtractedMetadata()
        )
        
        # Save to database
        print("Saving test audio chunk...")
        collection.add(
            ids=[test_chunk.chunk_id],
            documents=[test_chunk.text],
            metadatas=[test_chunk.to_persistence_payload()]
        )
        
        # Test 1: Verify metadata is stored correctly in database
        print("\n[Test 1] Retrieving audio chunk from database...")
        results = collection.get(
            ids=[test_chunk.chunk_id],
            include=["metadatas", "documents"]
        )
        
        retrieved_meta = results["metadatas"][0]
        
        print(f"Retrieved metadata: {retrieved_meta}")
        
        assert "start_time" in retrieved_meta, "start_time not found in retrieved metadata"
        assert "end_time" in retrieved_meta, "end_time not found in retrieved metadata"
        assert retrieved_meta["start_time"] == 0.5, f"Expected start_time=0.5, got {retrieved_meta['start_time']}"
        assert retrieved_meta["end_time"] == 5.2, f"Expected end_time=5.2, got {retrieved_meta['end_time']}"
        
        print("✓ Test 1 passed: start_time and end_time are stored in database")
        
        # Test 2: Verify metadata is properly reconstructed in ChunkRecord via retrieval pipeline
        print("\n[Test 2] Testing retrieval pipeline reconstruction...")
        retrieval_pipeline = MultiQueryRetrievalPipeline()
        
        # Query for the test chunk
        retrieved_chunks = retrieval_pipeline.retrieve_documents(
            sub_queries=["machine learning artificial intelligence"],
            k_per_query=5
        )
        
        # Find our test chunk in the results
        test_chunk_found = None
        for chunk in retrieved_chunks:
            if chunk.chunk_id == test_chunk.chunk_id:
                test_chunk_found = chunk
                break
        
        assert test_chunk_found is not None, "Test chunk not found in retrieval results"
        
        print(f"Retrieved chunk via pipeline: {test_chunk_found.chunk_id}")
        print(f"  start_time: {test_chunk_found.start_time}")
        print(f"  end_time: {test_chunk_found.end_time}")
        
        assert test_chunk_found.start_time == 0.5, f"Expected start_time=0.5, got {test_chunk_found.start_time}"
        assert test_chunk_found.end_time == 5.2, f"Expected end_time=5.2, got {test_chunk_found.end_time}"
        
        print("✓ Test 2 passed: ChunkRecord properly reconstructed with start_time and end_time")
        
    finally:
        # Clean up
        print("\nCleaning up test data...")
        collection.delete(ids=[test_chunk.chunk_id])
    
    return True

if __name__ == "__main__":
    try:
        test_audio_metadata_storage_and_retrieval()
        print("\n✅ All tests passed!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
