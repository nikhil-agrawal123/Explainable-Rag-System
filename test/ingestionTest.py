import asyncio
# Point this to a real PDF on your computer
TEST_FILE = "data/uploads/2017-ISWC-20170727.pdf" 

from app.pipeline.stage_1_ingestion import IngestionPipeline

async def main():
    pipeline = IngestionPipeline()
    await pipeline.process_document(TEST_FILE)

if __name__ == "__main__":
    asyncio.run(main())