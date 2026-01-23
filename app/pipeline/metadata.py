from app.models.schemas import Relation, ExtractedMetadata
from langsmith import traceable
from app.utils.llm import domain_classification, extract_relations, extract_entities
from dotenv import load_dotenv

load_dotenv(override=False)

class MetadataExtractor:
    def __init__(self):
        pass

    @traceable(name="Extract Metadata", run_type="tool", save_result=True, use_cache=True)
    def extract_metadata(self, chunck: str) -> ExtractedMetadata:

        found_entities = extract_entities(chunck)

        #Local LLM call for domain classification
        domain_guess = domain_classification(chunck) 

        # Use LLM for relation extraction instead of SpaCy dependency parsing
        raw_relations = extract_relations(chunck)
        relations = [
            Relation(subject=rel[0], predicate=rel[1], object=rel[2])
            for rel in raw_relations
        ]

        return ExtractedMetadata(
            entities=set(found_entities),
            relations=relations,
            domain=domain_guess  
        )
