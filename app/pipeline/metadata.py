import spacy
from app.models.schemas import Relation, ExtractedMetadata
from langsmith import traceable
from app.utils.llm import domain_classification
from dotenv import load_dotenv

load_dotenv(override=False)

class MetadataExtractor:
    def __init__(self):
        print("Loading Spacy model for metadata extraction...")
        try:
            self.nlp = spacy.load("en_core_web_trf")
            print("Spacy model loaded successfully.")
        except Exception as e:
            print(f"Error loading Spacy model: {e}")
            raise


    @traceable(name="Extract Metadata", run_type="tool", save_result=True, use_cache=True)
    def extract_metadata(self, chunck: str) -> ExtractedMetadata:
        doc = self.nlp(chunck)  

        found_entities = [ent.text for ent in doc.ents]

        #Local LLM call for domain classification
        domain_guess = domain_classification(chunck)  
        # domain_guess = "General Knowledge"  

        relations = []        
        # Iterating over verbs to find who did what
        for token in doc:
            if token.pos_ == "VERB":
                subj = [child.text for child in token.children if child.dep_ == "nsubj"]
                obj = [child.text for child in token.children if child.dep_ == "dobj"]
                
                if subj and obj:
                    relations.append(
                        Relation(
                            subject=subj[0], 
                            predicate=token.lemma_,
                            object=obj[0]
                        )
                    )

        return ExtractedMetadata(
            entities=set(found_entities),
            relations=relations,
            domain=domain_guess  
        )
