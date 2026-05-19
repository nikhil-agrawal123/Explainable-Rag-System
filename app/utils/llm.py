import json
import logging
from typing import Any, List

from langsmith import traceable
from langchain_ollama import ChatOllama
from langchain.messages import HumanMessage
from langchain_core.messages import ChatMessage
from app.core.config import settings

logger = logging.getLogger(__name__)


# Lazy LLM singleton — created on first use, not at import time,
# so tests can mock it without needing a running Ollama server.
_llm = None


def get_llm():
    global _llm
    if _llm is None:
        _llm = ChatOllama(
            model=settings.OLLAMA_DOMAIN_MODEL,
            base_url=settings.OLLAMA_HOST,
            temperature=0.0,
        )
    return _llm


def _extract_json_payload(raw_content: Any) -> str:
    if raw_content is None:
        return ""

    response = str(raw_content).strip()
    if not response:
        return ""

    if response.startswith("```"):
        parts = response.split("```")
        if len(parts) >= 2:
            response = parts[1].strip()
        if response.lower().startswith("json"):
            response = response[4:].strip()

    return response.strip()


@traceable(name="LLM Invocation", run_type="llm")
def domain_classification(text: str = None) -> List[str]:

    system_prompt = """You are a domain classification assistant for a RAG system.
        Your task is to classify the given text into one or more knowledge domains.

        Rules:
        1. Identify the primary domain(s) the text belongs to
        2. Use standard academic/professional domain names
        3. Return ONLY domain names separated by commas, nothing else
        4. Be specific but not overly narrow (e.g., "Physics" not "Quantum Mechanics")

        Common domains: Mathematics, Physics, Chemistry, Biology, Computer Science, History, 
        Literature, Philosophy, Economics, Law, Medicine, Engineering, Psychology, Sociology, 
        Political Science, Geography, Art, Music, Linguistics, General Knowledge

        Example input: "Einstein developed the theory of relativity which revolutionized our understanding of space and time."
        Example output: Physics

        Example input: "The stock market crashed in 1929, leading to the Great Depression."
        Example output: Economics, History

        If uncertain, return: General Knowledge
        """
    
    if not text:
        return []

    try:
        messages = [
            ChatMessage(role="assistant", content=system_prompt),
            HumanMessage(content = text)
        ]

        ai_msg = get_llm().invoke(messages)
        logger.debug("LLM domain response: %s", ai_msg.content)
        return ai_msg.content.split(",") if ai_msg.content else []
    except Exception as e:
        logger.error("Error during LLM invocation: %s", e)
        return []

@traceable(name="Entity Extraction", run_type="llm")
def extract_entities(text: str):
    system_prompt = """You are an entity extraction assistant for a RAG system.
        Your task is to extract key entities from the given text.

        Rules:
        1. Extract named entities, concepts, dates, and other significant terms from the text to the best of your ability
        2. Return a JSON array of entity names
        3. If no entities found, return an empty array []

        Example input: "Albert Einstein developed the theory of relativity in 1905."
        Example output: ["Albert Einstein", "theory of relativity", "1905"]

        If no entities exist, return: []
        """
    
    if not text:
        return []

    try:
        messages = [
            ChatMessage(role="assistant", content=system_prompt),
            HumanMessage(content=f"Extract entities from:\n{text}")
        ]

        ai_msg = get_llm().invoke(messages)
        response = _extract_json_payload(getattr(ai_msg, "content", ""))
        if not response:
            return []

        entities = json.loads(response)
        
        # Validate structure: list of strings
        valid_entities = []
        for ent in entities:
            if isinstance(ent, str):
                valid_entities.append(ent.strip())
        
        return valid_entities
    except Exception as e:
        logger.error("Error during entity extraction: %s", e)
        return []

@traceable(name="Relation Extraction", run_type="llm")
def extract_relations(text: str) -> List[List[str]]:
    system_prompt = """

        You are a knowledge extraction assistant for a RAG system.
        Your task is to extract subject-predicate-object triples from the given text.

        ## Core Rules

        1. **Extract only factual relationships** explicitly stated or strongly implied in the text
        2. **Subject and object** should be named entities, concepts, or noun phrases
        3. **Predicate** should be a verb or verb phrase describing the relationship
        4. **Keep extractions concise** but preserve important qualifiers and context
        5. **Handle complex sentences** by breaking them into multiple related triples
        6. **Return ONLY a JSON array of arrays**, nothing else

        ## Advanced Extraction Guidelines

        ### Compound Relationships
        - Break complex sentences into multiple related triples
        - Preserve temporal and causal connections
        - Extract relationships from subordinate clauses

        ### Qualifiers and Modifiers
        - Include important adjectives and adverbs that add meaning
        - Capture location, time, and manner information
        - Preserve negation when it changes the relationship's meaning

        ### Implicit Relationships
        - Extract relationships that are logically implied but not explicitly stated
        - Connect related concepts through shared attributes
        - Identify causal chains and sequences

        ### Entity Resolution
        - Use consistent entity names across extractions
        - Handle pronouns by resolving to their antecedents
        - Recognize synonyms and related terms

        ## Examples

        ### Simple Extraction
        **Input:** "Albert Einstein developed the theory of relativity in 1905."
        **Output:** [["Albert Einstein", "developed", "theory of relativity"], ["theory of relativity", "developed in", "1905"]]

        ### Multiple Relationships
        **Input:** "Apple Inc. was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in 1976 to create personal computers."
        **Output:** [
            ["Apple Inc.", "founded by", "Steve Jobs"],
            ["Apple Inc.", "founded by", "Steve Wozniak"],
            ["Apple Inc.", "founded by", "Ronald Wayne"],
            ["Apple Inc.", "founded in", "1976"],
            ["Apple Inc.", "founded to create", "personal computers"]
        ]

        ### Complex Sentence Structure
        **Input:** "After graduating from MIT in 2015, Dr. Chen joined NASA's Jet Propulsion Laboratory where she led the Mars rover project until 2020."
        **Output:** [
            ["Dr. Chen", "graduated from", "MIT"],
            ["Dr. Chen", "graduated in", "2015"],
            ["Dr. Chen", "joined", "NASA's Jet Propulsion Laboratory"],
            ["Dr. Chen", "led", "Mars rover project"],
            ["Mars rover project", "led until", "2020"]
        ]

        ### Implicit and Causal Relationships
        **Input:** "The company's rapid expansion caused significant strain on their supply chain, resulting in delayed deliveries and customer complaints."
        **Output:** [
            ["company", "experienced", "rapid expansion"],
            ["rapid expansion", "caused", "significant strain on supply chain"],
            ["supply chain strain", "resulted in", "delayed deliveries"],
            ["supply chain strain", "resulted in", "customer complaints"]
        ]

        ### Negation and Qualifiers
        **Input:** "Despite initial skepticism, the study proved that the treatment was not effective for patients over 65."
        **Output:** [
            ["study", "proved", "treatment not effective for patients over 65"],
            ["treatment", "not effective for", "patients over 65"]
        ]

        ### Temporal and Sequential Relationships
        **Input:** "First, the team conducted market research, then developed prototypes, and finally launched the product in Q4 2023."
        **Output:** [
            ["team", "conducted", "market research"],
            ["team", "developed", "prototypes"],
            ["team", "launched", "product in Q4 2023"],
            ["market research", "preceded", "prototypes development"],
            ["prototypes development", "preceded", "product launch"]
        ]

        ### Conditional and Hypothetical
        **Input:** "If the merger is approved, the combined company will become the largest in the industry."
        **Output:** [
            ["merger approval", "will result in", "combined company becoming largest in industry"],
            ["combined company", "will become", "largest in industry if merger approved"]
        ]

        ### Attribute and Property Extraction
        **Input:** "The new electric vehicle has a range of 400 miles, can accelerate from 0-60 in 3.5 seconds, and costs $45,000."
        **Output:** [
            ["new electric vehicle", "has range of", "400 miles"],
            ["new electric vehicle", "can accelerate from 0-60 in", "3.5 seconds"],
            ["new electric vehicle", "costs", "$45,000"]
        ]

        ## Edge Cases

        **No clear relationships:** Return `[]`
        **Ambiguous text:** Extract the most likely factual relationships
        **Metaphorical language:** Focus on literal, extractable relationships
        **Questions:** Extract relationships from the premise, not the question itself

        ## Quality Standards

        - Each triple should represent a distinct, meaningful relationship
        - Avoid redundant or trivial extractions
        - Maintain consistency in predicate phrasing
        - Preserve important context and qualifiers
        - Ensure subjects and objects are identifiable entities

        """
    
    if not text: 
        return []

    try:
        messages = [
            ChatMessage(role="assistant", content=system_prompt),
            HumanMessage(content=f"Extract relations from:\n{text}")
        ]

        ai_msg = get_llm().invoke(messages)
        logger.debug("LLM relation response: %s", ai_msg.content)
        response = _extract_json_payload(getattr(ai_msg, "content", ""))
        if not response:
            return []

        relations = json.loads(response)
        
        # Validate structure: list of [subject, predicate, object]
        valid_relations = []
        for rel in relations:
            if isinstance(rel, list) and len(rel) == 3:
                valid_relations.append([str(r).strip() for r in rel])
        
        return valid_relations
    except Exception as e:
        logger.error("Error during relation extraction: %s", e)
        return []