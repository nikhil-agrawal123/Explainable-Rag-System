from typing import List
from langsmith import traceable
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain.messages import HumanMessage
from langchain_core.messages import ChatMessage
import json

load_dotenv(override=False)

llm = ChatOllama(model="qwen2.5:7b", temperature=0.0)


@traceable(name="LLM Invocation", run_type="llm", save_result=True, use_cache=True)
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

    try:
        messages = [
            ChatMessage(role="assistant", content=system_prompt),
            HumanMessage(content = text)
        ]

        ai_msg = llm.invoke(messages)
        return ai_msg.content.split(",") if ai_msg.content else []
    except Exception as e:
        print(f"Error during LLM invocation: {e}")
        return []

@traceable(name="Entity Extraction", run_type="llm", save_result=True, use_cache=True)
def extract_entities(text: str) -> List[str]:
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

    try:
        messages = [
            ChatMessage(role="assistant", content=system_prompt),
            HumanMessage(content=f"Extract entities from:\n{text}")
        ]

        ai_msg = llm.invoke(messages)
        response = ai_msg.content.strip()
        
        if response.startswith("```"):
            response = response.split("```")[1]
            if response.startswith("json"):
                response = response[4:]
        response = response.strip()
        
        entities = json.loads(response)
        
        # Validate structure: list of strings
        valid_entities = []
        for ent in entities:
            if isinstance(ent, str):
                valid_entities.append(ent.strip())
        
        return valid_entities
    except Exception as e:
        print(f"Error during entity extraction: {e}")
        return []

@traceable(name="Relation Extraction", run_type="llm", save_result=True, use_cache=True)
def extract_relations(text: str) -> List[List[str]]:
    system_prompt = """You are a knowledge extraction assistant for a RAG system.
        Your task is to extract subject-predicate-object triples from the given text.

        Rules:
        1. Extract only factual relationships explicitly stated in the text
        2. Subject and object should be named entities, concepts, or noun phrases
        3. Predicate should be a verb or verb phrase describing the relationship
        4. Keep extractions concise and meaningful
        5. Return ONLY a JSON array of arrays, nothing else

        Example input: "Albert Einstein developed the theory of relativity in 1905."
        Example output: [["Albert Einstein", "developed", "theory of relativity"], ["theory of relativity", "developed in", "1905"]]

        If no clear relations exist, return: []
        """

    try:
        messages = [
            ChatMessage(role="assistant", content=system_prompt),
            HumanMessage(content=f"Extract relations from:\n{text}")
        ]

        ai_msg = llm.invoke(messages)
        response = ai_msg.content.strip()
        
        # Parse JSON response
        import json
        if response.startswith("```"):
            response = response.split("```")[1]
            if response.startswith("json"):
                response = response[4:]
        response = response.strip()
        
        relations = json.loads(response)
        
        # Validate structure: list of [subject, predicate, object]
        valid_relations = []
        for rel in relations:
            if isinstance(rel, list) and len(rel) == 3:
                valid_relations.append([str(r).strip() for r in rel])
        
        return valid_relations
    except Exception as e:
        print(f"Error during relation extraction: {e}")
        return []