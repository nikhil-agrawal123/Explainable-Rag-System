# Stage 5: Answer generation and validation
# - LLM Generation (constrained by Evidence Graph)
# - Claim validation against evidence
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langsmith import traceable
from typing import List
from app.models.schemas import ChunkRecord

class GenerationEngine:
    def __init__(self, model_name: str = "qwen2.5:7b"):
        print(f"Loading Generator ({model_name})...")
        self.llm = ChatOllama(model=model_name, temperature=0.1) # Low temp = factual
        
        # --- THE EVIDENCE PROMPT ---
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """
            You are DataForge, an evidence-based AI assistant.
            
            CORE INSTRUCTIONS:
            1. Answer the user's question using ONLY the provided Context and Knowledge Graph.
            2. CITATIONS ARE MANDATORY: Every claim you make must immediately be followed by its Source ID in square brackets. 
               Example: "Bernoulli published Ars Conjectandi in 1713 [DataForge_CH5]."
            3. USE THE GRAPH: Use the provided relationships to connect concepts.
            4. If the context is empty or irrelevant, say "I cannot find evidence for that in the provided documents."
            
            """),
            ("human", """
            --- TEXT CONTEXT ---
            {text_context}
            
            --- GRAPH CONTEXT (Verified Relations) ---
            {graph_context}
            
            --- USER QUESTION ---
            {question}
            
            Answer:
            """)
        ])
        
        self.chain = self.prompt | self.llm | StrOutputParser()

    @traceable(name="Stage5_Generation", run_type="chain")
    def generate_answer(self, query: str, chunks: List[ChunkRecord], graph_summary: str) -> str:
        print(f"Generating answer for: '{query}'")
        
        # 1. Format the Text Context
        text_context = "\n\n".join(
            [f"[{c.chunk_id}] {c.text}" for c in chunks]
        )
        
        # 2. Run the Chain
        response = self.chain.invoke({
            "question": query,
            "text_context": text_context,
            "graph_context": graph_summary
        })
        
        return response