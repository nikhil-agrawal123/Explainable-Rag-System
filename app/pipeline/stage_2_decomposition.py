# Stage 2: User query decomposition
# - Break complex query into sub-queries
# - Domain mapping

from langchain_ollama import ChatOllama
from langchain.messages import AIMessage, HumanMessage
from langsmith import traceable
from dotenv import load_dotenv

load_dotenv(override=False)

message = []

# Query Decomposition Pipeline
class QueryDecompositionPipeline:
    def __init__(self):
        self.llm = ChatOllama(
            model = "qwen2.5:7b",
            temperature=0
        )

        self.system_prompt = """
        You are an expert query decomposition agent in a Retrieval-Augmented Generation (RAG) pipeline.
        Your task is to transform a user’s complex or ambiguous query into exactly five or less simpler, self-contained sub-queries that together fully cover the original intent.

            1. Decomposition Rules:
            2 .Each sub-query must be clear, specific, and easy to understand on its own.
            3. Sub-queries should be retrieval-optimized:   
            4. Avoid vague language, pronouns, or implicit references.
            5. Prefer concrete terms that are likely to appear in documents.
            6. Each sub-query must only be a small breakdown of the original query not going beyond its scope.

        Collectively, the five sub-queries must:

            1. Preserve the full semantic intent of the original query.
            2. Cover different aspects, constraints, or dimensions of the problem.
            3. Do not introduce assumptions, opinions, or information not present in the original query.
            4. Do not answer the query, summarize documents, or explain your reasoning.
            5. If possible, or the query is simple, you may return less number of sub-queries than five to optimize on the compute.

        Output Format:

            1. Return only the five sub-queries.
            2. Use a numbered list (1–5).
            3. Each item must be a single well-formed question or retrieval statement.
            4. Your output will be directly used for document retrieval. Precision and clarity are critical.
        """

    @traceable(name="Decompose Query", run_type="tool", save_result=True, use_cache=True)
    def decompose_query(self, query: str) -> list:
        messages = [
            AIMessage(content=self.system_prompt),
            HumanMessage(content=query)
        ]

        try:
            ai_msg = self.llm.invoke(messages)

            sub_queries = []
            for line in ai_msg.content.split("\n"):
                line = line.strip()
                if line and any(char.isdigit() for char in line[:2]):
                    # Remove numbering
                    sub_query = line.split(".", 1)[1].strip() if "." in line else line
                    sub_queries.append(sub_query)

            sub_queries.append(query)

            return sub_queries
        except Exception as e:
            print(f"Error during query decomposition: {e}")
            return []
