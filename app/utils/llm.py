from sympy import content
import ollama
from langsmith import traceable
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain.messages import HumanMessage
from langchain_core.messages import ChatMessage

load_dotenv(override=False)

@traceable(name="LLM Invocation", run_type="llm", save_result=True, use_cache=True)
def domain_classification(text: str = None) -> str:
    llm = ChatOllama(model="qwen2.5:7b", temperature=0.0)

    system_prompt = """
        You are a helpful assistant. Your primary task is domain classification.
        Given an input text, identify the domain(s) it belongs to.
        Reply only with the domain name(s) and nothing else.
        If the text belongs to multiple domains, return them in a list format.
        Ensure the classification is accurate and concise.
        """

    try:
        messages = [
            ChatMessage(role="assistant", content=system_prompt),
            HumanMessage(content = text)
        ]

        ai_msg = llm.invoke(messages)
        return ai_msg.content
    except ollama.OllamaError as e:
        print(f"Error during LLM invocation: {e}")
        return "Unknown"