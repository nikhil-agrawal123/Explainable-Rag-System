import ollama
from langsmith import traceable
from dotenv import load_dotenv
import os

load_dotenv(override=False)

@traceable(name="Domain Classification", run_type="tool", save_result=True, use_cache=True)
def domain_classification(text: str) -> str:
    model = "qwen2.5:7b"
    prompt = (
        "You are a professional domain classifier. Classify the following text into a single knowledge domain "
        "(e.g., Computer Science, Mathematics, History). Return ONLY the domain name.\n\n"
        f"{text}\n\nDomain:"
    )
    
    try:
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.0, "num_predict": 10},
        )

        content = response.message.content.strip()

        if "Domain:" in content:
            domain = content.split("Domain:")[-1].strip()
        elif "domain:" in content:
            domain = content.split("domain:")[-1].strip()
        else:
            lines = [l.strip() for l in content.splitlines() if l.strip()]
            domain = lines[-1] if lines else "General Knowledge"

        # Clean up any extra punctuation
        domain = domain.strip('."\'')

        return domain or "General Knowledge"
    except Exception as e:
        print(f"Error during domain classification: {e}")
        return "General Knowledge"