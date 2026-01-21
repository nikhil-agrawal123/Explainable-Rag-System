import ollama
from langsmith import traceable

@traceable(name="Domain Classification", run_type="tool", save_result=True, use_cache=True)
def domain_classification(text: str) -> str:
    prompt = f"You are a profession domain classifier given the text classify the following text into a knowledge domain (e.g., 'Computer Science', 'Mathematics', 'History', etc.):\n\n{text}\n\nDomain:"
    
    try:
        response = ollama.chat(
            model="quen2.5:7b",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0.0
        )
        domain = response['choices'][0]['message']['content'].strip()
        return domain
    except Exception as e:
        print(f"Error during domain classification: {e}")
        return "General Knowledge"