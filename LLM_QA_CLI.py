# LLM_QA_CLI.py
import os
import re
import sys
import json
import openai

# Ensure your OPENAI_API_KEY is set in the environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("Error: Set OPENAI_API_KEY environment variable before running.")
    sys.exit(1)

openai.api_key = OPENAI_API_KEY

def preprocess(text):
    """Lowercase, remove punctuation (keeps basic characters), simple tokenization."""
    text = text.strip().lower()
    text = re.sub(r"[^\w\s]", " ", text)      # remove punctuation (replace w/ space)
    tokens = [t for t in text.split() if t]
    processed = " ".join(tokens)
    return processed, tokens

def construct_prompt(processed_question):
    prompt = (
        "You are a helpful, concise question-answering assistant.\n\n"
        "Question (processed):\n"
        f"{processed_question}\n\n"
        "Provide a clear, concise answer. If the question seems ambiguous, ask a single short clarifying question."
    )
    return prompt

def query_openai_chat(prompt, model="gpt-3.5-turbo", max_tokens=300, temperature=0.2):
    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant for an academic project."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    # Extract text from response
    text = response["choices"][0]["message"]["content"].strip()
    return text, response

def main():
    print("LLM QA CLI — enter a question (type 'exit' to quit)\n")
    while True:
        q = input("Question: ").strip()
        if not q:
            continue
        if q.lower() in ("exit", "quit"):
            print("Goodbye.")
            break

        processed, tokens = preprocess(q)
        print("\n--- Preprocessed Question ---")
        print(processed)
        print("\n--- Tokens ---")
        print(tokens)

        prompt = construct_prompt(processed)
        print("\nSending to LLM...")
        try:
            answer, raw = query_openai_chat(prompt)
        except Exception as e:
            print("Error calling LLM API:", str(e))
            continue

        print("\n--- LLM Raw Response (truncated) ---")
        # Print first ~500 chars of raw JSON for debugging
        print(json.dumps(raw, indent=2)[:1000] + ("..." if len(json.dumps(raw)) > 1000 else ""))
        print("\n--- Final Answer ---")
        print(answer)
        print("\n" + "="*40 + "\n")

if __name__ == "__main__":
    main()
