# app.py
import os
import re
from flask import Flask, render_template, request, redirect, url_for, flash
import openai

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET", "change_this_for_prod")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Set OPENAI_API_KEY environment variable before running.")
openai.api_key = OPENAI_API_KEY

def preprocess(text):
    text = text.strip().lower()
    text = re.sub(r"[^\w\s]", " ", text)
    tokens = [t for t in text.split() if t]
    return " ".join(tokens), tokens

def construct_prompt(processed_question):
    return (
        "You are a helpful, concise question-answering assistant.\n\n"
        "Question (processed):\n"
        f"{processed_question}\n\n"
        "Provide a clear, concise answer suitable for a student."
    )

def query_openai_chat(prompt, model="gpt-3.5-turbo", max_tokens=300, temperature=0.2):
    resp = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant for an academic project."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    text = resp["choices"][0]["message"]["content"].strip()
    return text, resp

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        question = request.form.get("question", "").strip()
        if not question:
            flash("Please enter a question.", "warning")
            return redirect(url_for("index"))

        processed, tokens = preprocess(question)
        prompt = construct_prompt(processed)

        try:
            answer, raw = query_openai_chat(prompt)
        except Exception as e:
            flash(f"LLM API error: {e}", "danger")
            return render_template("index.html", question=question, processed=processed, tokens=tokens, answer=None, raw=None)

        # Prepare a small safe-to-render raw extract
        raw_short = {
            "model": raw.get("model"),
            "usage": raw.get("usage"),
            "prompt_len": len(prompt),
        }
        return render_template("index.html",
                               question=question,
                               processed=processed,
                               tokens=tokens,
                               answer=answer,
                               raw=raw_short)

    return render_template("index.html", question=None, processed=None, tokens=None, answer=None, raw=None)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.getenv("PORT", 5000)))
