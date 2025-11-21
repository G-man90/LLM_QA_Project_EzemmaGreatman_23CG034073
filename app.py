from flask import Flask, render_template, request
import os
import openai
import re

app = Flask(__name__)

# Get your OpenAI API key from environment variable
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Set OPENAI_API_KEY environment variable before running.")

openai.api_key = OPENAI_API_KEY

# Simple tokenizer function
def tokenize(text):
    text = text.lower()                  # lowercase
    text = re.sub(r'[^\w\s]', '', text)  # remove punctuation
    tokens = text.split()                 # split by whitespace
    return tokens

@app.route("/", methods=["GET", "POST"])
def index():
    processed_question = ""
    tokens = []
    answer = ""
    response_metadata = None

    if request.method == "POST":
        question = request.form.get("question", "")
        processed_question = question.lower().strip()
        tokens = tokenize(processed_question)

        try:
            # OpenAI Completion API call
            response = openai.Completion.create(
                engine="text-davinci-003",
                prompt=processed_question,
                max_tokens=150
            )
            answer = response['choices'][0]['text'].strip()
            response_metadata = {
                "id": response.get("id"),
                "model": response.get("model"),
                "created": response.get("created")
            }
        except Exception as e:
            answer = f"Error: {str(e)}"

    return render_template("index.html",
                           processed_question=processed_question,
                           tokens=tokens,
                           answer=answer,
                           response_metadata=response_metadata)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
