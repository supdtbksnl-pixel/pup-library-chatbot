import os
import json
import numpy as np
import cohere
from flask import Flask, render_template, request, session, redirect
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired
import secrets

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)

# -----------------------------------
# LOAD API
# -----------------------------------

api_key = os.getenv("COHERE_API_KEY")
if not api_key:
    raise ValueError("COHERE_API_KEY not set.")

co = cohere.ClientV2(api_key)

# -----------------------------------
# LOAD KNOWLEDGE BASE
# -----------------------------------

with open("university_data.json", "r", encoding="utf-8") as f:
    knowledge_base = json.load(f)

# -----------------------------------
# CREATE DOCUMENT EMBEDDINGS
# -----------------------------------

texts = [entry["content"] for entry in knowledge_base]

response = co.embed(
    model="embed-english-v3.0",
    input_type="search_document",
    texts=texts
)

document_embeddings = np.array(response.embeddings.float)

# -----------------------------------
# COSINE SIMILARITY
# -----------------------------------

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# -----------------------------------
# RETRIEVAL FUNCTION
# -----------------------------------

def retrieve_relevant_doc(query):

    response = co.embed(
        model="embed-english-v3.0",
        input_type="search_query",
        texts=[query]
    )

    query_embedding = np.array(response.embeddings.float[0])

    similarities = [
        cosine_similarity(query_embedding, doc_embedding)
        for doc_embedding in document_embeddings
    ]

    best_index = int(np.argmax(similarities))
    best_score = similarities[best_index]

    return knowledge_base[best_index], best_score

# -----------------------------------
# FORM
# -----------------------------------

class Form(FlaskForm):
    text = StringField("", validators=[DataRequired()])
    submit = SubmitField("Send")

# -----------------------------------
# ROUTE
# -----------------------------------

@app.route("/", methods=["GET", "POST"])
def home():

    form = Form()

    # Reset chat automatically when chatbot opens
    if request.args.get("reset") == "1":
        session.pop("chat_history", None)
        return redirect("/")

    if "language" not in session:
        session["language"] = "en"

    # Language switch clears chat
    if request.args.get("lang"):
        new_lang = request.args.get("lang")

        if session.get("language") != new_lang:
            session["language"] = new_lang
            session.pop("chat_history", None)

        return redirect("/")

    if "chat_history" not in session:
        session["chat_history"] = []

    if form.validate_on_submit():

        user_input = form.text.data.strip()
        lower_input = user_input.lower()

        # -----------------------------------
        # BOOK SEARCH HANDLER
        # -----------------------------------

        book_keywords = ["book", "books", "catalog", "catalogue"]

        if any(word in lower_input for word in book_keywords):

            if session["language"] == "pa":
                assistant_reply = (
                    "ਕਿਤਾਬਾਂ ਖੋਜਣ ਲਈ Web OPAC ਵਰਤੋਂ ਕਰੋ। "
                    "ਤੁਸੀਂ title, author ਜਾਂ subject ਨਾਲ ਖੋਜ ਕਰ ਸਕਦੇ ਹੋ.\n\n"
                    "http://10.10.85.51/"
                )
            else:
                assistant_reply = (
                    "To search for books available in the library, "
                    "please use Web OPAC.\n\n"
                    "http://10.10.85.51/"
                )

        else:

            greetings = ["hi", "hello", "hey", "hlo"]

            if lower_input in greetings:

                if session["language"] == "pa":
                    assistant_reply = "ਸਤ ਸ੍ਰੀ ਅਕਾਲ! ਮੈਂ ਭਾਈ ਕਾਨ੍ਹ ਸਿੰਘ ਨਾਭਾ ਲਾਇਬ੍ਰੇਰੀ ਦਾ AI ਸਹਾਇਕ ਹਾਂ। ਮੈਂ ਤੁਹਾਡੀ ਕਿਵੇਂ ਮਦਦ ਕਰ ਸਕਦਾ ਹਾਂ?"
                else:
                    assistant_reply = "Hello! I am the AI assistant of Bhai Kahn Singh Nabha Library. How can I assist you?"

            else:

                best_doc, score = retrieve_relevant_doc(user_input)

                if score >= 0.40:

                    context_text = best_doc["content"]

                    if session["language"] == "pa":

                        system_prompt = f"""
ਤੁਸੀਂ ਭਾਈ ਕਾਨ੍ਹ ਸਿੰਘ ਨਾਭਾ ਲਾਇਬ੍ਰੇਰੀ ਦੇ AI ਸਹਾਇਕ ਹੋ।

ਹੇਠਾਂ ਦਿੱਤੀ ਜਾਣਕਾਰੀ ਅਧਿਕਾਰਕ ਹੈ:

{context_text}

ਉਪਭੋਗਤਾ ਦੇ ਸਵਾਲ ਦਾ ਜਵਾਬ ਸਿਰਫ ਪੰਜਾਬੀ ਵਿੱਚ ਦਿਓ।
3-4 ਵਾਕਾਂ ਵਿੱਚ ਸਪਸ਼ਟ ਜਵਾਬ ਦਿਓ।
"""

                    else:

                        system_prompt = f"""
You are the official AI assistant of Bhai Kahn Singh Nabha Library.

Use ONLY this official information:

{context_text}

Answer clearly in 3-4 sentences.
"""

                    response = co.chat(
                        model="command-a-03-2025",
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_input}
                        ],
                        max_tokens=200,
                        temperature=0.2
                    )

                    assistant_reply = response.message.content[0].text.strip()

                else:

                    if session["language"] == "pa":
                        assistant_reply = "ਮਾਫ ਕਰਨਾ, ਇਸ ਸਵਾਲ ਲਈ ਜਾਣਕਾਰੀ ਉਪਲਬਧ ਨਹੀਂ ਹੈ।"
                    else:
                        assistant_reply = "Sorry, information for this query is not available."

        # -----------------------------------
        # STORE CHAT
        # -----------------------------------

        session["chat_history"].append({
            "role": "user",
            "content": user_input
        })

        session["chat_history"].append({
            "role": "assistant",
            "content": assistant_reply
        })

        session.modified = True
        form.text.data = ""

    return render_template(
        "home.html",
        form=form,
        chat_history=session.get("chat_history", []),
        language=session["language"]
    )

# -----------------------------------
# RUN
# -----------------------------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)