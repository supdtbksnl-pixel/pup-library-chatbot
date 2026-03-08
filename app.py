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
# CREATE DOCUMENT EMBEDDINGS (ONCE)
# -----------------------------------

texts = [entry["content"] for entry in knowledge_base]

BATCH_SIZE = 90
document_embeddings = []

for i in range(0, len(texts), BATCH_SIZE):

    batch = texts[i:i + BATCH_SIZE]

    response = co.embed(
        model="embed-english-v3.0",
        input_type="search_document",
        texts=batch
    )

    document_embeddings.extend(response.embeddings.float)

document_embeddings = np.array(document_embeddings)

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

    if "language" not in session:
        session["language"] = "en"

    if request.args.get("clear") == "1":
        session.pop("chat_history", None)
        return redirect("/")

    if request.args.get("lang"):
        session["language"] = request.args.get("lang")
        return redirect("/")

    if "chat_history" not in session:
        session["chat_history"] = []

    if form.validate_on_submit():

        user_input = form.text.data.strip()
        lower_input = user_input.lower()

        # -----------------------------------
        # BOOK SEARCH HANDLER (NEW RULE)
        # -----------------------------------

        book_keywords = ["book", "books", "find book", "search book", "available books", "catalogue", "catalog"]

        if any(word in lower_input for word in book_keywords):

            if session["language"] == "pa":
                assistant_reply = (
                    "ਲਾਇਬ੍ਰੇਰੀ ਵਿੱਚ ਉਪਲਬਧ ਕਿਤਾਬਾਂ ਨੂੰ ਖੋਜਣ ਲਈ ਕਿਰਪਾ ਕਰਕੇ "
                    "Web OPAC (Online Public Access Catalogue) ਦੀ ਵਰਤੋਂ ਕਰੋ। "
                    "ਤੁਸੀਂ title, author ਜਾਂ subject ਨਾਲ ਖੋਜ ਕਰ ਸਕਦੇ ਹੋ।\n\n"
                    "Search here: http://10.10.85.51/"
                )
            else:
                assistant_reply = (
                    "To search for books available in the University Library, "
                    "please use the Web OPAC (Online Public Access Catalogue). "
                    "You can search by title, author, or subject.\n\n"
                    "Search here: http://10.10.85.51/"
                )

        else:

            # -----------------------------------
            # GREETING HANDLER
            # -----------------------------------

            greetings = ["hi", "hello", "hlo", "hey"]

            if lower_input in greetings:

                if session["language"] == "pa":
                    assistant_reply = "ਸਤ ਸ੍ਰੀ ਅਕਾਲ! ਮੈਂ ਭਾਈ ਕਾਨ੍ਹ ਸਿੰਘ ਨਾਭਾ ਲਾਇਬ੍ਰੇਰੀ ਦਾ AI ਸਹਾਇਕ ਹਾਂ। ਮੈਂ ਤੁਹਾਡੀ ਕਿਵੇਂ ਮਦਦ ਕਰ ਸਕਦਾ ਹਾਂ?"
                else:
                    assistant_reply = "Hello! I am the AI assistant of Bhai Kahn Singh Nabha Library. How can I assist you today?"

            else:

                best_doc, score = retrieve_relevant_doc(user_input)

                # -----------------------------------
                # HIGH CONFIDENCE → UNIVERSITY DATA
                # -----------------------------------

                if score >= 0.40:

                    response_type = best_doc.get("response_type", "direct")
                    page_title = best_doc.get("title", "relevant page")

                    if response_type == "redirect_only":

                        if session["language"] == "pa":
                            assistant_reply = f"{best_doc['content']} ਕਿਰਪਾ ਕਰਕੇ {page_title} ਪੇਜ ਵੇਖੋ।"
                        else:
                            assistant_reply = f"{best_doc['content']} Please visit the {page_title} page."

                    elif response_type == "redirect_with_summary":

                        if session["language"] == "pa":
                            assistant_reply = f"{best_doc['content']} ਵਧੇਰੇ ਜਾਣਕਾਰੀ ਲਈ ਕਿਰਪਾ ਕਰਕੇ {page_title} ਪੇਜ ਵੇਖੋ।"
                        else:
                            assistant_reply = f"{best_doc['content']} For more details, please visit the {page_title} page."

                    elif response_type == "redirect":

                        if session["language"] == "pa":
                            assistant_reply = f"ਕਿਰਪਾ ਕਰਕੇ {page_title} ਪੇਜ ਵੇਖੋ।"
                        else:
                            assistant_reply = f"Please visit the {page_title} page."

                    else:

                        context_text = best_doc["content"]

                        if session["language"] == "pa":
                            system_prompt = f"""
ਤੁਸੀਂ ਭਾਈ ਕਾਨ੍ਹ ਸਿੰਘ ਨਾਭਾ ਲਾਇਬ੍ਰੇਰੀ ਦੇ ਸਰਕਾਰੀ AI ਸਹਾਇਕ ਹੋ।

ਹੇਠਾਂ ਦਿੱਤੀ ਜਾਣਕਾਰੀ ਅਧਿਕਾਰਕ ਹੈ:

{context_text}

ਸਿਰਫ ਇਸ ਜਾਣਕਾਰੀ ਦੇ ਆਧਾਰ 'ਤੇ 3-4 ਵਾਕਾਂ ਵਿੱਚ ਜਵਾਬ ਦਿਓ।
"""
                        else:
                            system_prompt = f"""
You are the official AI assistant of Bhai Kahn Singh Nabha Library.

Use ONLY this official information:

{context_text}

Answer in 3-4 complete sentences.
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

                        assistant_reply = response.message.content[0].text

                else:

                    general_response = co.chat(
                        model="command-a-03-2025",
                        messages=[
                            {"role": "system", "content": "You are a helpful academic assistant."},
                            {"role": "user", "content": user_input}
                        ],
                        max_tokens=300,
                        temperature=0.4
                    )

                    assistant_reply = general_response.message.content[0].text

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
    app.run()