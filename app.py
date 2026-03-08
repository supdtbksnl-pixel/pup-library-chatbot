import os
import json
import numpy as np
import faiss
import cohere
from flask import Flask, render_template, request, session, redirect
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired
import secrets

# -----------------------------------
# APP SETUP
# -----------------------------------

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)

# -----------------------------------
# LOAD COHERE
# -----------------------------------

api_key = os.getenv("COHERE_API_KEY")
co = cohere.ClientV2(api_key)

# -----------------------------------
# LOAD KNOWLEDGE BASE
# -----------------------------------

with open("university_data.json", "r", encoding="utf-8") as f:
    knowledge_base = json.load(f)

embeddings = np.load("embeddings.npy")
index = faiss.read_index("faiss_index.index")

# -----------------------------------
# CONFIG
# -----------------------------------

TOP_K = 3
MEMORY_LENGTH = 4

# -----------------------------------
# FORM
# -----------------------------------

class ChatForm(FlaskForm):
    text = StringField("", validators=[DataRequired()])
    submit = SubmitField("Send")

# -----------------------------------
# VECTOR SEARCH
# -----------------------------------

def retrieve_docs(query):

    response = co.embed(
        model="embed-english-v3.0",
        input_type="search_query",
        texts=[query]
    )

    query_embedding = np.array(response.embeddings.float)

    distances, indices = index.search(query_embedding, TOP_K)

    docs = [knowledge_base[i]["content"] for i in indices[0]]

    return "\n".join(docs)

# -----------------------------------
# PROMPT BUILDER
# -----------------------------------

def build_prompt(language, context):

    if language == "pa":

        return f"""
ਤੁਸੀਂ ਭਾਈ ਕਾਨ੍ਹ ਸਿੰਘ ਨਾਭਾ ਲਾਇਬ੍ਰੇਰੀ ਦੇ ਸਰਕਾਰੀ AI ਸਹਾਇਕ ਹੋ।

ਹੇਠਾਂ ਦਿੱਤੀ ਜਾਣਕਾਰੀ ਦੇ ਆਧਾਰ 'ਤੇ ਉਪਭੋਗਤਾ ਦੇ ਸਵਾਲ ਦਾ ਜਵਾਬ ਦਿਓ।

ਜਾਣਕਾਰੀ:
{context}

ਨਿਯਮ:
• ਸਿਰਫ ਪੰਜਾਬੀ ਵਿੱਚ ਜਵਾਬ ਦਿਓ
• ਸਪਸ਼ਟ ਅਤੇ ਛੋਟਾ ਜਵਾਬ ਦਿਓ
"""

    else:

        return f"""
You are the official AI assistant of Bhai Kahn Singh Nabha Library.

Use the information below to answer the user.

Information:
{context}

Rules:
• Answer clearly
• Do not repeat instructions
"""

# -----------------------------------
# ROUTE
# -----------------------------------

@app.route("/", methods=["GET", "POST"])
def home():

    form = ChatForm()

    if request.args.get("reset"):
        session.pop("chat_history", None)
        return redirect("/")

    if "language" not in session:
        session["language"] = "en"

    if request.args.get("lang"):
        new_lang = request.args.get("lang")

        if new_lang != session["language"]:
            session["language"] = new_lang
            session.pop("chat_history", None)

        return redirect("/")

    if "chat_history" not in session:
        session["chat_history"] = []

    if form.validate_on_submit():

        user_input = form.text.data.strip()

        # -----------------------------------
        # RETRIEVE KNOWLEDGE
        # -----------------------------------

        context = retrieve_docs(user_input)

        # -----------------------------------
        # MEMORY
        # -----------------------------------

        conversation_memory = session["chat_history"][-MEMORY_LENGTH:]

        messages = []

        for msg in conversation_memory:
            messages.append(msg)

        messages.append({
            "role": "user",
            "content": user_input
        })

        # -----------------------------------
        # BUILD PROMPT
        # -----------------------------------

        system_prompt = build_prompt(session["language"], context)

        response = co.chat(
            model="command-a-03-2025",
            messages=[
                {"role": "system", "content": system_prompt},
                *messages
            ],
            temperature=0,
            max_tokens=200
        )

        answer = response.message.content[0].text.strip()

        # -----------------------------------
        # STORE CHAT
        # -----------------------------------

        session["chat_history"].append({
            "role": "user",
            "content": user_input
        })

        session["chat_history"].append({
            "role": "assistant",
            "content": answer
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