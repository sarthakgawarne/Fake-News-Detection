# rebuild trigger
import streamlit as st
import pickle
import os
import requests
import json
from bs4 import BeautifulSoup
from dotenv import load_dotenv

# Load API key from .env
load_dotenv()

# ---------------- LOAD ML MODEL ----------------
model = pickle.load(open("fake_news_model.pkl", "rb"))
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

# ---------------- PAGE UI ----------------
st.set_page_config(page_title="Fake News Detection System", page_icon="📰", layout="centered")

st.title("📰 FAKE NEWS DETECTION SYSTEM")
st.caption("Gawarne Sarthak Sunil")

st.caption("Naive Bayes Classification with AI-based Explanation")

user_input = st.text_area(
    "Enter News (URL or Text):",
    height=200
)

# ---------------- URL DETECTOR ----------------
def is_url(text):
    text = text.strip().lower()
    return text.startswith(("http://", "https://", "www."))

# ---------------- ARTICLE EXTRACTION ----------------
def extract_text_from_url(url):
    try:
        if url.startswith("www."):
            url = "https://" + url

        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=10)

        soup = BeautifulSoup(r.text, "html.parser")

        # remove unwanted parts
        for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside"]):
            tag.decompose()

        paragraphs = soup.find_all("p")
        article = " ".join(p.get_text() for p in paragraphs)
        article = " ".join(article.split())

        return article[:1500]

    except Exception:
        return None

# ---------------- OPENROUTER LLM EXPLANATION ----------------
def get_llm_explanation(text, label, confidence):

    api_key = os.getenv("OPENROUTER_API_KEY")

    url = "https://openrouter.ai/api/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    if label == "REAL":
        prompt = f"""
You are an AI assistant inside a Fake News Detection System.

The ML model classified the article as REAL with {confidence:.1f}% confidence.

Respond STRICTLY in this format:

Summary:
- line 1
- line 2

Tone: Neutral / Emotional / Biased

Article:
{text}
"""
    else:
        prompt = f"""
You are an AI assistant inside a Fake News Detection System.

The ML model classified the article as FAKE with {confidence:.1f}% confidence.

Respond STRICTLY in this format:

Tone: <one word>

Why it looks fake:
- reason 1
- reason 2
- reason 3

Article:
{text}
"""

    payload = {
        "model": "openai/gpt-3.5-turbo",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2
    }

    response = requests.post(url, headers=headers, data=json.dumps(payload))

    if response.status_code != 200:
        return "⚠️ AI explanation service not responding. Try again."

    result = response.json()
    return result["choices"][0]["message"]["content"]

# ---------------- MAIN ANALYSIS ----------------
if st.button("Analyze"):

    if user_input.strip() == "":
        st.warning("Please enter news text or URL.")
        st.stop()

    # Handle URL or Text
    if is_url(user_input):
        st.info("🔗 Reading article from link...")
        extracted_text = extract_text_from_url(user_input)

        if extracted_text is None or len(extracted_text) < 200:
            st.error("Could not extract article from this link.")
            st.stop()

        final_text = extracted_text
        st.success("Article extracted successfully.")

    else:
        final_text = user_input

    # -------- ML Prediction --------
    input_data = vectorizer.transform([final_text])
    prediction = model.predict(input_data)[0]

    decision_score = model.decision_function(input_data)[0]
    confidence = min(abs(decision_score) * 20, 99)

    if prediction == 0:
        st.error(f"⚠️ News is FAKE ({confidence:.1f}% confidence)")
        label = "FAKE"
    else:
        st.success(f"✅ News is REAL ({confidence:.1f}% confidence)")
        label = "REAL"

    # -------- AI Explanation --------
    st.subheader("🧠 AI Explanation")

    with st.spinner("Analyzing with AI..."):
        explanation = get_llm_explanation(final_text[:1200], label, confidence)

    st.code(explanation)
