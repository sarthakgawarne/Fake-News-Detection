Fake News Detection System (Explainable AI)

A Machine Learning + AI powered web application that detects whether a news article is **REAL or FAKE** and also provides a human-readable explanation for the prediction.

🔗 **Live Demo:**
https://fake-news-detector-ku0r.onrender.com

---

## 📌 Project Overview

With the rapid growth of social media and online news platforms, misinformation spreads faster than verified information.
Traditional fake news classifiers only give a label (REAL / FAKE), but users often do not trust the result because there is no reasoning.

This project solves that problem by combining:

* Machine Learning prediction
* Confidence scoring
* AI-generated explanation

The system not only predicts the authenticity of a news article but also **explains WHY the news appears real or fake**, making it more transparent and trustworthy.

This approach falls under the concept of **Explainable Artificial Intelligence (XAI).**

---

## 🚀 Features

* Detects fake or real news from user-provided text
* Calculates prediction confidence
* Provides AI-generated explanation
* User-friendly web interface
* Works directly in browser (no installation needed)
* Real-time analysis

---

## 🏗️ System Architecture

User Input
⬇
Text Preprocessing
⬇
TF-IDF Vectorization
⬇
Naive Bayes Classifier
⬇
Prediction + Confidence Score
⬇
AI Language Model Explanation

---

## 🧪 Technologies Used

### Machine Learning

* Naive Bayes Classifier
* TF-IDF Vectorizer
* Scikit-learn

### Web Application

* Streamlit (Frontend + Backend)

### AI Explanation

* Large Language Model (via OpenRouter API)

### Programming Language

* Python

---

## ⚙️ How the Model Works

1. The user pastes a news article.
2. The text is converted into numerical features using **TF-IDF Vectorization**.
3. A trained **Naive Bayes classifier** predicts whether the article is real or fake.
4. The system calculates confidence of prediction.
5. An AI language model analyzes the article and explains the reasoning in simple English.

---

## 📊 Dataset

The model was trained on a labeled news dataset containing thousands of news articles categorized as:

* Real News
* Fake News

The dataset allowed the model to learn patterns such as:

* sensational wording
* emotional claims
* lack of credible sources
* neutral journalistic tone

## 🔐 Environment Variables

Create a `.env` file and add your API key:

```
OPENROUTER_API_KEY=your_api_key_here
```

---

## 📸 Example Output

The system provides:

Prediction:
REAL or FAKE

Confidence Score:
Model confidence percentage

AI Explanation:

* Short summary of the article
* Writing tone analysis
* Reasons supporting the decision

---

## 🎯 Use Cases

* Fact-checking news articles
* Checking WhatsApp forwards
* Educational demonstration of AI
* Research on misinformation detection
* Journalism support tool

---

## 📚 Concepts Demonstrated

* Natural Language Processing (NLP)
* Text Classification
* Explainable AI (XAI)
* Machine Learning Deployment
* API Integration
* Web Application Development

---

## 👨‍💻 Author

**Gawarne Sarthak Sunil**
B.E. Information Technology (SPPU)

---

## 📄 License

This project is open-source and available for educational and research purposes.

---

## ⭐ Future Improvements

* Support news links automatically
* Improve dataset size
* Add multilingual detection
* Improve model accuracy using deep learning (BERT)
* Add source credibility scoring

---

If you found this project interesting, consider giving it a ⭐!
