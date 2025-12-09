from transformers import pipeline
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

class NLPPipeline:
    def __init__(self, device=0):
        # Sentiment pipeline
        self.sentiment = pipeline(
            task="sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment",
            truncation=True,
            max_length=256,
            device=device
        )

        # Summarizer
        try:
            self.summarizer = pipeline(
                "summarization",
                model="facebook/bart-large-cnn",
                device=device
            )
        except Exception:
            self.summarizer = None

        # Topic Modeling / Embeddings
        try:
            self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
            self.topic_model = BERTopic(
                embedding_model=self.embedder,
                verbose=False
            )
            self.use_bertopic = True
        except Exception:
            print("BERTopic unavailable — falling back to LDA.")
            self.topic_model = None
            self.use_bertopic = False
            self.count_vectorizer = CountVectorizer(
                stop_words="english",
                max_features=2000
            )

        self.stop_words = set(stopwords.words("english"))

        # Topic map for human-readable labels
        self.topic_map = {
            "Interest Rates": ["fed", "interest rate", "treasury", "yield", "monetary policy", "rate hike", "federal reserve"],
            "Inflation": ["inflation", "cpi", "consumer prices", "cost of living", "price index", "inflation expectations"],
            "Stock Market": ["stocks", "market", "equities", "s&p 500", "nasdaq", "dow jones", "bullish", "bearish"],
            "Earnings Reports": ["earnings", "eps", "revenue", "quarterly report", "guidance", "profit", "loss"],
            "Cryptocurrency": ["bitcoin", "ethereum", "crypto", "blockchain", "altcoins", "defi", "nft"],
            "Commodities": ["gold", "oil", "silver", "copper", "commodities", "crude", "wti", "natural gas"],
            "Forex": ["forex", "usd", "eur", "jpy", "currency", "exchange rate", "fx market"],
            "Mergers and Acquisitions": ["m&a", "acquisition", "merger", "deal", "takeover", "buyout", "corporate"],
            "Government Policy": ["regulation", "policy", "fiscal", "tax", "budget", "stimulus", "legislation"],
            "Banking & Finance": ["bank", "lending", "loan", "interest", "credit", "financial institution", "capital"],
            "Tech Stocks": ["tech", "apple", "microsoft", "google", "ai", "semiconductors", "software"],
            "Consumer Sector": ["retail", "consumer", "spending", "demand", "sales", "products", "brands"],
            "Energy Market": ["energy", "oil", "gas", "renewable", "coal", "electricity", "supply", "demand"],
            "Housing Market": ["real estate", "housing", "mortgage", "home prices", "construction", "property"],
            "Miscellaneous": ["other", "misc", "various", "news", "update", "trending", "market"]
        }

    # ---------------- Sentiment ---------------- #
    def analyze_sentiment(self, text):
        return self.sentiment(text, truncation=True)

    # ---------------- Summarization ---------------- #
    def summarize(self, text, max_length=20):
        if not self.summarizer:
            return text[:160] + ("..." if len(text) > 160 else "")
        s = self.summarizer(text, max_length=max_length, min_length=20, do_sample=False)
        return s[0]["summary_text"]

    # ---------------- Topic Helpers ---------------- #
    def get_topic_label(self, text):
        """Return a topic label from the map, based on keywords in text."""
        if not text:
            return "Miscellaneous"

        text_lower = text.lower()
        for label, keywords in self.topic_map.items():
            for kw in keywords:
                if kw in text_lower:
                    return label
        return "Miscellaneous"

    # ---------------- Topic Fitting ---------------- #
    def fit_topics(self, texts):
        """
        Fit BERTopic (or fallback LDA) and return mapped topic labels only.
        """
        if self.use_bertopic:
            topics, probs = self.topic_model.fit_transform(texts)
            # Map topics using topic_map based on original texts
            topic_labels = [self.get_topic_label(t) for t in texts]
            return topic_labels, probs
        else:
            # Fallback LDA-like keyword extraction
            X = self.count_vectorizer.fit_transform(texts)
            feature_names = self.count_vectorizer.get_feature_names_out()
            top_terms = []
            for row in X:
                arr = row.toarray().ravel()
                idx = arr.argmax()
                text_term = feature_names[idx] if arr.sum() > 0 else "Miscellaneous"
                top_terms.append(self.get_topic_label(text_term))
            return top_terms, None

    # ---------------- Topic Summaries ---------------- #
    def get_topic_info(self, texts, topics):
        info = {}
        for text, topic in zip(texts, topics):
            info.setdefault(topic, []).append(text)

        topic_summaries = {}
        for t, docs in info.items():
            joined = " ".join(docs[:10])
            summary = self.summarize(joined)
            topic_summaries[t] = {"label": t, "count": len(docs), "summary": summary}
        return topic_summaries
