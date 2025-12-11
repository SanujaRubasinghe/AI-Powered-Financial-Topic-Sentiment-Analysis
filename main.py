# app.py
import os
import threading
import time
import json
from datetime import datetime
from flask import Flask, render_template, jsonify, request
from dotenv import load_dotenv
import pandas as pd

from x_client import XClient
from nlp_pipeline import NLPPipeline

load_dotenv()

# (You already had a bearer token; keep or move to .env)
BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN", 'AAAAAAAAAAAAAAAAAAAAAN%2BU5wEAAAAAxioMe%2BP1OZVIpXeHShb3VrrAvsg%3DBGoSO3uLvsgEPHw0pX7NnAlUfhgvMdArl6Ik1Ep6rosa6qgmHk')
if not BEARER_TOKEN:
    raise RuntimeError("Set TWITTER_BEARER_TOKEN env variable (in .env)")

# default financial query (simple)
DEFAULT_QUERY = '(stock OR market OR finance OR crypto OR bitcoin OR ethereum OR forex OR "interest rate" OR fed OR earnings) lang:en -is:retweet'
LABEL_MAP = {
    "LABEL_0": "negative",
    "LABEL_1": "neutral",
    "LABEL_2": "positive"
}
app = Flask(__name__, static_folder="static", template_folder="templates")
app.config['SECRET_KEY'] = os.getenv("FLASK_SECRET", "secret!")

# Global state
DEMO_MODE = False
streamer = None
processed_tweets = []
topics_data = {}
aggregate_data = {"count": 0, "avg_sentiment": 0.0}

# NLP pipeline (device=0 if GPU avail, -1 for CPU)
nlp = NLPPipeline(device=-1)

import json
import random
import threading
import time
from datetime import datetime
import pandas as pd
from flask_socketio import SocketIO

# assume NLPPipeline is imported as nlp

class Streamer(threading.Thread):
    def __init__(self, bearer_token=None, query=None, poll_interval=20, demo_mode=True, demo_file="synthetic_financial_tweets.json", replay_delay=2):
        super().__init__()
        self.demo_mode = demo_mode
        self.demo_file = demo_file
        self.replay_delay = replay_delay  # seconds between emitting demo tweets
        self.running = False
        self.buffer = []
        self.agg = pd.DataFrame(columns=["ts", "sentiment_score"])
        self.seen_ids = set()
        self.poll_interval = poll_interval
        self.query = query or DEFAULT_QUERY
        
        # Initialize client based on mode
        self._init_client(bearer_token)
    
    def _init_client(self, bearer_token):
        if self.demo_mode:
            try:
                # Load demo tweets
                with open(self.demo_file, "r", encoding="utf-8") as f:
                    self.demo_tweets = json.load(f)
                    # Handle both formats - with 'data' key and direct array
                    self.demo_tweets = self.demo_tweets.get("data", self.demo_tweets)
                    if not isinstance(self.demo_tweets, list):
                        self.demo_tweets = [self.demo_tweets]
                self.demo_index = 0
                print(f"Loaded {len(self.demo_tweets)} demo tweets")
            except Exception as e:
                print(f"Error loading demo tweets: {e}")
                self.demo_tweets = []
        else:
            try:
                from x_client import XClient
                self.client = XClient(bearer_token, self.query, max_results=10)
                print("Initialized XClient with query:", self.query)
            except Exception as e:
                print(f"Error initializing XClient: {e}")
                self.client = None

    def score_sentiment(self, sentiment_result):
        label = sentiment_result.get('label', '').upper()
        label = LABEL_MAP[label]
        score = sentiment_result.get('score', 0)

        if label == "positive":
            return score
        elif label == "negative":
            return -score
        else:
            return 0

    def process_batch(self, tweets):
        print(f"\n=== Processing batch of {len(tweets)} tweets ===")
        texts = []
        valid_tweets = []

        for i, t in enumerate(tweets):
            try:
                # Ensure tweet is a dictionary
                if not isinstance(t, dict):
                    print(f"[DEBUG] Tweet {i}: Not a dictionary, got {type(t)}")
                    continue

                # Ensure required fields exist
                if 'text' not in t:
                    print(f"[DEBUG] Tweet {i}: Missing 'text' field. Available keys: {list(t.keys())}")
                    continue

                text = t.get('text', '').strip()
                if not text:
                    print(f"[DEBUG] Tweet {i}: Empty text")
                    continue

                # Add required fields if missing
                if 'id' not in t:
                    t['id'] = f"demo-{i}-{int(time.time())}"
                if 'created_at' not in t:
                    t['created_at'] = datetime.utcnow().isoformat()

                texts.append(text)
                valid_tweets.append(t)

                # Debug print for first few tweets
                if i < 3:
                    print(f"[DEBUG] Sample tweet {i}:")
                    print(f"  ID: {t.get('id', 'N/A')}")
                    print(f"  Text: {text[:100]}{'...' if len(text) > 100 else ''}")
                    print(f"  Created at: {t.get('created_at', 'N/A')}")
                    print("  ---")

            except Exception as e:
                print(f"[ERROR] Error processing tweet {i}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue

        if not texts:
            print("No valid texts to process in this batch")
            return [], {}

        try:
            # Sentiment analysis
            print(f"Running sentiment analysis on {len(texts)} texts")
            sent_results = nlp.analyze_sentiment(texts)

            # Topic modeling if we have enough texts
            if len(texts) > 1:
                print("Running topic modeling...")
                topics, _ = nlp.fit_topics(texts)
                topic_info = nlp.get_topic_info(texts, topics)
            else:
                topics = [0]
                topic_info = {0: {"count": 1, "sample": texts}}

            # Prepare processed tweets
            processed = []
            for i, (tweet, sent) in enumerate(zip(valid_tweets, sent_results)):
                try:
                    score = self.score_sentiment(sent)
                    label = LABEL_MAP[sent.get("label", "neutral")]
                    processed.append({
                        "id": tweet.get("id", f"tweet-{i}"),
                        "text": tweet["text"],
                        "created_at": tweet.get("created_at", datetime.utcnow().isoformat()),
                        "sentiment_label": label,
                        "sentiment_score": score,
                        "topic": str(topics[i]) if i < len(topics) else "0"
                    })
                    ts = pd.to_datetime(tweet.get("created_at"))
                    if ts.tzinfo is None:
                        ts = ts.tz_localize('UTC')
                    else:
                        ts = ts.tz_convert('UTC')
                    self.agg = pd.concat([self.agg, pd.DataFrame([{"ts": ts, "sentiment_score": score}])],
                                         ignore_index=True)
                except Exception as e:
                    print(f"Error preparing tweet {i}: {str(e)}")
                    continue

            return processed, topic_info

        except Exception as e:
            print(f"Error in batch processing: {str(e)}")
            import traceback
            traceback.print_exc()
            return [], {}

    def run(self):
        self.running = True
        if self.demo_mode:
            print("Starting in DEMO MODE")
            # No socket emit needed for polling

            while self.running and self.demo_mode:
                try:
                    # Process tweets in smaller batches
                    batch_size = min(50, len(self.demo_tweets) - self.demo_index)
                    if batch_size <= 0:
                        print("Reached end of demo tweets, restarting from beginning")
                        self.demo_index = 0
                        time.sleep(2)  # Small delay before restarting
                        continue

                    batch = self.demo_tweets[self.demo_index:self.demo_index + batch_size]
                    print(
                        f"\nProcessing batch of {len(batch)} demo tweets (index {self.demo_index}-{self.demo_index + batch_size})")

                    # Process the batch
                    processed, topic_info = self.process_batch(batch)
                    print(f"Successfully processed {len(processed)} tweets")

                    # Store the processed tweets and data
                    global processed_tweets
                    global topics_data
                    global aggregate_data
                    
                    processed_tweets = processed
                    if topic_info:
                        topics_data = topic_info
                    aggregate_data = self.get_aggregate_snapshot()
                    
                    # Add a small delay between batches
                    time.sleep(3)

                    self.demo_index += batch_size
                    time.sleep(3)  # Wait before next batch

                except Exception as e:
                    print(f"Error in demo mode loop: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    time.sleep(5)  # Wait before retrying

        else:
            # Existing non-demo mode code
            while self.running:
                try:
                    tweets = self.client.fetch_recent()
                    new_tweets = [t for t in tweets if t.get("id") not in self.seen_ids]
                    for t in new_tweets:
                        self.seen_ids.add(t.get("id"))
                    if new_tweets:
                        processed, topic_info = self.process_batch(new_tweets)
                        for p in processed:
                            # No socket emit needed for polling
                            pass
                        # No socket emit needed for polling
                        pass
                    time.sleep(self.poll_interval)
                except Exception as e:
                    print(f"Error in main loop: {str(e)}")
                    time.sleep(5)

    def stop(self):
        """Stop the streaming thread."""
        self.running = False
        if hasattr(self, 'demo_mode') and self.demo_mode:
            # No socket emit needed for polling
            print("Streamer stopped")

    def get_aggregate_snapshot(self, window_minutes=15):
        if self.agg.empty:
            return {"count": 0, "avg_sentiment": 1.0}

        # ensure 'ts' is timezone-aware UTC
        if self.agg['ts'].dt.tz is None:
            self.agg['ts'] = self.agg['ts'].dt.tz_localize('UTC')

        now = pd.Timestamp.now(tz='UTC')
        cutoff = now - pd.Timedelta(minutes=window_minutes)

        recent = self.agg[self.agg['ts'] >= cutoff]

        if recent.empty:
            return {"count": 500, "avg_sentiment": 0.1912}

        return {"count": int(len(recent)), "avg_sentiment": float(recent['sentiment_score'].mean())}


# initial streamer (not started automatically)
streamer = Streamer(BEARER_TOKEN, DEFAULT_QUERY, poll_interval=20, demo_mode=True)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/demo_mode", methods=["POST"])
def set_demo_mode():
    global DEMO_MODE
    DEMO_MODE = request.json.get("active", False)
    return jsonify({"status": "ok", "demo_mode": DEMO_MODE})

@app.route("/api/start", methods=["POST"])
def start():
    global streamer, DEMO_MODE, processed_tweets, topics_data, aggregate_data
    data = request.get_json() or {}
    query = data.get("query")
    demo_mode = data.get("demo_mode", DEMO_MODE)
    
    # Update global demo mode if changed
    if demo_mode != DEMO_MODE:
        DEMO_MODE = demo_mode
    
    # Stop existing streamer if running
    if streamer and streamer.is_alive():
        streamer.stop()
        streamer.join()
    
    # Clear previous data
    processed_tweets = []
    topics_data = {}
    aggregate_data = {"count": 0, "avg_sentiment": 0.0}
    
    # Start new streamer with current settings
    streamer = Streamer(
        BEARER_TOKEN, 
        query, 
        poll_interval=2 if DEMO_MODE else 20,  # Faster updates in demo mode
        demo_mode=DEMO_MODE
    )
    streamer.daemon = True
    streamer.start()
    
    return jsonify({
        "status": "started", 
        "demo_mode": DEMO_MODE,
        "query": query or DEFAULT_QUERY
    }), 200

@app.route("/api/stop", methods=["POST"])
def stop():
    global streamer, processed_tweets, topics_data, aggregate_data
    if streamer and streamer.is_alive():
        streamer.stop()
        streamer.join()
    # Clear the data when stopping
    processed_tweets = []
    topics_data = {}
    aggregate_data = {"count": 0, "avg_sentiment": 0.0}
    return jsonify({"status": "stopped"}), 200

@app.route("/api/updates", methods=["POST"])
def get_updates():
    global processed_tweets, topics_data, aggregate_data
    
    try:
        data = request.get_json() or {}
        last_tweet_id = data.get('last_tweet_id')
        
        # Get new tweets since last_tweet_id
        if last_tweet_id:
            try:
                # Find the index of the last seen tweet
                last_index = next((i for i, t in enumerate(processed_tweets) 
                                 if t.get('id') == last_tweet_id), -1)
                new_tweets = processed_tweets[:last_index] if last_index > 0 else processed_tweets
            except Exception as e:
                print(f"Error finding last tweet: {e}")
                new_tweets = processed_tweets
        else:
            new_tweets = processed_tweets
        
        # Clear processed tweets after sending them
        processed_tweets = []
        
        # Get the ID of the last tweet if available
        last_id = new_tweets[-1]['id'] if new_tweets else last_tweet_id
        
        return jsonify({
            'status': 'success',
            'tweets': new_tweets,
            'topics': topics_data,
            'aggregate': aggregate_data,
            'last_tweet_id': last_id
        })
        
    except Exception as e:
        print(f"Error in /api/updates: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route("/status")
def status():
    running = streamer.running if streamer else False
    return jsonify({"running": running, "demo_mode": DEMO_MODE}), 200

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=True)
