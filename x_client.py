import os
import requests
import time

X_RECENT_SEARCH = "https://api.x.com/2/tweets/search/recent"

class XClient:
    def __init__(self, bearer_token: str, query: str, max_results=10):
        self.bearer_token = bearer_token
        self.query = query
        self.max_results = max_results
        self.next_token = None

    def _headers(self):
        return {"Authorization": f"Bearer {self.bearer_token}"}

    def build_params(self):
        params = {
            "query": self.query,
            "tweet.fields": "created_at,lang,public_metrics,author_id",
            "max_results": str(self.max_results)
        }
        if self.next_token:
            params["next_token"] = self.next_token
        return params

    def fetch_recent(self):
        while True:
            try:
                r = requests.get(X_RECENT_SEARCH, headers=self._headers(), params=self.build_params(), timeout=10)
                print(r.request.path_url)
                if r.status_code == 429:
                    reset = int(r.headers.get("x-rate-limit-reset", time.time() + 60))
                    sleep_time = max(reset - int(time.time()), 1)
                    print(f"Rate limit hit. Sleeping for {sleep_time} seconds...")
                    time.sleep(sleep_time)
                    continue
                elif r.status_code != 200:
                    print("Twitter API error:", r.status_code, r.text)
                    time.sleep(5)
                    continue
                data = r.json()
                self.next_token = data.get("meta", {}).get("next_token")
                return data.get("data", [])
            except Exception as e:
                print("Error fetching tweets:", e)
                time.sleep(5)