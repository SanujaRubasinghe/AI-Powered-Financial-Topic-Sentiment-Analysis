import requests
import json

url = "https://api.x.com/2/tweets/search/recent?max_results=30&query=(stock%20OR%20stocks%20OR%20market%20OR%20economy%20OR%20inflation%20OR%20earnings)%20lang%3Aen%20-is%3Aretweet&tweet.fields=created_at,lang,public_metrics,author_id"

headers = {"Authorization": "Bearer AAAAAAAAAAAAAAAAAAAAAN%2BU5wEAAAAAxioMe%2BP1OZVIpXeHShb3VrrAvsg%3DBGoSO3uLvsgEPHw0pX7NnAlUfhgvMdArl6Ik1Ep6rosa6qgmHk"}
response = requests.get(url, headers=headers)

# Save as JSON
with open("twitter_response_2.json", "w", encoding="utf-8") as f:
    json.dump(response.json(), f, indent=4, ensure_ascii=False)

print("Saved to twitter_response.json")
