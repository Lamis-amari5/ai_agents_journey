# summarizer_agent.py
# Week 1  Text Summarizer Agent
# Framework: Agno | Model: HuggingFace (new InferenceClient)
import time
import json
import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

# --- Load your secret token from .env ---
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
print(f"DEBUG — Token loaded: {HF_TOKEN[:8] if HF_TOKEN else 'NOT FOUND'}")  # add this


# ─────────────────────────────────────────
# THE TOOL: calls HuggingFace new API
# ─────────────────────────────────────────
 

def summarize_text(text: str) -> str:
    """
    Calls HuggingFace summarization with automatic retry on timeout.
    """
    client = InferenceClient(
        provider="hf-inference",
        api_key=HF_TOKEN,
    )

    max_retries = 3  # try up to 3 times

    for attempt in range(max_retries):
        try:
            result = client.summarization(
                text,
                model="facebook/bart-large-cnn",
            )
            return result.summary_text

        except Exception as e:
            if attempt < max_retries - 1:
                wait = 10 * (attempt + 1)  # wait 10s, then 20s, then 30s
                print(f"   ⚠️  Server busy, retrying in {wait}s... (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait)
            else:
                return f"Failed after {max_retries} attempts: {str(e)}"

# ─────────────────────────────────────────
# THE AGENT: loops through articles
# ─────────────────────────────────────────
def run_summarizer_agent():
    with open("articles.json", "r") as f:
        articles = json.load(f)

    print(f"📚 Found {len(articles)} articles to summarize\n")

    results = []

    for article in articles:
        print(f"🔄 Processing: {article['title']}...")

        summary = summarize_text(article["text"])

        result = {
            "id": article["id"],
            "title": article["title"],
            "original_length": len(article["text"].split()),
            "summary": summary,
            "summary_length": len(summary.split())
        }

        results.append(result)
        print(f"   ✅ Done! Reduced from {result['original_length']} → {result['summary_length']} words")
        print(f"   📝 Summary: {summary[:120]}...\n")

    with open("output_week1.json", "w") as f:
        json.dump(results, f, indent=2)

    print("💾 All summaries saved to output_week1.json")
    print(f"🎉 Agent finished! Processed {len(results)} articles.")


# ─────────────────────────────────────────
# RUN IT
# ─────────────────────────────────────────
if __name__ == "__main__":
    run_summarizer_agent()