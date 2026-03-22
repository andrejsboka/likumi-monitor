"""
likumi_monitor.py
-----------------
This script checks the Latvian law website (likumi.lv) every day.
It finds new laws, analyzes them with Google Gemini AI,
and sends a daily summary to a Discord channel.

How it runs:
- GitHub Actions triggers this script every day on a schedule
- Prefect Cloud tracks every run, shows logs and status in the UI

Required packages:
    pip install prefect requests beautifulsoup4 google-generativeai
"""

import os
import json
import hashlib
import requests
from datetime import date, datetime
from bs4 import BeautifulSoup
from prefect import flow, task, get_run_logger
import google.generativeai as genai


# ──────────────────────────────────────────────
# SETTINGS
# These values come from environment variables.
# Never put real API keys directly in code.
# In GitHub: go to Settings → Secrets → Actions → New secret
# ──────────────────────────────────────────────

GEMINI_API_KEY  = os.environ["GEMINI_API_KEY"]       # key from Google AI Studio
DISCORD_WEBHOOK = os.environ["DISCORD_WEBHOOK_URL"]  # webhook URL from Discord

# Prefect Cloud connection — get this from app.prefect.cloud → Settings → API Keys
# Set it as PREFECT_API_KEY secret in GitHub
PREFECT_API_URL = os.environ.get("PREFECT_API_URL", "")  # your Prefect workspace URL
PREFECT_API_KEY = os.environ.get("PREFECT_API_KEY", "")  # your Prefect API key

LIKUMI_URL = "https://likumi.lv/ta/jaunie"  # page that lists new laws
SEEN_FILE  = "seen_laws.json"               # file that stores already processed laws

# This is the question we ask Gemini for each law.
# {title} and {url} will be replaced with real values later.
GEMINI_PROMPT = """
You are an expert in Latvian law and an IT security auditor assistant.
Read this law title and give a short analysis.
Reply ONLY with valid JSON, no extra text, no markdown.

Law title: {title}
URL: {url}

{{
  "summary": "2-3 sentences: what this law is about and what it changes",
  "sector": "finance / IT / general / other",
  "relevance": "high / medium / low",
  "relevance_reason": "short reason why this matters or not for a bank or IT security",
  "keywords": ["keyword 1", "keyword 2", "keyword 3"]
}}
"""


# ──────────────────────────────────────────────
# STEP 1: SCRAPING
# Download the page and get a list of new laws
# ──────────────────────────────────────────────

@task(name="fetch-laws", retries=3, retry_delay_seconds=30)
def fetch_laws() -> list[dict]:
    """
    Downloads likumi.lv/ta/jaunie and returns a list of laws.
    Each law is a dict with: title, url, date, id.

    retries=3 means Prefect will try again up to 3 times if this task fails.
    retry_delay_seconds=30 means it waits 30 seconds between retries.
    """
    logger = get_run_logger()

    # Tell the server we are a normal browser, not a bot
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }

    # Send a GET request — same as opening the page in a browser
    response = requests.get(LIKUMI_URL, headers=headers, timeout=20)

    # Stop here if the server returned an error (like 404 or 500)
    response.raise_for_status()

    # Parse the HTML so we can search inside it
    soup = BeautifulSoup(response.text, "html.parser")

    laws = []

    # Find all links on the page.
    # Law links on likumi.lv always contain "/ta/id/" in the URL.
    for a_tag in soup.find_all("a", href=True):
        href  = a_tag["href"]
        title = a_tag.get_text(strip=True)

        # Skip links that are not law documents
        if "/ta/id/" not in href:
            continue

        # Skip links with very short text (probably navigation buttons)
        if len(title) < 10:
            continue

        # Skip "Latvijas Vēstnesis" — this is the name of the gazette (newspaper),
        # not an actual law. It appears as a link on every entry on the page.
        if title.strip() == "Latvijas Vēstnesis":
            continue

        # Build a full URL if the href is a relative path like /ta/id/123
        if href.startswith("/"):
            full_url = "https://likumi.lv" + href
        else:
            full_url = href

        # Create a short unique ID for this law using MD5 hash of the URL.
        # MD5 turns any text into a fixed short string like "a1b2c3d4...".
        # We use it as a unique key — not for security.
        law_id = hashlib.md5(full_url.encode()).hexdigest()

        laws.append({
            "title": title,
            "url":   full_url,
            "date":  str(date.today()),
            "id":    law_id
        })

    # Remove duplicates — the same law can appear more than once on the page
    seen_ids    = set()
    unique_laws = []
    for law in laws:
        if law["id"] not in seen_ids:
            seen_ids.add(law["id"])
            unique_laws.append(law)

    logger.info(f"Found {len(unique_laws)} laws on the page")
    return unique_laws


# ──────────────────────────────────────────────
# STEP 2: DEDUPLICATION
# Only process laws we have not seen before
# ──────────────────────────────────────────────

@task(name="load-seen-ids")
def load_seen_ids() -> set:
    """
    Reads the list of already processed law IDs from a JSON file.
    Returns an empty set if the file does not exist yet (first run).
    """
    if os.path.exists(SEEN_FILE):
        with open(SEEN_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return set(data.get("ids", []))
    return set()


@task(name="save-seen-ids")
def save_seen_ids(seen_ids: set):
    """
    Saves the updated list of processed IDs back to the JSON file.
    The next run will read this file and skip these laws.
    """
    with open(SEEN_FILE, "w", encoding="utf-8") as f:
        json.dump({
            "ids":      list(seen_ids),
            "last_run": datetime.now().isoformat()
        }, f, ensure_ascii=False, indent=2)


@task(name="filter-new-laws")
def filter_new_laws(laws: list[dict], seen_ids: set) -> list[dict]:
    """
    Returns only the laws that are NOT in seen_ids yet.
    """
    logger = get_run_logger()
    new_laws = [law for law in laws if law["id"] not in seen_ids]
    skipped  = len(laws) - len(new_laws)
    logger.info(f"New: {len(new_laws)}  |  Already known (skipped): {skipped}")
    return new_laws


# ──────────────────────────────────────────────
# STEP 3: AI ANALYSIS
# Send each new law to Google Gemini and get a short analysis
# ──────────────────────────────────────────────

@task(name="analyze-law", retries=2, retry_delay_seconds=10)
def analyze_with_gemini(law: dict) -> dict:
    """
    Sends the law title to Google Gemini AI.
    Returns a dict with: summary, sector, relevance, relevance_reason, keywords.
    """
    logger = get_run_logger()

    # Connect to the Gemini API using our key
    genai.configure(api_key=GEMINI_API_KEY)

    # Use the fast free model — good enough for text analysis
    model = genai.GenerativeModel("gemini-2.0-flash")

    # Fill in the law title and URL into our prompt template
    prompt = GEMINI_PROMPT.format(title=law["title"], url=law["url"])

    try:
        response = model.generate_content(prompt)

        # The model sometimes wraps JSON in ```json ... ``` — remove that
        text = response.text.strip()
        text = text.lstrip("```json").lstrip("```").rstrip("```").strip()

        # Parse the JSON string into a Python dict
        analysis = json.loads(text)
        logger.info(f"Done: {law['title'][:60]}...")
        return analysis

    except Exception as e:
        # If anything goes wrong, return a safe default value
        logger.warning(f"Error for '{law['title'][:40]}': {e}")
        return {
            "summary":          "Analysis not available",
            "sector":           "unknown",
            "relevance":        "low",
            "relevance_reason": "Error during analysis",
            "keywords":         []
        }


# ──────────────────────────────────────────────
# STEP 4: FORMAT AND SEND TO DISCORD
# ──────────────────────────────────────────────

# Discord embed colors — stored as integers (hex color codes)
RELEVANCE_COLOR = {
    "high":   0xE74C3C,  # red    — important
    "medium": 0xF39C12,  # orange — worth knowing
    "low":    0x95A5A6,  # gray   — for reference only
}

RELEVANCE_EMOJI = {
    "high":   "🔴",
    "medium": "🟡",
    "low":    "⚪",
}


def build_law_embed(item: dict) -> dict:
    """
    Builds one Discord embed card for a single law.
    This is a plain function — not a Prefect task.
    It is called inside send_to_discord.
    """
    law      = item["law"]
    analysis = item["analysis"]

    relevance = analysis.get("relevance", "low")
    emoji     = RELEVANCE_EMOJI.get(relevance, "⚪")
    color     = RELEVANCE_COLOR.get(relevance, 0x95A5A6)
    keywords  = ", ".join(analysis.get("keywords", []))

    return {
        # Clicking the title in Discord opens the law on likumi.lv
        "title":       f"{emoji} {law['title'][:200]}",
        "url":         law["url"],
        "description": analysis.get("summary", ""),
        "color":       color,
        "fields": [
            {"name": "Sector",       "value": analysis.get("sector", "—"),                    "inline": True},
            {"name": "Relevance",    "value": f"{emoji} {relevance}",                         "inline": True},
            {"name": "Keywords",     "value": keywords if keywords else "—",                  "inline": False},
            {"name": "Why it matters","value": analysis.get("relevance_reason", "—")[:500],   "inline": False}
        ]
    }


def post_embeds_to_discord(embeds: list[dict]):
    """
    Sends one POST request to Discord with a list of embeds.
    Max 10 embeds per request — Discord will reject more.
    """
    response = requests.post(
        DISCORD_WEBHOOK,
        json={"embeds": embeds},
        headers={"Content-Type": "application/json"},
        timeout=15
    )
    if response.status_code != 204:
        raise requests.HTTPError(f"Discord error {response.status_code}: {response.text}")


@task(name="send-to-discord", retries=3, retry_delay_seconds=15)
def send_to_discord(results: list[dict]):
    """
    Sends all laws to Discord. Splits into multiple messages if needed.
    Discord allows max 10 embeds per message (1 header + 9 laws).
    Example: 20 laws → message 1 has 9, message 2 has 9, message 3 has 2.

    retries=3 means Prefect will retry if Discord is temporarily unavailable.
    """
    logger = get_run_logger()
    today  = date.today().strftime("%d.%m.%Y")
    count  = len(results)

    # Build all law embed cards
    all_embeds = [build_law_embed(item) for item in results]

    # Split into groups of 9 (each group + 1 header = 10 max per message)
    chunk_size = 9
    chunks = [all_embeds[i:i + chunk_size] for i in range(0, len(all_embeds), chunk_size)]

    for index, chunk in enumerate(chunks):

        # First message gets the main header with total count
        if index == 0:
            header = {
                "title":       f"📋 Likumi.lv digest — {today}",
                "description": f"**{count}** new laws found today",
                "color":       0x3498DB,
                "timestamp":   datetime.utcnow().isoformat() + "Z",
                "footer":      {"text": "likumi.lv monitor · automatic daily check"}
            }
        else:
            # Later messages get a simpler "continued" header
            header = {
                "title":       f"📋 Likumi.lv digest — {today} (part {index + 1})",
                "description": f"Laws {index * chunk_size + 1}–{index * chunk_size + len(chunk)}",
                "color":       0x3498DB
            }

        # Send: 1 header + up to 9 law cards = max 10 embeds
        post_embeds_to_discord([header] + chunk)
        logger.info(f"Sent part {index + 1}/{len(chunks)} to Discord")

    logger.info(f"Done — {count} laws sent to Discord.")


@task(name="send-no-news")
def send_no_news():
    """
    Sends a short message to Discord when there are no new laws today.
    This way you always get a daily confirmation that the script ran.
    """
    payload = {
        "embeds": [{
            "title":       f"📋 Likumi.lv — {date.today().strftime('%d.%m.%Y')}",
            "description": "No new laws found today.",
            "color":       0x95A5A6,
            "footer":      {"text": "likumi.lv monitor"}
        }]
    }
    requests.post(DISCORD_WEBHOOK, json=payload, timeout=15)


# ──────────────────────────────────────────────
# MAIN FLOW
# @flow tells Prefect to track this as one workflow.
# GitHub Actions runs this file. Prefect Cloud records the result.
# ──────────────────────────────────────────────

@flow(name="likumi-monitor", log_prints=True)
def likumi_monitor_flow():
    """
    Main Prefect flow. Runs all tasks in order.
    Prefect Cloud tracks every run, shows logs and task status in the UI.
    GitHub Actions triggers this flow every day on a schedule.
    """

    # Task 1: Get all laws from the website
    all_laws = fetch_laws()
    if not all_laws:
        print("Could not get any laws from the website. Stopping.")
        return

    # Task 2: Load already processed law IDs
    seen_ids = load_seen_ids()

    # Task 3: Keep only laws we have not processed before
    new_laws = filter_new_laws(all_laws, seen_ids)

    if not new_laws:
        print("No new laws today.")
        send_no_news()
        return

    # Task 4: Ask Gemini to analyze each new law
    results = []
    for law in new_laws:
        analysis = analyze_with_gemini(law)
        results.append({"law": law, "analysis": analysis})

    # Sort by relevance — show high relevance laws first
    # order dict maps relevance text to a sort number (0 = first)
    order = {"high": 0, "medium": 1, "low": 2}
    results.sort(key=lambda x: order.get(x["analysis"].get("relevance", "low"), 2))

    # Task 5: Send the results to Discord
    send_to_discord(results)

    # Task 6: Save the IDs of today's laws so we skip them next time.
    # The | operator merges two sets: {1, 2} | {3} = {1, 2, 3}
    new_ids = {law["id"] for law in new_laws}
    save_seen_ids(seen_ids | new_ids)

    print(f"Done. Processed {len(new_laws)} new laws.")


# GitHub Actions runs this file directly — so we call the flow here.
# No serve() needed — GitHub Actions handles the schedule.
if __name__ == "__main__":
    likumi_monitor_flow()
