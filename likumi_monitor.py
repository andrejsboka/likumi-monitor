"""
likumi_monitor.py
-----------------
This script checks the Latvian law website (likumi.lv) every day.
It finds new laws that took effect today (or the most recent day with laws),
analyzes them with Google Gemini AI,
and sends a daily summary to a Discord channel.

How it runs:
- Prefect Cloud triggers this script every day on a schedule
- Prefect Cloud tracks every run and shows logs in the UI

Required packages:
    pip install prefect requests beautifulsoup4 google-generativeai
"""

import os
import json
import hashlib
import time
import requests
from datetime import date, datetime, timedelta
from bs4 import BeautifulSoup
from prefect import flow, task, get_run_logger
import google.generativeai as genai


# ──────────────────────────────────────────────
# SETTINGS
# ──────────────────────────────────────────────

GEMINI_API_KEY  = os.environ["GEMINI_API_KEY"]       # key from Google AI Studio
DISCORD_WEBHOOK = os.environ["DISCORD_WEBHOOK_URL"]  # webhook URL from Discord

# Mobile version of likumi.lv — cleaner HTML, easier to parse.
# URL format: ?type=sst&date=DD.MM.YYYY
# type=sst = "stājas spēkā today" = laws that take effect on this date
LIKUMI_MOBILE_URL = "https://m.likumi.lv/jaunakie.php"

SEEN_FILE = "seen_laws.json"  # stores IDs of already processed laws

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
# ──────────────────────────────────────────────

def fetch_laws_for_date(check_date: date) -> list[dict]:
    """
    Downloads the mobile likumi.lv page for one specific date.
    Returns a list of laws that took effect on that date.
    Returns an empty list if there are no laws for that date.
    """
    date_str = check_date.strftime("%d.%m.%Y")  # format: 20.03.2026

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }

    url      = f"{LIKUMI_MOBILE_URL}?type=sst&date={date_str}"
    response = requests.get(url, headers=headers, timeout=20)
    response.raise_for_status()

    # Force UTF-8 encoding — the site uses UTF-8 but requests sometimes guesses wrong
    response.encoding = "utf-8"
    soup = BeautifulSoup(response.text, "html.parser")
    laws = []

    # Every law on the mobile page is a link with "/ta/id/" in the href
    for a_tag in soup.find_all("a", href=True):
        href  = a_tag["href"]
        title = a_tag.get_text(strip=True)

        if "/ta/id/" not in href:
            continue
        if len(title) < 10:
            continue

        # Convert to full URL on the main site (not mobile)
        if href.startswith("/"):
            full_url = "https://likumi.lv" + href
        else:
            full_url = href.replace("m.likumi.lv", "likumi.lv")

        # MD5 hash of the URL = unique ID for this law
        law_id = hashlib.md5(full_url.encode()).hexdigest()

        laws.append({
            "title": title,
            "url":   full_url,
            "date":  str(check_date),
            "id":    law_id
        })

    # Remove duplicates by ID
    seen_ids    = set()
    unique_laws = []
    for law in laws:
        if law["id"] not in seen_ids:
            seen_ids.add(law["id"])
            unique_laws.append(law)

    return unique_laws


@task(name="fetch-laws", retries=3, retry_delay_seconds=30)
def fetch_laws() -> list[dict]:
    """
    Tries to fetch laws for today.
    If today has no laws (weekend, holiday, or not yet published),
    goes back one day at a time — up to 7 days back.
    This way the digest always shows the most recent available laws.
    """
    logger = get_run_logger()

    for days_back in range(7):
        check_date = date.today() - timedelta(days=days_back)
        laws = fetch_laws_for_date(check_date)

        if laws:
            if days_back == 0:
                logger.info(f"Found {len(laws)} laws for today ({check_date})")
            else:
                logger.info(f"No laws today. Showing {len(laws)} laws from {days_back} day(s) ago ({check_date})")
            return laws

    logger.warning("No laws found in the last 7 days")
    return []


# ──────────────────────────────────────────────
# STEP 2: DEDUPLICATION
# ──────────────────────────────────────────────

@task(name="load-seen-ids")
def load_seen_ids() -> set:
    """
    Reads already processed law IDs from a JSON file.
    Returns empty set on first run.
    """
    if os.path.exists(SEEN_FILE):
        with open(SEEN_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return set(data.get("ids", []))
    return set()


@task(name="save-seen-ids")
def save_seen_ids(seen_ids: set):
    """Saves processed IDs so we skip them next run."""
    with open(SEEN_FILE, "w", encoding="utf-8") as f:
        json.dump({
            "ids":      list(seen_ids),
            "last_run": datetime.now().isoformat()
        }, f, ensure_ascii=False, indent=2)


@task(name="filter-new-laws")
def filter_new_laws(laws: list[dict], seen_ids: set) -> list[dict]:
    """Returns only laws not yet processed."""
    logger   = get_run_logger()
    new_laws = [law for law in laws if law["id"] not in seen_ids]
    skipped  = len(laws) - len(new_laws)
    logger.info(f"New: {len(new_laws)}  |  Already known (skipped): {skipped}")
    return new_laws


# ──────────────────────────────────────────────
# STEP 3: AI ANALYSIS
# ──────────────────────────────────────────────

@task(name="analyze-law", retries=2, retry_delay_seconds=10)
def analyze_with_gemini(law: dict) -> dict:
    """
    Sends the law title to Google Gemini AI and returns a structured analysis.
    Waits 5 seconds after each call to stay within the free tier rate limit
    (free tier allows ~15 requests per minute).
    """
    logger = get_run_logger()
    genai.configure(api_key=GEMINI_API_KEY)
    model  = genai.GenerativeModel("gemini-2.5-flash")
    prompt = GEMINI_PROMPT.format(title=law["title"], url=law["url"])

    try:
        response = model.generate_content(prompt)

        # Remove ```json ... ``` wrapping if present
        text = response.text.strip()
        text = text.lstrip("```json").lstrip("```").rstrip("```").strip()

        analysis = json.loads(text)
        logger.info(f"Done: {law['title'][:60]}...")

        # Wait 5 seconds — free tier limit is ~15 requests/minute
        time.sleep(5)

        return analysis

    except Exception as e:
        logger.warning(f"Error for '{law['title'][:40]}': {e}")
        return {
            "summary":          "Analysis not available",
            "sector":           "unknown",
            "relevance":        "low",
            "relevance_reason": "Error during analysis",
            "keywords":         []
        }


# ──────────────────────────────────────────────
# STEP 4: SEND TO DISCORD
# ──────────────────────────────────────────────

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
    """Builds one Discord embed card for a single law."""
    law      = item["law"]
    analysis = item["analysis"]

    relevance = analysis.get("relevance", "low")
    emoji     = RELEVANCE_EMOJI.get(relevance, "⚪")
    color     = RELEVANCE_COLOR.get(relevance, 0x95A5A6)
    keywords  = ", ".join(analysis.get("keywords", []))

    return {
        "title":       f"{emoji} {law['title'][:200]}",
        "url":         law["url"],
        "description": analysis.get("summary", ""),
        "color":       color,
        "fields": [
            {"name": "Sector",        "value": analysis.get("sector", "—"),                   "inline": True},
            {"name": "Relevance",     "value": f"{emoji} {relevance}",                        "inline": True},
            {"name": "Keywords",      "value": keywords if keywords else "—",                 "inline": False},
            {"name": "Why it matters","value": analysis.get("relevance_reason", "—")[:500],   "inline": False}
        ]
    }


def post_embeds_to_discord(embeds: list[dict]):
    """
    Sends one POST request to Discord with a list of embeds.
    Discord allows max 10 embeds per message.
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
def send_to_discord(results: list[dict], laws_date: str):
    """
    Sends all laws to Discord. Splits into multiple messages if needed.
    Discord allows max 10 embeds per message (1 header + 9 laws).
    """
    logger = get_run_logger()
    today  = date.today().strftime("%d.%m.%Y")
    count  = len(results)

    # Build all law embed cards
    all_embeds = [build_law_embed(item) for item in results]

    # Split into groups of 9 (1 header + 9 laws = 10 max per message)
    chunk_size = 9
    chunks = [all_embeds[i:i + chunk_size] for i in range(0, len(all_embeds), chunk_size)]

    for index, chunk in enumerate(chunks):
        if index == 0:
            # Show note if we are showing laws from a previous day
            note = f"\n_(Laws from {laws_date} — most recent available)_" if laws_date != today else ""
            header = {
                "title":       f"📋 Likumi.lv digest — {today}",
                "description": f"**{count}** laws found{note}",
                "color":       0x3498DB,
                "timestamp":   datetime.utcnow().isoformat() + "Z",
                "footer":      {"text": "likumi.lv monitor · automatic daily check"}
            }
        else:
            header = {
                "title":       f"📋 Likumi.lv digest — {today} (part {index + 1})",
                "description": f"Laws {index * chunk_size + 1}–{index * chunk_size + len(chunk)}",
                "color":       0x3498DB
            }

        post_embeds_to_discord([header] + chunk)
        logger.info(f"Sent part {index + 1}/{len(chunks)} to Discord")

    logger.info(f"Done — {count} laws sent to Discord.")


@task(name="send-no-news")
def send_no_news():
    """Sends a short message when no laws were found in the last 7 days."""
    payload = {
        "embeds": [{
            "title":       f"📋 Likumi.lv — {date.today().strftime('%d.%m.%Y')}",
            "description": "No new laws found in the last 7 days.",
            "color":       0x95A5A6,
            "footer":      {"text": "likumi.lv monitor"}
        }]
    }
    requests.post(DISCORD_WEBHOOK, json=payload, timeout=15)


# ──────────────────────────────────────────────
# MAIN FLOW
# ──────────────────────────────────────────────

@flow(name="likumi-monitor", log_prints=True)
def likumi_monitor_flow():
    """
    Main Prefect flow. Runs all tasks in order.
    Prefect Cloud tracks every run and shows logs in the UI.
    """

    # Task 1: Fetch laws — today or most recent day with laws
    all_laws = fetch_laws()
    if not all_laws:
        send_no_news()
        return

    # Remember which date these laws are from (for the Discord header note)
    laws_date = all_laws[0]["date"] if all_laws else str(date.today())

    # Task 2: Skip laws already sent before
    seen_ids = load_seen_ids()
    new_laws = filter_new_laws(all_laws, seen_ids)

    if not new_laws:
        print("No new laws — all already sent before.")
        send_no_news()
        return

    # Task 3: Analyze each law with Gemini AI
    results = []
    for law in new_laws:
        analysis = analyze_with_gemini(law)
        results.append({"law": law, "analysis": analysis})

    # Sort: high relevance first
    order = {"high": 0, "medium": 1, "low": 2}
    results.sort(key=lambda x: order.get(x["analysis"].get("relevance", "low"), 2))

    # Task 4: Send to Discord
    send_to_discord(results, laws_date)

    # Task 5: Save IDs so we don't send the same laws again tomorrow
    new_ids = {law["id"] for law in new_laws}
    save_seen_ids(seen_ids | new_ids)

    print(f"Done. Processed {len(new_laws)} new laws.")


if __name__ == "__main__":
    likumi_monitor_flow()
