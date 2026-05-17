"""
News Headlines Bot
Searches for yesterday's top 3 news headlines (with summaries) and sends them
to a Kindroid AI companion via the Kindroid send-message API.

Supported provider: Anthropic (Claude) with built-in web search

ℹ️  Model is set via the ANTHROPIC_MODEL env var, or the HAIKU_MODEL constant
    below. Update HAIKU_MODEL manually when Anthropic releases a new Haiku version
    (typically once or twice a year).
"""

import os
import re
import json
import hashlib
import logging
import time
from datetime import datetime
from pathlib import Path

import requests


# ── Config ──────────────────────────────────────────────────────────────────

def load_config():
    config_path = Path(__file__).parent / "config.json"
    with open(config_path) as f:
        return json.load(f)

CONFIG = load_config()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("news-bot")


# ── System Prompt ────────────────────────────────────────────────────────────
#
# Passed as the `system` parameter (highest-priority instructions) to every
# API call. Keeping formatting rules here — separate from the search task in
# the user prompt — gives them the strongest possible weight with Haiku.

SYSTEM_PROMPT = """You are a news aggregation bot. You MUST use the web_search tool to find current news articles — do not use your training data for headlines or summaries.

When you have finished all your searches, your text response must contain ONLY the 3 formatted news stories — absolutely nothing else. Do not write anything before or after the stories.

Output format — exactly 3 stories, with one blank line between each:

**Headline text here**
One to two sentence summary here.

Output rules:
- Do NOT write any preamble, transition, or closing remark of any kind
- Do NOT write phrases like "Here are", "Let me", "I found", "Based on my research", "Here's what I found", or anything similar before or between stories
- Do NOT number or bullet the stories
- Your entire text response must be exactly 3 stories in exactly the above format and nothing else
- Just perform your search silently and then output ONLY the final correctly formatted results."""


# ── Model ───────────────────────────────────────────────────────────────────
#
# Update this string when Anthropic releases a new Haiku version.
# Can be overridden at runtime via the ANTHROPIC_MODEL environment variable.

HAIKU_MODEL = "claude-haiku-4-5-20251001"


# ── Category Rotation ───────────────────────────────────────────────────────

def get_todays_categories(cfg: dict) -> list:
    """Primary categories + today's rotating picks (deterministic by date)."""
    primary = cfg.get("primary_categories", [])
    rotating = cfg.get("rotating_categories", [])
    per_run = cfg.get("rotating_per_run", 3)

    if not rotating or per_run <= 0:
        return primary

    today_str = datetime.utcnow().strftime("%Y-%m-%d")
    day_hash = int(hashlib.md5(today_str.encode()).hexdigest(), 16)
    n = len(rotating)
    start = day_hash % n
    picked = [rotating[(start + i) % n] for i in range(per_run)]

    log.info(f"Categories: {primary + picked}")
    return primary + picked


# ── Prompt ──────────────────────────────────────────────────────────────────

def build_prompt(categories: list, locations: list, omit_topics: list) -> str:
    """Builds the user-turn prompt. Formatting rules live in SYSTEM_PROMPT above."""
    today = datetime.utcnow().strftime("%B %d, %Y")

    location_labels = {
        "world": "international/world", "canada": "Canada", "us": "United States",
        "uk": "United Kingdom", "eu": "European Union", "local": "local/regional",
    }
    loc_descriptions = [location_labels.get(l, l) for l in locations]

    omit_block = ""
    if omit_topics:
        omit_block = (
            "\nEXCLUDE stories primarily about: "
            + ", ".join(omit_topics) + ".\n"
        )

    return f"""Today is {today}.

Search the web for recent (preferably yesterday's) important news from the last 7 days, across these categories:
{chr(10).join(f'- {cat}' for cat in categories)}

Geographic focus: {', '.join(loc_descriptions)}
{omit_block}
Rules for the stories you select:
- Every headline and summary MUST come directly from a real article found in search results. NEVER invent, speculate, or extrapolate.
- If a search returns no relevant results for a category, skip that category rather than fabricating a story.

Return exactly 3 stories. Important: Each story must be two lines: a **bold** headline, then a plain 1-2 sentence summary. One blank line between stories. No narration, no numbers, no bullets."""


# ── Provider ─────────────────────────────────────────────────────────────────

def search_anthropic(prompt: str, cfg: dict) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    # Prefer env override, else use the HAIKU_MODEL constant defined above
    model = os.environ.get("ANTHROPIC_MODEL") or HAIKU_MODEL

    messages = [{"role": "user", "content": prompt}]
    tools = [{"type": "web_search_20250305", "name": "web_search"}]

    for turn in range(15):
        log.info(f"Claude {model} turn {turn + 1}...")
        response = None
        for attempt in range(5):
            try:
                response = client.messages.create(
                    model=model,
                    max_tokens=1024,
                    system=SYSTEM_PROMPT,
                    tools=tools,
                    messages=messages,
                )
                break
            except anthropic.RateLimitError:
                wait = 60 * (attempt + 1)
                log.warning(f"Rate limited ({attempt+1}/5). Waiting {wait}s...")
                time.sleep(wait)

        if response is None:
            log.error("Failed after 5 retries.")
            return ""

        if response.stop_reason == "end_turn":
            return "\n".join(
                b.text for b in response.content if hasattr(b, "text")
            )

        messages.append({"role": "assistant", "content": response.model_dump()["content"]})
        tool_results = []
        for block in response.content:
            bd = block.model_dump() if hasattr(block, "model_dump") else {}
            if bd.get("type") == "tool_use":
                tool_results.append({
                    "type": "tool_result", "tool_use_id": bd["id"],
                    "content": "Search completed.",
                })
        if tool_results:
            messages.append({"role": "user", "content": tool_results})
        else:
            parts = [b.text for b in response.content if hasattr(b, "text")]
            return "\n".join(parts) if parts else ""

    return ""


PROVIDERS = {"anthropic": search_anthropic}


# ── Parsing ────────────────────────────────────────────────────────────────

def parse_headlines(raw: str) -> list[dict]:
    """Parse bold headline + summary pairs from model output.

    Anchors on **bold** lines as headlines, so stray narration lines or
    misplaced numbers are automatically discarded rather than corrupting
    the pairing. Leading numbers and bullets are stripped from all lines.
    """
    def clean(line: str) -> str:
        """Strip leading list markers like '1.' or '-' or '•'."""
        return re.sub(r'^[\d]+\.\s*|^[-•]\s*|^\*\s+', '', line).strip()

    def is_headline(line: str) -> bool:
        c = clean(line)
        return c.startswith("**") and c.endswith("**") and len(c) > 4

    results = []
    lines = [l.strip() for l in raw.strip().splitlines() if l.strip()]

    i = 0
    while i < len(lines) and len(results) < 3:
        if is_headline(lines[i]):
            headline = clean(lines[i])
            summary = ""
            # Advance to the next non-empty line and take it as the summary,
            # unless it turns out to be another headline (story has no summary).
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j < len(lines) and not is_headline(lines[j]):
                summary = clean(lines[j])
                i = j + 1
            else:
                i += 1
            results.append({"headline": headline, "summary": summary})
        else:
            i += 1  # Skip narration or any other non-headline line

    return results


# ── Kindroid Delivery ───────────────────────────────────────────────────────

def send_to_kindroid(entries: list[dict], cfg: dict):
    """Send headlines + summaries as a chat message to Kindroid."""
    kin_id = os.environ.get("KINDROID_AI_ID")
    api_key = os.environ.get("KINDROID_API_KEY")

    if not kin_id or not api_key:
        log.info("KINDROID_AI_ID or KINDROID_API_KEY not set — skipping.")
        return

    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

    parts = []
    for entry in entries:
        story = entry["headline"]
        if entry.get("summary"):
            story += f"\n{entry['summary']}"
        parts.append(story)

    body = "\n\n".join(parts)
    message = cfg.get("kindroid_message", "Today's top headlines:")
    resp = requests.post(
        "https://api.kindroid.ai/v1/send-message",
        headers=headers,
        json={"ai_id": kin_id, "message": f"{message}\n\n{body}"},
    )

    if resp.ok:
        log.info("Sent to Kindroid")
    else:
        log.error(f"Kindroid failed: {resp.status_code} - {resp.text}")


# ── Main ────────────────────────────────────────────────────────────────────

def run():
    log.info("Fetching top 3 headlines...")
    cfg = CONFIG
    categories = get_todays_categories(cfg)

    provider = os.environ.get("NEWS_PROVIDER", cfg.get("provider", "anthropic")).lower()
    search_fn = PROVIDERS.get(provider)
    if not search_fn:
        log.error(f"Unknown provider '{provider}'")
        return

    prompt = build_prompt(
        categories,
        cfg.get("locations", ["us", "world"]),
        cfg.get("omit_topics", []),
    )
    raw = search_fn(prompt, cfg)

    if not raw:
        log.warning("No headlines generated.")
        return

    log.info(f"Raw response from model:\n{raw}")
    entries = parse_headlines(raw)[:3]

    if not entries:
        log.warning("No headlines parsed — nothing to send.")
        return

    for entry in entries:
        print(f"\n{entry['headline']}")
        if entry.get("summary"):
            print(f"   {entry['summary']}")

    send_to_kindroid(entries, cfg)


if __name__ == "__main__":
    run()
