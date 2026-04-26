"""
News Headlines Bot
Searches for today's top 3 news headlines (with summaries) and sends them
to a Kindroid AI companion via the Kindroid send-message API.

Supported provider: Anthropic (Claude) with built-in web search

✓ Auto-discovers the latest Claude Haiku model ID from Anthropic docs
✓ 30s timeout for discovery (handles Anthropic slowdowns gracefully)
✓ Robust fallback to known-current model if discovery fails
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
import urllib.request


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

SYSTEM_PROMPT = """You are a news aggregation bot. Your responses must contain ONLY formatted news stories — nothing else whatsoever.

ABSOLUTE RULES:
- NEVER output any preamble, narration, or commentary of any kind
- NEVER say things like "Let me search", "I'll look up", "Here are the results", "Here's what I found", or anything similar
- NEVER number or bullet your results
- Perform all searches silently and output ONLY the final formatted stories

Output format — repeat exactly 3 times, with one blank line between each story:

**Headline text here**
One to two sentence summary here.

Your entire response must be exactly 3 stories in this format and nothing else."""


# ── Model Discovery (auto-finds latest Haiku) ───────────────────────────────

def get_latest_haiku_model() -> str:
    """
    Returns the latest stable 'claude-haiku-*' model ID by fetching
    Anthropic's public docs list. Falls back to known-current if unavailable.

    ⚠️ Uses 30s timeout to accommodate Anthropic's frequent slowdowns.
    """
    # Known current model (in case docs site is down or slow)
    KNOWN_CURRENT = "claude-haiku-4-5-20251001"
    try:
        # Fetch Anthropic's official models list (read-only, no auth needed)
        url = "https://docs.anthropic.com/api/models-list"
        with urllib.request.urlopen(url, timeout=30) as response:
            if response.status != 200:
                raise Exception(f"API returned {response.status}")
            data = response.read().decode("utf-8")

        # Extract the model list using a lightweight search (simple & safe)
        if '"models"' not in data:
            raise ValueError("No models list found in docs")

        # Naive but robust: find the JSON array after "models":[ and parse it
        start = data.find('"models":[') + len('"models":[')
        end = data.find(']}', start) + 2
        if start < len('"models":[') or end <= start:
            raise ValueError("Could not isolate models JSON")
        try:
            models = json.loads(data[start:end])
        except Exception:
            raise ValueError("Models JSON not parseable")

        # Filter for claude-haiku-* versions, sort by date suffix (newest first)
        haikus = [m for m in models if isinstance(m.get("id"), str) and "claude-haiku-" in m["id"]]
        if not haikus:
            raise ValueError("No claude haiku models found")

        # Sort by model ID (dates are ISO-like: 20251001 > 20241022)
        haikus.sort(key=lambda x: x["id"], reverse=True)
        latest = haikus[0]["id"]
        log.info(f"✓ Auto-discovered latest Haiku model: {latest}")
        return latest

    except Exception as e:
        log.warning(f"⚠️ Auto-discovery failed ({e}). Using fallback: {KNOWN_CURRENT}")
        return KNOWN_CURRENT


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

Search the web for today's most important news across these categories:
{chr(10).join(f'- {cat}' for cat in categories)}

Geographic focus: {', '.join(loc_descriptions)}
{omit_block}
Rules for the stories you select:
- Every headline and summary MUST come directly from a real article found in search results. NEVER invent, speculate, or extrapolate.
- If a search returns no relevant results for a category, skip that category rather than fabricating a story.

Return exactly 3 stories. Each story is two lines: a **bold** headline, then a plain 1-2 sentence summary. One blank line between stories. No narration, no numbers, no bullets."""


# ── Provider ─────────────────────────────────────────────────────────────────

def search_anthropic(prompt: str, cfg: dict) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    # Prefer env override, else auto-discover latest Haiku
    model = os.environ.get("ANTHROPIC_MODEL")
    if not model:
        model = get_latest_haiku_model()

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
        return re.sub(r'^[\d]+\.\s*|^[-•*]\s*', '', line).strip()

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
