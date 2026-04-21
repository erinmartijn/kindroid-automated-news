"""
News Headlines Bot
Searches for today's top 3 news headlines and sends them and a summary to a Kindroid AI
via message or profile update (configured by KINDROID_DELIVERY env var).

Supported providers: Anthropic (Claude), OpenAI (GPT), xAI (Grok)

✓ Auto-discovers the latest Claude Haiku model ID from Anthropic docs
✓ 30s timeout for discovery (handles Anthropic slowdowns gracefully)
✓ Robust fallback to known-current model if discovery fails
"""

import os
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

def get_yesterdays_categories(cfg: dict) -> list:
    """Primary categories + yesterday's rotating picks (deterministic by date)."""
    primary = cfg.get("primary_categories", [])
    rotating = cfg.get("rotating_categories", [])
    per_run = cfg.get("rotating_per_run", 2)

    if not rotating or per_run <= 0:
        return primary

    yesterday_str = datetime.utcnow().strftime("%Y-%m-%d")
    day_hash = int(hashlib.md5(yesterday_str.encode()).hexdigest(), 16)
    n = len(rotating)
    start = day_hash % n
    picked = [rotating[(start + i) % n] for i in range(per_run)]

    log.info(f"Categories: {primary + picked}")
    return primary + picked


# ── Prompt ──────────────────────────────────────────────────────────────────

def build_prompt(categories: list, locations: list, omit_topics: list) -> str:
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

    return f"""You are a news researcher. Today is {today}.

Search the web for today's most important news across these categories:
{chr(10).join(f'- {cat}' for cat in categories)}

Geographic focus: {', '.join(loc_descriptions)}
{omit_block}
IMPORTANT RULES:
- Do NOT narrate your process. Do NOT say things like "I'll search for news"
  or "Let me look up articles". Just perform your search silently and then
  output ONLY the final formatted results.
- Every headline and summary MUST come directly from a real article you found
  in search results. NEVER invent, speculate, or extrapolate.
- If a search returns no relevant results for a category, skip that category
  rather than fabricating a story.

Return EXACTLY 3 stories using this two-line format, with a blank line between each story:

Line 1: The article headline | article URL (the URL must appear ONLY after the |, nowhere else)
Line 2: A 1-2 sentence summary of the article in your own words.

Example:
Scientists discover new exoplanet in habitable zone | https://example.com/article
Researchers at MIT announced the discovery of a rocky exoplanet orbiting within the habitable zone of a nearby star, raising hopes for signs of liquid water.

No numbering, no bullets, no headers, no commentary outside this format."""


# ── Providers ───────────────────────────────────────────────────────────────

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
                    model=model, max_tokens=1024, tools=tools, messages=messages,
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


def search_openai(prompt: str, cfg: dict) -> str:
    from openai import OpenAI, RateLimitError
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    model = os.environ.get("OPENAI_MODEL", "gpt-4.1")

    response = None
    for attempt in range(5):
        try:
            response = client.responses.create(
                model=model, tools=[{"type": "web_search_preview"}], input=prompt,
            )
            break
        except RateLimitError:
            wait = 60 * (attempt + 1)
            log.warning(f"Rate limited ({attempt+1}/5). Waiting {wait}s...")
            time.sleep(wait)

    if response is None:
        return ""

    parts = []
    for item in response.output:
        if hasattr(item, "content"):
            for part in item.content:
                if hasattr(part, "text"):
                    parts.append(part.text)
    return "\n".join(parts)


def search_grok(prompt: str, cfg: dict) -> str:
    from openai import OpenAI, RateLimitError
    client = OpenAI(api_key=os.environ["XAI_API_KEY"], base_url="https://api.x.ai/v1")
    model = os.environ.get("GROK_MODEL", "grok-4-1-fast-reasoning")

    response = None
    for attempt in range(5):
        try:
            response = client.chat.completions.create(
                model=model, messages=[{"role": "user", "content": prompt}],
                 extra_body={"search_mode": "auto"},
            )
            break
        except RateLimitError:
            wait = 60 * (attempt + 1)
            log.warning(f"Rate limited ({attempt+1}/5). Waiting {wait}s...")
            time.sleep(wait)

    if response is None:
        return ""
    return response.choices[0].message.content or ""


PROVIDERS = {"anthropic": search_anthropic, "openai": search_openai, "grok": search_grok}


# ── Verification ───────────────────────────────────────────────────────────

def parse_headlines(raw: str) -> list[dict]:
    """Parse 'headline | url' + summary lines into structured dicts.

    Handles both compact format (summary immediately after headline) and
    spaced format (blank line between headline and summary).
    """
    results = []
    lines = [l.strip() for l in raw.strip().splitlines()]
    i = 0
    while i < len(lines):
        line = lines[i]
        if "|" in line:
            headline, url = line.rsplit("|", 1)
            # Advance to the next non-empty line for the summary
            j = i + 1
            while j < len(lines) and not lines[j]:
                j += 1
            summary = ""
            if j < len(lines) and "|" not in lines[j]:
                summary = lines[j]
                i = j + 1
            else:
                i += 1
            results.append({
                "headline": headline.strip(),
                "url": url.strip(),
                "summary": summary.strip(),
            })
        else:
            i += 1
    return results


def verify_headline(entry: dict, timeout: int = 10) -> bool:
    """Check that the source URL exists and responds."""
    url = entry.get("url")
    if not url or not url.startswith("http"):
        log.warning(f"No valid URL for: {entry['headline']}")
        return False
    try:
        resp = requests.head(url, allow_redirects=True, timeout=timeout,
                             headers={"User-Agent": "Mozilla/5.0 NewsBot/1.0"})
        if resp.status_code < 400:
            return True
        # Some sites block HEAD, try GET
        resp = requests.get(url, allow_redirects=True, timeout=timeout, stream=True,
                            headers={"User-Agent": "Mozilla/5.0 NewsBot/1.0"})
        return resp.status_code < 400
    except requests.RequestException as e:
        log.warning(f"URL check failed for {url}: {e}")
        return False


def verify_headlines(entries: list[dict]) -> list[dict]:
    """Return only headlines with reachable source URLs."""
    verified = []
    for entry in entries:
        if verify_headline(entry):
            log.info(f"✓ Verified: {entry['headline']}")
            verified.append(entry)
        else:
            log.warning(f"✗ Dropped (unverifiable): {entry['headline']}")
    return verified


# ── Kindroid Delivery ───────────────────────────────────────────────────────

def send_to_kindroid(headlines: str, cfg: dict):
    """Send 3 numbered headlines as a chat message to Kindroid."""
    kin_id = os.environ.get("KINDROID_AI_ID")
    api_key = os.environ.get("KINDROID_API_KEY")

    if not kin_id or not api_key:
        log.info("KINDROID_AI_ID or KINDROID_API_KEY not set — skipping.")
        return

    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

    lines = headlines.strip().splitlines()
    numbered = "\n".join(f"{i+1}. {line.strip()}" for i, line in enumerate(lines))
    message = cfg.get("kindroid_message", "Today's top headlines:")
    resp = requests.post(
        "https://api.kindroid.ai/v1/send-message",
        headers=headers,
        json={"ai_id": kin_id, "message": f"{message}\n\n{numbered}"},
    )

    if resp.ok:
        log.info("Sent to Kindroid")
    else:
        log.error(f"Kindroid failed: {resp.status_code} - {resp.text}")


# ── Main ────────────────────────────────────────────────────────────────────

def run():
    log.info("Fetching top 3 headlines...")
    cfg = CONFIG
    categories = get_yesterdays_categories(cfg)

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
    verified = verify_headlines(entries)

    if not verified:
        log.warning("All headlines failed verification — nothing to send.")
        return

    headlines = "\n".join(e["headline"] for e in verified)
    print(f"\n{headlines}\n")

    send_to_kindroid(headlines, cfg)


if __name__ == "__main__":
    run()

    run()
