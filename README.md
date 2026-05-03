# Kindroid News

Supports **Anthropic (Claude)** as the provider. No news API key needed — the AI does the web searching itself.

## What It Does
**Searches** the web for today's top news across your configured categories
**Summarizes** chosen categories into headlines and brief summary paragraphs.
**Sends** the three basic news snippets to your Kindroid AI companion

## Customize

1. **config.json** choose categories you want shuffled plus omit topics, customize kindroid message opener
2. **railway** enter environment variables into railway:

NEWS_PROVIDER="..." <-- anthropic
*PROVIDER*_API_KEY="sk..." <--- replace *provider* w/ anthropic, openai, or xai
KINDROID_AI_ID="..."
KINDROID_API_KEY="kn..."
SCHEDULE_CRON=" 0 11 * * * " <-- daily @ 7:00PM EST

3. **default models:** claude-haiku-4-5-20251001

## Prompt Hardening
URL verification — After the LLM responds, verify_headlines does an HTTP HEAD/GET on each source URL. Headlines with dead or missing URLs get dropped and logged. 

## For Extra Help
Feed this repo into Claude Code or OpenAI Codex if you'd like guided setup or edits for personalization.
