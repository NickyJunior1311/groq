import os
import time
import logging
import urllib.request
import urllib.parse
import json
import concurrent.futures
from datetime import datetime
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from groq import AsyncGroq
from duckduckgo_search import DDGS

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("inveniq")

# ── Groq client (async, with timeout) ─────────────────────────────────────────
GROQ_API_KEY = "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"
client = AsyncGroq(
    api_key=GROQ_API_KEY,
    timeout=45.0,  # don't let a hung upstream hold a worker forever
)

app = FastAPI(title="InvenIQ Backend", version="4.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request models ────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    messages: list
    model: str = "llama-3.3-70b-versatile"
    temperature: float = 0.1
    max_tokens: int = 8192

class SignalSearchRequest(BaseModel):
    category: str   # "weather", "calendar", "news", "raw"
    location: str = "Malaysia"
    context: str = ""  # CSV-derived keywords e.g. "beverages, snacks, sugar"

# ── Web search tool definition (OpenAI / Groq function-calling schema) ────────
WEB_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Search the web for current news, events, calendar dates, market trends, "
            "weather, or any unstructured data the user asks about. Use this whenever "
            "the user's question requires up-to-date or real-world information."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to look up on the web",
                }
            },
            "required": ["query"],
        },
    },
}


# ── DuckDuckGo helper with hard wall-clock timeout ────────────────────────────
# DDGS().text() can occasionally hang on a slow upstream; we wrap it in a
# thread + future.result(timeout=...) so the request can NEVER stall a worker
# longer than the budget below.
_DDGS_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=4)


def _ddgs_call_once(query: str, max_results: int) -> list:
    """Single blocking DDGS call. Run inside a thread via the executor."""
    with DDGS() as ddgs:
        return list(ddgs.text(query, max_results=max_results))


def _ddgs_search(
    query: str,
    max_results: int = 5,
    max_retries: int = 1,
    per_call_timeout: float = 8.0,
) -> tuple[list, str | None]:
    """
    DuckDuckGo text search with retry + rate-limit handling + hard timeout.
    Returns (results, error). On success, error is None.

    `max_retries=1` means a single attempt (no retries). Bump it for the
    /search-signal endpoint where the user is explicitly waiting.
    """
    last_err: str | None = None
    for attempt in range(max_retries):
        future = _DDGS_EXECUTOR.submit(_ddgs_call_once, query, max_results)
        try:
            results = future.result(timeout=per_call_timeout)
            if results:
                return results, None
            # empty results — short pause then maybe retry
            last_err = "empty"
            if attempt < max_retries - 1:
                time.sleep(1.5 * (attempt + 1))
        except concurrent.futures.TimeoutError:
            future.cancel()
            last_err = "timeout"
            logger.warning("DDGS timeout after %.1fs for query: %s", per_call_timeout, query)
            if attempt < max_retries - 1:
                time.sleep(1.0)
        except Exception as e:
            err_msg = str(e).lower()
            is_rate_limit = any(
                kw in err_msg for kw in ["429", "rate", "limit", "too many", "throttl", "captcha"]
            )
            last_err = "rate_limit" if is_rate_limit else f"error: {e}"
            if attempt < max_retries - 1:
                wait = (3 if is_rate_limit else 1.5) * (attempt + 1)
                logger.warning("DDGS %s — waiting %ss before retry %d", last_err, wait, attempt + 1)
                time.sleep(wait)
    return [], last_err


def perform_web_search(query: str) -> str:
    """
    Web search used by the AI's tool-use loop. Kept FAST and FAILS FAST so the
    chat endpoint can never get stuck. Returns a formatted markdown string the
    LLM can read (or a clear error message it can summarize).
    """
    # Single attempt only inside the chat loop — the loop itself can retry
    # with a different query if it wants to.
    results, err = _ddgs_search(query, max_results=5, max_retries=1, per_call_timeout=8.0)
    if results:
        return "\n\n".join(
            f"- **{r['title']}**\n  {r['body']}\n  Source: {r['href']}"
            for r in results
        )
    if err == "rate_limit":
        return ("Web search is rate-limited right now. Answer the user's question "
                "using your existing knowledge and clearly note that live data was "
                "unavailable.")
    if err == "timeout":
        return ("Web search timed out. Answer the user's question using your "
                "existing knowledge and clearly note that live data was unavailable.")
    if err and err != "empty":
        return (f"Web search failed ({err}). Answer using existing knowledge and "
                "note that live data was unavailable.")
    return ("No web results found for this query. Answer using existing knowledge "
            "and note that live data was unavailable.")


# ── System prompt (the AI's personality) ──────────────────────────────────────
MASTER_PROMPT = """You are InvenIQ, an elite, professional Inventory Intelligence AI.
Your primary job is to help the user manage their stock, analyze sales data, and predict inventory shortages.

RULES:
1. Be highly analytical, precise, and professional.
2. Format your answers clearly using bullet points or short paragraphs.
3. When the user asks about current news, events, market trends, weather, calendar info, or any real-time unstructured data, use the web_search tool to fetch up-to-date information before answering.
4. If web_search returns an error or rate-limit message, DO NOT call it again. Answer using your existing knowledge and clearly tell the user that live data was unavailable.
5. If you need more data (like a CSV file or numbers) to answer a question, ask the user to provide it.
"""

# ── Signal query templates (year is filled in dynamically) ────────────────────
def _signal_queries(year: int) -> dict[str, list[str]]:
    return {
        "weather": [],  # handled by wttr.in API directly
        "calendar": [
            f"upcoming Malaysia public holidays {year}",
            f"Malaysia school holidays calendar {year}{{ctx}}",
            f"Malaysia events and festivals {year}{{ctx}}",
        ],
        "news": [
            f"Malaysia{{ctx}}retail market news {year}",
            "Malaysia{ctx}supply chain business news",
            f"Malaysia consumer market updates {year}",
        ],
        "raw": [
            f"Malaysia{{ctx}}consumer trends economic outlook {year}",
            f"Malaysia GDP inflation food prices {year}",
            f"Malaysia{{ctx}}retail industry forecast {year}",
        ],
    }


# ── Weather (wttr.in) ─────────────────────────────────────────────────────────
def fetch_weather_wttr(location: str = "Malaysia") -> str:
    """Fetch real weather from wttr.in — free, no API key, fast."""
    cities = ["Kuala Lumpur", "Johor Bahru", "Penang", "Kota Kinabalu"]
    parts = []
    for city in cities:
        try:
            url = f"https://wttr.in/{urllib.parse.quote(city)}?format=j1"
            req = urllib.request.Request(url, headers={"User-Agent": "curl/7.68.0"})
            with urllib.request.urlopen(req, timeout=8) as resp:
                data = json.loads(resp.read().decode())

            current = data.get("current_condition", [{}])[0]
            weather_desc = current.get("weatherDesc", [{}])[0].get("value", "N/A")
            temp_c = current.get("temp_C", "N/A")
            humidity = current.get("humidity", "N/A")
            feels = current.get("FeelsLikeC", "N/A")

            forecast_lines = []
            for day in data.get("weather", []):
                date = day.get("date", "")
                max_t = day.get("maxtempC", "")
                min_t = day.get("mintempC", "")
                hourly = day.get("hourly", [])
                desc = (
                    hourly[4].get("weatherDesc", [{}])[0].get("value", "")
                    if len(hourly) > 4
                    else "N/A"
                )
                forecast_lines.append(f"  {date}: {min_t}°C–{max_t}°C, {desc}")
            forecast_str = "\n".join(forecast_lines[:5])

            parts.append(
                f"📍 {city}\n"
                f"  Now: {temp_c}°C (feels {feels}°C), {weather_desc}, Humidity {humidity}%\n"
                f"  Forecast:\n{forecast_str}"
            )
        except Exception as e:
            logger.warning("Weather fetch failed for %s: %s", city, e)
            parts.append(f"📍 {city}: Weather data unavailable")
    return "\n\n".join(parts)


# ── Tiny in-memory cache for /search-signal (5 min TTL) ───────────────────────
# Massively reduces DDGS hits when the user clicks the panel buttons repeatedly,
# which is the #1 cause of rate-limiting in this app.
_SIGNAL_CACHE: dict[str, tuple[float, dict]] = {}
_SIGNAL_CACHE_TTL = 300.0  # seconds


def _cache_get(key: str) -> dict | None:
    entry = _SIGNAL_CACHE.get(key)
    if not entry:
        return None
    ts, payload = entry
    if time.time() - ts > _SIGNAL_CACHE_TTL:
        _SIGNAL_CACHE.pop(key, None)
        return None
    return payload


def _cache_set(key: str, payload: dict) -> None:
    _SIGNAL_CACHE[key] = (time.time(), payload)


# ── Signal search endpoint ────────────────────────────────────────────────────
@app.post("/search-signal")
def search_signal(req: SignalSearchRequest):
    cache_key = f"{req.category}|{req.location}|{req.context}"
    cached = _cache_get(cache_key)
    if cached:
        logger.info("Signal cache HIT for %s", cache_key)
        return cached

    # ── Weather: use wttr.in directly ──
    if req.category == "weather":
        try:
            weather = fetch_weather_wttr(req.location)
            payload = {"results": weather, "query": "wttr.in API"}
            _cache_set(cache_key, payload)
            return payload
        except Exception as e:
            logger.exception("Weather endpoint failed")
            return {"results": f"Weather fetch failed: {e}", "query": "wttr.in API"}

    # ── Other categories: aggregate across all template queries ──
    queries_map = _signal_queries(datetime.now().year)
    templates = queries_map.get(req.category, queries_map["raw"])

    ctx_part = f" {req.context} " if req.context else " "

    rendered_queries: list[str] = []
    all_results: list[str] = []
    any_rate_limited = False

    for template in templates:
        query = " ".join(template.format(location=req.location, ctx=ctx_part).split())
        rendered_queries.append(query)

        # Inside /search-signal we allow 2 retries because the user is
        # explicitly waiting and clicked a button.
        results, err = _ddgs_search(
            query,
            max_results=6,
            max_retries=2,
            per_call_timeout=8.0,
        )
        if err == "rate_limit":
            any_rate_limited = True
        for r in results:
            all_results.append(f"• {r['title']}: {r['body']}")

    # Deduplicate while preserving order, cap at 8
    seen: set[str] = set()
    unique: list[str] = []
    for r in all_results:
        if r in seen:
            continue
        seen.add(r)
        unique.append(r)
        if len(unique) >= 8:
            break

    primary_query = rendered_queries[0] if rendered_queries else ""
    if not unique:
        msg = ("Search rate-limited by DuckDuckGo — try again in ~30 seconds."
               if any_rate_limited
               else "Search temporarily unavailable — try again in a moment.")
        # Don't cache failures — we want the next click to retry.
        return {"results": msg, "query": primary_query}

    payload = {"results": "\n".join(unique), "query": primary_query}
    _cache_set(cache_key, payload)
    return payload


# ── Chat endpoint (async + tool-use loop, OpenAI/Groq schema) ─────────────────
@app.post("/chat")
async def chat(req: ChatRequest):
    try:
        # Compose system prompt: master + any system messages from history.
        system_msg = MASTER_PROMPT
        chat_messages: list[dict] = []
        for msg in req.messages:
            if msg.get("role") == "system":
                system_msg += "\n" + msg.get("content", "")
            else:
                chat_messages.append(msg)

        full_messages = [{"role": "system", "content": system_msg}] + chat_messages

        common_kwargs = {
            "model": req.model,
            "temperature": req.temperature,
            "max_tokens": req.max_tokens,
            "tools": [WEB_SEARCH_TOOL],
            "tool_choice": "auto",
        }

        # First call
        response = await client.chat.completions.create(messages=full_messages, **common_kwargs)

        # Tool-use loop — capped tighter to keep total latency bounded
        max_iterations = 2
        for iteration in range(max_iterations):
            assistant_msg = response.choices[0].message
            tool_calls = assistant_msg.tool_calls or []
            if not tool_calls:
                break

            full_messages.append({
                "role": "assistant",
                "content": assistant_msg.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in tool_calls
                ],
            })

            unknown_tool_seen = False
            for tc in tool_calls:
                if tc.function.name == "web_search":
                    try:
                        args = json.loads(tc.function.arguments or "{}")
                    except json.JSONDecodeError:
                        args = {}
                    query = args.get("query", "")
                    logger.info("Tool web_search query: %s", query)
                    search_result = perform_web_search(query)
                else:
                    logger.warning("Unknown tool requested: %s", tc.function.name)
                    search_result = f"Unknown tool: {tc.function.name}"
                    unknown_tool_seen = True

                full_messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": search_result,
                })

            response = await client.chat.completions.create(messages=full_messages, **common_kwargs)

            if unknown_tool_seen:
                break
        else:
            logger.warning("Tool-use loop hit max_iterations=%d", max_iterations)

        final_text = response.choices[0].message.content or ""

        # Safety net: if after the loop the model is STILL trying to call tools,
        # synthesize a plain answer rather than returning empty content.
        if not final_text and response.choices[0].message.tool_calls:
            final_text = ("I tried to look up live information but the search service "
                          "kept failing. Please try again in a moment, or rephrase your "
                          "question so I can answer from existing knowledge.")

        return {
            "content": final_text,
            "model": response.model,
            "usage": {
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
            },
        }
    except Exception as e:
        logger.exception("Groq API ERROR")
        return {"error": str(e)}


# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "online",
        "model": "llama-3.3-70b-versatile",
        "provider": "Groq",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
