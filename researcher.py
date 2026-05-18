"""
ResearchAgent: Gemini-powered stock analyst using the REST API directly.

Uses function calling to gather news (Alpaca) and fundamental data (SEC EDGAR),
then returns a structured recommendation:
    {"signal": "buy"|"sell"|"hold", "confidence": float, "reasoning": str}

Add to config.py:
    gemini_key = "AIza..."  # from https://aistudio.google.com/apikey
"""

import json
import re
import requests

import config

_ALPACA_NEWS_URL = "https://data.alpaca.markets/v1beta1/news"
_ALPACA_HEADERS = {
    "APCA-API-KEY-ID": config.public,
    "APCA-API-SECRET-KEY": config.secret,
}

MODEL = "gemini-2.5-flash-lite"
_BASE_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent"

_SYSTEM = """\
You are a quantitative stock research analyst assisting an automated trading system.

You will receive:
- A stock symbol to analyze
- Current technical indicator values (SMA ratio, Bollinger %B, Momentum, PPO histogram)
- The technical model's directional signal (+1 = buy, -1 = sell, 0 = hold)
- Current portfolio exposure in the stock

Your workflow:
1. Call get_stock_news to assess recent news and market sentiment.
2. Call get_sec_data to review fundamental financial health.
3. Weigh the qualitative and fundamental factors against the technical signal.
4. Return ONLY a JSON object — no other text — in this exact format:
   {"signal": "buy" | "sell" | "hold", "confidence": <0.0-1.0>, "reasoning": "<2-3 sentences>"}

Guidelines:
- Only use "buy" or "sell" when confidence >= 0.6. Otherwise use "hold".
- "hold" is always valid when evidence is mixed or unclear.
- confidence reflects your conviction, not the certainty of profit.
- Be conservative: this system only executes when both technical and LLM signals agree.
"""

_TOOLS = [
    {
        "functionDeclarations": [
            {
                "name": "get_stock_news",
                "description": (
                    "Fetch recent news headlines and summaries for a stock ticker. "
                    "Use this to assess recent events, analyst sentiment, and market opinion."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "Stock ticker symbol (e.g. 'AAPL', 'SPY')",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Number of articles to return (max 20, default 10)",
                        },
                    },
                    "required": ["symbol"],
                },
            },
            {
                "name": "get_sec_data",
                "description": (
                    "Fetch key financial metrics from SEC EDGAR for a company: "
                    "the most recent revenue, net income, and EPS from 10-K/10-Q filings."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "Stock ticker symbol (e.g. 'AAPL', 'SPY')",
                        },
                    },
                    "required": ["symbol"],
                },
            },
        ]
    }
]


# ── Tool implementations ───────────────────────────────────────────────────────

def _fetch_news(symbol: str, limit: int = 10) -> dict:
    try:
        resp = requests.get(
            _ALPACA_NEWS_URL,
            headers=_ALPACA_HEADERS,
            params={"symbols": symbol, "limit": min(limit, 20), "sort": "desc"},
            timeout=10,
        )
        resp.raise_for_status()
        articles = [
            {
                "headline": art.get("headline", ""),
                "summary": art.get("summary", ""),
                "published": art.get("created_at", ""),
                "source": art.get("source", ""),
            }
            for art in resp.json().get("news", [])
        ]
        return {"articles": articles}
    except Exception as exc:
        return {"error": str(exc)}


def _fetch_sec_data(symbol: str) -> dict:
    headers = {"User-Agent": "MLTrader research-bot contact@example.com"}
    try:
        tickers_resp = requests.get(
            "https://www.sec.gov/files/company_tickers.json",
            headers=headers,
            timeout=10,
        )
        tickers_resp.raise_for_status()

        cik_padded = None
        company_name = symbol
        for entry in tickers_resp.json().values():
            if entry["ticker"].upper() == symbol.upper():
                cik_padded = str(entry["cik_str"]).zfill(10)
                company_name = entry["title"]
                break

        if not cik_padded:
            return {"error": f"CIK not found for {symbol} — may be an ETF or fund without SEC filings"}

        facts_resp = requests.get(
            f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik_padded}.json",
            headers=headers,
            timeout=15,
        )
        if facts_resp.status_code == 404:
            return {
                "note": f"{company_name} ({symbol}) has no XBRL financial data on SEC EDGAR. "
                        "This is common for ETFs and index funds — focus on news sentiment instead."
            }
        facts_resp.raise_for_status()
        us_gaap = facts_resp.json().get("facts", {}).get("us-gaap", {})

        result: dict = {"company": company_name}
        wanted = [
            ("Revenues",              "USD",        "revenue"),
            ("NetIncomeLoss",         "USD",        "net_income"),
            ("EarningsPerShareBasic", "USD/shares", "eps_basic"),
        ]
        for gaap_key, unit_key, label in wanted:
            entries = us_gaap.get(gaap_key, {}).get("units", {}).get(unit_key, [])
            filed = sorted(
                [e for e in entries if e.get("form") in ("10-K", "10-Q") and "val" in e],
                key=lambda e: e.get("end", ""),
                reverse=True,
            )
            result[label] = [
                {"period": e["end"], "value": e["val"], "form": e["form"]}
                for e in filed[:3]
            ]
        return result

    except Exception as exc:
        return {"error": str(exc)}


def _dispatch(name: str, args: dict) -> dict:
    if name == "get_stock_news":
        return _fetch_news(args.get("symbol", ""), args.get("limit", 10))
    if name == "get_sec_data":
        return _fetch_sec_data(args.get("symbol", ""))
    return {"error": f"Unknown function: {name}"}


# ── REST API helper ────────────────────────────────────────────────────────────

def _call_gemini(contents: list, force_tools: bool = False, no_tools: bool = False) -> dict:
    body: dict = {
        "systemInstruction": {"parts": [{"text": _SYSTEM}]},
        "tools": _TOOLS,
        "contents": contents,
    }
    if force_tools:
        # First call: require at least one tool call before answering
        body["toolConfig"] = {"functionCallingConfig": {"mode": "ANY"}}
    elif no_tools:
        # After tool results: emit final text, no more tool calls
        body["toolConfig"] = {"functionCallingConfig": {"mode": "NONE"}}

    resp = requests.post(
        _BASE_URL,
        params={"key": config.gemini_key},
        json=body,
        timeout=30,
    )
    data = resp.json()
    if "error" in data:
        raise RuntimeError(data["error"].get("message", "Gemini API error"))
    return data


# ── Public API ─────────────────────────────────────────────────────────────────

def research(
    symbol: str,
    technical_signal: float,
    indicators: dict,
    portfolio_qty: int,
) -> dict:
    """
    Run the Gemini research agent for the given symbol.

    Returns:
        signal     — "buy" | "sell" | "hold"
        confidence — 0.0–1.0
        reasoning  — human-readable explanation
    """
    sig_label = {1.0: "BUY", -1.0: "SELL"}.get(float(technical_signal), "HOLD")

    def _fmt(v):
        return f"{v:.4f}" if isinstance(v, float) else str(v)

    user_text = (
        f"Analyze {symbol} and provide a trade recommendation.\n\n"
        f"Technical model signal : {sig_label} ({technical_signal:+.0f})\n"
        f"Current indicators:\n"
        f"  SMA ratio    : {_fmt(indicators.get('SMA', 'n/a'))}\n"
        f"  Bollinger %B : {_fmt(indicators.get('BBP', 'n/a'))}\n"
        f"  Momentum     : {_fmt(indicators.get('MOM', 'n/a'))}\n"
        f"  PPO histogram: {_fmt(indicators.get('PPO', 'n/a'))}\n"
        f"Current position       : {portfolio_qty:+d} shares\n\n"
        "Use the tools to research the stock, then return your JSON recommendation."
    )

    contents = [{"role": "user", "parts": [{"text": user_text}]}]

    # Agentic tool-use loop
    first_call = True
    for _ in range(5):  # max 5 rounds
        try:
            data = _call_gemini(contents, force_tools=first_call)
        except RuntimeError as exc:
            return {"signal": "hold", "confidence": 0.0, "reasoning": str(exc)}
        first_call = False

        candidate = data["candidates"][0]
        parts = candidate["content"]["parts"]

        # Collect any function calls in this response
        func_calls = [p["functionCall"] for p in parts if "functionCall" in p]

        if not func_calls:
            # No tool calls → extract final text and parse JSON
            final_text = "".join(p.get("text", "") for p in parts)
            match = re.search(r"\{[^{}]+\}", final_text, re.DOTALL)
            if match:
                try:
                    rec = json.loads(match.group())
                    return {
                        "signal": rec.get("signal", "hold").lower(),
                        "confidence": float(rec.get("confidence", 0.5)),
                        "reasoning": rec.get("reasoning", ""),
                    }
                except json.JSONDecodeError:
                    pass
            return {
                "signal": "hold",
                "confidence": 0.0,
                "reasoning": "Could not parse Gemini recommendation — defaulting to hold.",
            }

        # Execute all function calls and build response parts
        contents.append({"role": "model", "parts": parts})
        tool_response_parts = [
            {
                "functionResponse": {
                    "name": fc["name"],
                    "response": _dispatch(fc["name"], fc.get("args", {})),
                }
            }
            for fc in func_calls
        ]
        contents.append({"role": "user", "parts": tool_response_parts})

        # After tool results, force text-only final answer
        try:
            data = _call_gemini(contents, no_tools=True)
        except RuntimeError as exc:
            return {"signal": "hold", "confidence": 0.0, "reasoning": str(exc)}

        parts = data["candidates"][0]["content"]["parts"]
        final_text = "".join(p.get("text", "") for p in parts)
        match = re.search(r"\{[^{}]+\}", final_text, re.DOTALL)
        if match:
            try:
                rec = json.loads(match.group())
                return {
                    "signal": rec.get("signal", "hold").lower(),
                    "confidence": float(rec.get("confidence", 0.5)),
                    "reasoning": rec.get("reasoning", ""),
                }
            except json.JSONDecodeError:
                pass
        return {
            "signal": "hold",
            "confidence": 0.0,
            "reasoning": "Could not parse Gemini recommendation — defaulting to hold.",
        }

    return {
        "signal": "hold",
        "confidence": 0.0,
        "reasoning": "Research agent exceeded maximum tool-call rounds — defaulting to hold.",
    }
