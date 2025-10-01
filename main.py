import os
import requests
import json
import statistics
from typing import List, Dict, Any
from forecasting_platform_sdk import ForecastingBot, ReasonedPrediction, BinaryPrediction, PredictedOptionList, NumericDistribution

# =============================
#  CONFIG
# =============================

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
EXA_API_KEY = os.getenv("EXA_API_KEY")
SERP_API_KEY = os.getenv("SERP_API_KEY")

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

ALLOWED_TOURNAMENTS = [
    "32813",
    "minibench",
    "market-pulse-25q4"
]

# Hard-coded models
PROPONENT_MODEL = "openrouter/openai/gpt-5"
OPPONENT_MODEL = "openrouter/openai/gpt-4o"
SYNTH_MODELS = [
    "openrouter/openai/gpt-5",
    "openrouter/openai/gpt-4o",
    "openrouter/openai/gpt-4o-mini"
]
PARSER_MODEL = "openrouter/openai/gpt-4o-mini"

# =============================
#  UTILITIES
# =============================

def call_openrouter(model: str, messages: List[Dict[str, str]]) -> str:
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.7
    }
    r = requests.post(OPENROUTER_URL, headers=headers, data=json.dumps(payload))
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]

def fetch_exa(query: str) -> List[Dict[str, str]]:
    headers = {"Authorization": f"Bearer {EXA_API_KEY}"}
    url = f"https://api.exa.ai/search?q={query}&type=news&num_results=5"
    r = requests.get(url, headers=headers)
    r.raise_for_status()
    results = r.json().get("results", [])
    return [{"title": res.get("title"), "url": res.get("url"), "snippet": res.get("snippet")} for res in results]

def fetch_serp(query: str) -> List[Dict[str, str]]:
    url = f"https://serpapi.com/search.json?engine=google_news&q={query}&api_key={SERP_API_KEY}"
    r = requests.get(url)
    r.raise_for_status()
    news = r.json().get("news_results", [])
    return [{"title": n.get("title"), "url": n.get("link"), "snippet": n.get("snippet")} for n in news]

def aggregate_research(query: str) -> (str, List[Dict[str, str]]):
    exa_results = fetch_exa(query)
    serp_results = fetch_serp(query)
    combined = exa_results + serp_results
    sources = []
    for i, item in enumerate(combined[:10]):
        sid = f"S{i+1}"
        sources.append({"id": sid, "title": item["title"], "url": item["url"], "snippet": item["snippet"]})
    summary = "\n".join([f"{s['id']}: {s['title']} - {s['snippet']}" for s in sources])
    return summary, sources

def parse_probability(response: str) -> float:
    """Extracts probability from model response (expects 0-100%)."""
    try:
        for tok in response.split():
            if tok.replace("%", "").replace(".", "").isdigit():
                val = float(tok.replace("%", ""))
                return min(max(val, 0.0), 100.0)
    except:
        pass
    return 50.0

# =============================
#  BOT
# =============================

class DebateForecastBot2025(ForecastingBot):

    def run_research(self, question) -> (str, List[Dict[str, str]]):
        return aggregate_research(question.question_text)

    def _debate(self, question, research: str, sources: List[Dict[str, str]]) -> (float, str):
        # Proponent
        pro_msg = [
            {"role": "system", "content": "You are the Proponent. Argue why the event is likely to happen."},
            {"role": "user", "content": f"Question: {question.question_text}\nResolution: {question.resolution_criteria}\nSources:\n{research}"}
        ]
        pro_out = call_openrouter(PROPONENT_MODEL, pro_msg)

        # Opponent
        opp_msg = [
            {"role": "system", "content": "You are the Opponent. Argue why the event is unlikely to happen."},
            {"role": "user", "content": f"Question: {question.question_text}\nResolution: {question.resolution_criteria}\nSources:\n{research}"}
        ]
        opp_out = call_openrouter(OPPONENT_MODEL, opp_msg)

        # Synthesizers
        synth_probs = []
        synth_justs = []
        for m in SYNTH_MODELS:
            synth_msg = [
                {"role": "system", "content": "You are the Synthesizer. Evaluate proponent vs opponent, use sources, and give a final probability 0-100% with a justification citing S#."},
                {"role": "user", "content": f"Question: {question.question_text}\nResolution: {question.resolution_criteria}\nSources:\n{research}\n\nProponent:\n{pro_out}\n\nOpponent:\n{opp_out}"}
            ]
            out = call_openrouter(m, synth_msg)
            p = parse_probability(out)
            synth_probs.append(p)
            synth_justs.append(out)

        final_prob = statistics.median(synth_probs)
        justification = "\n---\n".join(synth_justs)
        return final_prob, justification

    def _run_forecast_on_binary(self, question):
        if question.tournament_id not in ALLOWED_TOURNAMENTS:
            raise ValueError(f"Tournament {question.tournament_id} not supported")

        research, sources = self.run_research(question)
        prob, just = self._debate(question, research, sources)
        pred = BinaryPrediction.yes_probability(prob / 100.0)
        return ReasonedPrediction(pred, justification=just, sources=sources)

    def _run_forecast_on_multiple_choice(self, question):
        if question.tournament_id not in ALLOWED_TOURNAMENTS:
            raise ValueError(f"Tournament {question.tournament_id} not supported")

        research, sources = self.run_research(question)
        # For MC: distribute prob based on debate strength (simplified: equal unless biased)
        prob, just = self._debate(question, research, sources)
        n = len(question.options)
        base = 1.0 / n
        probs = [base for _ in question.options]
        # bias first option by prob (just for demo; can be smarter)
        probs[0] = prob / 100.0
        remain = 1.0 - probs[0]
        if n > 1:
            for i in range(1, n):
                probs[i] = remain / (n - 1)
        pred = PredictedOptionList(options=question.options, probabilities=probs)
        return ReasonedPrediction(pred, justification=just, sources=sources)

    def _run_forecast_on_numeric(self, question):
        if question.tournament_id not in ALLOWED_TOURNAMENTS:
            raise ValueError(f"Tournament {question.tournament_id} not supported")

        research, sources = self.run_research(question)
        prob, just = self._debate(question, research, sources)
        # Simple numeric: set median at prob, spread 10%
        p10 = max(0.0, (prob - 10) / 100.0)
        p50 = prob / 100.0
        p90 = min(1.0, (prob + 10) / 100.0)
        dist = NumericDistribution.from_percentiles([0.1, 0.5, 0.9], [p10, p50, p90])
        return ReasonedPrediction(dist, justification=just, sources=sources)
