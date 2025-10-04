import argparse
import asyncio
import logging
from datetime import datetime
from typing import Literal, List, Union, Any
from statistics import median, mean

from forecasting_tools import (
    BinaryQuestion,
    ForecastBot,
    GeneralLlm,
    MetaculusApi,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericQuestion,
    BinaryPrediction,
    PredictedOptionList,
    NumericDistribution,
    Percentile,
    ReasonedPrediction,
    SmartSearcher,
    clean_indents,
    structure_output,
)

logger = logging.getLogger(__name__)

# === CONFIGURATION ===
FORECASTER_MODELS = [
    "openrouter/openai/gpt-5",
    "openrouter/anthropic/claude-sonnet-4",
    "openrouter/anthropic/claude-opus-4.1",
]
PARSER_MODEL = "openrouter/openai/gpt-4.1-mini"
RESEARCHER_MODEL = "openrouter/openai/gpt-5"

OUTPUT_FILE = "forecasts.txt"

# === BOT CLASS ===
class EnsembleForecastBot(ForecastBot):
    _max_concurrent_questions = 1
    _concurrency_limiter = asyncio.Semaphore(_max_concurrent_questions)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._final_predictions_log = []

    async def run_research(self, question: MetaculusQuestion) -> str:
        async with self._concurrency_limiter:
            searcher = SmartSearcher(
                model=RESEARCHER_MODEL,
                temperature=0.3,
                num_searches_to_run=2,
                num_sites_per_search=10,
                use_advanced_filters=True,
            )
            prompt = clean_indents(
                f"""
                You are a world-class research assistant for a superforecaster.
                Provide a concise, factual, and up-to-date summary relevant to forecasting the following question.

                Question: {question.question_text}
                Resolution Criteria: {question.resolution_criteria}
                Fine Print: {question.fine_print}
                """
            )
            research = await searcher.invoke(prompt)
            logger.info(f"Research for {question.page_url}:\n{research}")
            return research

    # --- BINARY ---
    async def _get_single_binary_forecast(self, question: BinaryQuestion, research: str, model: str) -> float:
        llm = GeneralLlm(model=model, temperature=0.3, timeout=45, allowed_tries=2)
        prompt = clean_indents(
            f"""
            You are a professional forecaster.
            Question: {question.question_text}
            Background: {question.background_info}
            Resolution Criteria: {question.resolution_criteria}
            Fine Print: {question.fine_print}
            Research Summary: {research}
            Today: {datetime.now().strftime('%Y-%m-%d')}

            Write a short rationale, then output ONLY: "Probability: XX%" where XX is 0–100.
            """
        )
        reasoning = await llm.invoke(prompt)
        pred: BinaryPrediction = await structure_output(
            reasoning, BinaryPrediction, model=GeneralLlm(model=PARSER_MODEL)
        )
        return max(0.01, min(0.99, pred.prediction_in_decimal))

    async def _run_forecast_on_binary(self, question: BinaryQuestion, research: str) -> ReasonedPrediction[float]:
        forecasts_pct = []
        raw_reasonings = []

        for model in FORECASTER_MODELS:
            try:
                prob = await self._get_single_binary_forecast(question, research, model)
                forecasts_pct.append(prob * 100)
                raw_reasonings.append(f"{model.split('/')[-1]}: {prob:.3f}")
            except Exception as e:
                logger.error(f"Binary forecast failed for {model}: {e}")
                forecasts_pct.append(50.0)
                raw_reasonings.append(f"{model.split('/')[-1]}: ERROR → 0.5")

        if max(forecasts_pct) - min(forecasts_pct) > 20:
            final_pct = mean(forecasts_pct)
        else:
            final_pct = median(forecasts_pct)

        final_prob = max(0.01, min(0.99, final_pct / 100.0))
        combined_reasoning = " | ".join(raw_reasonings) + f" → Final: {final_prob:.4f}"
        self._log_prediction(question, final_prob)
        return ReasonedPrediction(prediction_value=final_prob, reasoning=combined_reasoning)

    # --- MULTIPLE CHOICE ---
    async def _get_single_mc_forecast(self, question: MultipleChoiceQuestion, research: str, model: str) -> PredictedOptionList:
        llm = GeneralLlm(model=model, temperature=0.3, timeout=45, allowed_tries=2)
        prompt = clean_indents(
            f"""
            Forecast the probabilities for each option.
            Question: {question.question_text}
            Options: {question.options}
            Background: {question.background_info}
            Resolution Criteria: {question.resolution_criteria}
            Research: {research}
            Today: {datetime.now().strftime('%Y-%m-%d')}
            Output ONLY lines like: Option: Probability (0–1)
            """
        )
        reasoning = await llm.invoke(prompt)
        parsing_instructions = f"Valid options: {question.options}"
        pred: PredictedOptionList = await structure_output(
            reasoning,
            PredictedOptionList,
            model=GeneralLlm(model=PARSER_MODEL),
            additional_instructions=parsing_instructions,
        )
        return pred

    async def _run_forecast_on_multiple_choice(self, question: MultipleChoiceQuestion, research: str) -> ReasonedPrediction[PredictedOptionList]:
        all_preds = []
        raw_reasonings = []

        for model in FORECASTER_MODELS:
            try:
                pred = await self._get_single_mc_forecast(question, research, model)
                all_preds.append(pred)
                raw_reasonings.append(f"{model.split('/')[-1]}: {pred}")
            except Exception as e:
                logger.error(f"MC forecast failed for {model}: {e}")
                uniform = {opt: 1.0 / len(question.options) for opt in question.options}
                all_preds.append(PredictedOptionList(predicted_options=uniform))
                raw_reasonings.append(f"{model.split('/')[-1]}: ERROR → uniform")

        # Average per option
        avg_probs = {}
        for opt in question.options:
            probs = [p.predicted_options.get(opt, 0.0) for p in all_preds]
            avg_probs[opt] = sum(probs) / len(probs)
        final_pred = PredictedOptionList(predicted_options=avg_probs)
        combined_reasoning = " | ".join(raw_reasonings) + f" → Averaged"
        self._log_prediction(question, final_pred)
        return ReasonedPrediction(prediction_value=final_pred, reasoning=combined_reasoning)

    # --- NUMERIC ---
    def _create_bound_messages(self, q: NumericQuestion) -> tuple[str, str]:
        ub = q.nominal_upper_bound if q.nominal_upper_bound is not None else q.upper_bound
        lb = q.nominal_lower_bound if q.nominal_lower_bound is not None else q.lower_bound
        upper = f"Max possible: {ub}" if not q.open_upper_bound else f"Unlikely above {ub}"
        lower = f"Min possible: {lb}" if not q.open_lower_bound else f"Unlikely below {lb}"
        return lower, upper

    async def _get_single_numeric_forecast(self, question: NumericQuestion, research: str, model: str) -> NumericDistribution:
        llm = GeneralLlm(model=model, temperature=0.3, timeout=50, allowed_tries=2)
        lower_msg, upper_msg = self._create_bound_messages(question)
        prompt = clean_indents(
            f"""
            Question: {question.question_text}
            Units: {question.unit_of_measure or 'inferred'}
            {lower_msg}; {upper_msg}
            Research: {research}
            Today: {datetime.now().strftime('%Y-%m-%d')}
            Output percentiles: 10,20,40,60,80,90 as "Percentile X: value"
            """
        )
        reasoning = await llm.invoke(prompt)
        percentiles: List[Percentile] = await structure_output(
            reasoning, list[Percentile], model=GeneralLlm(model=PARSER_MODEL)
        )
        return NumericDistribution.from_question(percentiles, question)

    async def _run_forecast_on_numeric(self, question: NumericQuestion, research: str) -> ReasonedPrediction[NumericDistribution]:
        medians = []
        all_dists = []
        raw_reasonings = []

        for model in FORECASTER_MODELS:
            try:
                dist = await self._get_single_numeric_forecast(question, research, model)
                p50 = dist.get_percentile_value(50)
                medians.append(p50)
                all_dists.append(dist)
                raw_reasonings.append(f"{model.split('/')[-1]}: p50={p50:.1f}")
            except Exception as e:
                logger.error(f"Numeric forecast failed for {model}: {e}")
                fallback_p50 = (question.lower_bound + question.upper_bound) / 2
                medians.append(fallback_p50)
                raw_reasonings.append(f"{model.split('/')[-1]}: ERROR → midpoint")

        scale = max(question.upper_bound - question.lower_bound, 1)
        if (max(medians) - min(medians)) > 0.2 * scale:
            final_p50 = mean(medians)
        else:
            final_p50 = median(medians)

        base = all_dists[0] if all_dists else NumericDistribution.from_question([], question)
        shift = final_p50 - base.get_percentile_value(50)
        shifted = [Percentile(p.percentile, p.value + shift) for p in base.declared_percentiles]
        final_dist = NumericDistribution.from_question(shifted, question)

        combined_reasoning = " | ".join(raw_reasonings) + f" → Final p50: {final_p50:.1f}"
        self._log_prediction(question, final_dist)
        return ReasonedPrediction(prediction_value=final_dist, reasoning=combined_reasoning)

    # --- LOGGING & COMMENTING ---
    def _log_prediction(self, question: MetaculusQuestion, pred: Any):
        entry = {
            "url": question.page_url,
            "title": question.question_text,
            "type": question.type_name,
            "prediction": str(pred),
        }
        self._final_predictions_log.append(entry)

    async def forecast_questions(self, questions: List[MetaculusQuestion], **kwargs):
        reports = await super().forecast_questions(questions, **kwargs)
        # Post comments for successfully forecasted questions
        for report in reports:
            if not isinstance(report, Exception) and report.prediction is not None:
                try:
                    comment = self._format_comment(report)
                    await MetaculusApi.post_comment_on_question(report.question, comment)
                    logger.info(f"Posted comment on {report.question.page_url}")
                except Exception as e:
                    logger.warning(f"Failed to comment on {report.question.page_url}: {e}")
        return reports

    def _format_comment(self, report) -> str:
        q = report.question
        pred_str = str(report.prediction.prediction_value)
        models_used = ", ".join(m.split("/")[-1] for m in FORECASTER_MODELS)
        return clean_indents(
            f"""
            **Ensemble Forecast (3 models)**: {models_used}
            **Aggregation**: Median (or mean if range >20%)
            **Prediction**: {pred_str}

            Rationale based on latest research and trend analysis.
            Full methodology: Exa+SERP research → 3-model ensemble → rule-based aggregation.
            """
        )

    def save_predictions_to_file(self):
        with open(OUTPUT_FILE, "w") as f:
            f.write("ENSEMBLE FORECAST RESULTS — Submitted to Metaculus\n")
            f.write("=" * 60 + "\n\n")
            for entry in self._final_predictions_log:
                f.write(f"URL: {entry['url']}\n")
                f.write(f"Question: {entry['title']}\n")
                f.write(f"Type: {entry['type']}\n")
                f.write(f"Prediction: {entry['prediction']}\n")
                f.write("-" * 60 + "\n\n")
        logger.info(f"Final predictions saved to {OUTPUT_FILE}")

# === MAIN ===
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)

    parser = argparse.ArgumentParser(description="Ensemble bot with Metaculus submission & commenting")
    parser.add_argument("--mode", default="tournament", choices=["tournament"])
    args = parser.parse_args()

    bot = EnsembleForecastBot(
        research_reports_per_question=1,
        predictions_per_research_report=1,  # ensembling handled internally
        use_research_summary_to_forecast=False,
        publish_reports_to_metaculus=True,  # ✅ SUBMIT TO METACULUS
        skip_previously_forecasted_questions=False,
    )

    tournament_ids = [
        MetaculusApi.CURRENT_MINIBENCH_ID,   # minibench
        32813,                               # numeric ID
        "market-pulse-25q4",                # string ID
    ]

    all_reports = []
    for tid in tournament_ids:
        try:
            reports = asyncio.run(bot.forecast_on_tournament(tid, return_exceptions=True))
            all_reports.extend(reports)
        except Exception as e:
            logger.error(f"Tournament {tid} failed: {e}")

    bot.log_report_summary(all_reports)
    bot.save_predictions_to_file()
