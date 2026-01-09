import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Optional

import emoji
import numpy as np
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, field_validator

from src.bot import CliBot, StructuredAnswer
from src.prompts.style_config import StyleConfig

load_dotenv()


# Configurable evaluation parameters
@dataclass
class EvalConfig:
    rule_weight: float = 0.4
    llm_weight: float = 0.6
    emoji_penalty: int = 20
    exclamation_penalty: int = 10
    length_penalty: int = 10
    max_length: int = 600
    passing_threshold: int = 80
    min_llm_notes_length: int = 20  # Minimum justification length for low scores

    # Rate limiting
    max_concurrent_requests: int = 5  # Max parallel requests
    delay_between_batches: float = 1.0  # Seconds between batches
    delay_between_requests: float = 0.2  # Seconds between individual requests



class Grade(BaseModel):
    score: int = Field(..., ge=0, le=100)
    notes: str

    @field_validator("notes")
    @classmethod
    def notes_not_empty(cls, v: str) -> str:
        if not v or len(v.strip()) < 5:
            raise ValueError("Notes must contain meaningful feedback")
        return v


class BotStyleEvaluator:
    def __init__(
        self,
        model_name: str,
        api_key: str,
        api_url: str,
        person: StyleConfig,
        reports_dir: Path,
        bot: CliBot,
        config: Optional[EvalConfig] = None,
    ):
        self.style = person
        self.reports_dir = reports_dir
        self.bot = bot
        self.config = config or EvalConfig()

        self.llm = ChatOpenAI(
            model=model_name,
            temperature=0.3,
            base_url=api_url,
            api_key=api_key,
            timeout=15,
        )

        self.grade_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    f"Ты — строгий ревьюер соответствия голосу бренда {self.style.brand}",
                ),
                (
                    "system",
                    f"Тон: {self.style.current_person_description}. Избегай: {', '.join(self.style.current_person_avoid)}. "
                    f"Обязательно: {', '.join(self.style.current_person_must_include)}.",
                ),
                (
                    "human",
                    "Ответ ассистента:\n{answer}\n\nДай целочисленный score 0..100 и краткие заметки почему.",
                ),
            ]
        )

    def rule_checks(self, text: str) -> dict:
        """Returns detailed rule checking results with violations list"""
        score = 100
        violations = []

        # 1) Emoji detection
        if self._has_emoji(text):
            score -= self.config.emoji_penalty
            violations.append("emoji_found")

        # 2) Excessive exclamation marks
        if "!!!" in text:
            score -= self.config.exclamation_penalty
            violations.append("excessive_exclamation")

        # 3) Length check
        if len(text) > self.config.max_length:
            score -= self.config.length_penalty
            violations.append("too_long")

        return {
            "score": max(score, 0),
            "violations": violations,
            "checks": {
                "has_emoji": self._has_emoji(text),
                "has_triple_exclamation": "!!!" in text,
                "length": len(text),
                "exceeds_max_length": len(text) > self.config.max_length,
            },
        }

    @staticmethod
    def _has_emoji(text: str) -> bool:
        """Improved emoji detection using emoji library"""
        return any(char in emoji.EMOJI_DATA for char in text)

    def llm_grade(self, text: str) -> Grade:
        """LLM grading with validation and sanity checks"""
        try:
            parser = self.llm.with_structured_output(Grade)
            grade: Grade = (self.grade_prompt | parser).invoke({"answer": text}) # type: ignore

            # Sanity check: low scores should have substantial justification
            if grade.score < 50 and len(grade.notes) < self.config.min_llm_notes_length:
                grade.notes += (
                    f" [WARNING: Low score ({grade.score}) with minimal justification. "
                    "This grade may be unreliable.]"
                )

            # Sanity check: perfect scores should explain why
            if grade.score >= 95 and "perfect" not in grade.notes.lower():
                grade.notes += " [Note: Near-perfect score assigned]"

            return grade

        except Exception as e:
            # Return a low-confidence grade instead of crashing
            return Grade(
                score=50,
                notes=f"[ERROR] Grading failed: {str(e)}. Assigned neutral score.",
            )

    async def _ask_bot_async(self, prompt: str) -> StructuredAnswer:
        """Async wrapper for bot interaction"""
        session_id = self.bot.get_new_session_id(user_id="style_eval")
    
        bot_reply, _ = await asyncio.to_thread(
            self.bot.ask,
            user_text=prompt,
            session_id=session_id
        )
        return bot_reply

    async def _eval_single_async(self, prompt: str, semaphore: asyncio.Semaphore) -> dict:
        """Evaluate a single prompt asynchronously with rate limiting"""
        async with semaphore:  # Limits concurrent requests
            try:
                # Optional: add delay between requests
                if self.config.delay_between_requests > 0:
                    await asyncio.sleep(self.config.delay_between_requests)
                
                reply = await self._ask_bot_async(prompt)
                
                # Run rule checks
                rule_result = self.rule_checks(reply.answer)
                
                # Run LLM grading (wrap in executor if needed)
                loop = asyncio.get_event_loop()
                llm_score = await loop.run_in_executor(
                    None, self.llm_grade, reply.answer
                )

                # Calculate final score
                final = int(
                    self.config.rule_weight * rule_result["score"]
                    + self.config.llm_weight * llm_score.score
                )

                return {
                    "user": prompt,
                    "assistant.answer": reply.answer,
                    "assistant.actions": reply.actions,
                    "assistant.tone": reply.tone,
                    "rule_score": rule_result["score"],
                    "rule_violations": rule_result["violations"],
                    "rule_checks": rule_result["checks"],
                    "llm_score": llm_score.score,
                    "llm_notes": llm_score.notes,
                    "final": final,
                    "passed": final >= self.config.passing_threshold,
                    "error": None,
                }

            except Exception as e:
                return {
                    "user": prompt,
                    "assistant.answer": None,
                    "error": str(e),
                    "final": 0,
                    "passed": False,
                }

    async def eval_batch_async(self, prompts: list[str]) -> dict:
        """Parallel async evaluation with rate limiting"""
        # Create semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(self.config.max_concurrent_requests)
        
        # Option 1: Process all with concurrency limit
        if self.config.delay_between_batches == 0:
            tasks = [self._eval_single_async(p, semaphore) for p in prompts]
            results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Option 2: Process in chunks with delays between chunks
        else:
            results = []
            chunk_size = self.config.max_concurrent_requests
            
            for i in range(0, len(prompts), chunk_size):
                chunk = prompts[i:i + chunk_size]
                print(f"Processing batch {i//chunk_size + 1}/{(len(prompts)-1)//chunk_size + 1} "
                      f"({len(chunk)} prompts)...")
                
                tasks = [self._eval_single_async(p, semaphore) for p in chunk]
                chunk_results = await asyncio.gather(*tasks, return_exceptions=True)
                results.extend(chunk_results)
                
                # Delay between batches (except after last batch)
                if i + chunk_size < len(prompts):
                    await asyncio.sleep(self.config.delay_between_batches)

        # Separate successful results from errors
        valid_results = []
        error_results = []
        
        for result in results:
            if isinstance(result, Exception):
                error_results.append({"error": str(result)})
            elif result.get("error"):
                error_results.append(result)
            else:
                valid_results.append(result)

        if not valid_results:
            return {
                "summary": {"error": "All evaluations failed"},
                "errors": error_results,
                "cases": [],
            }

        return self._compile_report(valid_results, error_results)
    
    
    def _compile_report(self, valid_results: list[dict], error_results: list[dict]) -> dict:
        """Compile comprehensive evaluation report with advanced metrics"""
        
        final_scores = [r["final"] for r in valid_results]
        rule_scores = [r["rule_score"] for r in valid_results]
        llm_scores = [r["llm_score"] for r in valid_results]

        # Calculate advanced statistics
        summary = {
            "config": {
                "rule_weight": self.config.rule_weight,
                "llm_weight": self.config.llm_weight,
                "passing_threshold": self.config.passing_threshold,
            },
            "total_cases": len(valid_results) + len(error_results),
            "successful_evaluations": len(valid_results),
            "failed_evaluations": len(error_results),
            "pass_rate": round(
                sum(1 for r in valid_results if r["passed"]) / len(valid_results) * 100, 2
            ),
            # Mean scores
            "mean_final": round(mean(final_scores), 2),
            "mean_rule_score": round(mean(rule_scores), 2),
            "mean_llm_score": round(mean(llm_scores), 2),
            # Distribution metrics
            "std_final": round(stdev(final_scores), 2) if len(final_scores) > 1 else 0,
            "min_final": min(final_scores),
            "max_final": max(final_scores),
            "median_final": round(np.median(final_scores), 2),
            "p25_final": round(np.percentile(final_scores, 25), 2),
            "p75_final": round(np.percentile(final_scores, 75), 2),
            "p95_final": round(np.percentile(final_scores, 95), 2),
            # Violation analysis
            "violations_count": sum(len(r["rule_violations"]) for r in valid_results),
            "common_violations": self._analyze_violations(valid_results),
            # Failures
            "failures": [r for r in valid_results if r["final"] < self.config.passing_threshold],
        }

        # Add worst and best cases
        if valid_results:
            summary["worst_case"] = min(valid_results, key=lambda x: x["final"])
            summary["best_case"] = max(valid_results, key=lambda x: x["final"])

        report = {
            "summary": summary,
            "cases": valid_results,
        }

        if error_results:
            report["errors"] = error_results

        # Save to file
        output_path = self.reports_dir / "style_eval.json"
        output_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        # Also save a summary-only file for quick review
        summary_path = self.reports_dir / "style_eval_summary.json"
        summary_path.write_text(
            json.dumps({"summary": summary}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        return report

    def _analyze_violations(self, results: list[dict]) -> dict:
        """Analyze most common rule violations"""
        violation_counts = {}
        for result in results:
            for violation in result.get("rule_violations", []):
                violation_counts[violation] = violation_counts.get(violation, 0) + 1

        return dict(sorted(violation_counts.items(), key=lambda x: x[1], reverse=True))

    def eval_batch(self, prompts: list[str]) -> dict:
        """Synchronous wrapper for async eval_batch"""
        return asyncio.run(self.eval_batch_async(prompts))
