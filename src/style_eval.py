import json
import re
from pathlib import Path
from statistics import mean

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from src.bot import CliBot, StructuredAnswer
from src.prompts.style_config import StyleConfig

load_dotenv()


# LLM-оценка
class Grade(BaseModel):
    score: int = Field(..., ge=0, le=100)
    notes: str


class BotEvaluator:
    def __init__(
        self,
        model_name: str,
        api_key: str,
        person: StyleConfig,
        reports_dir: Path,
        bot: CliBot,
    ):
        self.style = person
        self.reports_dir = reports_dir
        self.bot = bot

        self.llm = ChatOpenAI(
            model=model_name,
            temperature=0.3,
            base_url="https://openrouter.ai/api/v1",
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

    def rule_checks(self, text: str) -> int:
        score = 100
        # 1) Без эмодзи
        if re.search(r"[\U0001F300-\U0001FAFF]", text):
            score -= 20
        # 2) Без крика!!!
        if "!!!" in text:
            score -= 10
        # 3) Длина
        if len(text) > 600:
            score -= 10
        return max(score, 0)

    def llm_grade(self, text: str) -> Grade:
        parser = self.llm.with_structured_output(Grade)
        return (self.grade_prompt | parser).invoke({"answer": text})  # type: ignore

    def ask_bot(self, prompt: str) -> StructuredAnswer:
        session_id = self.bot.get_new_session_id(user_id="style_eval")
        bot_reply, _ = self.bot.ask(user_text=prompt, session_id=session_id)
        return bot_reply

    def eval_batch(self, prompts: list[str]) -> dict:
        results = []
        for prompt in prompts:
            reply = self.ask_bot(prompt)
            static_rule_score = self.rule_checks(reply.answer)
            llm_score = self.llm_grade(reply.answer)
            final = int(0.4 * static_rule_score + 0.6 * llm_score.score)
            results.append(
                {
                    "user": prompt,
                    "assistant.answer": reply.answer,
                    "assistant.actions": reply.actions,
                    "assistant.tone": reply.tone,
                    "rule_score": static_rule_score,
                    "llm_score": llm_score.score,
                    "llm_notes": llm_score.notes,
                    "final": final,                    
                }
            )

        mean_final = round(mean(r["final"] for r in results), 2)
        out = {
            "summary": {
                "mean_final": mean_final,
                "mean_rule_score": round(mean(r["rule_score"] for r in results), 2),
                "mean_llm_score": round(mean(r["llm_score"] for r in results), 2),
                "total_cases": len(results),
                "violations (final < 80)": len([r for r in results if r["final"] < 80])
            },            
            "cases": results,
        }
        (self.reports_dir / "style_eval.json").write_text(
            json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        return out
