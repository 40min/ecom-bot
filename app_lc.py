import datetime
import json
import logging
import os
from pathlib import Path

import click
from dotenv import load_dotenv

from src.bot import CliBot
from src.orders_db import load_orders
from src.prompts.style_config import StyleConfig
from src.style_eval import BotEvaluator

load_dotenv()


class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "message": record.getMessage(),
        }
        token_usage = getattr(record, "token_usage", None)
        if token_usage:
            log_record["token_usage"] = token_usage
        return json.dumps(log_record, ensure_ascii=False)


timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"./logs/session_{timestamp}.jsonl"

handler = logging.FileHandler(log_filename, encoding="utf-8")
handler.setFormatter(JsonFormatter(datefmt="%Y-%m-%d %H:%M:%S"))

logging.basicConfig(level=logging.INFO, handlers=[handler], force=True)


def get_common_config():
    """Get common configuration for both bot and evaluate commands."""
    model_name = os.getenv("OPENROUTER_API_MODEL", "gpt-4o-mini")
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY is not set")
    person_name = os.getenv("PERSON_NAME", "alex")

    # Load person configuration
    person = StyleConfig.load(person_name, "./data/style_guide.yaml")

    # Load orders data
    load_orders()

    return {"model_name": model_name, "api_key": api_key, "person": person}


@click.group()
def main():
    """E-commerce bot CLI with multiple modes."""
    pass


@main.command()
def bot():
    """Run the bot in interactive mode."""
    config = get_common_config()

    bot = CliBot(
        model_name=config["model_name"],
        api_key=config["api_key"],
        person=config["person"],
    )

    logging.info("=== New session ===")
    bot("user_123")


@main.command()
@click.option("--eval-model", default="gpt-4o-mini", help="Model to use for evaluation")
def evaluate(eval_model):
    """Run the bot in evaluation mode."""
    config = get_common_config()

    bot = CliBot(
        model_name=config["model_name"],
        api_key=config["api_key"],
        person=config["person"],
        silent=True,
    )

    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)

    evaluator = BotEvaluator(
        model_name=eval_model,
        api_key=config["api_key"],
        person=config["person"],
        reports_dir=reports_dir,
        bot=bot,
    )

    data_dir = Path("data")
    eval_prompts = (
        (data_dir / "eval_prompts.txt").read_text(encoding="utf-8").strip().splitlines()
    )

    report = evaluator.eval_batch(eval_prompts)

    summary = report["summary"]
    
    print("=" * 50)
    print("ОЦЕНКА СТИЛЯ - СВОДКА")
    print("=" * 50)
    print(f"📊 Общий средний балл: {summary['mean_final']:.2f}/100")
    print(f"📈 Процент прохождения: {summary['pass_rate']:.2f}%")
    print(f"✅ Успешные оценки: {summary['successful_evaluations']}/{summary['total_cases']}")
    print(f"❌ Неудачные оценки: {summary['failed_evaluations']}")
    
    print("\n📋 ДЕТАЛЬНЫЕ МЕТРИКИ:")
    print(f"  • Правила (rule-based): {summary['mean_rule_score']:.2f}")
    print(f"  • ИИ оценка (LLM-based): {summary['mean_llm_score']:.2f}")
    print(f"  • Стандартное отклонение: {summary['std_final']:.2f}")
    
    print(f"\n📊 РАСПРЕДЕЛЕНИЕ ОЦЕНОК:")
    print(f"  • Минимум: {summary['min_final']}")
    print(f"  • 25-й перцентиль: {summary['p25_final']}")
    print(f"  • Медиана: {summary['median_final']}")
    print(f"  • 75-й перцентиль: {summary['p75_final']}")
    print(f"  • 95-й перцентиль: {summary['p95_final']}")
    print(f"  • Максимум: {summary['max_final']}")
    
    if summary.get('violations_count', 0) > 0:
        print(f"\n⚠️  НАРУШЕНИЯ ПРАВИЛ:")
        print(f"  • Всего нарушений: {summary['violations_count']}")
        common_violations = summary.get('common_violations', {})
        if common_violations:
            for violation, count in list(common_violations.items())[:5]:
                print(f"  • {violation}: {count} раз(а)")
    
    print(f"\n📄 Полный отчёт сохранён: {reports_dir / 'style_eval.json'}")
    print(f"📋 Краткая сводка: {reports_dir / 'style_eval_summary.json'}")
    print("=" * 50)


if __name__ == "__main__":
    main()
