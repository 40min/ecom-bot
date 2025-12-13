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
        raise ValueError("OPENAI_API_KEY is not set")
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

    print("Средний балл:", report["mean_final"])
    print("Отчёт:", reports_dir / "style_eval.json")


if __name__ == "__main__":
    main()
