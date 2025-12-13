
import os
import logging
import json
import datetime

from dotenv import load_dotenv

from src.bot import CliBot
from src.orders_db import load_orders
from src.prompts.style_config import StyleConfig


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

handler = logging.FileHandler(log_filename, encoding='utf-8')
handler.setFormatter(JsonFormatter(datefmt="%Y-%m-%d %H:%M:%S"))

logging.basicConfig(
    level=logging.INFO,
    handlers=[handler],
    force=True
)
if __name__ == "__main__":
    
    shop_name = os.getenv("BRAND_NAME", "Магазин")
    model_name = os.getenv("OPENROUTER_API_MODEL", "gpt-4o-mini")
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set")
    person_name = os.getenv("PERSON_NAME", "alex")

    # Load person configuration
    person = StyleConfig.load(person_name, './data/style_guide.yaml')

    # Load orders data
    load_orders()

    bot = CliBot(
        model_name = model_name,
        api_key=api_key,        
        person=person,
    )

    mode = os.getenv("MODE", "evaluate")
    
    if mode == "evaluate":
        from src.style_eval import 
        eval_batch()
    else:
        logging.info("=== New session ===")
        bot("user_123")
