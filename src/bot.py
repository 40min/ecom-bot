import json
import time
import logging
from typing import Any
from venv import logger


from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from openai import (
    APITimeoutError,
    APIConnectionError,
    AuthenticationError
)

logger = logging.getLogger(__name__)

# Import the lookup_order_tool from order_db module
from src.orders_db import lookup_order_tool


# Создаём класс для CLI-бота
class CliBot():
    def __init__(self, 
                 model_name: str, 
                 api_key: str,
                 shop_name: str,
                 persona: str,
                 faq_file: str = './data/faq.json',

):        
        self.chat_model = ChatOpenAI(
            model=model_name,
            temperature=0.3,
            base_url="https://openrouter.ai/api/v1",       
            api_key=api_key,
            timeout=15,
        )

        # Load FAQ data
        faq_data = self._load_faq(faq_file)
        
        system_prompt: str = f"""
        Ты полезный сотрудник интернет-магазина {shop_name}.
        Отвечай на вопросы клиентов, используя информацию из базы данных магазина.
        Будь вежливым и корректным.
        Давай короткие ответы без лишних пояснения.

        База знаний (FAQ):
        {faq_data}

        Используй эту информацию для ответов на типичные вопросы клиентов.
        Если вопрос не покрывается FAQ, отвечай на основе общих знаний о работе интернет-магазинов.

        Когда клиент спрашивает о статусе заказа либо вводит команду "/order order_id" (например, "/order 12345"), 
        используй инструмент для поиска информации о заказе.
        """        

        self.checkpointer = InMemorySaver()

        self.agent = create_agent(
            model=self.chat_model,
            tools=[lookup_order_tool],
            system_prompt=system_prompt,
            checkpointer=self.checkpointer,
        )



    def _get_new_session_id(self, user_id: str) -> str:
        return f"{user_id}_{int(time.time())}"

    def _load_faq(self, file_path: str) -> str:
        """Load and format FAQ data from JSON file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                faq = json.load(f)
            
            # Format FAQ as text
            formatted = "\n\n".join([
                f"Вопрос: {item['q']}\nОтвет: {item['a']}"
                for item in faq
            ])
            
            return formatted
        except FileNotFoundError:
            print(f"Предупреждение: FAQ файл не найден по пути {file_path}")
            return "FAQ данные недоступны."
        except json.JSONDecodeError:
            print(f"Предупреждение: Неверный формат JSON в {file_path}")
            return "FAQ данные повреждены."
        except Exception as e:
            print(f"Ошибка при загрузке FAQ: {e}")
            return "FAQ данные недоступны."
        
    def _extract_token_usage(self, response: dict) -> Any:
        """Extract token usage information from the response."""
        last_msg = response['messages'][-1]
        if hasattr(last_msg, 'response_metadata'):
            return last_msg.response_metadata.get('token_usage').get('total_tokens')
        return None

    def __call__(self, user_id: str):
        session_id = self._get_new_session_id(user_id)
        while True:
            try:
                user_text = input("Вы: ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nБот: Завершение работы.")
                break
            if not user_text:
                continue

            logger.info(f"User: {user_text}")

            msg = user_text.lower()
            if msg in ("выход", "стоп", "конец"):
                print("Бот: До свидания!")
                break
            if msg == "сброс":
                # we neglect memory leak with old sessions as this is for testing purposes
                session_id = self._get_new_session_id(user_id)
                print("Бот: Контекст диалога очищен.")
                continue

            try:
                print("Sending request to API...")
                start_time = time.time()
                response = self.agent.invoke(
                    {"messages": [{"role": "user", "content": user_text}]},
                    config={"configurable": {"thread_id": session_id}}
                )
                end_time = time.time()

                bot_reply = response['messages'][-1].content

                token_usage = self._extract_token_usage(response)
                extra = {'token_usage': token_usage} if token_usage else {}
                logging.info(f"Bot: {bot_reply}", extra=extra)
                                
                print(f"Response time: {end_time - start_time:.2f} seconds")
                print('Бот:', bot_reply, "\n")
            except APITimeoutError as e:
                print("Бот: [Ошибка] Превышено время ожидания ответа.")
                continue
            except APIConnectionError as e:
                print("Бот: [Ошибка] Не удалось подключиться к сервису LLM.")
                continue
            except AuthenticationError as e:
                print("Бот: [Ошибка] Проблема с API‑ключом (неавторизовано).")
                break
            except Exception as e:
                print(f"Бот: [Неизвестная ошибка] {e}")
                continue
