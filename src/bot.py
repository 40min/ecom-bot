import json
import time
import logging
from typing import Any
from venv import logger


from langchain.agents import create_agent
from langchain_core.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from openai import (
    APITimeoutError,
    APIConnectionError,
    AuthenticationError
)
from pydantic import BaseModel, Field

from src.orders_db import lookup_order_tool
from src.prompts.style_config import StyleConfig
from src.prompts.examples import get_few_shots


logger = logging.getLogger(__name__)

class StructuredAnswer(BaseModel):
    answer: str = Field(description="Основная фраза-ответ на вопрос клиента")
    actions: list[str] = Field(description="Пошаговое описание процесса или дополнительные пояснения", default=[])
    tone: str = Field(description="Самоконтроль соответствия тона общения, не нарушается ли тон общения и ограничения, не выходит ли за рамки заданного стиля", default="")

    def __str__(self):
        return self.answer + "\n" + "\n".join(self.actions)


# Создаём класс для CLI-бота
class CliBot():
    def __init__(self,
                 model_name: str,
                 api_key: str,
                 person: StyleConfig,
                 faq_file: str = './data/faq.json',
                 examples_file: str = './data/few_shots_alex.jsonl',

):
        self.chat_model = ChatOpenAI(
            model=model_name,
            temperature=0.3,
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            timeout=15,
        )
                    
        self.checkpointer = InMemorySaver()

        # Load examples using get_few_shots method
        examples_file = f"./data/few_shots_{person.person_name}.jsonl"
        self.few_shots_examples = get_few_shots(examples_file)

        self.agent = self._create_agent(person, faq_file)

    def _create_agent(self, person: StyleConfig, faq_file: str):

        # Load FAQ data
        faq_data = self._load_faq(faq_file)
        
        # Generate person system prompt addition
        person_prompt = person.get_system_prompt_addition()    
        
        system_prompt: str = f"""
        {person_prompt}

Отвечай на вопросы клиентов, используя информацию из базы данных магазина.
База знаний (FAQ):
{faq_data}
Используй эту информацию для ответов на типичные вопросы клиентов.
Если вопрос не покрывается FAQ, отвечай на основе общих знаний о работе интернет-магазинов.
Ответ должен содержать основную фразу (answer) и пошаговое описание процесса или дополнительные пояснения (actions).
Также оцени свой ответ: не нарушается ли тон общения и ограничения, не выходит ли за рамки заданного стиля.

Когда клиент спрашивает о статусе заказа либо вводит команду "/order order_id" (например, "/order 12345"),
используй инструмент для поиска информации о заказе.

"""        
        
        return create_agent(
            model=self.chat_model,
            tools=[lookup_order_tool],
            system_prompt=system_prompt,
            checkpointer=self.checkpointer,
            response_format=StructuredAnswer,            
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
                self.checkpointer.delete_thread(session_id)
                session_id = self._get_new_session_id(user_id)
                print("Бот: Контекст диалога очищен.")
                continue

            try:
                examples = self.few_shots_examples.format(input=user_text)
                print("Sending request to API...")
                
                start_time = time.time()
                response = self.agent.invoke(
                    {
                        "messages": [                            
                            {"role": "system", "content": examples},
                            {"role": "user", "content": user_text}
                        ]
                    },
                    config={"configurable": {"thread_id": session_id}}
                )
                end_time = time.time()

                bot_reply: StructuredAnswer = response["structured_response"]

                token_usage = self._extract_token_usage(response)
                extra = {'token_usage': token_usage} if token_usage else {}
                
                logging.info(f"Bot: {bot_reply.model_dump_json(indent=2)}", extra=extra)
                                
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
