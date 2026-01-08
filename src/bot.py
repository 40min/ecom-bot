import logging
import time

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langchain.tools import tool
from langchain_community.vectorstores import FAISS
from openai import APIConnectionError, APITimeoutError, AuthenticationError
from pydantic import BaseModel, Field

from src.orders_db import lookup_order_tool
from src.prompts.examples import get_few_shots
from src.prompts.style_config import StyleConfig

logger = logging.getLogger(__name__)


class StructuredAnswer(BaseModel):
    answer: str = Field(description="Основная фраза-ответ на вопрос клиента")
    actions: list[str] = Field(
        description="Пошаговое описание процесса или дополнительные пояснения",
        default=[],
    )
    tone: str = Field(
        description="Самоконтроль соответствия тона общения, не нарушается ли тон общения и ограничения, не выходит ли за рамки заданного стиля",
        default="",
    )

    def __str__(self):
        return self.answer + "\n" + "\n".join(self.actions)


# Создаём класс для CLI-бота
class CliBot:
    def __init__(
        self,
        model_name: str,
        api_key: str,
        api_url: str,
        person: StyleConfig,
        vector_store: FAISS,
        faq_docs_to_load: int = 3,
        silent: bool = False,
    ):
        self.chat_model = ChatOpenAI(
            model=model_name,
            temperature=0.3,
            base_url=api_url,
            api_key=api_key,
            timeout=15,
        )

        self.vector_store = vector_store
        self.faq_docs_to_load = faq_docs_to_load

        self.silent = silent
        self.checkpointer = InMemorySaver()

        # Load examples using get_few_shots method
        examples_file = f"./data/few_shots_{person.person_name}.jsonl"
        self.few_shots_examples = get_few_shots(examples_file)

        self.agent = self._create_agent(person, faq_file)

    def say(self, txt: str) -> None:
        """Output text if silent mode is False"""
        if not self.silent:
            print(txt)

    def _create_faq_search_tool(self):
        """Create FAQ search tool that captures vector_store"""
        vector_store = self.vector_store  # Capture in closure
        docs_to_load = self.faq_docs_to_load
        
        @tool
        def search_faq(query: str) -> str:
            """Search the FAQ database for information about customer questions.
            Use this when the customer asks about policies, procedures, or common questions."""
            results = vector_store.similarity_search(query, k=docs_to_load)
            return "\n\n".join([doc.page_content for doc in results])
        
        return search_faq

    def _create_agent(self, person: StyleConfig, faq_file: str):        

        # Generate person system prompt addition
        person_prompt = person.get_system_prompt_addition()

        system_prompt: str = f"""
        {person_prompt}

Отвечай на вопросы клиентов, используя информацию из базы данных магазина.
База знаний (FAQ):
empty
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

    def get_new_session_id(self, user_id: str) -> str:
        return f"{user_id}_{int(time.time())}"
    
    def _extract_token_usage(self, response: dict) -> int:
        """Extract token usage information from the response."""
        last_msg = response["messages"][-1]
        if hasattr(last_msg, "response_metadata"):
            return int(last_msg.response_metadata.get("token_usage").get("total_tokens", 0))
        return 0

    def __call__(self, user_id: str) -> None:
        session_id = self.get_new_session_id(user_id)
        while True:
            try:
                user_text = input("Вы: ").strip()
            except (KeyboardInterrupt, EOFError):
                self.say("\nБот: Завершение работы.")
                break
            if not user_text:
                continue

            logger.info(f"User: {user_text}")

            msg = user_text.lower()
            if msg in ("выход", "стоп", "конец"):
                self.say("Бот: До свидания!")
                break
            if msg == "сброс":
                self.checkpointer.delete_thread(session_id)
                session_id = self.get_new_session_id(user_id)
                self.say("Бот: Контекст диалога очищен.")
                continue

            try:

                bot_reply, token_usage = self.ask(user_text, session_id)

                extra = {"token_usage": token_usage} if token_usage else {}
                logging.info(f"Bot: {bot_reply.model_dump_json(indent=2)}", extra=extra)

                self.say("Бот: " + str(bot_reply) + "\n")

            except APITimeoutError:
                self.say("Бот: [Ошибка] Превышено время ожидания ответа.")
                continue
            except APIConnectionError:
                self.say("Бот: [Ошибка] Не удалось подключиться к сервису LLM.")
                continue
            except AuthenticationError:
                self.say("Бот: [Ошибка] Проблема с API‑ключом (неавторизовано).")
                break
            except Exception as e:
                self.say(f"Бот: [Неизвестная ошибка] {e}")
                continue

    def ask(self, user_text: str, session_id: str) -> tuple[StructuredAnswer, int]:
        examples = self.few_shots_examples.format(input=user_text)

        self.say("Sending request to API...")
        start_time = time.time()

        response = self.agent.invoke(
            {
                "messages": [
                    {"role": "system", "content": examples},
                    {"role": "user", "content": user_text},
                ]
            },
            config={"configurable": {"thread_id": session_id}},
        )

        end_time = time.time()
        token_usage = self._extract_token_usage(response)        

        self.say(
            f"Response time: {end_time - start_time:.2f} seconds, tokens: {token_usage}"
        )

        bot_reply: StructuredAnswer = response["structured_response"]

        return bot_reply, token_usage
