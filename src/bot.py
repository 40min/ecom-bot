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


class Citation(BaseModel):
    source: str = Field(description="Источник информации из FAQ")
    page: int = Field(description="Номер страницы или секции", default=0)
    snippet: str = Field(description="Релевантный фрагмент текста из источника")


class StructuredAnswer(BaseModel):
    answer: str = Field(description="Основная фраза-ответ на вопрос клиента")
    actions: list[str] = Field(
        description="Пошаговое описание процесса или дополнительные пояснения",
        default=[],
    )
    citations: list[Citation] = Field(
        description="Массив цитат из FAQ, использованных для ответа",
        default=[],
    )
    confidence: str = Field(description="Уровень уверенности в ответе: low, medium или high")
    tone: str = Field(
        description="Самоконтроль соответствия тона общения, не нарушается ли тон общения и ограничения, не выходит ли за рамки заданного стиля",
        default="",
    )

    def __str__(self):
        result = self.answer
        if self.actions:
            result += "\n" + "\n".join(self.actions)
        return result


# Создаём класс для CLI-бота
class CliBot:
    def __init__(
        self,
        model_name: str,
        api_key: str,
        api_url: str,
        person: StyleConfig,
        vector_store: FAISS,
        faq_docs_to_load: int = 4,
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
        self.person = person

        self.silent = silent
        self.checkpointer = InMemorySaver()

        # Load examples using get_few_shots method
        examples_file = f"./data/few_shots_{person.person_name}.jsonl"
        self.few_shots_examples = get_few_shots(examples_file)

        self.agent = self._create_agent(person)

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
            """Найти информацию в базе знаний FAQ по запросу клиента.
            
            Используй этот инструмент когда клиент задаёт вопрос о:
            - политике магазина (возврат, обмен, гарантия)
            - процедурах и правилах
            - часто задаваемых вопросах
            - общих вопросах о работе магазина
            
            Args:
                query: Поисковый запрос на русском языке
                
            Returns:
                Релевантная информация из FAQ или сообщение об отсутствии данных
            """
            try:
                results = vector_store.similarity_search(query, k=docs_to_load)
                
                if not results:
                    return "NO_DATA_FOUND"
                
                # Format results with metadata for citation tracking
                formatted_results = []
                for i, doc in enumerate(results, 1):
                    metadata = doc.metadata
                    source = metadata.get("source", f"FAQ-документ-{i}")
                    page = metadata.get("page", 0)
                    
                    formatted_results.append(
                        f"[ИСТОЧНИК: {source} | СТРАНИЦА: {page}]\n{doc.page_content}\n"
                    )
                
                return "\n---\n".join(formatted_results)
                
            except Exception as e:
                logger.error(f"Error in FAQ search: {e}")
                return "NO_DATA_FOUND"
        
        return search_faq

    def _create_agent(self, person: StyleConfig):        

        # Generate person system prompt addition
        person_prompt = person.get_system_prompt_addition()
        fallback_response = person.no_info_fallback_response

        system_prompt: str = f"""
{person_prompt}

Отвечай на вопросы клиентов, используя инструмент search_faq для поиска информации в базе знаний 
либо lookup_order_tool если вопрос связан с заказом

Обработка результатов поиска:
- Если инструмент search_faq вернул "NO_DATA_FOUND" или результаты не релевантны вопросу:
  * Используй стандартную фразу: "{fallback_response}"
  * Установи confidence: "low"
  * Оставь citations пустым массивом
  
- Если найдена релевантная информация:
  * Сформируй ответ на основе найденных данных
  * Извлеки citations из результатов (ищи метки [ИСТОЧНИК: ... | СТРАНИЦА: ...])
  * Для каждого использованного фрагмента создай объект citation с полями:
    - source: название источника
    - page: номер страницы
    - snippet: релевантный фрагмент текста (30-100 символов)
  * Оцени confidence:
    - "high": информация точно отвечает на вопрос, найдено 2+ релевантных источника
    - "medium": информация частично отвечает на вопрос или найден 1 источник
    - "low": информация косвенно связана с вопросом

Структура ответа:
- answer: основная фраза-ответ клиенту (соблюдай стиль общения!)
- actions: пошаговые инструкции или дополнительные пояснения (если применимо)
- citations: массив источников, использованных для ответа
- confidence: уровень уверенности (low/medium/high)
- tone: самоконтроль стиля общения

Обработка заказов:
Когда клиент спрашивает о статусе заказа или вводит команду "/order order_id",
используй инструмент lookup_order_tool для поиска информации о заказе.

НЕ пытайся отвечать на вопросы о политиках и процедурах магазина без использования search_faq!
"""
        
        faq_search_tool = self._create_faq_search_tool()

        return create_agent(
            model=self.chat_model,
            tools=[lookup_order_tool, faq_search_tool],
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

                # Display citations if present
                output = "Бот: " + str(bot_reply)
                if bot_reply.citations:
                    output += f"\n\n[Уверенность: {bot_reply.confidence}]"
                    output += f"\n[Источники: {len(bot_reply.citations)}]"
                
                self.say(output + "\n")

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
                logger.exception("Unexpected error in bot")
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
