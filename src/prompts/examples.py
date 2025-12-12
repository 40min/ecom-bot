import json

from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

class FewShotExamplesException(Exception):
    """Custom exception for errors during few-shot examples loading."""
    pass

def get_few_shots(file_path: str) -> FewShotPromptTemplate:
    """Load examples from JSONL file and format them for few-shot prompting"""
    
    examples = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:  # Skip empty lines
                    example = json.loads(line)
                    examples.append({
                        "input": example["user"],
                        "output": example["assistant"]
                    })
    except FileNotFoundError:
        raise FewShotExamplesException(f"Предупреждение: Файл с примерами не найден по пути {file_path}")        
    except json.JSONDecodeError:
        raise FewShotExamplesException(f"Предупреждение: Неверный формат JSON в {file_path}")        
    except KeyError:
        raise FewShotExamplesException(f"Предупреждение: Отсутствуют обязательные поля 'user' или 'assistant' в {file_path}")        
    except Exception as e:
        raise FewShotExamplesException(f"Ошибка при загрузке примеров: {e}")        

    if not examples:
        raise FewShotExamplesException("Примеры ответов недоступны.")

    example_prompt = PromptTemplate.from_template(
        "user: {input}\nassistant: {output}"
    )

    example_selector = SemanticSimilarityExampleSelector.from_examples(
        examples=examples,
        embeddings=embeddings,
        vectorstore_cls=Chroma,
        k=1
    )

    prompt = FewShotPromptTemplate(
        example_selector=example_selector,
        example_prompt=example_prompt,
        prefix="Примеры ответов на вопросы клиентов:",
        suffix="",
        input_variables=["input"]
    )

    return prompt