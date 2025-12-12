from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate

def get_few_shots(file_path: str) -> str:
    
    # load examples from file in json
    examples = [
        {"user": "happy", "assistant": "sad"},
        {"user": "tall", "assistant": "short"}
    ]

    example_prompt = PromptTemplate.from_template(
        "user: {input}\assistant: {output}"
    )

    prompt = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        prefix="Примеры ответов на вопросы:",
        suffix="user: {input}\assistant:",
        # input_variables=["input"]
    )

    formatted_prompt = prompt.format()

    return formatted_prompt