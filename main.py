"""Example entry point that ties all helpers together."""

import os
import getpass

from prompt_utils import messages_to_prompt, completion_to_prompt
from model_setup import authenticate, load_llm, setup_service_context, build_index
from query_utils import classify_query


def main():
    hf_token = getpass.getpass("Вставьте ваш токен: ")
    openai_key = getpass.getpass("Введите OpenAI API Key:")
    authenticate(hf_token, openai_key)

    llm = load_llm()
    setup_service_context(llm)

    data_path = os.environ.get("DATA_PATH", "/kaggle/input/tanebaum-ostin")
    index = build_index(data_path)

    query = (
        "Серверы работают под управлением каких операционных систем? "
        "Поддерживаются ли UNIX и Windows?"
    )
    validation = classify_query(query)
    if validation != "Запрос корректный.":
        print(validation)
        return

    query_engine = index.as_query_engine(similarity_top_k=10)
    response = query_engine.query(query)

    message_template = f"""<s>system
Ты являешься моделью, которая отвечает только на основании предоставленных источников.
Отвечай строго на основе информации из текста.
Если нужной информации нет в источнике, ответь: 'я не знаю'. Не добавляй ничего, что не указано в тексте. Не придумывай и не добавляй лишние данные.
</s>
<s>user
Вопрос: {query}
Источник:
</s>
"""
    print("\nОтвет:")
    print(response.response)


if __name__ == "__main__":
    main()
