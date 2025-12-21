from mistralai import Mistral

key = 'e8vQnlZ9jgk7YQ5v1M2nIqioS7t38bXj'


def get_answer(promt: str, context=None):
    messages=[{"content": promt, "role": "user"}]
    if context:
        messages.append({"role": "system", "content": f"Для ответа на вопросы бери только информацию из контекста. Начало контекста <{context}>Конец контекста"})
    with Mistral(api_key=key) as mistral:
        response = mistral.chat.complete(
            model="mistral-small-latest",
            messages=messages,
            stream=False,
        )
        return response.choices[0].message.content
