from mistralai import Mistral
from config import settings


def get_answer(promt: str):
    with Mistral(api_key=settings.MISTRAL_API_KEY) as mistral:
        return mistral.chat.complete(
            model="mistral-small-latest",
            messages=[
                {
                    "content": promt,
                    "role": "user",
                },
            ],
            stream=False,
        )
