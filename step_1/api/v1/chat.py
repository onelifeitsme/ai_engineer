from fastapi import APIRouter
from schemas import chat as schemas
from utils.chat import get_answer


router = APIRouter()


@router.post("/chat", response_model=schemas.ChatResponse)
async def chat(
    chat_request: schemas.ChatRequest
):
    res = get_answer(promt=chat_request.promt)
    return schemas.ChatResponse(
        answer=res.choices[0].message.content,
        token_input_count=res.usage.prompt_tokens,
        token_output_count=res.usage.completion_tokens,
        token_total_count=res.usage.total_tokens
    )
