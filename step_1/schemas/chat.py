from pydantic import BaseModel


class ChatRequest(BaseModel):
    promt: str
    context: str | None = None


class ChatResponse(BaseModel):
    answer: str
    token_input_count: int
    token_output_count: int
    token_total_count: int
