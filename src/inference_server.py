from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.retriever import QARAGSystem
import torch

app = FastAPI()
system = QARAGSystem()

class QueryRequest(BaseModel):
    prompt: str
    use_rag: bool = True
    max_length: int = 200
    temperature: float = 0.7

@app.post("/generate")
async def generate_answer(request: QueryRequest):
    try:
        answer = system.generate_answer(
            question=request.prompt,
            use_rag=request.use_rag,
            n_docs=3
        )
        return {"response": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)