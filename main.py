from fastapi import FastAPI
from pydantic import BaseModel
from extractor import extract_entities

app = FastAPI()

class DocumentRequest(BaseModel):
    category: str
    text: str
@app.get("/")
def root():
    return {"message": "Service is running!"}

@app.post("/extract")
async def extract(req: DocumentRequest):
    output = extract_entities(req.category, req.text)
    return {"result": output}
