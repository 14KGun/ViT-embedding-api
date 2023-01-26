from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Union
from vit import ViT

app = FastAPI()
vit = ViT()

class EmbeddingBody(BaseModel):
    imageUrls: List[str]

@app.post("/embedding")
def embeddingHandler(body: EmbeddingBody):
    try:
        return {
            "vector": vit.embeddingFromUrl(body.imageUrls[0])
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail="error")
