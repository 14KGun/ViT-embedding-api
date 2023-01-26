import requests
import json

class ViT:
    def __init__(self):
        pass
    
    def embeddingFromUrl(self, imageUrl):
        res = requests.post("http://localhost:3000/predictions/vit", json={
            "url": imageUrl
        })

        if res.status_code != 200:
            raise ValueError("error")
        
        result = json.loads(res.text)
        return result["vector"]
