from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
from ml import obtain_image

class Item(BaseModel):
    name: str
    price: float
    tags: list[str] = []
    

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello" : "World"}

@app.get("/items/{item_id}")
def read_item(item_id: int):
    return {"item_id": item_id}

@app.post("/items")
def create_item(item: Item):
    return item

@app.get("/generate")
def generate_image(prompt: str):
    image = obtain_image(prompt=prompt, num_inference_steps=5, seed=1024)
    image.save("image.png")

    return FileResponse("image.png")
