
from fastapi import FastAPI
from datetime import datetime
app = FastAPI()

@app.get("/hi")
def say_hi():
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
    texto = f"Hola Jumping Girls!! son las {timestamp}"
    return {"message": texto}
