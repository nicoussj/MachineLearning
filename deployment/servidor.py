from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import joblib
import numpy as np

app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Cargar el modelo entrenado
model = joblib.load("modelo.pkl")

# Página principal con formulario
@app.get("/", response_class=HTMLResponse)
async def get_form(request: Request):
    return templates.TemplateResponse("formulario.html", {"request": request})

# Recibir datos y predecir
@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request,
                  TimeAlive: float = Form(...),         # Ahora en segundos
                  TravelledDistance: float = Form(...), # Ahora en kilómetros
                  RoundKills: float = Form(...),
                  MatchKills: float = Form(...),
                  RoundAssists: float = Form(...),
                  MatchAssists: float = Form(...),
                  RoundHeadshots: float = Form(...)):

    # Convertir segundos a milisegundos y km a metros
    TimeAlive *= 1000
    TravelledDistance *= 1000

    features = np.array([[TimeAlive, TravelledDistance, RoundKills, MatchKills,
                          RoundAssists, MatchAssists, RoundHeadshots]])

    resultado = model.predict(features)[0]
    probabilidad = model.predict_proba(features)[0][1]

    return templates.TemplateResponse("resultado.html", {
        "request": request,
        "resultado": "GANADOR" if resultado == 1 else "PERDEDOR",
        "prob": f"{probabilidad:.2%}"
    })