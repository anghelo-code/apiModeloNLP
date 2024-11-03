from fastapi import FastAPI
from pydantic import BaseModel
from nplProcess import procesar1
from fastapi.middleware.cors import CORSMiddleware
import joblib

# Cargar el modelo guardado
modelo = joblib.load("SVM_Kernel_RBF.pkl")

app = FastAPI()

origins = ["*"]

app.add_middleware(
  CORSMiddleware,
  allow_origins=origins,  # Puedes especificar dominios específicos si es necesario
  allow_credentials=True,
  allow_methods=["*"],  # Permitir todos los métodos (GET, POST, etc.)
  allow_headers=["*"],  # Permitir todos los encabezados
)

# Definir el esquema de entrada usando Pydantic
class InputData(BaseModel):
  texto: str

@app.post("/clasificar/")
async def clasificar_texto(data: InputData):
  # Aquí asumiendo que tu modelo espera un texto y hace una predicción
  texto_recibido = data.texto

  texto_recibido = procesar1(texto_recibido, 3)
  
  # Realizar la predicción usando el modelo
  resultado = modelo.predict(texto_recibido.toarray())  # Asegúrate de que la entrada sea en el formato correcto

  # Devolver el resultado en formato JSON
  return {"resultado": resultado.tolist()}  # Convertir a lista si es necesario
