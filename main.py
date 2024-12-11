from fastapi import FastAPI
from pydantic import BaseModel
from nplProcess import procesar1
from fastapi.middleware.cors import CORSMiddleware
from keras.models import load_model

# Cargar el modelo guardado
modelo = load_model('modelo_LSTM_fasttext.keras')

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
  # print(texto_recibido)
  resultado = modelo.predict(texto_recibido)[0].tolist()  # Asegúrate de que la entrada sea en el formato correcto
  # resultado = "  ddd"
  print(resultado)

  # Devolver el resultado en formato JSON
  return {
    "resultado_0": resultado[0],
    "resultado_1": resultado[1]
  }  # Convertir a lista si es necesario
