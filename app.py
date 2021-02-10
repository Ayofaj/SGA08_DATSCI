# 1. Library imports
import uvicorn
import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware

class DataType(BaseModel):
    type: str
    area: str
    bed: int
    bath: int
    toilet: int


#Create API and load model
app = FastAPI()
model = joblib.load("myModel.sav")

# Prediction
@app.get('/')
def index():
    return {"API" : "Ready to call"}

@app.post("/predict/")
def prediction(data: DataType):
    data = data.dict()
    type = data['type']
    area = data['area']
    bed = data['bed']
    bath = data['bath']
    toilet = data['toilet']
    result = model.predict([type, area, bed, bath, toilet])
    #result = np.exp(result)
    
    return {'Expected rent is': result}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    #allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 4. Run the API with uvicorn
#    API will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)