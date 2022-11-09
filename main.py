
from fastapi import FastAPI, Body,UploadFile,Header,File,Request
from joblib import load
import pandas as pd
import sys
import os

app = FastAPI(debug=True)

def resource_path(relative_path):
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    print(base_path)
    return os.path.join(base_path, relative_path)


model_Colon = load(resource_path('alone_cancer_RF.joblib'))
model_Lung = load(resource_path('lung_cancer_knn.joblib'))    
model_Liver = load(resource_path('Liver_cancer_RF.joblib'))    
model_Leukemia = load(resource_path('luekemia_cancer_svc.joblib'))    
model_Breast = load(resource_path('Breast_cancer_RF.joblib'))    



@app.post("/")
async def root():
    return {"message": 'cancer Api'}    
    

@app.post("/detectLung/",)
async def upload_file(file:UploadFile):
    
    data = pd.read_csv(file.file, header=None)
    if len(data.columns) == 2000:
        prediction =pd.Series(model_Lung.predict(data))
        print(prediction)
        result = pd.DataFrame(prediction)
        print(result)
        result = result.replace({0: 'Tumoral', 1: 'Normal'})
        result = result[0].iloc[0]
    else:
        result = "Please Upload Usable Data File"
    return {"prediction": result,}    

@app.post("/detectLiver/",)
async def upload_file(file:UploadFile):
    
    data = pd.read_csv(file.file, header=None)
    if len(data.columns) == 10000:
        prediction =pd.Series(model_Liver.predict(data))
        print(prediction)
        result = pd.DataFrame(prediction)
        print(result)
        result = result.replace({0: 'Tumoral', 1: 'Normal'})
        result = result[0].iloc[0]
    else:
        result = "Please Upload Usable Data File"
    return {"prediction": result,}    

@app.post("/detectLeukemia/",)
async def upload_file(file:UploadFile):
    
    data = pd.read_csv(file.file, header=None)
    if len(data.columns) == 15000:
        prediction =pd.Series(model_Leukemia.predict(data))
        print(prediction)
        result = pd.DataFrame(prediction)
        print(result)
        result = result.replace({0: 'Tumoral', 1: 'Normal'})
        result = result[0].iloc[0]
    else:
        result = "Please Upload Usable Data File"
    return {"prediction": result,}    

@app.post("/detectBreast/",)
async def upload_file(file:UploadFile):
    
    data = pd.read_csv(file.file, header=None)
    if len(data.columns) == 5000:
        prediction =pd.Series(model_Breast.predict(data))
        print(prediction)
        result = pd.DataFrame(prediction)
        print(result)
        result = result.replace({0: 'Normal', 1: 'Tumoral'})
        result = result[0].iloc[0]
    else:
        result = "Please Upload Usable Data File"
    return {"prediction": result,}    

@app.post("/detectColon/",)
async def upload_file(file:UploadFile):
    
    data = pd.read_csv(file.file, header=None)
    if len(data.columns) == 200:
        prediction =pd.Series(model_Colon.predict(data))
        print(prediction)
        result = pd.DataFrame(prediction)
        print(result)
        result = result.replace({0: 'Normal', 1: 'Tumoral'})
        result = result[0].iloc[0]
    else:
        result = "Please Upload Usable Data File"
    return {"prediction": result,}    

