import requests
import json
import pandas as pd

data_nueva = pd.read_csv("data/datos_prueba.csv", header=None)
data_nueva[4] = data_nueva[4].map(lambda x: int(x.replace("'", "")))
data_nueva[0] = pd.to_datetime(data_nueva[0].map(lambda x: str(x.replace("b'", ""))))
data_nueva = data_nueva.loc[data_nueva[4]==1, :]
data_nueva = data_nueva.iloc[:30, 1:4].values

data_nueva = data_nueva.tolist()

header = {'Content-Type': 'application/json','Accept': 'application/json'}
data = {"data": data_nueva}
#resp = requests.post(url = "http://localhost:8000/predict", data = json.dumps(data), headers= header)

resp = requests.post(url = "https://proyecto-app-web2.uc.r.appspot.com/predict", data = json.dumps(data), headers= header)

print(resp.json()["prueba"])

