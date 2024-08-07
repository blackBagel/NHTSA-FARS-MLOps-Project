import requests

passenger = {
    'VE_FORMS': 2,
    'STATE': 1,
    'COUNTY': 55,
    'DAY': 23,
    'MONTH': 1,
    'HOUR': 18,
    'AGE': 54,
    'SEX': 2,
    'SEAT_POS': 4,
    'REST_USE': 3,
    'REST_MIS': 0,
    'HELM_USE': 20,
    'HELM_MIS': 7,
    'AIR_BAG': 20,
    'EJECTION': 0
 }

url = 'http://localhost:9696/predict'
response = requests.post(url, json=passenger)
print(response.json())