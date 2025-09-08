import requests

url = "http://localhost:8000/score"

sample = {
    "age": 61,
    "hypertension": 0,
    "heart_disease": 0,
    "avg_glucose_level": 202.21,
    "bmi": None,
    "gender": "Female",
    "ever_married": "Yes",
    "work_type": "Self-employed",
    "Residence_type": "Rural",
    "smoking_status": "never smoked"
}

r = requests.post(url, json=sample, timeout=10)
print(r.status_code, r.json())
