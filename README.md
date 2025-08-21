# 🩺 Sepsis Prediction  

A simple **Flask + Machine Learning** web app to predict the risk of **Sepsis (blood infection)** from patient data.  

---

## 📌 About  
Sepsis is a serious medical condition caused by the body’s response to infection. This project uses a trained ML model (`.h5`) wrapped in a **Flask web application** with a doctor-friendly interface for early detection.  

---

## 🚀 Features  
- Web app built with **Flask**  
- Input form for patient details  
- Predicts **Sepsis risk** instantly  
- Easy to deploy on **Render / Hugging Face / Heroku**  

---

## 📂 Project Structure  
├── app.py / main.py # Flask app (entry point)
├── sepsis_model.h5 # Trained ML model
├── requirements.txt # Dependencies
├── Procfile # For deployment
├── templates/ # HTML files (frontend)
│ ├── index.html
│ └── result.html
├── static/ # (Optional) CSS/JS

Install requirements:
pip install -r requirements.txt


Run Flask app
python app.py


Deployment:
web: gunicorn app:app


---

## ⚙️ How to Run Locally  
1. Clone this repository  
   ```bash
   git clone https://github.com/your-username/sepsis-prediction.git
   cd sepsis-prediction
