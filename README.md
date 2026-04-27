# Stock-Price-Predictor
## 🚀 How to Run the Project Locally

Follow these steps to set up and run the Stock Price Prediction web app on your system.

---

### 📌 1. Clone the Repository

```bash
git clone https://github.com/Bhuvangoli/Stock-Price-Predictor.git
cd Stock-Price-Predictor
```

---

### 📌 2. Create a Virtual Environment

```bash
python -m venv venv
```

---

### 📌 3. Activate the Virtual Environment

#### ▶️ On Windows:

```bash
venv\Scripts\activate
```

#### ▶️ On Mac/Linux:

```bash
source venv/bin/activate
```

---

### 📌 4. Install Required Dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

---

### 📌 5. Run the Streamlit App

```bash
streamlit run app.py
```

---

### 🌐 6. Open in Browser

After running the above command, you will see:

```
Local URL: http://localhost:8501
```

Open this URL in your browser to use the app.

---

## ⚠️ Notes

* Make sure you are using **Python 3.8 – 3.10** for best compatibility.
* If you face issues with TensorFlow, try:

  ```bash
  pip install tensorflow==2.10
  ```
* Do **NOT** commit the `venv/` folder to GitHub.

---

## 🧹 Deactivate Virtual Environment (Optional)

```bash
deactivate
```

---
