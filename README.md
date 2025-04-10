## 📷 Fake Instagram Account Predictor

This project is a full-stack web application designed to detect fake Instagram accounts using machine learning. It features an interactive frontend built with **Node.js** and **EJS**, and a backend model developed in **Python (Flask)** for inference.

---

## 💡 Features

- 🌐 User-friendly web interface
- 🔍 Real-time fake account prediction
- 📊 Data insights from training dataset
- 🧠 Trained Random Forest model
- 📂 Separate frontend and backend structure

---

## 🚀 How to Run the Project

### ✅ Prerequisites

- Node.js & npm
- Python (preferably 3.8+)
- pip

---

### 1️⃣ Setup Backend (ML Model)

1. Open terminal and navigate to the `backend` directory:

```bash
cd backend
```

2. Install required Python packages:

```bash
pip install flask pandas scikit-learn joblib
```

3. Run the Flask API:

```bash
python app.py
```

This will start the backend server at `http://127.0.0.1:5000`.

---

### 2️⃣ Setup Frontend (Node.js App)

1. Open a new terminal and navigate to the `frontend` directory:

```bash
cd frontend
```

2. Install the dependencies:

```bash
npm install
```

3. Start the Node.js server:

```bash
node server.js
```

The frontend will be live at: `http://localhost:3000`

---

## 🔁 Prediction Workflow

1. The user fills out a form describing an Instagram account.
2. The frontend sends a POST request to the Flask backend.
3. The backend uses the Random Forest model to predict whether the account is fake.
4. The result is shown to the user in real-time.

---

## 📊 Model Insights

You can navigate to the **Model Insights** page to view training performance, correlation matrices, and the neural network structure used (if any).

---

Let me know if you'd like help creating the `README.md` with this or adding badges, live preview GIFs, or deployment steps!
