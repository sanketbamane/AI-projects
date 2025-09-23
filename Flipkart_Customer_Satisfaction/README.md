📦 Flipkart Customer Satisfaction Prediction








📌 Overview

This project predicts customer satisfaction on Flipkart using product reviews, ratings, and related features.

The pipeline includes:
✅ Exploratory Data Analysis (EDA)
✅ Feature Engineering & Preprocessing
✅ Model Training & Evaluation
✅ Explainability (Feature Importances)
✅ Deployment-Ready Scripts

This project can help Flipkart (or any e-commerce platform) understand what drives customer happiness, improve product quality, and boost customer retention.

📂 Project Structure
Flipkart_Customer_Satisfaction/
│── data/                     # Raw dataset(s)
│   └── flipkart.csv
│
│── eda/                      # EDA outputs
│   ├── describe.csv
│   └── head50.csv
│
│── models/                   # Trained models & preprocessors
│   ├── rf_model.joblib
│   └── preprocessor.joblib
│
│── model_building/           # Model training artifacts
│   ├── evaluation.txt
│   └── metrics.json
│
│── model_explainability/     # Feature importance & plots
│   ├── feature_importances.csv
│   └── feature_importances.png
│
│── notebooks/                # Jupyter Notebooks
│   └── Flipkart_Customer_Satisfaction.ipynb
│
│── reports/                  # Reports & outputs
│   ├── evaluation.txt
│   └── predictions_sample.csv
│
│── src/                      # Source code
│   ├── main.py               # Run inference
│   └── train.py              # Train model
│
│── tests/                    # Unit tests
│   └── test_model_exists.py
│
│── requirements.txt          # Dependencies
└── README.md                 # Documentation

🚀 Getting Started
🔹 1. Clone Repository
git clone https://github.com/yourusername/Flipkart_Customer_Satisfaction.git
cd Flipkart_Customer_Satisfaction

🔹 2. Create Virtual Environment
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

🔹 3. Install Dependencies
pip install -r requirements.txt

🧑‍💻 Usage
🔹 Train the Model
python src/train.py --data-path data/flipkart.csv
🔹 For complete dataset training
python src/train.py --data-path data/flipkart.csv --project-root . --sample 0 --max-features 3000 --n-estimators 100 --max-depth 20

🔹 Run Predictions
python src/main.py

🔹 Explore in Notebook
jupyter notebook notebooks/Flipkart_Customer_Satisfaction.ipynb

📊 Results & Explainability

✅ Evaluation metrics → model_building/metrics.json & reports/evaluation.txt

✅ Feature Importances:

CSV → model_explainability/feature_importances.csv

Plot → model_explainability/feature_importances.png

Example (Feature Importances Plot):


✅ Testing

Run unit tests:

pytest tests/

🔮 Future Work

🔹 Hyperparameter tuning with Optuna/GridSearchCV

🔹 Deep Learning models (LSTM/BERT for reviews)

🔹 Deploy via Flask / FastAPI / Streamlit

🔹 Interactive real-time dashboard

📜 License

This project is released under the MIT License
.

⚡ Made with ❤️ using Python & Machine Learning
