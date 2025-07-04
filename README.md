# Student Score Prediction System

## Overview

The **Student Score Prediction System** is an end-to-end machine learning project designed to predict students' mathematics scores using demographic and academic data. Leveraging seven regression algorithms, this system delivers accurate predictions through a user-friendly Flask web interface. The final model (CatBoost) achieves an RMSE of **4.576**, making it suitable for educational insights such as early intervention and personalized learning.

## Features

* **Score Prediction**: Predicts math scores based on seven input features:

  * Gender
  * Ethnicity
  * Parental level of education
  * Lunch type
  * Test preparation course status
  * Reading score
  * Writing score
* **Data Pipeline**: Processes 800 samples (80% training, 20% testing) with preprocessing for both numerical and categorical features.
* **Modeling**: Trains and evaluates 7 regression models using 3-fold GridSearchCV, selecting the best based on R² score (minimum threshold of 0.6).
* **Web Interface**: Provides a Flask-based UI with two HTML pages (`index.html` and `home.html`) and validated input fields for real-time predictions.
* **Performance Evaluation**: Reports training times, inference latency, and cross-validation metrics for model comparison.

## Tech Stack

* **Programming Language**: Python 3.7
* **Web Framework**: Flask
* **Machine Learning Libraries**: scikit-learn, CatBoost, XGBoost
* **Data Manipulation**: Pandas, NumPy
* **Visualization**: Seaborn, Matplotlib
* **Version Control**: Git



## Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/your-repo-link/student-score-prediction.git
   cd student-score-prediction
   ```

2. **Set Up Virtual Environment**

   ```bash
   python3 -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare Dataset**

   * Place `stud.csv` in the `artifacts/` directory.

## Usage

### Run the Flask Application

```bash
python app.py
```

* Open your browser and navigate to `http://localhost:5000`.
* Fill in all seven input fields and submit to view the predicted math score.

### Train Models Manually

```bash
python src/model_trainer.py
```

* Trains all seven regression models.
* Saves the best-performing model (CatBoost) as a pickle file in `artifacts/`.

## Performance

* **Training Time**: \~0.077 seconds for CatBoost (100 iterations).
* **Inference Latency**: Under 1 second for real-time web predictions.
* **Model Evaluation**: 3-fold cross-validation with R² ≥ 0.6; CatBoost achieves an RMSE of 4.576.

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository.
2. Create a new branch: `git checkout -b feature/YourFeature`.
3. Commit your changes: `git commit -m "Add YourFeature description"`.
4. Push to the branch: `git push origin feature/YourFeature`.
5. Open a Pull Request.

Please ensure all new code is covered by tests and follows the existing style guidelines.

## License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.
