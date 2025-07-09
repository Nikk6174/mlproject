# api/index.py
from flask import Flask, request, render_template
from src.pipeline.predict_pipeline import PredictPipeline, CustomData

app = Flask(
    __name__,
    template_folder="../templates",
    static_folder="../static",
    static_url_path="/static"
)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET','POST'])
def predict_datapoint():
    # … same logic …
    if request.method == 'GET':
        return render_template('home.html')
    data = CustomData(
        gender=request.form['gender'],
        race_ethnicity=request.form['ethnicity'],
        parental_level_of_education=request.form['parental_level_of_education'],
        lunch=request.form['lunch'],
        test_preparation_course=request.form['test_preparation_course'],
        reading_score=float(request.form['reading_score']),
        writing_score=float(request.form['writing_score'])
    )
    pred = PredictPipeline().predict(data.get_data_as_data_frame())[0]
    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        return render_template('_result_snippet.html', result=pred)
    return render_template('home.html', results=pred)

# Vercel’s Python runtime will look for a callable named "app"
