from flask import Flask, request, render_template, redirect
import os
import torch
from pathlib import Path

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', force_reload=True)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']
        filepath = os.path.join(UPLOAD_FOLDER, f.filename)
        f.save(filepath)
        results = model(filepath)
        results.save()
        return redirect('/result')
    return render_template('upload.html')

@app.route('/result')
def show_result():
    return '<h2>Detection Completed. Check the runs/detect folder for output.</h2>'

if __name__ == '__main__':
    app.run(debug=True)
