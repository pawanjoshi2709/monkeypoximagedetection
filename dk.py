from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
from pred1 import MonkeypoxPredictor
from flask import send_from_directory
#from pyngrok import ngrok
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'static'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp', 'tiff', 'bmp'}
#ngrok.set_auth_token('2xEndCq0oDen1jYtZoqR8cvqWDK_6z82o8JwM44DxckxeofKk')
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model_path = 'densenet121_best.h5'
predictor = MonkeypoxPredictor(model_path)

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html', filename=None)


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
@app.route('/upload', methods=['POST'])
def upload_file():
    # Clear previous images
    for filename in os.listdir(UPLOAD_FOLDER):
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        os.remove(file_path)

    # Clear previous prediction output
    prediction_path = os.path.join(OUTPUT_FOLDER, 'output_prediction.png')
    if os.path.exists(prediction_path):
        os.remove(prediction_path)

    file = request.files.get('file')

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Perform prediction
        prediction_path, prob = predictor.predict_image(filepath)

        # Return result to template
        return render_template('index.html', filename=filename, prediction_image=prediction_path)

    return redirect(url_for('index'))

if __name__ == '__main__':
 #   public_url = ngrok.connect(5000)
 #   print("Public URL:", public_url.public_url)
    app.run(port=5000, debug=True)
