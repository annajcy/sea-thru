import os
from flask import Flask, request, redirect, url_for, render_template, jsonify
from werkzeug.utils import secure_filename
from detect import run

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Ensure the upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

model_paths = {
    'yolov5s' : 'C:\\Users\\annaj\\PycharmProjects\\yolov5-master\\weight\\yolov5s.pt',
    'yolov5s_mod' : 'C:\\Users\\annaj\\PycharmProjects\\yolov5-master\\weight\\yolov5s_mod.pt',
    'yolov5s_biFPN_add' : 'C:\\Users\\annaj\\PycharmProjects\\yolov5-master\\weight\\yolov5s_biFPN_add.pt',
    'yolov5s_biFPN_concat' : 'C:\\Users\\annaj\\PycharmProjects\\yolov5-master\\weight\\yolov5s_biFPN_concat.pt'
}

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'success': False})
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False})
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            return jsonify({'success': True, 'filename': filename})
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    filename = request.form.get('filename')
    option = request.form.get('model')
    if not filename:
        return redirect(url_for('upload_file'))
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    print(file_path, model_paths[option])
    if option == 'yolov5s_mod':
        cmd = f'python seathru-mono-e2e.py --monodepth-add-depth 1.0 --monodepth-multiply-depth 3.0 --image {file_path} --output {file_path}'
        os.system(cmd)
    run(exist_ok=True, imgsz=(640, 360), conf_thres=0.25, source=file_path, destination=file_path, weights=model_paths[option])
    return render_template('index.html', filename=filename)

if __name__ == '__main__':
    app.run(debug=True)


