# app.py
from flask import Flask, render_template, request
import subprocess

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('dashboard.html')

@app.route('/start_realtime', methods=['POST'])
def start_realtime():
    model_type = request.form.get('model')
    subprocess.Popen(["python", "modules/realtime.py", model_type])
    return "实时监测已启动"

@app.route('/run_historical', methods=['POST'])
def run_historical():
    file = request.files['pcap_file']
    file.save('upload.pcap')
    subprocess.run(["python", "modules/historical.py", "upload.pcap"])
    return "历史分析完成"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)