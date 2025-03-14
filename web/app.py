from flask import Flask, render_template, request, jsonify
import subprocess
import threading
import os
import signal
import time
from collections import deque
import logging
from pathlib import Path
import sys
VENV_PYTHON = sys.executable

app = Flask(__name__)
app.logger.setLevel(logging.INFO)

# 全局状态管理
class RealtimeMonitor:
    def __init__(self):
        self.process = None
        self.output_buffer = deque(maxlen=1000)
        self.lock = threading.Lock()
        self.running = False

realtime_monitor = RealtimeMonitor()

# 路径配置
MODULES_DIR = Path(__file__).parent.parent / "modules"
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/realtime', methods=['POST'])
def start_realtime():
    """启动实时监测"""
    params = request.json
    model_type = params.get('model_type', 'DNN')
    
    # 停止现有进程
    if realtime_monitor.process:
        stop_realtime_process()

    # 构建命令
    cmd = [
        str(VENV_PYTHON),
        str(MODULES_DIR / "realtime_monitor.py"),
        "--model_type", model_type
    ]

    try:
        # 启动子进程
        realtime_monitor.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
        )
        realtime_monitor.running = True
        
        # 启动输出读取线程
        threading.Thread(target=read_output, daemon=True).start()
        
        return jsonify({
            "status": "started",
            "pid": realtime_monitor.process.pid
        })
    except Exception as e:
        app.logger.error(f"启动失败: {str(e)}")
        return jsonify({"error": str(e)}), 500

def read_output():
    """实时读取进程输出"""
    while realtime_monitor.running and realtime_monitor.process:
        try:
            line = realtime_monitor.process.stdout.readline()
            if line:
                with realtime_monitor.lock:
                    realtime_monitor.output_buffer.append(line.strip())
            else:  # 进程结束
                break
        except (ValueError, AttributeError):
            break
        except Exception as e:
            app.logger.error(f"输出读取错误: {str(e)}")
            break
    realtime_monitor.running = False

@app.route('/realtime/output')
def get_realtime_output():
    """获取实时输出"""
    with realtime_monitor.lock:
        return jsonify({
            "output": "\n".join(realtime_monitor.output_buffer),
            "active": realtime_monitor.running
        })

@app.route('/realtime', methods=['DELETE'])
def stop_realtime():
    """停止实时监测"""
    try:
        stop_realtime_process()
        return jsonify({"status": "stopped"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def stop_realtime_process():
    """安全的进程停止方法"""
    if realtime_monitor.process:
        # Windows专用终止方法
        if os.name == 'nt':
            subprocess.run(
                ['taskkill', '/F', '/T', '/PID', str(realtime_monitor.process.pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        else:
            realtime_monitor.process.terminate()
        
        realtime_monitor.process.wait(timeout=5)
        realtime_monitor.process = None
        realtime_monitor.running = False

@app.route('/historical', methods=['POST'])
def run_historical():
    """执行历史数据分析"""
    try:
        # 验证文件上传
        if 'file' not in request.files:
            return jsonify({"error": "未上传文件"}), 400
            
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "空文件名"}), 400
        
        # 验证文件类型
        if not file.filename.lower().endswith('.csv'):
            return jsonify({"error": "仅支持CSV文件"}), 400
            
        # 保存文件
        upload_path = UPLOAD_DIR / file.filename
        file.save(upload_path)
        
        # 验证参数
        model_type = request.form.get('model_type', 'DNN')
        mode = request.form.get('mode', 'predict')
        
        # 执行分析
        cmd = [
            str(VENV_PYTHON),
            str(MODULES_DIR / "historical_predictor.py"),
            "--model_type", model_type,
            "--data_path", str(upload_path),
            "--mode", mode
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5分钟超时
        )
        
        # 清理临时文件
        upload_path.unlink(missing_ok=True)
        
        return jsonify({
            "output": result.stdout,
            "error": result.stderr,
            "returncode": result.returncode
        })
        
    except subprocess.TimeoutExpired:
        return jsonify({"error": "分析超时"}), 500
    except Exception as e:
        app.logger.error(f"历史分析失败: {str(e)}")
        return jsonify({"error": str(e)}), 500

# 训练任务追踪
training_jobs = {}
training_lock = threading.Lock()

@app.route('/train', methods=['POST'])
def train_model():
    """启动模型训练"""
    params = request.json
    model_type = params.get('model_type')
    
    if model_type not in ["DNN", "LSTM"]:
        return jsonify({"error": "无效模型类型"}), 400
    
    # 生成任务ID
    job_id = f"{model_type}_{int(time.time())}"
    
    # 构建命令
    script_map = {
        "DNN": "testDNN_IDS.py",
        "LSTM": "testLSTM_IDS.py"
    }
    
    cmd = [
        str(VENV_PYTHON),
        str(MODULES_DIR / script_map[model_type]),
        "--epochs", str(params.get('epochs', 10)),
        "--batch_size", str(params.get('batch_size', 32))
    ]
    
    # 启动异步训练
    def train_task():
        log_path = UPLOAD_DIR / f"{job_id}.log"
        try:
            with open(log_path, 'w') as log_file:
                process = subprocess.Popen(
                    cmd,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    text=True
                )
                
                with training_lock:
                    training_jobs[job_id] = {
                        "status": "running",
                        "start_time": time.time(),
                        "log_path": log_path
                    }
                    
                process.wait()
                
                with training_lock:
                    training_jobs[job_id]["status"] = "completed" 
                    training_jobs[job_id]["end_time"] = time.time()
                    
        except Exception as e:
            with training_lock:
                training_jobs[job_id]["status"] = "failed"
                training_jobs[job_id]["error"] = str(e)
    
    threading.Thread(target=train_task).start()
    
    return jsonify({
        "status": "started",
        "job_id": job_id,
        "monitor_url": f"/train/status/{job_id}"
    })

@app.route('/train/status/<job_id>')
def get_train_status(job_id):
    """获取训练状态"""
    with training_lock:
        job = training_jobs.get(job_id)
        
    if not job:
        return jsonify({"error": "任务不存在"}), 404
        
    response = {
        "status": job["status"],
        "duration": time.time() - job["start_time"]
    }
    
    if job["status"] == "completed":
        try:
            with open(job["log_path"]) as f:
                response["log"] = f.read(4096)  # 返回最后4KB日志
        except Exception:
            response["log"] = ""
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)