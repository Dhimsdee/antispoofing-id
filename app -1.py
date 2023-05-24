# Flask app
from flask import Flask, request, jsonify, Response, render_template
from flask_cors import CORS
from livenessnet.pasif import livenessnet, is_real_fake
import aktif
import cv2

app = Flask(__name__)
CORS(app)

def generate_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/livenessnet', methods=['POST'])
def liveness_detection():
    data = request.json
    model_Path = data['model_Path']
    le_path = data['le_path']
    detector_folder = data['detector_folder']
    confidence = data['confidence'] 
    #label = livenessnet(model_Path, le_path, detector_folder, confidence)
    sequence_count_real, sequence_count_fake = livenessnet(model_Path, le_path, detector_folder, confidence)
    result = is_real_fake(sequence_count_real, sequence_count_fake)
    response = {
        'label': result
    }
    return jsonify(response)

@app.route("/activenessnet", methods=['POST'])
def activenessnet():
    cam = aktif.capture_frame()
    liveness = aktif.activenessnet(cam)
    label = liveness['label']
    cam.release()
    return jsonify({'result': label})

if __name__ == '__main__':
    app.run(debug=True)