from flask import Flask, render_template, Response, request
from flask import jsonify
import cv2
from deepface import DeepFace


app = Flask(__name__, template_folder='C:/Users/welcome/PycharmProjects/pythonProject2/project_directory/templates', static_folder='C:/Users/welcome/PycharmProjects/pythonProject2/project_directory/static')

# Load the pre-trained deep learning model for emotion recognition
model = DeepFace.build_model("Emotion")

# Open the webcam
cap = cv2.VideoCapture(0)

emotion_analysis_active = False

def generate_frames():
    while True:
        success, frame = cap.read()  # read the camera frame
        if not success:
            break
        elif emotion_analysis_active:
            # Detect faces in the frame
            faces = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detected_faces = faces.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            # Analyze each detected face for emotion
            for (x, y, w, h) in detected_faces:
                try:
                    face_img = frame[y:y + h, x:x + w]
                    result = DeepFace.analyze(face_img, actions=['emotion'], enforce_detection=False)

                    # Get the dominant emotion
                    emotion = result[0]['dominant_emotion']

                    # Draw rectangle and emotion text on the frame
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    cv2.putText(frame, f'Emotion: {emotion}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                except Exception as e:
                    print(f"Error analyzing face: {e}")

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result
        else:
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + b'\r\n')  # empty frame

@app.route('/')
def index():
    return render_template('index.html')  # render the HTML page

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_emotion_detection', methods=['POST'])
def start_emotion_detection():
    global emotion_analysis_active
    emotion_analysis_active = True
    return '', 204

@app.route('/get_emotion')
def get_emotion():
    global cap

    # Capture a single frame from the webcam
    ret, frame = cap.read()

    # Detect faces in the frame
    faces = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detected_faces = faces.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Analyze the first detected face for emotion
    if len(detected_faces) > 0:
        x, y, w, h = detected_faces[0]
        try:
            face_img = frame[y:y + h, x:x + w]
            result = DeepFace.analyze(face_img, actions=['emotion'], enforce_detection=False)

            # Get the dominant emotion
            detected_emotion = result[0]['dominant_emotion']
        except Exception as e:
            print(f"Error analyzing face: {e}")
            detected_emotion = "Unknown"
    else:
        detected_emotion = "No face detected"

    return jsonify({'emotion': detected_emotion})


if __name__ == "__main__":
    app.run(debug=True)
