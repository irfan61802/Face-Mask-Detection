from flask import Flask, render_template, Response, request, jsonify
from mask_id_detection import VideoCamera

app= Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/navpage')
def navpage():
    return render_template('navpage.html')

@app.route('/faceID')
def FaceID():
    return render_template('ID.html')


@app.route('/maskID')
def maskID():
	return render_template('maskID.html')


global pause, face_name
pause=False
face_name=""

def gen(camera):
  while True:
    frame=camera.get_frame()
    yield (b'--frame\r\n' 
           b'Content-Type: image/jpeg\r\n\r\n'+frame+b'\r\n\r\n')
    
def genBoth(camera):
  while True:
    frame=camera.get_both()
    yield (b'--frame\r\n' 
           b'Content-Type: image/jpeg\r\n\r\n'+frame+b'\r\n\r\n')
    
def genID(camera):
  while True:
    global pause, face_name
    frame=camera.get_frame_ID()
    if pause==True:
        camera.encode=True
        camera.face_name=face_name
        print("[INFO] Face Encoded")
        pause=False
    yield (b'--frame\r\n' 
           b'Content-Type: image/jpeg\r\n\r\n'+frame+b'\r\n\r\n')
    
@app.route('/video_feed')
def video_feed():
  return Response(gen(VideoCamera()), mimetype='multipart/x-mixed-replace; boundary=frame')
    
@app.route('/video_feed2')
def video_feed2():
    return Response(genBoth(VideoCamera()), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed3')
def video_feed3():
    return Response(genID(VideoCamera()), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/upload_face')
def upload_face():
    global pause, face_name
    pause=True
    face_name = request.args.get('name')
    return jsonify(face_name)

if __name__=='__main__':
    app.run(debug=False)
    