from flask import Response
from flask import Flask, request, redirect, url_for, jsonify
from flask import render_template
import threading
import argparse
import datetime
import time
import cv2
import torch
from tracker import *
import numpy as np
from pymongo import MongoClient


model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)


# initialize the output frame and a lock used to ensure thread-safe
# exchanges of the output frames (useful when multiple browsers/tabs
# are viewing the stream)
outputFrame = None
lock = threading.Lock()

# initialize a flask object
app = Flask(__name__)


client = MongoClient('mongodb://localhost:27017/')
db = client['people_counter']  # Tên database
collection = db['users']

# initialize the video stream and allow the camera sensor to
# warmup
#vs = VideoStream(usePiCamera=1).start()
username = ''
password = ''
ipcamera = ''
path = ''

# Tạo URL RTSP
rtsp_url = f'rtsp://{username}:{password}@{ipcamera}/{path}' if ipcamera else 0
vs = cv2.VideoCapture(rtsp_url)

time.sleep(2.0)


tracker = Tracker()

area_1 = [(2, 2), (2, 497), (1017, 497), (1017, 2)]
area1 = set()

first_time_visit = True

@app.route("/register")
def index():
	# return the rendered template
	return render_template("register.html")

@app.route("/login")
def index2():
	# return the rendered template
	return render_template("login.html")

@app.route("/")
def index1():
	global first_time_visit
	if first_time_visit:
		first_time_visit = False
		# Chuyển hướng đến trang login.html nếu là lần đầu tiên truy cập
		return redirect(url_for("index2"))
	# return the rendered template
	return render_template("index.html")

def detect_motion(frameCount):
	# grab global references to the video stream, output frame, and
	# lock variables
	global vs, outputFrame, lock

	# initialize the motion detector and the total number of frames
	# read thus far
	#md = SingleMotionDetector(accumWeight=0.1)

		# loop over frames from the video stream
	while True:
		# read the next frame from the video stream, resize it,
		# convert the frame to grayscale, and blur it
		ret, frame = vs.read()
		try:
			frame = cv2.resize(frame, (1020, 500))
			gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			gray = cv2.GaussianBlur(gray, (7, 7), 0)

			# grab the current timestamp and draw it on the frame
			timestamp = datetime.datetime.now()
			cv2.putText(frame, timestamp.strftime(
				"%A %d %B %Y %I:%M:%S%p"), (10, frame.shape[0] - 10),
						cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

			frame = cv2.resize(frame, (1020, 500))

			cv2.polylines(frame, [np.array(area_1, np.int32)], True, (0, 255, 0), 3)

			results = model(frame)
			#    frame = np.squeeze(results.render())
			list = []
			for index, row in results.pandas().xyxy[0].iterrows():
				x1 = int(row['xmin'])
				y1 = int(row['ymin'])
				x2 = int(row['xmax'])
				y2 = int(row['ymax'])
				b = str(row['name'])
				# cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
				# cv2.putText(frame, b, (x1, y1), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
				if 'person' in b:
					list.append([x1, y1, x2, y2])

			area1.clear()
			boxes_ids = tracker.update(list)
			for boxes_id in boxes_ids:
				x, y, w, h, id = boxes_id
				cv2.rectangle(frame, (x, y), (w, h), (255, 0, 255), 2)
				# cv2.putText(frame, str(id), (x, y), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 2)
				results = cv2.pointPolygonTest(np.array(area_1, np.int32), (int(w), int(h)), False)
				if results > 0:
					area1.add(id)

			p = len(area1)
			cv2.putText(frame, str(p), (500, 30), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 2)

			cv2.imshow('FRAME', frame)
			if cv2.waitKey(1) & 0xFF == 27:
				break

			with lock:
				outputFrame = frame.copy()

		except cv2.error as e:
			# Update global variables
			global username, password, ipcamera, rtsp_url
			username = ''
			password = ''
			ipcamera = ''
			rtsp_url = f'rtsp://{username}:{password}@{ipcamera}/{path}' if ipcamera else 0

			vs.release()
			# Create a new VideoCapture
			vs = cv2.VideoCapture(rtsp_url)

			print(f"Error resizing frame: {e}")
			continue


def generate():
	# grab global references to the output frame and lock variables
	global outputFrame, lock

	# loop over frames from the output stream
	while True:
		# wait until the lock is acquired
		with lock:
			# check if the output frame is available, otherwise skip
			# the iteration of the loop
			if outputFrame is None:
				continue

			# encode the frame in JPEG format
			(flag, encodedImage) = cv2.imencode(".jpg", outputFrame)

			# ensure the frame was successfully encoded
			if not flag:
				continue

		# yield the output frame in the byte format
		yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
			bytearray(encodedImage) + b'\r\n')

@app.route("/video_feed")
def video_feed():
	# return the response generated along with the specific media
	# type (mime type)
	return Response(generate(),
		mimetype = "multipart/x-mixed-replace; boundary=frame")


@app.route('/connect_camera', methods=['POST'])
def connect_camera():
	# Release the current VideoCapture if it exists

	data = request.get_json()
	received_username = data.get('username', '')
	received_password = data.get('password', '')
	received_ipcamera = data.get('ip', '')
	received_path = data.get('path', '')

	# Update global variables
	global username, password, ipcamera, rtsp_url, vs, path
	username = received_username or ''
	password = received_password or ''
	ipcamera = received_ipcamera or ''

	if received_path == 'tapo':
		path = 'stream1'
	elif received_path == 'hikvision':
		path = 'Streaming/Channels/1'
	elif received_path == 'dahua':
		path = 'cam/realmonitor?1&subtype=0'
	elif received_path == 'axis':
		path = 'axis-media/media.amp?videocodec=h264&streamprofile=1'
	elif received_path == 'foscam':
		path = 'videoMain'
	elif received_path == 'avigilon':
		path = 'cam/realmonitor?channel=1&subtype=0'
	elif received_path == 'sony':
		path = 'cam/realmonitor?channel=1&subtype=0'
	elif received_path == 'vivotek':
		path = 'videoMain'
	elif received_path == 'uniFi':
		path = 'cam/realmonitor?channel=1&subtype=0'
	elif received_path == 'lorex':
		path = 'cam/realmonitor?channel=1&subtype=0'
	else:
		path = 'stream1'

	rtsp_url = f'rtsp://{username}:{password}@{ipcamera}/{path}' if ipcamera else 0

	vs.release()
	# Create a new VideoCapture
	vs = cv2.VideoCapture(rtsp_url)


	# Return a streaming response
	return redirect(url_for('video_feed'))

@app.route('/register', methods=['POST'])
def register():
	data = request.get_json()

	username = data.get('username', '')
	password = data.get('password', '')
	confirm_password = data.get('confirmPassword', '')


	# Kiểm tra xác nhận mật khẩu
	if password != confirm_password:
		return render_template('register.html', message='Password and Confirm Password do not match.')

	# Kiểm tra xem tên người dùng đã tồn tại chưa
	if db.users.find_one({'username': username}):
		return render_template('register.html', message='Username already exists. Choose a different one.')

	# Thêm người dùng vào collection 'users'
	user_data = {'username': username, 'password': password}
	collection.insert_one(user_data)

	return jsonify({"success": True, "message": "OK"})

@app.route('/login', methods=['POST'])
def login():
	data = request.get_json()
	username = data.get('username', '')
	password = data.get('password', '')

	user = db.users.find_one({'username': username})

	if user and user.get('password', '') == password:
		return jsonify({"success": True, "message": "OK"})
	else:
		# Đăng nhập thất bại
		return render_template('login.html', message='Invalid username or password')


@app.route('/logout', methods=['POST'])
def logout():
	global username, password, ipcamera, rtsp_url, vs
	username = ''
	password = ''
	ipcamera = ''
	rtsp_url = f'rtsp://{username}:{password}@{ipcamera}/{path}' if ipcamera else 0

	vs.release()
	# Create a new VideoCapture
	vs = cv2.VideoCapture(rtsp_url)

	return jsonify({"success": True, "message": "OK"})


# check to see if this is the main thread of execution
if __name__ == '__main__':
	# construct the argument parser and parse command line arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--ip", type=str, required=True,
		help="ip address of the device")
	ap.add_argument("-o", "--port", type=int, required=True,
		help="ephemeral port number of the server (1024 to 65535)")
	ap.add_argument("-f", "--frame-count", type=int, default=32,
		help="# of frames used to construct the background model")
	args = vars(ap.parse_args())

	# start a thread that will perform motion detection
	t = threading.Thread(target=detect_motion, args=(
		args["frame_count"],))
	t.daemon = True
	t.start()

	# start the flask app
	app.run(host=args["ip"], port=args["port"], debug=True,
		threaded=True, use_reloader=False)

# release the video stream pointer
vs.stop()


#  python main.py --ip 0.0.0.0 --port 8000
#  rtsp://dinhhuutai:31072002@192.168.1.7:554/stream1