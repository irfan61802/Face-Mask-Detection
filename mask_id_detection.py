import numpy as np
import imutils
import cv2
import os
import face_recognition
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

global face_names
face_names=[]

# connect to database
uri = "mongodb+srv://irfan61802:<PASSWORD HERE>@face-db.92meset.mongodb.net/?retryWrites=true&w=majority"
client = MongoClient(uri, server_api=ServerApi('1'))
db = client['face']
faces=db.face
print("[INFO] Connecting to database...")

global all_docs,known_face_encodings,known_face_names
# get all recorded face encodings from database
all_docs = list(faces.find({}))
known_face_names, known_face_encodings = [doc["name"] for doc in all_docs], [doc["embedding"] for doc in all_docs]
known_face_encodings = np.array(known_face_encodings)


# load serialized face detector model from disk
print("[INFO] loading face detector model...")
prototxtPath = os.path.sep.join(["face_detector", "deploy.prototxt"])
weightsPath = os.path.sep.join(["face_detector", "res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model from disk
print("[INFO] loading face mask detector model...")
maskNet = load_model("mask_detector.model")

class VideoCamera(object):
	def __init__(self):
		self.vs = cv2.VideoCapture(0)
		self.encode=False
		self.face_name=""

	def __del__(self):
		self.vs.release()
		cv2.destroyAllWindows()

	# add face encoding and name to database
	def encode_face(self, face_image,face_name):
		if len(face_recognition.face_encodings(face_image)) > 0:
			face_encoding = face_recognition.face_encodings(face_image)[0]
			faces.insert_one({"name": face_name, "embedding": face_encoding.tolist()})
		#reload afer encoding
		global all_docs,known_face_encodings,known_face_names
		all_docs = list(faces.find({}))
		known_face_names, known_face_encodings = [doc["name"] for doc in all_docs], [doc["embedding"] for doc in all_docs]
		known_face_encodings = np.array(known_face_encodings)


	def detect_and_predict_mask(self, frame, faceNet, maskNet):
		# grab the dimensions of the frame and then construct a blob from it
		(h, w) = frame.shape[:2]
		blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))

		# pass the blob through the network and obtain the face detections
		faceNet.setInput(blob)
		detections = faceNet.forward()

		# initialize our list of faces, their corresponding locations, and the list of predictions from our face mask network
		faces = []
		locs = []
		preds = []

		# loop over the detections
		for i in range(0, detections.shape[2]):
			# extract the confidence (i.e., probability) associated with the detection
			confidence = detections[0, 0, i, 2]

			# filter out weak detections by ensuring the confidence is greater than the minimum confidence
			if confidence > 0.5:
				# compute the (x, y)-coordinates of the bounding box for the object
				box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
				(startX, startY, endX, endY) = box.astype("int")

				# ensure the bounding boxes fall within the dimensions of the frame
				(startX, startY) = (max(0, startX), max(0, startY))
				(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

				# extract the face ROI, convert it from BGR to RGB channel ordering, resize it to 224x224, and preprocess it
				face = frame[startY:endY, startX:endX]
				if face.any():
					face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
					face = cv2.resize(face, (224, 224))
					face = img_to_array(face)
					face = preprocess_input(face)
					faces.append(face)
					locs.append((startX, startY, endX, endY))

		# only make a predictions if at least one face was detected
		if len(faces) > 0:
			# for faster inference we'll make batch predictions on all faces at the same time rather than one-by-one 
			faces = np.array(faces, dtype="float32")
			preds = maskNet.predict(faces, batch_size=32)

		# return a 2-tuple of the face locations and their corresponding locations
		return (locs, preds)

	def get_frame(self):
		global face_names
		# grab the frame from the threaded video stream and resize it to 1000px
		success,frame = self.vs.read()
		if not success:
			return
		else:
			frame = imutils.resize(frame, width=1000)
			
			# detect faces in the frame and determine if they are wearing a face mask or not
			(locs, preds) = self.detect_and_predict_mask(frame, faceNet, maskNet)

			# loop over the detected face locations and their corresponding locations
			for (box, pred) in zip(locs, preds):
				# unpack the bounding box and predictions
				(startX, startY, endX, endY) = box
				(mask, withoutMask) = pred

				# determine the class label and color we'll use to draw the bounding box and text
				label = "Mask" if mask > withoutMask else "No Mask"
				color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
					
				# include the probability in the label
				label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

				# display the label and bounding box rectangle on the output frame
				cv2.putText(frame, label, (startX, startY - 10),
					cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
				cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

			ret, buffer = cv2.imencode('.jpg', frame)
			frame = buffer.tobytes()
			
		return frame
	
	def get_frame_ID(self):

		success,frame=self.vs.read()
		if not success:
			return
		else:
			if self.encode:
				self.encode_face(frame,self.face_name)
				self.encode=False

			# Resize frame of video to 1/4 size for faster face recognition processing
			small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
			# Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
			rgb_small_frame = cv2.cvtColor(small_frame,cv2.COLOR_BGR2RGB)
			
			
			# Find all the faces and face encodings in the current frame of video
			face_locations = face_recognition.face_locations(rgb_small_frame) #batch size for optimiztion
			face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
			face_names = []
			for face_encoding in face_encodings:
				# See if the face is a match for the known face(s)
				matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
				name = "Unknown"
				# Or instead, use the known face with the smallest distance to the new face
				face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
				best_match_index = np.argmin(face_distances)
				if matches[best_match_index]:
					name = known_face_names[best_match_index]
				face_names.append(name)
				

			# Display the results
			for (top, right, bottom, left), name in zip(face_locations, face_names):
				# Scale back up face locations since the frame we detected in was scaled to 1/4 size
				top *= 4
				right *= 4
				bottom *= 4
				left *= 4

				# Draw a box around the face
				cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

				# Draw a label with a name below the face
				cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
				font = cv2.FONT_HERSHEY_DUPLEX
				cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)
			
			ret,buffer=cv2.imencode('.jpg',frame)
			return buffer.tobytes()
		
	def get_both(self):
		global face_names
		# grab the frame from the threaded video stream and resize it to 1000px
		success,frame = self.vs.read()
		if not success:
			return
		else:
	
			# Resize frame of video to 1/4 size for faster face recognition processing
			small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
			# Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
			rgb_small_frame = cv2.cvtColor(small_frame,cv2.COLOR_BGR2RGB)

			frame = imutils.resize(frame, width=1000)
			
			# detect faces in the frame and determine if they are wearing a face mask or not
			(locs, preds) = self.detect_and_predict_mask(frame, faceNet, maskNet)

			face_names=[]
			# Find all the faces and face encodings in the current frame of video
			face_locations = face_recognition.face_locations(rgb_small_frame)
			face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
			for face_encoding in face_encodings:
				# See if the face is a match for the known face(s)
				matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
				name = "Unknown"
				# Or instead, use the known face with the smallest distance to the new face
				face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
				best_match_index = np.argmin(face_distances)
				if matches[best_match_index]:
					name = known_face_names[best_match_index]
				face_names.append(name)
			print(face_names)

			# loop over the detected face locations and their corresponding locations
			for (box, pred) in zip(locs, preds):
				# unpack the bounding box and predictions
				(startX, startY, endX, endY) = box
				(mask, withoutMask) = pred

				# determine the class label and color we'll use to draw the bounding box and text
				label = "Mask" if mask > withoutMask else "No Mask"
				color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
					
				# include the probability in the label
				label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

				# display the label and bounding box rectangle on the output frame
				cv2.putText(frame, label, (startX, startY - 10),
					cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
				cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

			ret, buffer = cv2.imencode('.jpg', frame)
			frame = buffer.tobytes()
			
		return frame
  