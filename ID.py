import numpy as np
import imutils
import time
import cv2
import os
import face_recognition
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

#connect to database
uri = "mongodb+srv://irfan61802:zDZQo0yAcnvUqBhC@face-db.92meset.mongodb.net/?retryWrites=true&w=majority"
client = MongoClient(uri, server_api=ServerApi('1'))
db = client['face']
faces_db=db.face
print("[INFO] Connected to database")

#get all recorded face encodings from database
all_docs = list(faces_db.find({}))
known_face_names, known_face_encodings = [doc["name"] for doc in all_docs], [doc["embedding"] for doc in all_docs]
known_face_encodings = np.array(known_face_encodings)

def encode_face(face_image,face_name):
	face_encoding = face_recognition.face_encodings(face_image)[0]
	faces_db.insert_one({"name": face_name, "embedding": face_encoding.tolist()})

class VideoCamera(object):
  def __init__(self):
    self.video = cv2.VideoCapture(0)

  def __del__(self):
    self.video.release()
    cv2.destroyAllWindows()

  def get_frame(self):
    ret,frame=self.video.read()

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.2, fy=0.2)
    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = cv2.cvtColor(small_frame,cv2.COLOR_BGR2RGB)
    
    # Only process every other frame of video to save time
    
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
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
    
    ret,jpeg=cv2.imencode('.jpg',frame)
    return jpeg.tobytes()
  