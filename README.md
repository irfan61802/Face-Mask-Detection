# Face-Mask-Detection

## Project Description 
This project was inspired by three of my classes for my final semester. I did my CS 4630 Undergraduate Seminar presentation/essay on how facial recognition works using computer vision and machine learning. I am also taking CS 4210 Machine Learning this semester which went into further detail about how neural networks actually work as well as how to train them using python. Lastly, I wanted to learn how to integrate trained machine learning models into a web-app as I learned the entire development cycle to create a web application from scratch in CS 4800 Software Engineering. This project was intended to be a culmination of everything I had learned from class and my own research this semester. Machine Learning and AI is one of my major interests as a sub-field of computer science so it was a great opportunity to create something on my own. 

I decided to create a web application using computer vision and machine learning that is able to detect whether people are wearing masks in real time. I also added the ability the recognize the faces that are registered in the database. 

## Training the Mask Detector Model

### 1. Dataset
The Dataset I used was from [Kaggle](https://www.kaggle.com/datasets/omkargurav/face-mask-dataset) which consisted of 7553 RGB images. It consisted of 3725 images with masks and 3828 images without masks resulting in the classes being quite balanced. 

![Class Distribution](images/classes.jpg)

### 2. Model and Methodology Proposed

In order to classify objects in images, Convolutional Neural Networks (CNN) are most commonly used. Instead of creating one from scratch, I decided to use transfer learning with pre-trained models and fine-tune the final layers to fit my classification needs. Transfer learning is a machine learning technique that involves using knowledge gained from a previous task with a large dataset that is incorporated into a similar task in order to improve the performance of the new task as the model utilizes the previously learned information. By using a pre-trained model as a starting point, transfer learning significantly reduced the amount of data and time required to train a new model while increasing the accuracy. It also allows for testing multiple models and variables to efficiently find which model best suits our task. Since the models are already trained on image datasets of over a million images each, they can easily detect edges and features in the images while the additional layers and training will allow it to adhere to the specific task of mask detection on a person's face.

I decided to test the models ResNet50V2, MobileNetV2, and InceptionResNetV2 from the tensor flow keras library. 

### 3. Image Preprocessing

I used Keras' ImageDataGenerator class to perform Image augmentation and flow_from_dataframe for further preprocessing. Image augmentation is used to prevent overfitting in models, especially with smaller datasets. It creates a lot more variation in the data and exposes the models to many images that were not in the original dataset.

Hyperparameters:
- Normalization by dividing all pixel values by 255
- Rotation range of 20 degrees
- Vertical/Horizontal shift of 20%
- Slant of 20 degrees
- Allowing for horizontal flip

All images were also resized to 224 by 224 before using them to train the model.
 
### 4. Training

The final layers I added to fine-tune these models included:
- Flatten layer
- Dense layer with 1024 Neurons and reli activation function
- Dropout layer with a rate of 0.5
- Dense output layer with sigmoid activation function

The models were trained using an adam optimizer with a learning rate of 0.001 and binary_crossentropy loss function. I tested them all initially after running a couple epochs each but ended up choosing ResNet50V2 to run for 30 epochs for efficiency and accuracy. 

### 5. Results
Below shows the accuracy and loss per epoch after running ResNet50V2 for 30 epochs. This took around a day to train. 

![Accuracy and Loss](images/accloss.jpg)

Below shows the classification table and confusion matrix for the model after it finished training. It reached a test accuracy of 99.34%
![Classification Table](images/class.jpg)
![Confusion Matrix](images/conf.jpg)

The model reached over 99% accuracy after the 12th epoch but kept running even though I had an early stopping function. Since the minimum change or min_delta was set to 0, there may have been minute improvements in accuracy.

## Integrating the Model to Web-Application

### Python Open-CV and Flask
In order to read frames from a real-time video that could be used for further processing, I used python open-cv library. The frames generated are in array format of pixels which are used to actually do the computation to find faces and run through the model. To display it to the browser using python flask, I used cv2 imencode() function to convert to jpeg format, then convert to bytes and display the video to the user. Once the face is found and passed through the model, open-cv is also used to display the box around the found faces and the probability of whether they have a mask on or not. This is done for all faces found in the photo. 

### Face Detection and Mask Prediction
Face detection must be done before actually passing a frame into the trained model to determine where the faces are in the image and if there even are any faces in the frame. I experimented with Haar Cascade Classifiers and python's face recognition module to detect faces but ended up choosing the Caffe pre-trained model as it seemed to be faster and more efficient for detecting faces. To use the Caffe model, I first created blobs from the images to make it in the correct input format of the neural network model. The model is then able to give the locations for each face in the image that are above a confidence of 0.5. 

After getting the faces, some pre-processing must be done before putting it through the model to detect masks including:
- Converting Colors from BGR to RGB
- Resizing to 224 x 224
- Using ResNet's preprocessing function

The trained mask detector model is loaded by using TensorFlow's load_model function which loads the weights and layers of the trained model into Python thus making it available to make predictions on the frame images. It outputs the classification in an array as a probability of each class, for example: [0,99]. I parsed the probability and displayed it in the frame in real-time using OpenCV.

##Face Recognition

I added functionality that could also identify know face onto my web app. It uses Python's face_recognition library to find faces in the frame and encode them. I stored them in a mongoDB database which is loaded each time the Flask app is run. Faces are then checked against existing faces to identify all faces in an image. 


## Masked Face Recognition
Since the first program mentioned only does mask detection and not recognition and the second only does identification without masks, I wanted to create something that could recognize faces even when masked. I attempted to combine the programs of face mask detection and facial recognition but determined that a new model would need to be trained to accomplish such. The model would need to be trained to classify each face with multiple images of the same person with and without a mask. My experiments determined that the face_recognition model could not encode faces with masks on because they were not detected and the Caffe model locations did not work when put into the encoding algorithm either. Even in the program that accounts for both, it is extremely inefficient as it does too much computation on a single frame and therefore not realistic for actual use.

##Conclusion

If I were to continue my project, training another model to recognize masked faces would be my next step. It is possible to recognize faces with masks on as there are still enough features to recognize a person above the nose, but may not be suitable for a web application as the model may take longer to make a prediction. I tested some other implementations I found on GitHub using FaceNet but which worked pretty well but the probability given for each face was quite low, even without a mask on. 

For the training aspect of the project, there are a lot of variables I could have changed when training the model. I could have tested different optimizers, learning rates, final layers, and even models. However, the accuracy achieved was already quite high at 99.34% so I'm not sure how much more they would help. Using larger databases such as CASIA-WebFace and VGGFace2 for future training on recognizing masked faces is also a possibility. It would definitely take a lot longer and more computational power but would definitely yield better results than my attempt to combine both technologies. 



