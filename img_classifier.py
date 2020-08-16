import cv2
import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import tensorflow as tf
import pandas as pd


# dir="/home/nuc-obs-01/Downloads/personal/ML/image"

# categories = ['cats', 'dogs']
# data_label = []

# for category in categories:
# 	path = os.path.join(dir, category)
# 	label = categories.index(category)

# 	for image in os.listdir(path):
# 		img_path = os.path.join(path, image)
# 		img = cv2.imread(img_path, 0)
# 		# cv2.imshow("image", img)
# 		# cv2.waitKey(0)
# 		try:			
# 			resized_img = cv2.resize(img, (640, 480))
# 			array_img = np.array(resized_img).flatten()
# 			data_label.append([array_img, label])
# 		except Exception as e:
# 			pass

# # print(data_label)

# pick_in = open("data1.pickle", "wb")
# pickle.dump(data_label, pick_in)
# pick_in.close()

###Model Training
pick_in = open("data1.pickle", "rb")
data = pickle.load(pick_in)
pick_in.close()

random.shuffle(data)

features = []
labels = []

for feature, label in data:
	features.append(feature)
	labels.append(label)

# print(features, labels)

x_train, x_test, y_train, y_test = train_test_split(features,labels, test_size=0.01)

# model = SVC(C=1,kernel='poly', gamma='auto')
# model.fit(x_train, y_train)

# pick = open('model.sav', 'wb')
# pickle.dump(model,pick)
# pick.close()

###Model Training

##Prediction Training

pick = open('model.sav', 'rb')
model = pickle.load(pick)
pick.close()

prediction=model.predict(x_test)

accuracy=model.score(x_test, y_test)

categories = ['cats', 'dogs']

print("Accuracy", accuracy)

print("Prediction ", categories[prediction[0]])


pet = x_test[0].reshape(640,480)
plt.imshow(pet, cmap='gray')
plt.show()

##Prediction Training