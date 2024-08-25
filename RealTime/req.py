#Importing required packages and dependencies
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier

#Loading distances data
X_train = np.load('archive/X_train.npy')
X_test = np.load('archive/X_test.npy')
y_train = np.load('archive/y_train.npy')
y_test = np.load('archive/y_test.npy')

#Normalization
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Dimensionality reduction using PCA
pca = PCA(n_components = 0.99)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

#Classification using random Forest = Our best performed classifier
rf = RandomForestClassifier(criterion = 'entropy')
rf.fit(X_train_pca, y_train)
y_pred = rf.predict(X_test_pca)
categories = ['neutral', 'disgusted', 'fearful',  'surprised', 'happy', 'sad', 'angry', 'can not say']

#prediction pipeline
def predict(landmarks):
    x = features(landmarks)
    x = np.array([x])
    x = scaler.transform(x)
    x = pca.transform(x)
    return categories[rf.predict(x)[0]]

#distance measure - Euclidean distance
def distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    return ((x1-x2)**2 + (y1-y2)**2)**.5

#feature extaction
def features(landmarks):
    if len(landmarks) ==0:
        return np.array([0]*5184)
    else:
        landmarks = landmarks[0]
        points = []
        for i, v in landmarks.items():
            points.extend(v)
        distances = []
        for i in points:
            distances.extend([distance(i, k) for k in points])
        return np.array(distances)