import numpy as np
import os
import matplotlib.pyplot as plt

from zipfile import ZipFile
from PIL import Image
from numpy import sqrt
import random
import csv

from sklearn.metrics import mean_squared_error
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.metrics import normalized_mutual_info_score
import algorithms

# variables




path = ''+os.getcwd()+'/'
targetpath = path + 'data'
resultpath = path + 'result/'
rng = np.random.RandomState(13)

Iter_step = 500
skip_step = 50
error = 1e-4


# util functions

def unzip_file(sourcepath, targetpath):
  with ZipFile(sourcepath, 'r') as f:
    f.extractall(path=targetpath)

def load_data(root='data/CroppedYaleB', reduce=1):
    """
    Load ORL (or Extended YaleB) dataset to numpy array.

    Args:
        root: path to dataset.
        reduce: scale factor for zooming out images.

    """

    root = path + root



    images, labels = [], []

    for i, person in enumerate(sorted(os.listdir(root))):

        if not os.path.isdir(os.path.join(root, person)):
            continue

        for fname in os.listdir(os.path.join(root, person)):

            # Remove background images in Extended YaleB dataset.
            if fname.endswith('Ambient.pgm'):
                continue

            if not fname.endswith('.pgm'):
                continue

            # load image.
            img = Image.open(os.path.join(root, person, fname))
            img = img.convert('L') # grey image.

            # reduce computation complexity.
            img = img.resize([s//reduce for s in img.size])

            # TODO: preprocessing.

            # convert image to numpy array.
            img = np.asarray(img).reshape((-1,1))

            # collect data and label.
            images.append(img)
            labels.append(i)

    # concate all images and labels.
    images = np.concatenate(images, axis=1)
    labels = np.array(labels)

    return images, labels

# get_image_size
def get_image_size(datatype, reduce=1):

  if datatype == 'data/ORL':
    image_size = [i // reduce for i in (92, 112)]
  elif datatype == 'data/CroppedYaleB':
    image_size = [i // reduce for i in (168, 192)]
  return np.array(image_size)

# Load ORL dataset.
X_hat, Y_hat = load_data(root= 'data/ORL', reduce=2)
print('ORL dataset: X_hat.shape = {}, Y_hat.shape = {}'.format(X_hat.shape, Y_hat.shape))

# Load Extended YaleB dataset.
X_hat, Y_hat = load_data(root='data/CroppedYaleB',  reduce=4)
print('Extended YalB dataset: X_hat.shape = {}, Y_hat.shape = {}'.format(X_hat.shape, Y_hat.shape))

# define noise addition methods

# add block noise
def add_block_noise(X, block_size, datatype, reduce):

  n_features, n_samples = X.shape
  image_size = get_image_size(datatype, reduce)
  X = X.copy().reshape((image_size[1], image_size[0], n_samples))
  head_index_y, head_index_x = image_size - block_size + 1
  for i in range(n_samples):
    index_x = rng.randint(head_index_x)
    index_y = rng.randint(head_index_y)
    X[index_x:index_x+block_size,index_y:index_y+block_size,i] = 255
  X = X.reshape((n_features, n_samples))
  return X

# add salt and pepper noise on X
def add_salt_pepper_noise(X, percentage):

    n_features, n_samples = X.shape
    X = X.copy()
    size = int(n_features * percentage)
    middle = size//2+1
    for i in range(n_samples):
      index = np.arange(n_features)
      rng.shuffle(index)
      X[index[:middle],i] = 0
      X[index[middle:size], i] = 255
    return X

# add random noise on X
def add_random_noise(X, beta=40):

    n_features, n_samples = X.shape
    X = X.copy()
    X_noise = np.random.rand(*X.shape) * beta
    X = X + X_noise
    return X

# evaluation methods
# Evaluate Root Means Square Errors

def get_RMSE(D, R, X_hat):
  return sqrt(mean_squared_error(X_hat, D.dot(R)))

def assign_cluster_label(X, Y):
    kmeans = KMeans(n_clusters=len(set(Y)),n_init='auto').fit(X)
    Y_pred = np.zeros(Y.shape)
    for i in set(kmeans.labels_):
        ind = kmeans.labels_ == i
        Y_pred[ind] = Counter(Y[ind]).most_common(1)[0][0] # assign label.
    return Y_pred

def get_Acc_NMI(R, Y_hat):
  Y_pred = assign_cluster_label(R.T, Y_hat)
  acc = accuracy_score(Y_hat, Y_pred)
  nmi = normalized_mutual_info_score(Y_hat, Y_pred)
  return acc, nmi

def get_evaluation_matrics(D,R,X_hat,Y_hat):
  RMSE = get_RMSE(D,R,X_hat)
  ACC, NMI = get_Acc_NMI(R, Y_hat)
  return RMSE,ACC,NMI

# experiment function
# dataType : 'data/ORL' or 'data/CroppedYaleB'
# noiseType : 'b' for block noise, 'sp' for salt and pepper noise
# modelType : 'model_stdNMF' for NMF_standard, 'model_L1NMF' for NMF_L1norm, 'model_L21NMF' for NMF_L21norm,
# 'model_L1RNMF' for NMF_L1NormR, 'model_HCNMF' for HCNMF
def experiment(dataType, reduce, noiseType, modelType, block_size=10, percentage=0.05, index=10):
  # Load ORL dataset.
  if dataType == 'data/ORL':
    n_components = 40
    print("Experiment on ORL dataset...")
    print("Loading ORL dataset...")
    X_hat, Y_hat = load_data(root= dataType, reduce=reduce)
    # split the data
    indexORL = random.sample(range(400), 390)
    X_hat = X_hat[:, indexORL]
    Y_hat = Y_hat[indexORL]
    n_features, n_samples = X_hat.shape
    image_size = get_image_size('data/ORL', reduce=reduce)
    print('ORL dataset: X_hat.shape = {}, Y_hat.shape = {}'.format(X_hat.shape, Y_hat.shape))
  elif dataType == 'data/CroppedYaleB':
    n_components = 38
    print("Experiment on Extended YaleB dataset...")
    print("Loading Extended YaleB dataset...")
    X_hat, Y_hat = load_data(root= dataType, reduce=reduce)
    # split the data
    indexYal = random.sample(range(2414), 2172)
    X_hat = X_hat[:, indexYal]
    Y_hat = Y_hat[indexYal]
    n_features, n_samples = X_hat.shape
    image_size = get_image_size('data/CroppedYaleB', reduce=reduce)
    print('YaleB dataset: X_hat.shape = {}, Y_hat.shape = {}'.format(X_hat.shape, Y_hat.shape))

  # Add noise
  print("Adding noise...")
  if noiseType == 'b':
    # block noise with different block size (10,12,14)
    print(f"Adding block noise with block size {block_size}")
    X = add_block_noise(X_hat, block_size, dataType, reduce=reduce)
  elif noiseType == 'sp':
    print(f"Adding salt and pepper noise with percentage {percentage}")
    X = add_salt_pepper_noise(X_hat, percentage)

  # construct model
  print(f"Constructing {modelType} model...")
  if modelType == 'model_stdNMF':
    model = algorithms.NMF_standard(n_components,Iter_step)
  elif modelType == 'model_L1NMF':
    model = algorithms.NMF_L1norm(n_components,Iter_step)
  elif modelType == 'model_L21NMF':
    model = algorithms.NMF_L21norm(n_components,Iter_step)
  elif modelType == 'model_L1RNMF':
    model = algorithms.NMF_L1NormR(n_components,Iter_step)
  elif modelType == 'model_HCNMF':
    model = algorithms.HCNMF(n_components,Iter_step)

  # Training
  print("Training...")
  D, R = model.fit(X)

  #Evaluation
  print("Evaluating...")
  RMSE,ACC,NMI = get_evaluation_matrics(D,R,X_hat,Y_hat)
  print(f"Evaluation matrics on dataset:{dataType} noiseType:{noiseType} modelType:{modelType} Iter_step:{Iter_step}")
  print("  RMSE   |  ACC     |  NMI   ")
  print("{:^8.4f} | {:^8.4f} | {:^8.4f}".format(RMSE,ACC,NMI))
  print("recontruct image...")
#   X_reconstruct = D.dot(R)
#   plt.figure(figsize=(10,3))
#   plt.subplot(141)
#   plt.imshow(X_hat[:,index].reshape(image_size[1],image_size[0]), cmap=plt.cm.gray)
#   plt.axis('off')
#   plt.title('Image(original)')
#   plt.subplot(142)
#   plt.imshow(X[:,index].reshape(image_size[1],image_size[0]), cmap=plt.cm.gray)
#   plt.axis('off')
#   plt.title('Image(noised)')
#   plt.subplot(143)
#   plt.imshow(X_reconstruct[:,index].reshape(image_size[1],image_size[0]), cmap=plt.cm.gray)
#   plt.axis('off')
#   plt.title('Image(reconstruct)')
#   plt.show()

  return D,R,RMSE,ACC,NMI


