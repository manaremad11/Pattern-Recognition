from keras.datasets import mnist
import numpy as np

def extract_features(data, block_size=4):
  blocks_start_index = []
  img_height, img_width = data[0].shape
  for x in range(0, img_height-block_size+1, block_size):
    for y in range(0, img_width-block_size+1, block_size):
      blocks_start_index.append((x,y))

  features = []
  for img in data:
    img_feature = []
    for idx in blocks_start_index:
      total, total_x, total_y = 1e-9,0,0

      for x in range(idx[0], idx[0]+block_size):
        for y in range(idx[1], idx[1]+block_size):
          total += img[x][y]
          total_x += img[x][y]*x
          total_y += img[x][y]*y
      total_x /= total
      total_y /= total
      img_feature.append(total_x)
      img_feature.append(total_y)

    features.append(img_feature)
  return np.array(features)

(train_X, train_y), (test_X, test_y) = mnist.load_data()

print("train data")
print(train_X.shape)
print(train_y.shape)
print("test data")
print(test_X.shape)
print(test_y.shape)

train_X = train_X[:10000]/255.0
train_y = train_y[:10000]
test_X = test_X[:6000]/255.0
test_y = test_y[:6000]

train_X = extract_features(train_X, block_size=4)
test_X = extract_features(test_X, block_size=4)

print("train data")
print(train_X.shape)
print("test data")
print(test_X.shape)

def knn_predict(K=1):
  predicted_y = []
  for x in test_X:
    euclidean_distances = np.sqrt(np.sum(np.square(train_X-x), axis=1))
    voting = np.argsort(euclidean_distances)[:K]
    voting_classes = train_y[voting]
    winner = np.bincount(voting_classes).argmax()
    predicted_y.append(winner)

  return np.array(predicted_y)

def calculate_accuracy(pred_y, actual_y):
  acc=0
  for res in zip(pred_y, actual_y):
    if res[0]==res[1]:
      acc+=1

  acc = (acc/pred_y.shape[0])*100
  return acc

for k in range(1, 4):
  predict_y = knn_predict(k)
  acc = calculate_accuracy(predict_y, test_y)
  print("k: {0} - accuracy: {1:.2f}".format(k, acc))