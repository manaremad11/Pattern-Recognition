import numpy as np
import os
import cv2
from collections import Counter


def knn_predict(train_x, train_y, test_x, k):
    predicted_y = []
    for x in test_x:
        euclidean_distances = np.sqrt(np.sum(np.square(train_x - x), axis=1))
        voting = np.argsort(euclidean_distances)[:k]
        voting_classes = train_y[voting]
        winner = Counter(voting_classes).most_common(1)[0][0]
        predicted_y.append(winner)

    return np.array(predicted_y)


def calculate_accuracy(predicted_y, actual_y):
    acc = 0
    for res in zip(predicted_y, actual_y):
        if res[0] == res[1]:
            acc += 1

    acc = (acc / predicted_y.shape[0]) * 100
    return acc


def normalized_glcm_horizontal(img, n, d=1):
    g_matrix = np.array([np.array([0] * n)] * n)
    height, width = img.shape

    for y in range(height):
        for x in range(width - d):
            g_matrix[img[y][x], img[y][x + d]] += 1

    normalized_g_matrix = g_matrix / g_matrix.sum()

    return normalized_g_matrix


def normalized_glcm_vertical(img, n, d=1):
    g_matrix = np.array([np.array([0] * n)] * n)
    height, width = img.shape

    for x in range(width):
        for y in range(height - d):
            g_matrix[img[y][x], img[y + d][x]] += 1

    normalized_g_matrix = g_matrix / g_matrix.sum()

    return normalized_g_matrix


def correlation(g_matrix):
    n = g_matrix.shape[0]
    indexes = np.arange(1, n + 1)
    rows_sum, columns_sum = g_matrix.sum(axis=0), g_matrix.sum(axis=1)
    mu_i = indexes * columns_sum
    mu_j = indexes * rows_sum

    var_i = columns_sum * (np.subtract(indexes, mu_i) ** 2)
    var_j = rows_sum * (np.subtract(indexes, mu_j) ** 2)

    k = np.subtract(indexes, mu_i) / (np.sqrt(var_i) + 1e-9)
    k = k.reshape(n, 1)
    m = np.subtract(indexes, mu_j) / (np.sqrt(var_j) + 1e-9)
    res = k * m * g_matrix
    res = res.sum()
    return res


def contrast(g_matrix):
    n = g_matrix.shape[0]
    indexes = np.arange(1, n + 1)
    k = indexes.reshape(n, 1)
    m = np.subtract(indexes, k) ** 2
    res = np.sum(m * g_matrix)
    return res


def energy(g_matrix):
    res = np.sum(g_matrix ** 2)
    return res


def homogeneity(g_matrix):
    n = g_matrix.shape[0]
    indexes = np.arange(1, n + 1)
    k = indexes.reshape(n, 1)
    m = 1 / (1 + np.abs(np.subtract(indexes, k)))
    res = np.sum(m * g_matrix)
    return res


def entropy(g_matrix):
    res = -1 * np.sum(g_matrix * np.log2(g_matrix + 1e-9))
    return res


def max_pair(g_matrix):
    return np.amax(g_matrix)


def load_data():
    main_folder = './image'
    x = []
    y = []

    for folder_name in os.listdir(main_folder):
        folder = os.path.join(main_folder, folder_name)
        for image_name in os.listdir(folder):
            if not image_name.endswith(".jpg"):
                continue
            image_path = os.path.join(folder, image_name)
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            x.append(img)
            y.append(folder_name)

    x, y = np.array(x), np.array(y)
    n = x.shape[0]
    random_indexes = np.random.permutation(n)
    x = x[random_indexes]
    y = y[random_indexes]
    k = int(n * .7)

    return x[:k], y[:k], x[k:], y[k:]


def extract_glcm_features(data):
    x_features = []
    k = 1
    for img in data:
        k += 1
        glcm = [normalized_glcm_horizontal(img, 256), normalized_glcm_vertical(img, 256)]
        current_features = []
        for g in glcm:
            current_features.append(correlation(g))
            current_features.append(contrast(g))
            current_features.append(energy(g))
            current_features.append(homogeneity(g))
            current_features.append(entropy(g))
            current_features.append(max_pair(g))
        current_features = np.array(current_features)
        x_features.append(current_features)
        print(k)

    x_features = np.array(x_features)
    x_features = x_features / np.max(x_features, axis=0)
    return x_features


def main():
    train_data, train_Y, test_data, test_Y = load_data()
    print('data shape')
    print(train_data.shape)
    print(test_data.shape)

    train_X = extract_glcm_features(train_data)
    test_X = extract_glcm_features(test_data)
    print('features data shape')
    print(train_X.shape)
    print(test_X.shape)

    predicted_y = knn_predict(train_X, train_Y, test_X, 1)
    acc = calculate_accuracy(predicted_y, test_Y)
    print('accuracy: ', acc)


if __name__ == '__main__':
    main()
