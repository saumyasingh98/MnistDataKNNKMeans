import random
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
def read_data(file_name):
    data_set = []
    with open(file_name, 'rt') as f:
        for line in f:
            line = line.replace('\n', '')
            tokens = line.split(',')
            label = tokens[0]
            attribs = []
            for i in range(784):
                attribs.append(tokens[i + 1])
            data_set.append([label, attribs])
    return data_set


def show(file_name, mode):
    data_set = read_data(file_name)
    for obs in range(len(data_set)):
        for idx in range(784):
            if mode == 'pixels':
                if data_set[obs][1][idx] == '0':
                    print(' ', end='')
                else:
                    print('*', end='')
            else:
                print('%4s ' % data_set[obs][1][idx], end='')
            if (idx % 28) == 27:
                print(' ')
        print('LABEL: %s' % data_set[obs][0], end='')
        print(' ')



# returns Euclidean distance between vectors a dn b
def euclidean(a, b):
    len_a=len(a)
    len_b=len(b)
    dist = 0
    if len_a != len_b:
        print("Lengths of the vectors don't match. Retry again.")
    else:
        for i in range(len_a):
            dim_dist = (int(b[i]) - int(a[i])) ** 2
            dist = dist + dim_dist
        dist = dist ** 0.5
    return dist


# returns Cosine Similarity between vectors a dn b
def cosim(a, b):
    len_a = len(a)
    len_b = len(b)
    ab_dist, a_dist, b_dist = float(0), float(0), float(0)
    if len_a != len_b:
        print("Lengths of the vectors don't match. Retry again.")
    else:
        for i in range(len_a):
            ab_dist = ab_dist + (int(a[i]) * int(b[i]))
            a_dist = a_dist + (int(a[i]) ** 2)
            b_dist = b_dist + (int(b[i]) ** 2)
        if a_dist == 0 or b_dist == 0:
            print("Cannot calculate cosine distance. At least one of the vector is the zero vector.")
        else:
            a_dist = a_dist ** 0.5
            b_dist = b_dist ** 0.5
            dist = ab_dist / (a_dist * b_dist)
            return dist


train = read_data("train.csv")
train_knn= read_data("train.csv")
test = read_data("test.csv")
valid = read_data("valid.csv")



train_labels = []
train_features = []

for i in train:
    train_labels.append(i[0])
    train_features.append(i[1])



pca = PCA(n_components =75)
train1 = pca.fit_transform(train_features)


for i in range(len(train)):
    train[i][1]=train1[i]








def test_distances(a, b):
    euclidean_d = euclidean(a, b)
    cosim_d = cosim(a,b)

    sklearn_euclidean = euclidean_distances(a, b)
    sklearn_cosim = cosine_similarity

# returns a list of labels for the query dataset based upon labeled observations in the train dataset.
# metric is a string specifying either "euclidean" or "cosim".
# All hyper-parameters should be hard-coded in the algorithm.
def knn(train, query, metric):
    k = 10
    distances = list()
    for train_row in train:
        if metric == "euclid":
            dist = euclidean(query[1], train_row[1])
        else:
            dist = cosim(query[1], train_row[1])
        distances.append((train_row, dist))
    distances.sort(key=lambda tup: tup[1])
    neighbors = list()
    for i in range(k):
        neighbors.append(distances[i][0])
    return neighbors


def predict_classification(train, query, metric):
    if metric == "euclid":
        neighbours = knn(train, query, "euclid")
    else:
        neighbours = knn(train, query, "cosim")
    output_values = [row[0] for row in neighbours]
    prediction = max(set(output_values), key=output_values.count)
    return prediction


def knn_predictions(train, test, metric):
    predictions = list()
    count = 0
    for row in test:
        if metric == "euclid":
            print("Starting Calculating Predictions on test row " + str(count))
            output = predict_classification(train, row, "euclid")
        else:
            print("Starting Calculating Predictions on test row " + str(count))
            output = predict_classification(train, row, "cosim")
        predictions.append(output)
        count = count + 1
    return predictions


def confusion_matrix(actual, predict, labels):
    matrix = [[0 for i in range(len(labels))] for j in range(len(labels))]
    if len(actual) != len(predict):
        print("Lengths of estimate and predicted don't match. Try again")
    else:
        for i in range(len(labels)):
            for j in range(len(labels)):
                for k in range(len(actual)):
                    a = int(actual[k])
                    p = int(predict[k])
                    if a == i:
                        if p == j:
                            matrix[a][p] = matrix[a][p] + 1
    return matrix


print("Starting KNN Euclid - Test")
test_predictions = knn_predictions(train_knn, test, "euclid")

cnfsn_mtrx = confusion_matrix(
    actual=[row[0] for row in test],
    predict=test_predictions,
    labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
)

sum1 = 0
for i in range(len(cnfsn_mtrx)):
    sum1 = sum1 + sum(cnfsn_mtrx[i])

pre_accuracy = 0
for i in range(10):
    pre_accuracy = pre_accuracy + cnfsn_mtrx[i][i]

print("Confusion Matrix = " + str(cnfsn_mtrx))
print("Accuracy Percentage = " + str(pre_accuracy * 100 / len(test_predictions)))


# returns a list of labels for the query dataset based upon observations in the train dataset.
# labels should be ignored in the training set
# metric is a string specifying either "euclidean" or "cosim".
# All hyper-parameters should be hard-coded in the algorithm.
def initialize_centroids(data, k):
    initial_centroids = []

    for i in range(k):
        random.seed(i * 100)
        initial_centroids_index = random.randint(0, len(data) - 1)
        initial_centroids.append(data[initial_centroids_index])

    return initial_centroids


def assign_centroid(data, centroids, metric):
    centroid_assign = []
    centroid_errors = []

    for observation in data:
        errors = []
        for centroid in centroids:
            if metric == "euclid":
                error = round(euclidean(centroid[1], observation[1]), 3)
            else:
                error = round(cosim(centroid[1], observation[1]), 3)

            errors.append(error)

        centroid_error = min(errors)
        for i in range(len(errors)):
            if errors[i] == centroid_error:
                closest_centroid = i

        # Assign values to lists
        centroid_assign.append(closest_centroid)
        centroid_errors.append(centroid_error)

    return centroid_assign, centroid_errors


def kmeans(train, metric):
    k = 10
    num_iter = 50

    centroids = initialize_centroids(train, k)
    error = []
    exit = True
    iter = 0

    while exit and iter < num_iter:
        print(str(iter) + " iteration started")
        new_centroids, iter_error = assign_centroid(train, centroids, metric)
        error.append(sum(iter_error))
        temp_centroids = [[]] * 10
        for i in range(len(temp_centroids)):
            temp_centroids[i] = [i, [0] * 784, 0]
        for k in range(len(new_centroids)):
            for temp_centroid in temp_centroids:
                if new_centroids[k] == temp_centroid[0]:
                    temp_centroid[1] = [int(x) + int(y) for x, y in zip(temp_centroid[1], train[k][1])]
                    temp_centroid[2] = temp_centroid[2] + 1
        for k, temp_centroid in enumerate(temp_centroids):
            if temp_centroid[2] == 0:
                avg_pixels = initialize_centroids(train, 1)
                print(avg_pixels)
                avg_pixels = avg_pixels[0][1]
            else:
                avg_pixels = [round((pixel / temp_centroid[2]), 3) for pixel in temp_centroid[1]]
            centroids[k] = [temp_centroid[0], avg_pixels]

        if len(error) < 2:
            exit = True
        else:
            if error[iter] != error[iter - 1]:
                exit = True
            else:
                exit = False
        iter = iter + 1
        print(error)

    return centroids, error


print("Starting k-Means - Euclid")
centroids, error = kmeans(train, "euclid")
print(centroids)
print(error)


def main():
    show('valid.csv', 'pixels')


if __name__ == "__main__":
    main()
