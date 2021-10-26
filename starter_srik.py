import random


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
    len_a = len(a)
    len_b = len(b)
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
            ab_dist = ab_dist + (a[i] * b[i])
            a_dist = a_dist + (a[i] ** 2)
            b_dist = b_dist + (b[i] ** 2)
        if a_dist == 0 or b_dist == 0:
            print("Cannot calculate cosine distance. At least one of the vector is the zero vector.")
        else:
            a_dist = a_dist ** 0.5
            b_dist = b_dist ** 0.5
            dist = ab_dist / (a_dist * b_dist)
            return dist


train = read_data("train.csv")
test = read_data("test.csv")


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
            dist = cosim(query, train_row)
        distances.append((train_row, dist))
    distances.sort(key=lambda tup: tup[1])
    neighbors = list()
    for i in range(k):
        neighbors.append(distances[i][0])
    return neighbors


def predict_classification(train, query, metric):
    if metric == "euclid":
        neighbors = knn(train, query, "euclid")
    else:
        neighbors = knn(train, query, "cosim")
    output_values = [row[0] for row in neighbors]
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


ans_predictions = knn_predictions(train, test, "euclid")

cnfsn_mtrx = confusion_matrix(
    actual=[row[0] for row in test],
    predict=ans_predictions,
    labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
)
print([row[0] for row in test])
print(ans_predictions)
print(cnfsn_mtrx)
sum1 = 0
for i in range(len(cnfsn_mtrx)):
    sum1 = sum1 + sum(cnfsn_mtrx[i])
print(sum1)
pre_accuracy = 0
for i in range(10):
    pre_accuracy = pre_accuracy + cnfsn_mtrx[i][i]
print("Accuracy Percentage = " + str(pre_accuracy*100/len(ans_predictions)))

# returns a list of labels for the query dataset based upon observations in the train dataset.
# labels should be ignored in the training set
# metric is a string specifying either "euclidean" or "cosim".  
# All hyper-parameters should be hard-coded in the algorithm.
#def kmeans(train, query, metric):
#    k = 10

#    actual_labels = []
#    data = []
#    for i in range(len(train)):
#        actual_labels.append(train[i][0])
#        data.append(train[i][1])

#    initial_centroids = []
#    for i in range(10):
#        random.seed(i)
#        initial_centroids_index = random.randint(0, len(data) - 1)
#        initial_centroids.append(data[initial_centroids_index])

#    print(initial_centroids)

#    return (labels)


def main():
    show('valid.csv', 'pixels')


if __name__ == "__main__":
    main()
