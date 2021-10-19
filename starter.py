# returns Euclidean distance between vectors a dn b
def euclidean(a, b):
    len_a = len(a)
    len_b = len(b)
    dist = 0
    if len_a != len_b:
        print("Lengths of the vectors don't match. Retry again.")
    else:
        for i in range(len_a):
            dim_dist = (b[i] - a[i])**2
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
            a_dist = a_dist + (a[i]**2)
            b_dist = b_dist + (b[i]**2)
        if a_dist == 0 or b_dist == 0:
            print("Cannot calculate cosine distance. At least one of the vector is the zero vector.")
        else:
            a_dist = a_dist ** 0.5
            b_dist = b_dist ** 0.5
            dist = ab_dist / (a_dist * b_dist)
            return dist


# returns a list of labels for the query dataset based upon labeled observations in the train dataset.
# metric is a string specifying either "euclidean" or "cosim".  
# All hyper-parameters should be hard-coded in the algorithm.
def knn(train, query, metric):
    return(labels)


# returns a list of labels for the query dataset based upon observations in the train dataset. 
# labels should be ignored in the training set
# metric is a string specifying either "euclidean" or "cosim".  
# All hyper-parameters should be hard-coded in the algorithm.
def kmeans(train, query, metric):
    return(labels)


def read_data(file_name):
    data_set = []
    with open(file_name, 'rt') as f:
        for line in f:
            line = line.replace('\n', '')
            tokens = line.split(',')
            label = tokens[0]
            attribs = []
            for i in range(784):
                attribs.append(tokens[i+1])
            data_set.append([label, attribs])
    return data_set

train = read_data("train.csv")
print(train.shape)

#print(knn(train, , euclidean))

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


def main():
    show('valid.csv', 'pixels')


if __name__ == "__main__":
    main()
