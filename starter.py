import numpy as np
from math import sqrt
from collections import Counter
# returns Euclidean distance between vectors a dn b
def euclidean(a,b):
    dist = np.linalg.norm(b-a)
    return (dist)
        
# returns Cosine Similarity between vectors a dn b
def cosim(a,b):
    sum=0
    sum1=0
    sum2=0
    for i,j in zip(a,b):
        sum1+=i*i
        sum2+=j*j
        sum+=i*j
    dist = sum / ((sqrt(sum1)) * (sqrt(sum2)))
    return (dist)

# returns a list of labels for the query dataset based upon labeled observations in the train dataset.
# metric is a string specifying either "euclidean" or "cosim".  
# All hyper-parameters should be hard-coded in the algorithm.
def knn(train,query,metric):
    dis=[]
    k=3
    for i in train:
        for j in train[i]:
            d = euclidean(j,query)
            dis.append([d], i)
    lab=[i[1] for i in sorted(dis)[:k]]
    print(Counter(lab).most_common(1))
    labels=Counter(lab).most_common(1)[0][0]
    return(labels)

# returns a list of labels for the query dataset based upon observations in the train dataset. 
# labels should be ignored in the training set
# metric is a string specifying either "euclidean" or "cosim".  
# All hyper-parameters should be hard-coded in the algorithm.
def kmeans(train,query,metric):
    


    return(labels)

def read_data(file_name):
    
    data_set = []
    with open(file_name,'rt') as f:
        for line in f:
            line = line.replace('\n','')
            tokens = line.split(',')
            label = tokens[0]
            attribs = []
            for i in range(784):
                attribs.append(tokens[i+1])
            data_set.append([label,attribs])
    return(data_set)
        
def show(file_name,mode):
    
    data_set = read_data(file_name)
    for obs in range(len(data_set)):
        for idx in range(784):
            if mode == 'pixels':
                if data_set[obs][1][idx] == '0':
                    print(' ',end='')
                else:
                    print('*',end='')
            else:
                print('%4s ' % data_set[obs][1][idx],end='')
            if (idx % 28) == 27:
                print(' ')
        print('LABEL: %s' % data_set[obs][0],end='')
        print(' ')
            
def main():
    show('valid.csv','pixels')
    
if __name__ == "__main__":
    main()
    