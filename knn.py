import numpy as np
from collections import Counter
class KNN:
    def __init__(self, k, distance_function):
        """
        :param k: int
        :param distance_function
        """
        self.k = k
        self.distance_function = distance_function
        self.features_train = []
        self.labels_train = []
    # TODO: save features and lable to self
    def train(self, features, labels):
        """
        In this function, features is simply training data which is a 2D list with float values.
        For example, if the data looks like the following: Student 1 with features age 25, grade 3.8 and labeled as 0,
        Student 2 with features age 22, grade 3.0 and labeled as 1, then the feature data would be
        [ [25.0, 3.8], [22.0,3.0] ] and the corresponding label would be [0,1]

        For KNN, the training process is just loading of training data. Thus, all you need to do in this function
        is create some local variable in KNN class to store this data so you can use the data in later process.
        :param features: List[List[float]]
        :param labels: List[int]
        """
        if len(features)!= 0 and len(labels)!=0 and len(features)==len(labels):
            self.labels_train = labels
            self.features_train = features
        #self.labels_train = labels
        #print('features_train',self.features_train)
        # self.features_train = features
        # self.labels_train = labels
        

    # TODO: predict labels of a list of points
    def predict(self, features):
        """
        This function takes 2D list of test data points, similar to those from train function. Here, you need process
        every test data point, reuse the get_k_neighbours function to find the nearest k neighbours for each test
        data point, find the majority of labels for these neighbours as the predict label for that testing data point.
        Thus, you will get N predicted label for N test data point.
        This function need to return a list of predicted labels for all test data points.
        :param features: List[List[float]]
        :return: List[int]
        """
        # for every points call get neighbours
        import collections
        from collections import Counter
        if len(features)!=0 and len(features[0])!=0:
            predicted_label=[]
            for point,l in zip(features,self.labels_train):
                predict_result=Counter(self.get_k_neighbors(point)).most_common(1)[0][0]
                predicted_label.append(predict_result)
            return np.array(predicted_label)
        return []
        # predicted_label = []
        # for point in features:
        #     ones = 0
        #     zeros = 0
        #     for label in self.get_k_neighbors(point):
        #         if label:
        #             ones+=1
        #         else:
        #             zeros+=1
        #     if ones>zeros:
        #         predicted_label.append(1)
        #     else:
        #         predicted_label.append(0)
        # return predicted_label
        # raise NotImplementedError

    # TODO: find KNN of one point
    def get_k_neighbors(self, point):
        """
        This function takes one single data point and finds k-nearest neighbours in the training set.
        You already have your k value, distance function and you just stored all training data in KNN class with the
        train function. This function needs to return a list of labels of all k neighours.
        :param point: List[float]
        :return:  List[int]
        """
        #print('\n Computing distnances from all training points ...')
        # f_train=[]
        # l_train=[]
        # for g,h in zip(self.features_train,self.labels_train):
        #         f_train.append(g)
        #         l_train.append(h)
        distances=[]
        for f,i in zip(self.features_train,self.labels_train):
            dist=self.distance_function(point,f)
            distances.append([dist,i])
        votes=[i[1] for i in sorted(distances)[:self.k]]
        return votes
            #####################################################################
        # dist = dict()
        # for i in range(0,len(self.features_train)):
        #     print('points',point,self.features_train[i])
        #     print('dist',self.distance_function(point,self.features_train[i]))
        #     dist[i]=self.distance_function(point,self.features_train[i])

        # #print('\n Computed distances from all points.')
        # closest_neighbors = sorted(dist.items(), key=lambda p: p[1])
        # #print("\n Closest_neighbors, sorted : ", closest_neighbors)
        # k_closest_neighbors = []
        # #
        # for i in range(0,self.k):
        #     #print(self.labels_train[closest_neighbors[i][0]])
        #     k_closest_neighbors.append(self.labels_train[closest_neighbors[i][0]])
        # #print('\n k_closest_neighbors : ', k_closest_neighbors)
        # return k_closest_neighbors
        # raise NotImplementedError





if __name__ == '__main__':
    print(np.__version__)
