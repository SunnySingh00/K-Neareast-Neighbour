import numpy as np
from knn import KNN

############################################################################
# DO NOT MODIFY ABOVE CODES
############################################################################


# TODO: implement F1 score
def f1_score(real_labels, predicted_labels):
    """
    Information on F1 score - https://en.wikipedia.org/wiki/F1_score
    :param real_labels: List[int]
    :param predicted_labels: List[int]
    :return: float
    """
    if len(real_labels)!= 0 and len(predicted_labels)!=0 and len(real_labels)==len(predicted_labels):
        TP=0
        TN=0
        FP=0
        FN=0
        for i in range(0,len(real_labels)):
            if predicted_labels[i]:
                # we are looking at truely positives and falsely positives
                if real_labels[i]: # Truely positives
                    TP+=1
                else:
                    FP+=1
            else:
                # we are looking at truely negatives and falsely negatives
                if real_labels[i]==0:# Truely Negatives
                    TN+=1
                else:
                    FN+=1
        if TP+FP==0 or TP+FN==0 or TP==0:
            return 0
        P = (TP)/(TP+FP)
        R = (TP)/(TP+FN)
        F1_PR = (2*P*R)/(R+P) # Using Precision and Recall
        return F1_PR
        # F1 = (2*np.dot(real_labels,predicted_labels))/(np.sum(real_labels)+np.sum(predicted_labels))    
        # #print('F1 score: with TP, TN',F1_PR)
        # #print('F1 score:',F1)
        # return F1
    else:
        return 0
        raise NotImplementedError

class Distances:
    @staticmethod
    # TODO
    def minkowski_distance(point1, point2):
        """
        Minkowski distance is the generalized version of Euclidean Distance
        It is also know as L-p norm (where p>=1) that you have studied in class
        For our assignment we need to take p=3
        Information on Minkowski distance - https://en.wikipedia.org/wiki/Minkowski_distance
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        p=3
        p_sum = 0
        for i in range(0,len(point1)):
            p_sum+=(abs(point1[i]-point2[i]))**p
        return p_sum**(1. /p)
        raise NotImplementedError

    @staticmethod
    # TODO
    def euclidean_distance(point1, point2):
        """
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        p=2
        p_sum = 0
        for i in range(0,len(point1)):
            p_sum+=(abs(point1[i]-point2[i]))**p
        return p_sum**(1. /p)
        raise NotImplementedError

    @staticmethod
    # TODO
    def inner_product_distance(point1, point2):
        """
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        return np.dot(np.array(point1),np.array(point2))
        raise NotImplementedError

    @staticmethod
    # TODO
    def cosine_similarity_distance(point1, point2):
        """
       :param point1: List[float]
       :param point2: List[float]
       :return: float
       """
        return 1-(np.dot(np.array(point1),np.array(point2)))/(np.linalg.norm(np.array(point1))*np.linalg.norm(np.array(point2)))
        raise NotImplementedError

    @staticmethod
    # TODO
    def gaussian_kernel_distance(point1, point2):
        """
       :param point1: List[float]
       :param point2: List[float]
       :return: float
       """
        distance = [(x - y) ** 2 for x, y in zip(point1, point2)]
        distance = sum(distance)
        distance = -np.exp(-0.5 * distance)
        return distance


class HyperparameterTuner:
    def __init__(self):
        self.best_k = None
        self.best_distance_function = None
        self.best_scaler = None
        self.best_model = None

    # TODO: find parameters with the best f1 score on validation dataset
    def tuning_without_scaling(self, distance_funcs, x_train, y_train, x_val, y_val):
        """
        In this part, you should try different distance function you implemented in part 1.1, and find the best k.
        Use k range from 1 to 30 and increment by 2. Use f1-score to compare different models.

        :param distance_funcs: dictionary of distance functions you must use to calculate the distance.
            Make sure you loop over all distance functions for each data point and each k value.
            You can refer to test.py file to see the format in which these functions will be
            passed by the grading script
        :param x_train: List[List[int]] training data set to train your KNN model
        :param y_train: List[int] train labels to train your KNN model
        :param x_val:  List[List[int]] Validation data set will be used on your KNN predict function to produce
            predicted labels and tune k and distance function.
        :param y_val: List[int] validation labels

        Find(tune) best k, distance_function and model (an instance of KNN) and assign to self.best_k,
        self.best_distance_function and self.best_model respectively.
        NOTE: self.best_scaler will be None

        NOTE: When there is a tie, choose model based on the following priorities:
        Then check distance function  [euclidean > minkowski > gaussian > inner_prod > cosine_dist]
        If they have same distance fuction, choose model which has a less k.
        """
        
        # You need to assign the final values to these variables
        self.best_k = None
        self.best_distance_function = None
        self.best_model = None
        if len(x_train)!= 0 and len(x_train[0])!= 0 and len(y_train)!=0 and len(x_train)==len(y_train):
            f1 = -1
            dist_priority = dict({'euclidean':1,'minkowski':2,'gaussian':3,'inner_prod':4,'cosine_dist':5})
            dist = 0
            k_old = 1
            for k in range(1,min(len(y_train),30),2):
                for key,value in distance_funcs.items():
                    knn = KNN(k,value)
                    knn.train(x_train,y_train)
                    predicted_labels = knn.predict(x_val)
                    #print('predicted_labels',predicted_labels)
                    f1_new = f1_score(y_val,predicted_labels)
                    if f1_new==f1:
                        if dist_priority[key]<dist:
                            dist = dist_priority[key]
                            k_old=k
                        elif dist_priority[key]==dist:
                            if k<k_old:
                                k_old=k
                        else:
                            pass
                    elif f1_new>f1:
                        f1=f1_new
                        dist = dist_priority[key]
                        k_old=k

            self.best_k = int(k_old)
            for key, val in dist_priority.items():
                if val == dist:
                    self.best_distance_function = key
                    break

            model = KNN(self.best_k, distance_funcs[key])
            model.train(x_train,y_train)
            self.best_model = model

    # TODO: find parameters with the best f1 score on validation dataset, with normalized data
    def tuning_with_scaling(self, distance_funcs, scaling_classes, x_train, y_train, x_val, y_val):
        """
        This part is similar to Part 1.3 except that before passing your training and validation data to KNN model to
        tune k and disrance function, you need to create the normalized data using these two scalers to transform your
        data, both training and validation. Again, we will use f1-score to compare different models.
        Here we have 3 hyperparameters i.e. k, distance_function and scaler.

        :param distance_funcs: dictionary of distance funtions you use to calculate the distance. Make sure you
            loop over all distance function for each data point and each k value.
            You can refer to test.py file to see the format in which these functions will be
            passed by the grading script
        :param scaling_classes: dictionary of scalers you will use to normalized your data.
        Refer to test.py file to check the format.
        :param x_train: List[List[int]] training data set to train your KNN model
        :param y_train: List[int] train labels to train your KNN model
        :param x_val: List[List[int]] validation data set you will use on your KNN predict function to produce predicted
            labels and tune your k, distance function and scaler.
        :param y_val: List[int] validation labels

        Find(tune) best k, distance_funtion, scaler and model (an instance of KNN) and assign to self.best_k,
        self.best_distance_function, self.best_scaler and self.best_model respectively

        NOTE: When there is a tie, choose model based on the following priorities:
        For normalization, [min_max_scale > normalize];
        Then check distance function  [euclidean > minkowski > gaussian > inner_prod > cosine_dist]
        If they have same distance function, choose model which has a less k.
        """
        
        # You need to assign the final values to these variables
        self.best_k = None
        self.best_distance_function = None
        self.best_scaler = None
        self.best_model = None
        if len(x_train)!= 0 and len(x_train[0])!= 0 and len(y_train)!=0 and len(x_train)==len(y_train):
            f1 = -1
            dist_priority = dict({'euclidean':1,'minkowski':2,'gaussian':3,'inner_prod':4,'cosine_dist':5})
            scale_priority = dict({'min_max_scale':1,'normalize':2})
            dist = 0
            k_old = 1
            scale = 0
            for s_key,s_value in scaling_classes.items():
                scaler = s_value()
                train_scaled = scaler(x_train)
                test_scaled = scaler(x_val)
                for k in range(1,min(len(y_train),30),2):
                    for key,value in distance_funcs.items():
                        knn = KNN(k,value)
                        knn.train(train_scaled,y_train)
                        predicted_labels = knn.predict(test_scaled)
                        f1_new = f1_score(y_val,predicted_labels)
                        if f1_new==f1:
                            if scale_priority[s_key]<scale:
                                scale = scale_priority[s_key]
                                dist = dist_priority[key]
                                k_old=k
                            elif scale_priority[s_key]==scale:
                                if dist_priority[key]<dist:
                                    dist = dist_priority[key]
                                    k_old=k
                                elif dist_priority[key]==dist:
                                    if k<k_old:
                                        k_old=k
                                else:
                                    pass
                        elif f1_new>f1:
                            f1=f1_new
                            dist = dist_priority[key]
                            k_old=k
                            scale = scale_priority[s_key]

            self.best_k = int(k_old)
            for key, val in dist_priority.items():
                if val == dist:
                    self.best_distance_function = key
                    break
            for s_key, val in scale_priority.items():
                if val == scale:
                    self.best_scaler = s_key
                    break
            model = KNN(self.best_k, distance_funcs[key])
            model.train(x_train,y_train)
            self.best_model = model

class NormalizationScaler:
    def __init__(self):
        pass

    # TODO: normalize data
    def __call__(self, features):
        """
        Normalize features for every sample

        Example
        features = [[3, 4], [1, -1], [0, 0]]
        return [[0.6, 0.8], [0.707107, -0.707107], [0, 0]]

        :param features: List[List[float]]
        :return: List[List[float]]
        """
        normalized = []
        for feature in features:
            if all(x==0 for x in feature):
                normalized.append(feature)
            else:
                normalized.append([x/float(np.linalg.norm(feature)) for x in feature])
        return normalized


class MinMaxScaler:
    """
    Please follow this link to know more about min max scaling
    https://en.wikipedia.org/wiki/Feature_scaling
    You should keep some states inside the object.
    You can assume that the parameter of the first __call__
    will be the training set.

    Hints:
        1. Use a variable to check for first __call__ and only compute
            and store min/max in that case.

    Note:
        1. You may assume the parameters are valid when __call__
            is being called the first time (you can find min and max).

    Example:
        train_features = [[0, 10], [2, 0]]
        test_features = [[20, 1]]

        scaler1 = MinMaxScale()
        train_features_scaled = scaler1(train_features)
        # train_features_scaled should be equal to [[0, 1], [1, 0]]

        test_features_scaled = scaler1(test_features)
        # test_features_scaled should be equal to [[10, 0.1]]

        new_scaler = MinMaxScale() # creating a new scaler
        _ = new_scaler([[1, 1], [0, 0]]) # new trainfeatures
        test_features_scaled = new_scaler(test_features)
        # now test_features_scaled should be [[20, 1]]

    """
    

    def __init__(self):
        self.first = True
        self.min = []
        self.max = []
        pass

    def __call__(self, features):
        """
        normalize the feature vector for each sample . For example,
        if the input features = [[2, -1], [-1, 5], [0, 0]],
        the output should be [[1, 0], [0, 1], [0.333333, 0.16667]]

        :param features: List[List[float]]
        :return: List[List[float]]
        """
        a = np.array(features)
        if self.first:
            for i in range(0,len(a[0])):
                self.min.append(min(a[:,i]))
                self.max.append(max(a[:,i]))
            self.first=False
        assert len(self.min)!=0 and len(self.max)!=0
        for i in range(0,len(features)):
            for j in range(0,len(features[0])):
                if self.min[j]!=self.max[j]:
                    features[i][j] = (features[i][j]-self.min[j])/(self.max[j]-self.min[j])
                else:
                    features[i][j]=1
        return features
        raise NotImplementedError
