import numpy as np
import math




class my_KNN:

    def __init__(self, n_neighbors=3, metric="minkowski", p=2):
        # metric = {"minkowski", "euclidean", "manhattan", "cosine"}
        # p value only matters when metric = "minkowski"
        # notice that for "cosine", 1 is closest and -1 is furthest
        # therefore usually cosine_dist = 1- cosine(x,y)
        self.n_neighbors = int(n_neighbors)
        self.metric = metric
        self.p = p

    def minkowski__distance(self, a, b, p):
        try:
            return sum(abs(float(e1) - float(e2)) ** p for e1, e2 in zip(a, b)) ** (1 / self.p)
        except ValueError:
        # Handle the case where conversion to float is not possible
            return float('inf')  # or any other suitable value


    def new_manhattan__distance(self, v1, v2):
        n = len(v1)
        sum = 0

        for c in range(n):
            sum += abs(float(v1[c]) - float(v2[c]))
        return sum

    def edu(self, v1, v2):
        dist_f = np.sqrt(np.sum(np.square(v1 - v2)))
        return dist_f

    def cos(self, v1, v2):
        normv1 = self.norm(v1)
        normv2 = self.norm(v2)
        dist_f = 1 - (np.dot(v1, v2) / (normv1 * normv2))
        return dist_f

    def norm(self, vector):
        return math.sqrt(sum(x * x for x in vector))

    def distance(self, v1, v2):
        if self.metric == 'minkowski':
            dist_f = self.minkowski__distance(v1, v2, self.p)
        elif self.metric == 'euclidean':
            dist_f = self.edu(v1, v2)
        elif self.metric == 'manhattan':
            dist_f = self.new_manhattan__distance(v1, v2)
        elif self.metric == 'cos':
            dist_f = self.cos(v1, v2)
        return dist_f

    def bubbleSort(self, llll, cccc):
        n = len(llll)
        for i in range(n - 1):

            for j in range(n - i - 1):
                if llll[j] > llll[j + 1]:
                    llll[j], llll[j + 1] = llll[j + 1], llll[j]
                    cccc[j], cccc[j + 1] = cccc[j + 1], cccc[j]

        return llll, cccc

    def new_new_predicting_class(self, sorted_type_array):

        # _count_array = np.zeros((n_target_classes), dtype=np.intc)
        # n_neigh = 5
        num = 0
        distinct_classes = np.unique(self._y)
        count_list = [0] * len(distinct_classes)

        for i in range(len(sorted_type_array)):
            if (sum(count_list) < self.n_neighbors):
                for j in range(len(distinct_classes)):
                    if (sorted_type_array[i] == distinct_classes[j]):
                        count_list[j] = count_list[j] + 1

        max_vaal = max(count_list)

        for index, value in enumerate(count_list):
            if (value == max_vaal):
                num = index

        probab = max_vaal / self.n_neighbors

        final_class = distinct_classes[num]

        return final_class, probab

    def fit(self, X, y):
        # X: pd.DataFrame, independent variables, float
        # y: list, np.array or pd.Series, dependent variables, int or str
        self.classes_ = list(set(list(y)))
        self._y = y
        self._X = X
        # write your code below
        return

    def predict(self, X):
        return_list = []
        for i in range(len(X)):
            clas = self.predict_proba(X.iloc[i])
            return_list.append(clas)
        return_final = np.array(return_list)
        return return_final

    def predict_proba(self, X_test):

        new_dist_array = np.zeros((len(self._X)), dtype=np.float64)
        new_type_array = np.zeros((len(self._X)), dtype=object)

        v2 = X_test
        for j in range(len(self._X)):
            v1 = self._X.iloc[j]
            new_dist_array[j] = self.distance(v1, v2)
            new_type_array[j] = self._y.iloc[j]

        sorted_dist_array, sorted_type_array = self.bubbleSort(new_dist_array, new_type_array)
        ans_class, ans_probab = self.new_new_predicting_class(sorted_type_array)
        tt_list = [ans_class, ans_probab]

        return tt_list








