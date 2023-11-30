import os
from sklearn import svm
import numpy as np
from output_matrix import OutputMatrix
import joblib
from pprint import pprint


class AnomalyDetector:
    def __init__(self, unsup_models, sup_models):
        self.__unsup_clfs = unsup_models
        self.__sup_clfs = sup_models
        self.__outputMat = OutputMatrix()

    def __fit_unsupervised(self, data):
        unsuper_clfs = []
        for i in range(len(self.__unsup_clfs)):
            current_model = self.__unsup_clfs[i]
            saveloc = f'../data/{current_model[0]}.pkl'
            if os.path.exists(saveloc):
                current_model = (current_model[0], joblib.load(saveloc))
            else:
                current_model[1].fit(data)
                joblib.dump(current_model[1], saveloc)
            self.__unsup_clfs[i] = current_model
            unsuper_clfs.append(saveloc)

    def __fit_supervised(self, data_x, data_y):
        super_clfs = []
        for i in range(len(self.__sup_clfs)):
            current_model = self.__sup_clfs[i]
            saveloc = f'../data/{current_model[0]}.pkl'
            if os.path.exists(saveloc):
                current_model = (current_model[0], joblib.load(saveloc))
            else:
                current_model[1].fit(data_x, data_y.ravel())
                joblib.dump(current_model[1], saveloc)
            self.__sup_clfs[i] = current_model
            super_clfs.append(saveloc)

    def __classify(self, model, data):
        predictions = model[1].predict(data)
        return predictions

    def fit_supervised_models(self, sup_train_x, sup_train_y):
        self.__fit_supervised(sup_train_x, sup_train_y)

    def fit_unsupervised_models(self, data_x):
        self.__fit_unsupervised(data_x)

    def classify_supervised(self, dataset_name, in_data, out_label):
        all_preds = []
        for i in range(len(self.__sup_clfs)):
            model = self.__sup_clfs[i]
            preds = self.__classify(model, in_data)
            all_preds.append(preds)
        all_preds = np.array(all_preds)
        print(f'Supervised results for {dataset_name}')
        print('Order of results: ', [item[0] for item in self.__sup_clfs])
        self.call_output(out_label, all_preds)
        return all_preds

    def classify_unsupervised(self, dataset_name, in_data, out_label):
        all_preds = []
        for i in range(len(self.__unsup_clfs)):
            model = self.__unsup_clfs[i]
            preds = self.__classify(model, in_data)
            all_preds.append(preds)
        all_preds = np.array(all_preds)
        print(f'Unsupervised results for {dataset_name}')
        print('Order of results: ', [item[0] for item in self.__unsup_clfs])
        self.call_output(out_label, all_preds)
        return all_preds

    def call_output(self, true_y, pred_y):
        for item in pred_y:
            self.__outputMat.output_matrix(true_y, item)
