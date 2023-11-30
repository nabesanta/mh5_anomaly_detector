from pprint import pprint
import yaml
from sklearn import svm
import os
from data_process import DataProcess
from anomaly_detector import AnomalyDetector
from data_persist import DataPersist

class Interface:
    def __init__(self, input_file):
        print('\n\n', os.getcwd())

        with open(input_file, 'r') as fil:
            local = yaml.safe_load(fil)

            self.__fit_file = local["Files"]["fit_file"]
            self.__unsupervised_train_file = local["Files"]["unsupervised_train"]
            self.__supervised_train_file = local["Files"]["supervised_train"]
            self.__testing = local["Files"]["testing"]
            self.__processor = DataProcess(local["Operations"]["reduction"])

        self.__anomaly_classifier = None
        self.__data_persist = DataPersist()

        self.__ready_preprocessor()
        self.__preprocess_data_training()

    def __ready_preprocessor(self):
        split_x, split_y = self.__processor.split(self.__fit_file["name"],
                                                  self.__fit_file["input_col"],
                                                  self.__fit_file["output_col"])
        scaled_x = self.__processor.scaler_fit(split_x)
        reduced_x = self.__processor.reduction_fit(scaled_x)
        self.__data_persist.dump_data(self.__fit_file["name"], reduced_x, split_y)

    def __preprocess_data_training(self):
        if self.__unsupervised_train_file["name"]:
            self.__preprocess_data_helper(self.__unsupervised_train_file)

        if self.__supervised_train_file["name"]:
            self.__preprocess_data_helper(self.__supervised_train_file)

    def __preprocess_data_helper(self, item):
        split_x, split_y = self.__processor.split(item["name"],
                                                  item["input_col"],
                                                  item["output_col"])
        scaled_x = self.__processor.scaler_transform(split_x)
        reduced_x = self.__processor.reduction_transform(scaled_x)
        self.__data_persist.dump_data(item["name"], reduced_x, split_y)

    def __testing_predictions_helper(self, item):
        unsuper_preds, super_preds = [], []
        x, y = self.__data_persist.retrieve_dumped_data(item["name"])

        if self.__unsupervised_train_file["name"]:
            unsuper_preds = self.__anomaly_classifier.classify_unsupervised(
                item["name"], x, y)

        if self.__supervised_train_file["name"]:
            super_preds = self.__anomaly_classifier.classify_supervised(
                item["name"], x, y)

        return unsuper_preds, super_preds

    def print_all(self):
        print('fit\t', self.__fit_file, type(self.__fit_file))
        print('unsuper\t', self.__unsupervised_train_file, type(self.__unsupervised_train_file))
        print('super\t', self.__supervised_train_file, type(self.__supervised_train_file))
        print('test\t', self.__testing, type(self.__testing))
        pprint(self.__testing)
        print('\n')

    def genmodel_train(self, unsup_models, sup_models):
        self.__anomaly_classifier = AnomalyDetector(unsup_models, sup_models)

        if self.__unsupervised_train_file["name"]:
            print('=====Unsupervised=====')
            unsup_x, unsup_y = self.__data_persist.retrieve_dumped_data(
                self.__unsupervised_train_file["name"])
            self.__anomaly_classifier.fit_unsupervised_models(unsup_x)

        if self.__supervised_train_file["name"]:
            print('=====Supervised=====')
            sup_x, sup_y = self.__data_persist.retrieve_dumped_data(
                self.__supervised_train_file["name"])
            self.__anomaly_classifier.fit_supervised_models(sup_x, sup_y)

    def get_testing_predictions(self):
        u_preds = []
        s_preds = []

        for item in self.__testing:
            self.__preprocess_data_helper(item)
            usup_preds, sup_preds = self.__testing_predictions_helper(item)

            u_preds.append(usup_preds)
            s_preds.append(sup_preds)

        return u_preds, s_preds

    def retrieve_data(self, loc):
        return self.__data_persist.retrieve_dumped_data(loc)
