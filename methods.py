import random
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.spatial import distance
from sklearn.model_selection import KFold
import os
from config import Config
from sklearn.preprocessing import LabelEncoder
import smote_variants as sv
import glob
from sklearn.metrics import classification_report
from glob import glob

class Functions:
    def __init__(self):
        self.X_filename = Config.X_filename
        self.y_filename = Config.y_filename
        self.raw_data_dir = Config.raw_data_dir
        self.no_of_splits = Config.no_of_splits
        self.saving_dir = Config.saving_dir
        self.oversampled_data_dir = Config.oversampled_data_dir
        self.data_shape = Config.data_shape
        self.groups_to_analyse = Config.groups_to_analyse
        self.Groups = Config.Groups
        self.similarity_score_dir = Config.similarity_score_dir
        self.oversampled_data_based_on_similarity_dir=Config.oversampled_data_based_on_similarity_dir
        self.model_results=Config.model_results
        os.environ['PYTHONHASHSEED'] = '0'
        np.random.seed(2)
        random.seed(2)
        self.classifiers=Config.classifiers
        self.without_smote_results_dir=Config.without_smote_results_dir
        self.final_result_dir=Config.final_result_dir
    # creating kfold data
    def create_kfold_data(self):
        print("inside k fold")
        X = np.load(os.sep.join([self.raw_data_dir,self.X_filename]))
        y = np.load(os.sep.join([self.raw_data_dir,self.y_filename]))
        skf = KFold(n_splits=self.no_of_splits)
        count = 1
        try:
            os.mkdir(os.sep.join([self.saving_dir]))
        except:
            print("directory exists")
        for train_index, test_index in skf.split(X, y):
            X_train, y_train = X[train_index], y[train_index]
            X_test, y_test = X[test_index], y[test_index]
            np.save(os.sep.join([self.saving_dir,'Fold_{}_X_train.npy'.format(count)]), X_train)
            np.save(os.sep.join([self.saving_dir,'Fold_{}_y_train.npy'.format(count)]), y_train)
            np.save(os.sep.join([self.saving_dir,'Fold_{}_X_test.npy'.format(count)]), X_test)
            np.save(os.sep.join([self.saving_dir,'Fold_{}_y_test.npy'.format(count)]), y_test)
            count += 1

    def plotting(self):
        result=pd.read_csv(os.sep.join([self.final_result_dir,'final_results.csv']))
        fig = px.line(result, x="model_name", y="probablity", color='name', title='weighted_f1_score comparison')
        fig.show()

    def aggregated_results(self):
        all_dataframe=[]
        for file in glob(os.sep.join([self.model_results,'*','*.csv'])):
            dataframe=pd.read_csv(file)
            dataframe=dataframe.reset_index(drop=True).rename(columns={'Unnamed: 0':'model_name','0':'probablity'})
            dataframe['name']=os.path.basename(file).split('.')[0]
            all_dataframe.append(dataframe)
        for file in glob(os.sep.join([self.without_smote_results_dir,'*.csv'])):
            dataframe=pd.read_csv(file)
            dataframe=dataframe.reset_index(drop=True).rename(columns={'Unnamed: 0':'model_name','0':'probablity'})
            dataframe['name'] ='Without smote'
            all_dataframe.append(dataframe)
        final_results=pd.concat(all_dataframe)
        final_results=final_results.groupby(['name','model_name'])['probablity'].mean().reset_index()
        final_results.to_csv(os.sep.join([self.final_result_dir,'final_results.csv']),index=False)

    def without_smote_results(self):
        for fold in range(1, self.no_of_splits + 1):
            train_data = np.load(os.sep.join([self.saving_dir, 'Fold_{}_X_train.npy'.format(fold)]))
            train_data = train_data.reshape(-1, self.data_shape)
            train_onehot = np.load(os.sep.join([self.saving_dir, 'Fold_{}_y_train.npy'.format(fold)]))
            le_encoder = LabelEncoder()
            y_train = le_encoder.fit_transform(train_onehot)
            X_test = np.load(os.sep.join([self.saving_dir, 'Fold_{}_X_test.npy'.format(fold)]))
            X_test = X_test.reshape(-1, self.data_shape)
            test_onehot = np.load(os.sep.join([self.saving_dir, 'Fold_{}_y_test.npy'.format(fold)]))
            le_encoder = LabelEncoder()
            y_test = le_encoder.fit_transform(test_onehot)
            weighted_f1_score = {}
            for clf_name, clf1 in self.classifiers.items():
                clf1.fit(train_data, y_train)
                pred = clf1.predict(X_test)
                true = y_test
                weighted_f1_score[clf_name] = (classification_report(true, pred, output_dict=True))['weighted avg'][
                    'f1-score']
                pd.DataFrame.from_dict(weighted_f1_score, orient='index').to_csv(
                    os.sep.join([self.without_smote_results_dir, 'fold_' + str(fold) + '.csv']))

    def model_running(self):
        analysis_groups = self.groups_to_analyse
        for fold in range(1, self.no_of_splits + 1):
            X_test = np.load(os.sep.join([self.saving_dir,'Fold_{}_X_test.npy'.format(fold)]))
            X_test = X_test.reshape(-1, self.data_shape)
            test_onehot = np.load(os.sep.join([self.saving_dir,'Fold_{}_y_test.npy'.format(fold)]))
            le_encoder = LabelEncoder()
            y_test = le_encoder.fit_transform(test_onehot)
            for analysis in analysis_groups:
                weighted_f1_score = {}
                Training_X = pd.read_pickle(
                    os.sep.join([self.oversampled_data_based_on_similarity_dir, 'fold_'+str(fold), analysis, 'all_data.pkl']))
                Training_y = pd.read_pickle(
                    os.sep.join([self.oversampled_data_based_on_similarity_dir, 'fold_'+str(fold), analysis, 'y_train.pkl']))
                for clf_name, clf1 in self.classifiers.items():
                    clf1.fit(Training_X, Training_y)
                    pred = clf1.predict(X_test)
                    true = y_test
                    weighted_f1_score[clf_name] = (classification_report(true, pred, output_dict=True))['weighted avg'][
                        'f1-score']
                    try:
                        os.makedirs(os.sep.join([self.model_results,'fold_'+str(fold)]))
                    except:
                        print('directory exists')
                    pd.DataFrame.from_dict(weighted_f1_score,orient='index').to_csv(os.sep.join([self.model_results,'fold_'+str(fold),analysis+'.csv']))
                for method in self.Groups[analysis].keys():
                    weighted_f1_score={}
                    Training_X=pd.read_pickle(os.sep.join([self.oversampled_data_based_on_similarity_dir,'fold_'+str(fold),method,'all_data.pkl']))
                    Training_y=pd.read_pickle(os.sep.join([self.oversampled_data_based_on_similarity_dir,'fold_'+str(fold),method,'y_train.pkl']))
                    for clf_name,clf1 in self.classifiers.items():
                        clf1.fit(Training_X, Training_y)
                        pred = clf1.predict(X_test)
                        true = y_test
                        weighted_f1_score[clf_name]=(classification_report(true, pred, output_dict=True))['weighted avg']['f1-score']
                        pd.DataFrame.from_dict(weighted_f1_score, orient='index').to_csv(
                            os.sep.join([self.model_results, 'fold_' + str(fold), method + '.csv']))

    ## creating oversampled data
    def oversampling_data(self):
        for i in range(1,self.no_of_splits+1):
            train_data = np.load(os.sep.join([self.saving_dir,'Fold_{}_X_train.npy'.format(i)]))
            train_data = train_data.reshape(-1, self.data_shape)
            train_onehot = np.load(os.sep.join([self.saving_dir,'Fold_{}_y_train.npy'.format(i)]))
            le_encoder = LabelEncoder()
            y_train = le_encoder.fit_transform(train_onehot)
            analysis_groups = self.groups_to_analyse
            for analysis in analysis_groups:
                group=self.Groups[analysis]
                X = []
                Y = []
                try:
                    os.makedirs(os.sep.join([self.oversampled_data_dir, 'fold_' + str(i), analysis]))
                except Exception as e:
                    print('Directory already exists')
                for method in group.keys():
                    oversampler = sv.MulticlassOversampling(group[method])
                    x, y = oversampler.sample(train_data, y_train)
                    X.extend(x)
                    Y.extend(y)
                    x=pd.DataFrame(x)
                    x['label']=y
                    x.to_csv(os.sep.join([self.oversampled_data_dir,'fold_'+str(i),analysis,str(method)+'.csv']),index=False)
                X = pd.DataFrame(X)
                X['label'] = Y
                X.to_csv(os.sep.join([self.oversampled_data_dir,'fold_'+str(i),analysis,str(analysis)+'.csv']),index=False)

    def similarity_values(self):
        analysis_groups = self.groups_to_analyse
        for fold in range(1, self.no_of_splits + 1):
            train_data = np.load(os.sep.join([self.saving_dir, 'Fold_{}_X_train.npy'.format(fold)]))
            train_data = train_data.reshape(-1, self.data_shape)
            train_onehot = np.load(os.sep.join([self.saving_dir, 'Fold_{}_y_train.npy'.format(fold)]))
            le_encoder = LabelEncoder()
            y_train = le_encoder.fit_transform(train_onehot)
            orig_data = pd.DataFrame(train_data)
            orig_data['label'] = y_train
            for analysis in analysis_groups:
                for oversampled_file in glob.glob(os.sep.join([self.oversampled_data_dir,'fold_'+str(fold),analysis,'*.csv'])):
                    X=pd.read_csv(oversampled_file)
                    for k in X['label'].unique():
                        print('running similarity score calculation for', 'label',k,'for fold',fold,'for group or method',str(os.path.basename(oversampled_file).split('.')[0]))
                        final_similarity = []
                        for i in tqdm(range(len(X.loc[X.label == k, :]))):
                            dist = []
                            for j in range(0, 10):
                                p = random.randint(0, int(len(X.loc[X.label == k, :])))
                                a = list(X.loc[X.label == k, X.columns != 'label'].iloc[p - 1, :])
                                b = list(orig_data.loc[orig_data.label == k, X.columns != 'label'].iloc[j, :])
                                dist.append(distance.euclidean(a, b))
                            a = np.mean(dist)
                            final_similarity.append(a)
                        try:
                            os.makedirs(os.sep.join([self.similarity_score_dir,'fold_' + str(fold),str(os.path.basename(oversampled_file).split('.')[0])]))
                        except:
                            print('directory exists')
                        pd.DataFrame(final_similarity).to_csv(os.sep.join([self.similarity_score_dir,'fold_' + str(fold),str(os.path.basename(oversampled_file).split('.')[0]),'label_'+str(k)+'.csv']))

    def oversampled_data_based_on_similarity(self):
        analysis_groups = self.groups_to_analyse
        for fold in range(1, self.no_of_splits + 1):
            for analysis in analysis_groups:
                files_in_group=glob.glob(os.sep.join([self.oversampled_data_dir, 'fold_' + str(fold), analysis, '*.csv']))
                for oversampled_file in files_in_group:
                    X = pd.read_csv(oversampled_file)
                    if os.path.basename(oversampled_file).startswith('Group'):
                        oversampled_data_to_consider = X['label'].value_counts().values.min() / len(files_in_group)
                    else:
                        oversampled_data_to_consider = X['label'].value_counts().values.min()
                    considerable_data = []
                    for m in X['label'].unique():
                        label_data = pd.read_csv(os.sep.join([self.similarity_score_dir, 'fold_' + str(fold),
                                    str(os.path.basename(oversampled_file).split('.')[0]), 'label_' + str(m) + '.csv'])).rename(columns={'Unnamed: 0': 'index', '0': 'similarity'}).sort_values(by='similarity')['index'][
                                     0:int(oversampled_data_to_consider)].index
                        label_data = (X.loc[X.label == m, X.columns != 'label'].iloc[label_data, :])
                        label_data['label'] = m
                        considerable_data.append(label_data)
                    all_data = pd.concat(considerable_data)
                    all_data = all_data.reset_index(drop=True)
                    y_train = all_data.pop('label')
                    try:
                        os.makedirs(os.sep.join([self.oversampled_data_based_on_similarity_dir, 'fold_' + str(fold),
                                                    str(os.path.basename(oversampled_file).split('.')[0])]))
                    except:
                        print('directory available')
                    all_data.to_pickle(os.sep.join([self.oversampled_data_based_on_similarity_dir, 'fold_' + str(fold),
                                                    str(os.path.basename(oversampled_file).split('.')[0]),'all_data.pkl']))
                    pd.DataFrame(y_train).to_pickle(os.sep.join([self.oversampled_data_based_on_similarity_dir, 'fold_' + str(fold),
                                                    str(os.path.basename(oversampled_file).split('.')[0]), 'y_train.pkl']))












