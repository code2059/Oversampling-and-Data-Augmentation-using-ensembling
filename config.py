import smote_variants as sv
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier


class Config:
    model_names = ["RBF SVM", "Decision Tree", "Random Forest", "Neural Net", "LDA", "LogReg", "SVC", "KNN"]
    no_of_splits = 5
    Groups = {'Group_1': {'ProWSyn': sv.ProWSyn(), 'AND_SMOTE': sv.AND_SMOTE(), 'SMOTE': sv.SMOTE()},
              'Group_2': {'G_SMOTE': sv.G_SMOTE(), 'Random_SMOTE': sv.Random_SMOTE()},
              'Group_3': {'SMOTE_TomekLinks':sv.SMOTE_TomekLinks(proportion=1.0),'VIS_RST':sv.VIS_RST()},
              'Group_4':{'CBSO':sv.CBSO(),'SMOBD':sv.SMOBD(),'A_SUWO':sv.A_SUWO()}}

    classifiers = {"RBF SVM":SVC(gamma=2, C=1, max_iter=1000), "Decision Tree":DecisionTreeClassifier(max_depth=5)
                      , "Random Forest":RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)
                      , "Neural Net":MLPClassifier(alpha=1, max_iter=1000), "LDA":LinearDiscriminantAnalysis()
                      , "LogReg":LogisticRegression(), "SVC":SVC(kernel="linear", C=0.025),"KNN":KNeighborsClassifier(n_neighbors=3)}

    ## setting up directories
    raw_data_dir=r'C:\Users\shubh\Desktop\Methods\raw_data'
    X_filename=r'X_4_feature_3_sec_Acc_m_+Gyr_m1_Scale.npy'
    y_filename=r'y_4_feature_3_sec_Acc_m_+Gyr_m1_Scale.npy'
    saving_dir=r'C:\Users\shubh\Desktop\Methods\k_fold_data'
    oversampled_data_dir=r'C:\Users\shubh\Desktop\Methods\oversampled_data'
    data_shape=16
    similarity_score_dir=r'C:\Users\shubh\Desktop\Methods\similarity_score'
    oversampled_data_based_on_similarity_dir=r'C:\Users\shubh\Desktop\Methods\oversampled_data_based_on_similarity'
    model_results=r'C:\Users\shubh\Desktop\Methods\model_results'
    without_smote_results_dir=r'C:\Users\shubh\Desktop\Methods\without smote results'
    final_result_dir=r'C:\Users\shubh\Desktop\Methods\final_result'
    # user inputs
    groups_to_analyse=['Group_1']
