from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score
from sklearn import linear_model
from sklearn.grid_search import GridSearchCV
from sklearn import svm

from transfer_learning import DataLoader
from augmenter import Augmentation


class predictors(object):

    def __init__(self,load = True, model='ResNet50',
        x_eval = None,
        y_eval = None):
        self.model = model

    @staticmethod
    def accuracy(matrix):
        return (np.trace(matrix)) * 1.0 / np.sum(matrix)

    @staticmethod
    def pca(train_data_flat, num_features):
            pca = PCA(n_components = num_features)
            pca.fit(train_data_flat)
            train_data_flat_pca = pca.transform(train_data_flat)
            print(train_data_flat_pca.shape)
            return train_data_flat_pca	

    @staticmethod
    def eval_metrics(clf, x_eval=None  ,y_eval = None ):
        pred = clf.predict(x_eval)
        print(confusion_matrix(y_eval, pred))
        print(predictors.accuracy(confusion_matrix(y_eval, pred)))
        print('f1-score micro: {}'.format(f1_score(y_eval, pred, average= 'micro')))
        print('f1-score macro: {}'.format(f1_score(y_eval, pred, average= 'macro')))
        print('f1-score weighted: {}'.format(f1_score(y_eval, pred, average= 'weighted')))

    @staticmethod
    def lr(train_data, label):
            print('Logistic Regression\n')
            logistic_clf = linear_model.LogisticRegression(penalty="l2", 
                class_weight="balanced")
            logistic_clf.fit(train_data, label)
            return logistic_clf
    
    @staticmethod
    def svm(train_data, train_labels_augmented):
            svc = svm.SVC(C=0.5, kernel='linear')
            param_grid = [
                        {'C': [0.5, 1, 5], 'kernel': ['linear']},
                        {'C': [0.1, 1, 5], 'gamma': [0.001], 'kernel': ['rbf']},
                     ]
            kernel = ['linear', 'rbf']
            Cs = [0.1, 0.3, 1]    
            clf = GridSearchCV(estimator=svc,
            	param_grid=param_grid, 
            	cv = 10,
            	n_jobs = 2)

            clf.fit(train_data, train_labels_augmented)
            print('___________')
            print('\nSVM:\n')
            print('best_score_:',clf.best_score_)
            print('best_C:',clf.best_estimator_.C)
            print('best kernel:',clf.best_estimator_.kernel)
            print('best set of parameters:{}'.format(clf.best_params_))
    
    @staticmethod
    def svm_best(train_data, label):
        from sklearn import svm
        clf = svm.SVC(C=5, kernel='rbf', gamma = 0.001)
        clf.fit(train_data, label)
        return clf
    
    @staticmethod
    def random_forest(X, y):
            print('\nRandom Forest\n')
            k_fold = 10
            kf_total = KFold(n_splits=k_fold)
            forest = RandomForestClassifier(n_estimators=250,
                                                                        random_state=0)
            #estimators_list = [50, 100, 150, 250, 500, 800, 1000]  
            estimators_list = [50, 150, 500]  
            clf_forest = GridSearchCV(estimator=forest, 
                param_grid=dict(n_estimators=estimators_list, 
                    warm_start=[True, False]), 
                cv=k_fold, n_jobs=-1)
            cms = [confusion_matrix(eval_labels_aug, 
                clf_forest.fit(X,y).predict(eval_data_flat_pca)) for train, test in kf_total.split(X)]
            accuracies = []
            for cm in cms:
                    accuracies.append(accuracy(cm))
            print(accuracies)
            print(np.mean(accuracies))
    
    @staticmethod
    def calls(train_data_flat_pca,train_labels_aug,x_eval = None,y_eval = None):

        #clf = predictors.lr(train_data_flat_pca, train_labels_aug)
        #predictors.eval_metrics(clf,x_eval = x_eval, y_eval = y_eval)
        cv_results_ = predictors.svm(train_data_flat_pca, train_labels_aug) # gridsearch SVM later
        
        #clf = predictors.svm_best(train_data_flat_pca, train_labels_aug) # update best svm params
        																  # after gridsearch
        #predictors.eval_metrics(clf,x_eval = x_eval, y_eval = y_eval)
        #predictors.random_forest(train_data_flat_pca, train_labels_aug)