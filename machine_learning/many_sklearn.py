import os
import time
import random
import numpy as np
from sklearn import metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, roc_auc_score, r2_score, make_scorer
from sklearn import tree
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Lasso, LogisticRegression, ElasticNet
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import VotingClassifier, VotingRegressor
from sklearn.feature_selection import RFECV, RFE
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV


def log_info(*msg):
    print('[' + time.asctime(time.localtime(time.time())) + ']', *msg)


def auc_score(y_true, y_pred):
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred)
    auc = metrics.auc(fpr, tpr)
    return auc


def train_eval(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    try:
        y_pred_train = model.predict_proba(X_train)[:, 1]
    except:
        y_pred_train = model.predict(X_train)
    auc_train = auc_score(y_train, y_pred_train)
    try:
        y_pred_test = model.predict_proba(X_test)[:, 1]
    except:
        y_pred_test = model.predict(X_test)
    auc_test = auc_score(y_test, y_pred_test)
    scores = {
        'train_auc': auc_train,
        'test_auc': auc_test
    }
    return scores, y_pred_test


def run_exp(seed, X, y):
    random.seed(seed)
    np.random.seed(seed)
    #
    inds = list(range(X.shape[0]))
    random.shuffle(inds)
    inds = np.array(inds)
    num_folds = 5
    offset = (X.shape[0] + num_folds - 1) // num_folds
    fold_inds_list = [inds[i * offset : (i + 1) * offset] for i in range(num_folds)]
    assert np.all(np.array(sorted(np.concatenate(fold_inds_list).tolist())) == np.arange(X.shape[0]))
    test_X_list = [X[fold_inds] for fold_inds in fold_inds_list]
    test_y_list = [y[fold_inds] for fold_inds in fold_inds_list]
    train_X_list = []
    train_y_list = []
    for i, fold_inds in enumerate(fold_inds_list):
        train_inds = np.array(list(set(np.arange(X.shape[0])) - set(fold_inds)))
        train_X_list.append(X[train_inds])
        train_y_list.append(y[train_inds])
        assert len(set(train_inds) & set(fold_inds)) == 0
        print('Fold', i, 'with train size', len(train_inds), 'and test size', len(fold_inds))
    # For Lasso
    def scoring_roc_auc(y, y_pred):
        try:
            return roc_auc_score(y, y_pred)
        except:
            return 0.5
    robust_roc_auc = make_scorer(scoring_roc_auc)
    param_grid={
        # 'alpha' : [0.022, 0.021, 0.02, 0.019, 0.023, 0.024, 0.025, 0.026, 0.027, 0.029, 0.031],
        # 'tol'   : [0.0013, 0.0014, 0.001, 0.0015, 0.0011, 0.0012, 0.0016, 0.0017]      
        'alpha' : [0.010, 0.011, 0.012, 0.013, 0.014, 0.015, 0.016],
        'tol': [0.0001, 0.0004, 0.008, 0.0012]
    }
    #
    model2score = {}
    models = [
        # ('AdaBoost_5', AdaBoostClassifier(n_estimators=5)),
        # ('AdaBoost_10', AdaBoostClassifier(n_estimators=10)),  # good
        # ('AdaBoost_100', AdaBoostClassifier(n_estimators=100)),
        # ('AdaBoost_10_SAMME', AdaBoostClassifier(n_estimators=10, algorithm='SAMME')),
        
        # ('GradBoost_10_1.0', GradientBoostingClassifier(n_estimators=10, learning_rate=1.0, max_depth=1, random_state=0)),  # good
        # ('GradBoost_10_1.0_2', GradientBoostingClassifier(n_estimators=10, learning_rate=1.0, max_depth=2, random_state=0)),
        # ('GradBoost_100_1.0_1', GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)),
        # ('GradBoost_100_1.0_2', GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=2, random_state=0)),
        # ('GradBoost_1_1.0_1', GradientBoostingClassifier(n_estimators=1, learning_rate=1.0, max_depth=1, random_state=0)),
        # ('GradBoost_4_1.0_1', GradientBoostingClassifier(n_estimators=4, learning_rate=1.0, max_depth=1, random_state=0)),
        # ('GradBoost_8_1.0_1', GradientBoostingClassifier(n_estimators=8, learning_rate=1.0, max_depth=1, random_state=0)),
        # ('GradBoost_10_0.1_1', GradientBoostingClassifier(n_estimators=10, learning_rate=0.1, max_depth=1, random_state=0)),
        # ('GradBoost_10_0.01_1', GradientBoostingClassifier(n_estimators=10, learning_rate=0.01, max_depth=1, random_state=0)),
        
        # ('HistGradBoost_10', HistGradientBoostingClassifier(max_iter=10)),
        # ('HistGradBoost_100', HistGradientBoostingClassifier(max_iter=100)),

        # ('DT', tree.DecisionTreeClassifier()),  # bad
        # ('RF_10', RandomForestClassifier(n_estimators=10, random_state=1)),
        # ('RF_100', RandomForestClassifier(n_estimators=100, random_state=1)),

        # ('Vote_soft', VotingClassifier(estimators=(('AdaBoost_10', AdaBoostClassifier(n_estimators=10)), ('GradBoost_10_1.0', GradientBoostingClassifier(n_estimators=10, learning_rate=1.0, max_depth=1, random_state=0)), ('HistGradBoost_10', HistGradientBoostingClassifier(max_iter=10)), ('LogistReg', LogisticRegression(solver="liblinear"))), voting='soft')),  # good
        # ('Vote_hard', VotingClassifier(estimators=(('AdaBoost_10', AdaBoostClassifier(n_estimators=10)), ('GradBoost_10_1.0', GradientBoostingClassifier(n_estimators=10, learning_rate=1.0, max_depth=1, random_state=0)), ('HistGradBoost_10', HistGradientBoostingClassifier(max_iter=10))), voting='hard')),  # bad
     
        # ('AdaBoost_reg_10', AdaBoostRegressor(n_estimators=10)),  # bad
        # ('GradBoost_reg_10_1.0', GradientBoostingRegressor(n_estimators=10, learning_rate=1.0, max_depth=1, random_state=0)),
        # ('RF_reg_10', RandomForestRegressor(n_estimators=10)), # bad

        # ('Lasso_cyclic', Lasso(alpha=0.031, tol=0.01, selection='cyclic')),  # good
        ('Lasso_random', Lasso(alpha=0.014, tol=0.0017, selection='random')),  # best
        # ('LogistReg', LogisticRegression(solver="liblinear")),
        # ('SVC', SVC()),  # bad
        # ('Elastic', ElasticNet()),
        # ('NB', GaussianNB()),  # bad

        # ('GridSearch', GridSearchCV(Lasso(alpha=0.031, tol=0.01, random_state=seed, selection='random'), param_grid=param_grid, verbose=0, n_jobs=-1, scoring=robust_roc_auc, cv=10)),  # good

        # ('Vote_reg', VotingRegressor([('Lasso_random', Lasso(alpha=0.014, tol=0.0017, selection='random')), ('Elastic', ElasticNet()), ('GradBoost_reg_10_1.0', GradientBoostingRegressor(n_estimators=10, learning_rate=1.0, max_depth=1, random_state=0))])),


    ]
    for name, model in models:
        log_info(model)
        all_test_y = []
        all_pred_y = []
        for i, (train_X, train_y, test_X, test_y) in enumerate(zip(train_X_list, train_y_list, test_X_list, test_y_list)):
            scores, pred_y = train_eval(model, train_X, train_y, test_X, test_y)
            log_info('Fold', i, scores)
            all_test_y.extend(test_y.tolist())
            all_pred_y.extend(pred_y.tolist())
        assert len(all_test_y) == len(all_pred_y) == X.shape[0]
        #
        all_test_y = np.array(all_test_y)
        all_pred_y = np.array(all_pred_y)
        auc = auc_score(all_test_y, all_pred_y)
        thres = 0.5
        eps = 1e-12
        all_pred_y = (all_pred_y > thres).astype(all_test_y.dtype)
        TP = np.sum(all_pred_y * all_test_y)
        FP = np.sum(all_pred_y * (1 - all_test_y))
        FN = np.sum((1 - all_pred_y) * all_test_y)
        TN = np.sum((1 - all_pred_y) * (1 - all_test_y))
        TPR = TP / (TP + FN + eps)
        SPC = TN / (TN + FP + eps)
        ACC = np.mean(all_pred_y == all_test_y)
        #
        model2score[name] = {
            'AUC': auc,
            'TPR': TPR,
            'SPC': SPC,
            'ACC': ACC,
        }
        log_info(model2score[name])
        print()
    return model2score


def main():
    X = []
    y = []
    with open('merge_cleared_with_condition.csv', 'r') as reader:
        lines = reader.readlines()
    head_line = lines[0]
    _, age, _, _, _, u_score, _, _, _, bad_ending, *_, c1, c2, c3, c4, c5 = head_line.strip().split(',')
    log_info('Feats:', age, u_score, c1, c2, c3, c4, c5)
    log_info('Label:', bad_ending)
    lines = lines[1 :]
    for line in lines:
        _, age, _, _, _, u_score, _, _, _, bad_ending, *_, c1, c2, c3, c4, c5 = line.strip().split(',')
        X.append([int(age), int(u_score), int(c1), int(c2), int(c3), int(c4), int(c5)])
        # X.append([int(c1), int(c2), int(c3), int(c4), int(c5)])
        # X.append([int(c1)])
        # X.append([int(u_score)])
        y.append(int(bad_ending))
    X = np.array(X)
    y = np.array(y)
    #
    seeds = [1, 5, 13, 41, 42, 43, 233, 255, 349, 488, 628, 666, 686, 784, 799, 822, 937, 946, 988, 65535]
    model2scores = {}
    for seed in seeds:
        model2score = run_exp(seed, X, y)
        for model, score in model2score.items():
            if not model in model2scores:
                model2scores[model] = {}
            for key, value in score.items():
                if not key in model2scores[model]:
                    model2scores[model][key] = []
                model2scores[model][key].append(value)
    for model, scores in model2scores.items():
        for key, values in scores.items():
            log_info(f'{model}\t{key}\tmean {round(np.mean(values), 4)}\tstd {round(np.std(values), 4)}')


if __name__ == '__main__':
    main()
