import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest,r_regression,mutual_info_regression
from dcor import distance_correlation
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline



# wrap the distance_correlation function to be used in SelectKBest
def distance_correlation_score(X, y):
    return np.array([distance_correlation(X[:, i], y.squeeze()) for i in range(X.shape[1])])

def variance_score(X, *args):
    # y is not necessary
    # less variance, more important
    return 1/np.var(X, axis=0)

def get_select_k_best(f, k, x_df, y_df):
    selector = SelectKBest(f, k=k)
    x_new = selector.fit_transform(x_df, y_df)
    print("ok")
    x_new_df = pd.DataFrame(x_new, columns=x_df.columns[selector.get_support()])
    scores = list(zip(x_df.columns,selector.scores_))
    scores_df = pd.DataFrame(data = scores, columns=['Feat_names', 'Scores']).sort_values(by='Scores', ascending=False)
    return x_new_df, scores_df

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def greedy_search_k_best(model, x_df, y_df, k, method):
    selected_features = []
    removed_features = []
    scores = []
    if method == 'forward':
        remaining_features = list(x_df.columns)
        while len(selected_features) < k:
            best_score = np.inf
            best_feature = None
            for feature in remaining_features:
                current_features = selected_features + [feature]
                score = cross_val_score(model, x_df[current_features], y_df, cv=5, scoring=make_scorer(rmse)).mean()
                if len(selected_features)  == 0:
                    scores.append((feature,score))
                if score < best_score:
                    best_score = score
                    best_feature = feature
            if best_feature is None:
                break
            selected_features.append(best_feature)
            remaining_features.remove(best_feature)
            print(f"Selected {best_feature}, RMSE: {best_score:.4f}")
        scores_df = pd.DataFrame(data = scores, columns=['Feat_names', 'Scores']).sort_values(by='Scores', ascending=True)
        return selected_features, scores_df

    elif method == 'backward':
        current_features = list(x_df.columns)
        while len(current_features) > len(x_df.columns) - k:
            worst_score = -np.inf
            worst_feature = None
            for feature in current_features:
                temp_features = [f for f in current_features if f != feature]
                score = cross_val_score(model, x_df[temp_features], y_df, cv=5, scoring=make_scorer(rmse)).mean()
                if score > worst_score:
                    worst_score = score
                    worst_feature = feature
            if worst_feature is None:
                break
            current_features.remove(worst_feature)
            removed_features.append(worst_feature)
            print(f"Removed {worst_feature}, RMSE: {worst_score:.4f}")
        return removed_features,scores

    else:
        raise ValueError("Method should be either 'forward' or 'backward'")


if __name__ == '__main__':
    data_url = "http://lib.stat.cmu.edu/datasets/boston"
    raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
    data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
    target = raw_df.values[1::2, 2]
    x_df = pd.DataFrame(data, columns=["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT"])
    y_df = pd.DataFrame(target, columns=["MEDV"])
    # Data preprocessing: handle missing values and scaling
    x_df.fillna(x_df.mean(), inplace=True)
    y_df.fillna(y_df.mean(), inplace=True)
    # pearson
    print("Using Pearson:")
    x_new_df,scores_df = get_select_k_best(r_regression, 5, x_df, y_df)
    print(scores_df.sort_values(by='Scores', ascending=False))
    print(x_new_df.columns)
    # mutual_info_regression
    print("Using Mutual Information:")
    x_new_df,scores_df = get_select_k_best(mutual_info_regression, 5, x_df, y_df)
    print(scores_df.sort_values(by='Scores', ascending=False))
    print(x_new_df.columns)
    # distance_correlation
    print("Using Distance Correlation:")
    x_new_df,scores_df = get_select_k_best(distance_correlation_score, 5, x_df, y_df)
    print(scores_df.sort_values(by='Scores', ascending=False))
    print(x_new_df.columns)
    # variance
    print("Using inverse Variance:")
    x_new_df,scores_df = get_select_k_best(variance_score, 5, x_df, y_df)
    print(scores_df.sort_values(by='Scores', ascending=False))
    print(x_new_df.columns)
    # lightgbm
    print("Using LightGBM:")
    model_lgbm = LGBMRegressor(verbose=-1)
    print("Forward Selection:")
    best_features_forward_lgbm,scores = greedy_search_k_best(model_lgbm, x_df, y_df, k=5, method='forward')
    print("Best features (forward):", best_features_forward_lgbm)
    print("\nBackward Elimination:")
    best_features_backward_lgbm, _ = greedy_search_k_best(model_lgbm, x_df, y_df, k=5, method='backward')
    print("Best features (backward):", best_features_backward_lgbm)
    print("Embedding methods, RMSE as metric: ")
    print(scores[:5])