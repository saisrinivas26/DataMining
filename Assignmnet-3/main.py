# %%
from helpers import *
import datetime as dt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
import pickle
from collections import defaultdict
from sklearn.utils import shuffle
from sklearn.decomposition import PCA
from dateutil.relativedelta import relativedelta
import pickle_compat
pickle_compat.patch()
pd.options.mode.chained_assignment = None
carbs = 'BWZ Carb Input (grams)'
glucose = 'Sensor Glucose (mg/dL)'


cgm = pd.read_csv('CGMData.csv')
cgm = cgm[['Date', 'Time', glucose]]
cgm['datetime'] = cgm.apply(lambda x: pd.to_datetime(
    '{} {}'.format(x['Date'], x['Time'])), axis=1)
cgm = cgm.sort_values("datetime", ascending=True)
cgm.set_index('datetime', inplace=True)
cgm.head()

insulin = pd.read_csv('InsulinData.csv')
insulin = insulin[['Date', 'Time', carbs]]
insulin['datetime'] = insulin.apply(lambda x: pd.to_datetime(
    '{} {}'.format(x['Date'], x['Time'])), axis=1)

insulin = insulin.sort_values("datetime", ascending=True)
print('total {} entries'.format(len(insulin)))
insulin.dropna(inplace=True, axis=0)
insulin = insulin.loc[insulin[carbs] != 0.0]
insulin.head()


def process_meal_data(insulin):
    def is_meal_data(row):
        lastMealWas = abs(row.lastMealWas)
        nextMealIs = abs(row.nextMealIs)
        if lastMealWas >= 30 and nextMealIs >= 120:
            return 1
        else:
            return 0

    insulin['nextMealIs'] = insulin.datetime.diff(
        periods=-1).apply(lambda x: x / np.timedelta64(1, 'm')).fillna(0).astype('int64')
    insulin['lastMealWas'] = insulin.datetime.diff().apply(
        lambda x: x / np.timedelta64(1, 'm')).fillna(0).astype('int64')
    insulin['meal'] = insulin.apply(is_meal_data, axis=1)
    print('number of meal datapoints are {}'.format(insulin.meal.sum()))
    insulin = insulin.loc[insulin.meal==1]
    return insulin


insulin_meal = process_meal_data(insulin)
insulin_meal.sample(5)


def extract_meal_data(cgm, insulin_meal):
    import traceback
    meal_feats = pd.DataFrame()
    missed_feats = 0
    for , (, row) in enumerate(insulin_meal.iterrows()):
        start_meal = pd.Timestamp(row.Time) - dt.timedelta(minutes=30)
        end_meal = pd.Timestamp(row.Time) + dt.timedelta(minutes=120)
        
        # for the same day, select all cgm points which lie within the range
        meal_data = cgm.loc[cgm.Date == row.Date].between_time(
            start_meal.strftime('%H:%M:%S'),
            end_meal.strftime('%H:%M:%S')
        )
        try:
            missing = meal_data[glucose].isna().sum()
            if missing > 0.2 * len(meal_data):
                # print('skipping because {} data is missing'.format(missing))
                missed_feats += 1
                continue
            meal_data[glucose].interpolate(method='quadratic', inplace=True)
            features = extract_features(meal_data[glucose].values)
            if not features:
                missed_feats += 1
                # print('skipping because no features')
                continue
            
            # add the carbs level here to features
            features['carbs'] = row[carbs]
            
            fdf = pd.DataFrame(features)
            meal_feats = meal_feats.append(fdf)
        except Exception as e:
            print('-'*100)
            print(meal_data[glucose].values)
            traceback.print_exc()
            # exit(-1)
            pass
        
    print('skipped {} possible data points'.format(missed_feats))
    # print('meal')
    # print(meal_feats.describe().transpose())
    # print(meal_feats.isna().sum())


    meal_dropped = meal_feats.dropna(axis=0)

    # import matplotlib.pyplot as plt
    # meal_dropped['max_difference_cgm'].plot.hist()
    # plt.show()
    meal_filter = (40 <= meal_dropped['max_difference_cgm']) & (meal_dropped['max_difference_cgm']  <= 100)
    # meal_filter = meal_dropped['max_difference_cgm'] >= 0
    features_df = meal_dropped.loc[meal_filter]

    print('total features', len(features_df))
    return features_df

# %%
meal_data = extract_meal_data(cgm, insulin_meal)
print('len meal datapoints', len(meal_data))
meal_data.head()

# %%
min_carb = int(meal_data.carbs.min())
max_carb = int(meal_data.carbs.max())
bin_range = 20
print('min carbs {} and max carb {}'.format(min_carb, max_carb))
edges = [x for x in range(min_carb-1, max_carb+bin_range, bin_range)]
print(edges)

meal_data['bin'] = pd.cut(meal_data.carbs, edges)
meal_data.head()

# %%
meal_data['bin'] = meal_data.bin.cat.codes
print(meal_data.bin.unique())
meal_data.head()

# %%
X = meal_data.drop(['bin','carbs'],axis=1)
X.head()

# %%
y = meal_data[['bin']]
y.head()

# %%
X_normalized = X.apply(normalize_np, axis=0)

# %%
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_normalized)

# %%
# import matplotlib.pyplot as plt
# plt.scatter(X_pca[::,0], X_pca[::,1])
# plt.show()

# %%
kmeans = KMeans(n_clusters = len(edges)-1).fit(X_normalized)

# %%
def compute_sse(centers, num_classes, X, labels):
    cluster_centers = list(centers)
    # print(cluster_centers)
    clusterwise_sse = [0 for _ in range(num_classes)]
    for point, label in zip(X, labels):
        # print(point, cluster_centers[label])
        clusterwise_sse[label] += np.square(point - cluster_centers[label]).sum()
    total_sse = np.sum(clusterwise_sse)
    return total_sse

# %%
def compute_entropy_purity_matrix(labels, ground_truths, num_classes):
    matrix = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    if len(ground_truths)!= len(labels):
        print('invalid lengths of labels and ground_truths', len(labels), len(ground_truths))
        return matrix
    
    for i in range(len(ground_truths)):
        matrix[int(labels[i])][int(ground_truths[i])] += 1
    
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if matrix[i][j] == 0:
                matrix[i][j] = 1
    
    return matrix

# %%
import math
def compute_entropy_purity(matrix):
    entropies = [0 for _ in matrix] 
    purities = [0 for _ in matrix] 
    weighted_entropy = 0
    weighted_purity = 0
    num_samples = 0
    for i in range(len(matrix)):
        sum_pj = np.sum(matrix[i])
        # todo: check this because we initialize with 1 instead of 0
        num_samples += sum_pj
        for j in range(len(matrix[i])):
            frac = matrix[i][j] / float(sum_pj)
            purities[i] = max(purities[i], frac)
            entropies[i] += (-1 * frac * math.log(frac))
        weighted_entropy += sum_pj * entropies[i]
        weighted_purity += sum_pj * purities[i]
    total_entropy = weighted_entropy / num_samples
    total_purity = weighted_purity / num_samples
    return total_entropy, total_purity

# %%
import random
dbscan = DBSCAN(eps=.155).fit(X_pca)
max_dbscan = max(dbscan.labels_)
label_range = [i for i in range(len(edges)-1)]
db_labels = [x if x != -1 else random.choice(label_range) for x in dbscan.labels_ ]
print(dbscan.labels_, max(dbscan.labels_), max(db_labels), len(edges)-1)
half = 0.5
# %%
def get_dbscan_cluster_centers(X, labels):
    clusters = defaultdict(list)
    for x, y in zip(X, labels):
        clusters[y].append(x)
    
    cluster_centers = []
    for key in sorted(clusters):
        cluster_centers.append(np.mean(clusters[key], axis=0))
    return cluster_centers

# %%
dbscan_cluster_centers = get_dbscan_cluster_centers(X_pca, db_labels)
dbscan_sse = compute_sse(dbscan_cluster_centers, len(edges)-1, X_pca, db_labels)
print('total dbscan sse is', dbscan_sse)
db_matrix = compute_entropy_purity_matrix(db_labels, y.values, len(edges)-1)
print(db_matrix)
db_entropies, db_purities = compute_entropy_purity(db_matrix)
print('entropy is', db_entropies)
print('purity is', db_purities)



kmeans_sse = compute_sse(kmeans.cluster_centers_, len(edges)-1, X_normalized.values, kmeans.labels_)
print('total kmeans sse is', kmeans_sse)
matrix = compute_entropy_purity_matrix(kmeans.labels_, y.values, len(edges)-1)
print(matrix)
entropies, purities = compute_entropy_purity(matrix)
print('entropy is', entropies)
print('purity is', purities)
# %%
output= {
    0: [kmeans_sse],
    1: [dbscan_sse],
    2: [entropies],
    3: [db_entropies],
    4: [purities],
    5: [db_purities],
}
odf = pd.DataFrame(output)
odf.to_csv('Results.csv', header=False, index=False,)