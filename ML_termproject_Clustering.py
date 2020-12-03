import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, Normalizer, StandardScaler, RobustScaler
from sklearn.cluster import DBSCAN, KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples, silhouette_score


# Read the dataset
indicators_df = pd.read_csv('Indicators.csv')
series_df = pd.read_csv('Series.csv')

# Preprocessing
# Find topics related to 'Health'
# Print topic list
# topic_list = series_df['Topic'].unique()
# print(topic_list)

health_topics = series_df[series_df['Topic'].str.contains('Health|health')]
health_indicator_list = health_topics['IndicatorName'].unique()

# Drop indicators that don't include the 'Health' topic
health_indicators = indicators_df[indicators_df['IndicatorName'].isin(health_indicator_list)]
health_indicators = health_indicators.drop(['CountryCode', 'IndicatorCode'], axis=1)
health_indicators = health_indicators.dropna(axis=1)        # No NaN value

# Restructure the dataframe
restructured_df = pd.DataFrame(index=health_indicators['CountryName'].unique(),
                               columns=health_indicators['IndicatorName'].unique())

# Use the most recent data between 2011 and 2015
lower_year = 2011
health_indicators = health_indicators[health_indicators['Year'] >= lower_year]
health_indicators = health_indicators.drop_duplicates(subset=['CountryName', 'IndicatorName'], keep='last')

for row in health_indicators.itertuples():
    restructured_df[row.IndicatorName].loc[restructured_df.index == row.CountryName] = row.Value

# Remove columns which have too many NaN. (NaN proportion >= 0.11)
restructured_df = restructured_df[restructured_df.columns[(((restructured_df.isna() == True).sum() /
                                                            len(restructured_df)) <= 0.11)].to_numpy()]

# Replace NaN value with the 'mean' OR 'median'  of each rows
for col in restructured_df.columns:
    restructured_df[col].fillna(restructured_df[col].mean(), inplace=True)
    # restructured_df[col].fillna(restructured_df[col].median(), inplace=True)

# Scaling
scaler = MinMaxScaler()
scaled_df = scaler.fit_transform(restructured_df)
scaled_df = pd.DataFrame(data=scaled_df, columns=restructured_df.columns)

# DBSCAN

eps = [0.05, 0.1, 0.2, 0.5]
min_samples = [3, 5, 10, 15, 20, 30, 50, 100]


def dbscan(df):
    labels = list()

    for n in eps:
        for m in min_samples:
            db = DBSCAN(eps=n, min_samples=m).fit(df)
            labels.append(db.labels_)

    return labels


# k-means
n_clusters = [2, 3, 4, 5, 6]
Kmax_iter = [50, 100, 200, 300]


def kMeans(df):
    labels = list()

    for n in n_clusters:
        for m in Kmax_iter:
            kmeans = KMeans(n_clusters=n, max_iter=m)
            labels.append(kmeans.fit_predict(df))

    return labels


# EM
n_components = [2, 3, 4, 5, 6]
EMmax_iter = [50, 100, 200, 300]


def EM(df):
    labels = list()

    for n in n_components:
        for m in EMmax_iter:
            gmm = GaussianMixture(n_components=n, max_iter=m)
            labels.append(gmm.fit_predict(df))

    return labels


dbscan_labels = dbscan(scaled_df)
kMeans_labels = kMeans(scaled_df)
EM_lables = EM(scaled_df)


# Visualization: PCA
def plot_pca(df, label):
    pca = PCA(n_components=2)
    # pca_result = pca.fit_transform(df.drop(['label'], axis=1, inplace=False))
    pca_result = pca.fit_transform(df)
    result_df = pd.DataFrame(data=pca_result, columns=['pca1', 'pca2'])

    return result_df


# Evaluation: silhouette
def silhouette(df, labels):
    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed clusters
    silhouette_avg = silhouette_score(df, labels)
    return silhouette_avg


# Result: plot pca and print the average of silhouette score
def result(modelName, df, labels, param1, param2):
    count = 1
    for i, n in enumerate(param1):
        for j, m in enumerate(param2):
            plt.subplot(len(param1), len(param2), count)
            result_df = plot_pca(df, labels)
            plt.scatter(result_df['pca1'], result_df['pca2'], c=labels[count-1], s=3)
            try:
                sil = silhouette(df, labels[count - 1])
            except ValueError as e:  # 'metric': 'hamming' and 'algorithm': 'kd_tree' cannot coexist.
                plt.title("{}, {}, sil='error'".format(n, m))
            else:
                plt.title("{}, {}, sil={}".format(n, m, '%.3f' % sil))
            count += 1
    plt.suptitle(modelName, fontsize=30)
    plt.show()


result("DBSCAN", scaled_df, dbscan_labels, eps, min_samples)
result("k-means", scaled_df, kMeans_labels, n_clusters, Kmax_iter)
result("EM", scaled_df, EM_lables, n_components, EMmax_iter)
