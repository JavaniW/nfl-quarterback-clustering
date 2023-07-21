import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
# import sklearn.cluster.hierarchical as hclust
from sklearn import preprocessing
import seaborn as sns

quarterback_stats = pd.read_csv("dataset/quarterback_stats.csv")
features = quarterback_stats.drop(["name", "win_percentage"], axis=1)

scalar = preprocessing.MinMaxScaler()
features_normal = scalar.fit_transform(features)

kmeans = KMeans(4, init="random", n_init='auto').fit(features_normal)
labels = pd.DataFrame(kmeans.labels_)
labeled_quarterbacks = pd.concat((features,labels),axis=1)
labeled_quarterbacks = labeled_quarterbacks.rename({0:'labels'},axis=1)
sns.pairplot(labeled_quarterbacks,hue='labels', palette="bright")
plt.savefig("output/clusters.png", bbox_inches="tight")
plt.show()

labeled_quarterbacks['Constant'] = "Data"
f, axes = plt.subplots(1, 5, figsize=(20, 25), sharex=False)
f.subplots_adjust(hspace=0.2, wspace=0.7)
for i in range(0,len(list(labeled_quarterbacks))-2):
    col = labeled_quarterbacks.columns[i]
    ax = sns.swarmplot(x=labeled_quarterbacks['Constant'],y=labeled_quarterbacks[col].values,hue=labeled_quarterbacks['labels'],ax=axes[i], palette="bright")
    ax.set_title(col)
plt.savefig("output/swarm-plots")
plt.show()

quarterbacks = quarterback_stats[['name', 'win_percentage']]
quarterbacks = pd.concat((quarterbacks,labels),axis=1)
quarterbacks = quarterbacks.rename({0:'Cluster'},axis=1)
sorted_quarterbacks = quarterbacks.sort_values(['Cluster'])
averages = sorted_quarterbacks[["Cluster", "win_percentage"]].groupby("Cluster", as_index=False).mean().rename(columns = {"win_percentage": "Average Win Percentage"})
averages["Average Win Percentage"] = averages["Average Win Percentage"].apply(lambda x: round((x * 100), 2))
averages.to_csv("output/clusters_avg_win_percentage.csv", index=False)

pd.set_option('display.max_rows', 1000)
sorted_quarterbacks.to_csv("output/labels.csv", index=False)
print(sorted_quarterbacks)