#Clusters / lists of patient_IDs
#cluster_algo = AgglomerativeClustering(n_clusters=n_clus, linkage='single')

import numpy as np
import pandas as pd
import math, statistics

def jaccard_similarity(list1, list2):
    s1 = set(list1)
    s2 = set(list2)
    return len(s1.intersection(s2)) / len(s1.union(s2))

def clusters(clustering, patient_ids):
    clusters = []
    for j in range(len(np.unique(clustering.labels_))):
        clus = []
        for i in range(len(clustering.labels_)):
            if clustering.labels_[i] == j:        #Extract patient numbers
                clus.append(patient_ids.to_numpy()[i])
        clusters.append(clus)
    return clusters

def clusterboot(clus, cluster_algo, info, patient_ids, frac):
    i = 0
    runs = 100
    n_clus = len(clus)
    bootresult = [ [ None for _ in range(runs) ] for _ in range(n_clus)]
    bootmean = []
    while i < runs:
        info_boot = pd.DataFrame()
        info['patient_ids'] = patient_ids
        cols = info.columns
        while info_boot.empty:
            info_temp = info
            samp = info.sample(frac=frac, axis='rows') #frac = fraction of original data to be replaced
            replace = info.loc[~info.index.isin(samp.index)].sample(samp.shape[0])
            info_temp.loc[replace.index] = samp.values
            info_boot = info_temp
        clustering = cluster_algo.fit(info_boot.drop(['patient_ids'], axis=1))
        cluster_boot = clusters(clustering, patient_ids)
        for k in range(n_clus):
            bootresult[k][i] = max(jaccard_similarity(clus[k], j) for j in cluster_boot)

        i+= 1

    for h in range(n_clus):
        bootmean.append(statistics.mean(bootresult[h]))

    return bootmean