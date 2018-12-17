import pandas as pd
from sklearn.cluster import KMeans # not allowed
from sklearn.cluster import AgglomerativeClustering # not allowed
import matplotlib.pyplot as plt # not allowed
from sklearn.preprocessing import scale # not allowed
import argparse
import random
from scipy.spatial.distance import euclidean
from scipy.spatial.distance import pdist
from scipy.spatial import distance_matrix
import math
import numpy as np
import time
from Queue import PriorityQueue


# 'Clustering Script for CSV datasets with 4 Numeric Variables'
def main():
    # get dataset path, k and model
    parser = argparse.ArgumentParser(prog='Cluster')
    parser.add_argument('file_path', help='Location of data in csv format')
    parser.add_argument('K', help='Value for k, the number of clusters to form', type=int)
    opts = ["km", "ac", '2a', '2c', '2d', '3a']
    parser.add_argument('model', help='The model used to generate clusters', choices=opts)
    args = parser.parse_args()

    # read dataset into matrix
    data = pd.read_csv(args.file_path, sep=',', quotechar='"', header=0)
    data = data[['latitude', 'longitude', 'reviewCount', 'checkins']]
    data_m = data.as_matrix()

    if args.model == "km":
        plot_params = {'plotit': False}
        my_c, wc_sse = k_means( data_m , args.K, plot_params)
        # base_c = baseline_km( data_m , args.K)
        # compare_km_centroids(my_c, base_c)
    elif args.model == "ac":
        agglomerative(data_m , args.K)
        # baseline_ac(data_m , args.K)
    elif args.model == "2a":
        for i in range(5):
            QTwo(data_m, title='k_v_wcsee{}'.format(i) )
    elif args.model == "2c":
        # log transformation
        data_m[:,2] = np.log(data_m[:,2])
        data_m[:,3] = np.log(data_m[:,3])
        for i in range(5):
            QTwo(data_m, title='log_tansform_k_v_wcsee{}'.format(i) )
    elif args.model == "2d":
        data_m = scale(data_m)
        # k_means( data_m , 4)
        for i in range(5):
            QTwo(data_m, title='scale_k_v_wcsee{}'.format(i) )
    elif args.model == "3a":
        k = 4
        plot_params = {'plotit':True, 'x': 'latitude', 'y': 'longitude', 'show': True, 'title' : 'km_k_{}'.format(k) }
        k_means(data_m, k, plot_params=plot_params)
        plot_params = {'plotit':True, 'x': 'latitude', 'y': 'longitude', 'show': True, 'title' : 'ac_k_{}'.format(k) }
        baseline_ac(data_m , k, plot_params=plot_params)
    return


def QTwo(data_matrix, title='K vs WC-SSE', show=False):
    ks = [2, 4, 8, 16, 32, 64]
    results = []
    success = False
    for k in ks:
        while not success:
            try:
                plot_params= {'plotit': False}
                c, wc_sse = k_means(data_matrix , k, plot_params)
                results.append(wc_sse)
                success = True
            except IOError:
                print 'IO ERROR at k = {}'.format(k)
                continue
        success = False
    plt.plot(ks, results)
    plt.yscale('linear')
    plt.title(title)
    plt.grid(True)
    plt.savefig('plots/' +title)
    if show:
        plt.show()
    plt.clf()
    return

def compare_km_centroids(calculated, baseline):
    print ( '***Comparing calculated centers and baseline centers***')
    k = len(calculated)
    labels = [0]*k # red = new
    labels += [1]*k # blue = baseline
    data = np.append(calculated, baseline, axis=0)
    print 'Centroid Array: {}'.format(data)
    # plot(data, labels, 'Centroid Comparison' )
    return

def k_means(data_matrix, k,
    plot_params={'plotit':True,
        'x': 'latitude',
        'y': 'longitude',
        'show': True}):

    beginningTime = time.time()
    # print('***Applying K-Means with K={}***'.format(k))
    # print('Number of observations : {}'.format(len(data_matrix)))

    # Generate k random seeds from data
    centroids = []
    for i in range(0, k):
        centroids.append( data_matrix[ random.randint( 0, len(data_matrix)-1 )] )

    n = len(data_matrix)
    # Final label assignment
    labels = []
    for i in range(n):
        labels.append(-1)
    # error from point to its center
    sse = []
    for i in range(n):
        sse.append(float('inf'))
    wc_sse = 0
    old_wc_sse = float('inf')
    assignment_changed = True
    while assignment_changed:
        # make assignments based on centroids
        # recompute centroids using a mean of existing assignments
        # recompute within cluster sum squared distance
        assignment_changed = False
        # go over all points
        for i in range(n):
            min_dist = float('inf')
            min_i = -1
            # update closest centroid
            for j in range(k):
                try: # NaN value protection
                    ji_dist = euclidean(centroids[j], data_matrix[i] )
                except ValueError:
                    ji_dist = float('inf')
                if ji_dist < min_dist:
                    min_dist = ji_dist
                    closest = j
            if labels[i] != closest:
                assignment_changed = True
                labels[i] = closest
                sse[i] = min_dist**2
        # updating centroids
        for center in range(k):
            cluster = []
            # get all points assigned to center'th cluster
            for p in range(n):
                if(labels[p] == center):
                    cluster.append(data_matrix[p])
            centroids[center] = np.mean(cluster, axis=0)
    # no more reassignments left
    print( 'WC-SSE={}'.format( np.sum(sse) ) )
    for i in range(k):
        print( 'Centroid{}={}'.format(i+1, list(centroids[i]) ) )

    endTime = time.time() - beginningTime
    # print('K-Means runtime: {}'.format(endTime))
    if plot_params['plotit']:
        x = plot_params['x']
        y = plot_params['y']
        plot(data_matrix, labels, title=plot_params['title'], show=plot_params['show'], xlabel=x, ylabel=y )
    return centroids, np.sum(sse)

def baseline_km(data_matrix, k):
    # Baseline KM clusters using banned library
    model = KMeans(n_clusters=k)
    labels = model.fit_predict(data_matrix)
    print('Model intertia (WC-SSE): {}'.format( model.inertia_))
    for i in range(k):
        print( 'Base Centroid{}={}'.format(i+1, model.cluster_centers_[i] ))
    x = 'reviewCount'
    y = 'checkins'
    plot(data_matrix, labels, title='baseline_scale_{}_{}_km_k_{}'.format(x, y, k), show=True, xlabel=x, ylabel=y )
    return model.cluster_centers_


def agglomerative(data_matrix, k):
    # start all points as individual clusters
    # avg linkage: sum of distance for each point in X to each point in Y / (|X|*|Y|)
    n = len(data_matrix)
    pq = PriorityQueue()

    distance_m = distance_matrix( data_matrix, data_matrix)
    merged = []
    for i in range(n):
        dv = distance_m[i]
        min_d = np.min( dv[np.nonzero(dv)] )
        closest_i = np.where( dv == min_d  )
        # cluster index, current cluster, nearest neighbor, distance vector
        new_cluster = (i, [i], closest_i, np.square(dv) )
        pq.put( (min_d, new_cluster) )
    while pq.qsize() > k:
        c1 = pq.get()
        while c1[1] in merged:
            c1 = pq.get()
        # get the c1[2]'th cluster save as c2
        neighbor = c1[2]
        while not pq.empty():

        # merge c1 & c2
        # append c2[1] and c1[1]
        # update distance vector
        # update nearest neighbor
        # put c1c2 back in pq

        # what if another cluster refers to c1 or c2 as the nearest nighbor?
    return

def baseline_ac(data_matrix, k, plot_params={'plotit':True,
        'x': 'latitude',
        'y': 'longitude',
        'show': True}):
    # Baseline AC clusters using banned library
    model = AgglomerativeClustering(n_clusters=k, affinity='euclidean', linkage='average')
    labels = model.fit_predict(data_matrix)
    print('Model Params: ')
    print( model.get_params() )
    if plot_params['plotit']:
        x = plot_params['x']
        y = plot_params['y']
        plot(data_matrix, labels, title=plot_params['title'], show=plot_params['show'], xlabel=x, ylabel=y )
    return

def plot(data_matrix, labels, title = 'Plot', show = False, xlabel = 'latitude', ylabel = 'longitude'):
    try:
        x = ['latitude', 'longitude', 'reviewCount', 'checkins'].index(xlabel)
        y = ['latitude', 'longitude', 'reviewCount', 'checkins'].index(ylabel)
    except ValueError:
        print 'invalid labels: {} {}'.format(xlabel, ylabel)
        x = 0; y = 1
    xa = data_matrix[:,x]
    ya = data_matrix[:,y]
    plt.scatter(xa, ya, s=10, c=labels)
    plt.title(title)
    plt.xlabel('x-axis: ' + xlabel)
    plt.ylabel('y-axis: ' + ylabel)
    plt.savefig(title + '.png')
    if show:
        plt.show()
    plt.clf()
    return




if __name__ == '__main__':
    main()

