import numpy as np, pandas as pd
import seaborn as sns, matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.datasets import make_moons
import random
import sys
sys.executable


def euclidean_distance(point1, point2):
    point1 = np.array(point1)
    point2 = np.array(point2)
    return np.linalg.norm(point1 - point2)
def pearson_correlation_distance(point1, point2):
    point1 = np.array(point1)
    point2 = np.array(point2)
    
    # Calculate the means of each point
    mean1 = np.mean(point1)
    mean2 = np.mean(point2)
    
    # Calculate the Pearson correlation coefficient
    numerator = np.sum((point1 - mean1) * (point2 - mean2))
    denominator = np.sqrt(np.sum((point1 - mean1) ** 2) * np.sum((point2 - mean2) ** 2))
    
    # Handle division by zero case
    if denominator == 0:
        return 0
    else:
        return 1 - (numerator / denominator)
def create_array(rows, cols):
    # Calculate the total number of elements
    total_elements = rows * cols
    
    # Create an array filled with zeros
    array = np.zeros(total_elements)
    
    # Reshape the array to the desired shape
    array = array.reshape((rows, cols))
    
    return array
def min_of_each_column(array_2d):
    # Convert the list of lists to a NumPy array
    array_2d = np.array(array_2d)
    
    # Find the minimum element of each column
    min_values = np.min(array_2d, axis=0)
    
    return min_values
def max_of_each_column(array_2d):
    # Convert the list of lists to a NumPy array
    array_2d = np.array(array_2d)
    
    # Find the minimum element of each column
    min_values = np.max(array_2d, axis=0)
    
    return min_values
def generate_random_arr(min,max,clusternum):#to get random cluster centroid within specified range of data and returns it
    arr2=[]
    for j in range(clusternum):
        arr1=[]
        for i in range(len(min)):
            arr1.append(np.random.uniform(min[i], max[i]))

        arr2.append(arr1)
    return np.array(arr2)
def GUC_Distance ( Cluster_Centroids, Data_points, Distance_Type ):
# ## write code here for the Distance function here #
    arr=create_array(len(Data_points),len(Cluster_Centroids))
    for j in range(len(Data_points)):                
        for i in range(len(Cluster_Centroids)):
            if Distance_Type == "euclidean":
        
                arr[j][i]=euclidean_distance(Data_points[j],Cluster_Centroids[i])
                #print("distance between "+str(Data_points[j])+"and "+str(Cluster_Centroids[i])+"is "+str(euclidean_distance(Data_points[j],Cluster_Centroids[i])))
            else:
                arr[j][i]=pearson_correlation_distance(Data_points[j],Cluster_Centroids[i])
    return arr   

def find_minimum_index_in_each_row(arr):#this function is to get the cluster of each point in our data 
    #and it is returning an array containing cluster heads for each data point (accprding to their order in array)
    min_indices = []
    for row in arr:
        min_index = np.argmin(row)
        min_indices.append(min_index)
    return np.array(min_indices)
def getting_mean_square(min_index_arr,distance_arr,clusternum): # this one is returning mean square for every cluster using an array data type  
    #(every element is the mean square of corresponding cluster)
    total=0
    for k in range(clusternum):
        sum=0
        count=0
        for i in min_index_arr:
            if min_index_arr[i]==k:
                sum=sum+distance_arr[i][k]**2
                count=count+1
        total=total+sum/count
    return total
def mean_of_array_of_arrays(array_of_arrays): # this one is used to update the centroid of each cluster depending on the mean of its data points
    # Convert the list of lists to a NumPy array
    array_2d = np.array(array_of_arrays)

    # Calculate the mean along the rows (axis=0)
    mean_values = np.mean(array_2d, axis=0)
    
    return mean_values
def getting_new_centroids(min_index_arr,data,clusternum,old_centers): # this one is returning new centroid heads based on data in each cluster 
    #(array containing new clusetr heads)
    new_centroids=[]
    for k in range(clusternum):
        arr=[]
        for j in range(len(min_index_arr)):
            if min_index_arr[j]==k:
                arr.append(data[j])
        if (len(arr)>0):
            new_centroids.append(mean_of_array_of_arrays(arr))
        else:
            new_centroids.append(old_centers[k])
    return np.array(new_centroids)
def distortion(data, centroids, assignments): ## takes min_idx_Array(assignment),list of centroids,list of data and returns distortion of the whole data from their corresponding cluster heads
    total_distortion = 0
    for i in range(len(centroids)):
        cluster_points = [data[j] for j in range(len(data)) if assignments[j] == i]
        for point in cluster_points:
            total_distortion += euclidean_distance(point, centroids[i])**2
    return total_distortion

def GUC_Kmean ( Data_points, Number_of_Clusters,  Distance_Type ): 
        #this lines is for generating random cluster heads at the start of the algorithm
        min = min_of_each_column(Data_points)
        max = max_of_each_column(Data_points)
        old_clusters = generate_random_arr(min,max,Number_of_Clusters)
        print("arriv")
        prev_res = 1000000000
        ###
        while (True):
            Distance =GUC_Distance ( old_clusters, Data_points, Distance_Type )
            min_idx = find_minimum_index_in_each_row(Distance)
            new_res=distortion(Data_points,old_clusters,min_idx)
            if abs(new_res-prev_res) <= 1e-4 :
                break
            prev_res = new_res
            clusters=getting_new_centroids(min_idx,Data_points,Number_of_Clusters,old_clusters)
            old_clusters=clusters
            # Distance =GUC_Distance ( clusters, Data_points, Distance_Type )
            # min_idx = find_minimum_index_in_each_row(Distance)
            # prev_res=getting_mean_square(min_idx,Distance,Number_of_Clusters)
        SSE=new_res
        

        return [Data_points,min_idx, clusters , SSE ]   


    # cluster_arr=generate_random_arr(min_of_each_column(Data_points),max_of_each_column(Data_points),Number_of_Clusters)
    # GUC_Distance=GUC_Distance(cluster_arr,Data_points,Distance_Type)
    # minIdx=find_minimum_index_in_each_row(GUC_Distance)
    # CLUSTER_MEAN_SQR=getting_mean_square(minIdx,GUC_Distance,Number_of_Clusters)
def display_cluster(X, data_centers=None, data_labels=None, num_clusters=0):
    color = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#00FFFF', '#FF00FF', '#800000', '#008000', '#000080', '#808000', '#008080', '#800080']
    no_of_features = X.shape[1]
    plt.rcParams['figure.figsize'] = [10,3]

    if no_of_features != 2:
        for i in range(no_of_features-1):
            for j in range(i+1, no_of_features):
                fig,ax = plt.subplots()  
                if data_centers is not None and data_labels is not None:  
                    for k in range(num_clusters):
                        ax.scatter(data_centers[:, i][k], data_centers[:, j][k], c=color[k], marker='s', s=20)
                        ax.scatter(X[data_labels == k, i], X[data_labels == k, j], c=color[k], alpha=0.5, s=20)
                        ax.set_xlabel('Feature ' + str(i+1))
                        ax.set_ylabel('Feature ' + str(j+1))
                        
                    ax.set_title('Clustered Data with ' + str(num_clusters) + ' Clusters')
                else:
                     ax.scatter(X[:, i], X[:, j], c=color[0], alpha=0.5, s=20)
                     ax.set_xlabel('Feature ' + str(i+1))
                     ax.set_ylabel('Feature ' + str(j+1))
                     ax.set_title('Original Data')
                plt.show();
    
    else:  
        fig, ax = plt.subplots()  
        print("arrived1")
        if data_centers is not None and data_labels is not None:  
            for k in range(num_clusters):
                ax.scatter(data_centers[:, 0][k], data_centers[:, 1][k], c=color[k], marker='s', s=20)
                print("arrived2")
                ax.scatter(X[data_labels == k, 0], X[data_labels == k, 1], c=color[k], alpha=0.5, s=20)
                print("arrived2")
                ax.set_xlabel('Feature 1')
                ax.set_ylabel('Feature 2')
            print("arrived3")
            ax.set_title('Clustered Data with ' + str(num_clusters) + ' Clusters')
        else:
            ax.scatter(X[:, 0], X[:, 1], c=color[0], alpha=0.5, s=20)
            ax.set_xlabel('Feature 1')
            ax.set_ylabel('Feature 2')
            ax.set_title('Original Data')
        print("arrived4")
        plt.show();




plt.rcParams['figure.figsize'] = [8,8]
sns.set_style("whitegrid")
sns.set_context("talk")
# Produce a data set that represent the x and y o coordinates of a circle 
# this part can be replaced by data that you import froma file 
angle = np.linspace(0,2*np.pi,20, endpoint = False)
X_1 = np.append([np.cos(angle)],[np.sin(angle)],0).transpose()
for k in range(2,11):
    Data_points,cluster_assignments, centroids , SSE = GUC_Kmean(X_1,k,'pearson')
    display_cluster(X_1,centroids,cluster_assignments,num_clusters=k)

# min = min_of_each_column(X_1)
# max = max_of_each_column(X_1)
# clusters = generate_random_arr(min,max,3)# 2 for cluster num
# print("initial clusters are")
# print(clusters)
# Distance =GUC_Distance ( clusters, X_1, "euclidean" )
# min_idx = find_minimum_index_in_each_row(Distance)
# print("min idx is")
# print(min_idx)
# new_res=distortion(X_1,clusters,min_idx)
# print("new res")
# print(new_res)
# clusters=getting_new_centroids(min_idx,X_1,3)
# print("new clusters are")
# print(clusters)
# Distance =GUC_Distance ( clusters, X_1, "euclidean" )
# min_idx = find_minimum_index_in_each_row(Distance)
# print("min idx is")
# print(min_idx)
# new_res=distortion(X_1,clusters,min_idx)
# print("new res")
# print(new_res)
# clusters=getting_new_centroids(min_idx,X_1,3)
# print("new clusters are")
# print(clusters)
# Distance =GUC_Distance ( clusters, X_1, "euclidean" )
# min_idx = find_minimum_index_in_each_row(Distance)
# print("min idx is")
# print(min_idx)
# new_res=distortion(X_1,clusters,min_idx)
# print("new res")
# print(new_res)

# display_cluster(X_1,clusters,min_idx,num_clusters=0)
# display_cluster(X_1,clusters,min_idx,num_clusters=2)
# helper function that allows us to display data in 2 dimensions an highlights the clusters

# Data is displayed 
# to display the data only it is assumed that the number of clusters is zero which is the default of the fuction 
#display_cluster(X_1)
#cluster_assignments , example_centroids, example_SSE=GUC_Kmean( X_1, 2,  'euclidean' )
#display_cluster(X_1, data_centers=example_centroids, data_labels=cluster_assignments, num_clusters=2)
# print(mean_of_array_of_arrays([[0.42082277163707005, -0.127599467763853], [0.775901809800422, 0.9073270100394752]]))
# def GUC_Kmean ( Data_points, Number_of_Clusters,  Distance_Type ):

#     centroids = Data_points[np.random.choice(Data_points.shape[0], Number_of_Clusters, replace=False)];
#     old_centroids= np.zeros((Number_of_Clusters, Data_points.shape[0]));
#     cluster_distances = np.zeros((Data_points.shape[0], Number_of_Clusters));
#     cluster_assignments = np.zeros(Data_points.shape[0]);

#     for m in range(1000) :
#        if(np.array_equal(old_centroids, centroids)):
#            break;
#        old_centroids=np.copy(centroids);
#        cluster_distances=GUC_Distance(Data_points, centroids, Distance_Type);
#        cluster_assignments = np.argmin(cluster_distances, axis=0);
#        for i in range(Number_of_Clusters):
#         cluster_points=np.zeros(np.shape(Data_points));
#         index=0;
#         for j in range(len(cluster_assignments)):
#                 if(cluster_assignments[j]==i):
#                     cluster_points[index]=Data_points[j];
#                     index=index+1;         
#         if(index!=0):
#             summation=np.sum(cluster_points, axis=0);
#             mean=summation/index;
#             centroids[i]=mean;

#     data_with_assignments=np.insert(Data_points,Data_points.shape[1],cluster_assignments,axis=1);
#     SSE_elements=cluster_distances.min(axis=0);
#     SSE_elements=np.square(SSE_elements);
#     SSE=0;
#     l=0;
#     for element in SSE_elements:
#         l=l+1;
#         SSE=SSE+element;
#     SSE=SSE/l;
    
#     return [ Data_points,cluster_assignments, centroids , SSE ]    
# Data_points,cluster_assignments, centroids , SSE=GUC_Kmean(X_1,3,'euclidean')
# print(Data_points,cluster_assignments, centroids , SSE)
# display_cluster(Data_points,centroids,cluster_assignments,num_clusters=3)
