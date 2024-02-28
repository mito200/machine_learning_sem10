import numpy as np, pandas as pd
import seaborn as sns, matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.datasets import make_moons
import random
import sys
sys.executable


#    return Cluster_Distance 
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
    min_indices = []
    for row in arr:
        min_index = np.argmin(row)
        min_indices.append(min_index)
    return min_indices


# data=[[3,4],[5,6],[1,2]]
# arr=[]
# for i in range(len(data)):
#     arr.append(data[i])
    
# random_int_array_2d = np.random.randint([1,2], [5,6], size=(3, 3))
# print(random_int_array_2d)
def column_ranges(array_of_arrays):
    # Convert the list of 1D arrays into a 2D NumPy array
    array_2d = np.array(array_of_arrays)
    
    # Calculate the range for each column (dimension)
    ranges = np.ptp(array_2d, axis=0)
    
    return ranges

array_of_arrays = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 10]
]
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
def generate_random_arr(min,max,clusternum):
    
    arr2=[]
    for j in range(clusternum):
        arr1=[]
        for i in range(len(min)):
            arr1.append(random.randint(min[i], max[i]))

        arr2.append(arr1)
    return arr2
# Example usage:

print(min_of_each_column(array_of_arrays))
print(max_of_each_column(array_of_arrays))
print(generate_random_arr(min_of_each_column(array_of_arrays),max_of_each_column(array_of_arrays),4))



column_ranges_array = column_ranges(array_of_arrays)
print("Range of each column:", column_ranges_array)
import numpy as np



# Example usage:


# Cluster_Centroids=[[1,2],[3,4],[5,6]]
# data=[[3,4],[5,6],[1,2]]
# dist_arr=GUC_Distance(Cluster_Centroids,data,"euclidean")
# print(dist_arr)
# min_index_in_each_row = find_minimum_index_in_each_row(dist_arr)
# print(min_index_in_each_row)
# def getting_mean_square(min_index_arr,distance_arr,clusternum):
#     array=[]
#     for k in range(clusternum):
#         sum=0
#         count=0
#         for i in min_index_arr:
#             if min_index_arr[i]==k:
#                 sum=sum+distance_arr[i][k]**2
#                 count=count+1
#         array.append(sum/count)
#     return array
# print(getting_mean_square(min_index_in_each_row,dist_arr,3))
# def GUC_Kmean ( Data_points, Number_of_Clusters,  Distance_Type ):
#          dim=len(Data_points[i])

             
    
#     return [ Final_Cluster_Distance , Cluster_Metric ]  
# # min_index_in_each_row = find_minimum_index_in_each_row(two_d_array)
# # print("Index of minimum number in each row:", min_index_in_each_row)
# # array=[]
# # for k in clusters:
# #     sum=0
# #     count =0
# #     for i in mindist:
# #         if mindist[i]==k:
# #             sum=sum+square(twod[i][k])
# #             count=count+1
# #     array.append(sum/count)