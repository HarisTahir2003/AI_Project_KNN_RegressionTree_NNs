# %% [markdown]
# ### **Problem Description**
# 
# In this project, you will be solving a problem where micro robots are tasked with navigating within a wooden box environment which has obstacles. Your objective is to predict the trajectory of the robot over a specified duration based on historical video data.
# 
# ### **Time Series and Lookback**
# 
# The data we have is $\color{orange}{\text{time-series}}$. You will come across this data type very often in this field, since it forms the basis for many AI-based tasks. Let us learn some concepts regarding that. <br/>
# A time series dataset is a collection of data points (or observations) recorded at successive, equally spaced intervals of time, with the intention of analyzing patterns/trend, and behavior of a feature over time.
# For example, if we have a feature's values across a 10 timestep window i.e., `[t-10, t-9, t-8, ... t-1]` we can predict its value at time step `t` based on aggregated information from the previous timesteps. </br>
# 
# #### Lookback
# The lookback concept in time series analysis refers to the number of previous observations used as input when predicting the next value in the series. It determines how far back in time the model "looks" to gather information for making predictions. For example, with a lookback of one, only the immediately preceding observation is used for prediction, while with a lookback of two, the two previous observations are considered, and so on. Consider these examples: <br/>
# 
# **Lookback of One** <br/>
# Input: (502, 59) -> Output: (506, 59)
# 
# **Look-back of Two** <br/>
# Input: ((496, 63), (502, 59)) -> Output: (506, 59)
# 
# ### **Dataset Overview**
# 
# You have the following format for train and test data in this project:
# 
# * Training Data
#   - A 1200-second video recording `(training_data.mp4)` of the robot's movement within the wooden box environment. This video is captured at 30 frames per second (fps).
#   - A text file  `(training_data.txt)`containing the robot's coordinates, with 30 values recorded for each second (since video is 30 fps).
# 
# * Testing Data
#   - A test video `(test01.mp4)`, 60 seconds long recorded at 30 fps.
#   - A test txt file `(test01.txt)` following the same format as the `training_data.txt` file.
# 
# ### **Objective**
# 
# Your goal is to forecast the positions of the robot using KNN, Regression Tree (decision trees that can take continuous values instead of class labels), and Neural Networks.

# %% [markdown]
# # **Part 1A: KNN from Scratch [30 marks]**
# 
# You are $\color{red}{\text{NOT}}$ allowed to use scikit-learn or any other machine learning toolkit for this part (Part 1A). You have to implement your own kNN model. However, you can use numpy, matplotlib, seaborn, and other standard Python libraries for this part. Contact the TAs if you want to use any other libraries.
# 
# ### Imports

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %% [markdown]
# ### Loading and Preprocessing the Dataset
# 
# Get the training and testing data `(trainX and testX)` (you may use the function below to preprocess it). Further, think about how you would form the predictions array `(trainY and testY)` using a lookback of 1.

# %%
# Do not edit this cell
def process_data(file_name):
    with open(file_name, 'r') as file:
        lines = file.readlines()
    data = []
    for line in lines:          # process the lines to extract x and y coordinates
        x_str, y_str = line.strip().split(',')
        x = int(x_str)
        y = int(y_str)
        data.append([x, y])     # store x and y coordinates for each time step
    data = np.array(data)
    return data

train_data = process_data('training_data.txt')
test_data = process_data('test01.txt')

trainX = train_data[:-1]
trainY = train_data[1:]

testX = test_data[:-1]
testY = test_data[1:]

# %% [markdown]
# 
# ### Implement K-Nearest Neighbors (KNN) from Scratch
# 
#   * Develop the KNN algorithm.
#   * This involves creating functions for calculating distances between data points, finding nearest neighbors, and making predictions.
#   * For values of k where a tie occurs, you will need to break the tie by backing off to k-1 value. In case the tie persists, you will need to keep descreasing k till you get a clear winner.
#   * Use Euclidean distance in your kNN classifier
# $$
# d_{\text{Euclidean}}(\vec{p},\vec{q}) = \sqrt{(p_1 - q_1)^2 + (p_2 - q_2)^2 + (p_3 - q_3)^2 + ... + (p_n - q_n)^2}
# $$
# 

# %%
def euclideanDistance(p, q):
    result = np.sqrt(np.sum((p - q) ** 2))
    return result

def KNN(X_train, Y_train, X_test, k):
    d = np.array([euclideanDistance(X_test, x_train) for x_train in X_train])
    sorted_arr = np.argsort(d)
    
    original_k = k
    current_k = k

    while (current_k > 1):
        nearest_indices = sorted_arr[:current_k]
        nearest_outputs = Y_train[nearest_indices]
        mean_nearest_outputs = np.mean(nearest_outputs, axis=0)
        #print(f"k={current_k}: mean output = {mean_nearest_outputs}")
        current_k = current_k - 1

    nearest_indices = sorted_arr[:original_k]
    nearest_outputs = Y_train[nearest_indices]
    return np.mean(nearest_outputs, axis=0)

# %% [markdown]
# ### Evaluation and Analysis
# 
# *  Plot a line graph to evaluate your model's performance (using RMSE) across a reasonable range of K values `i.e., 2, 4, 6, 8, 10`, and explain the trend in your graph. Be sure to also identify optimal k value and add reasoning for your choice. <br/>
# 
# * Note that our model predicts both x and y coordinates at once, thus the RMSE needs to account for that. (This is a useful trick to remember for your future AI-based projects!)
# 
# $$
# \text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (x_i - \hat{x}i)^2} + \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}
# $$

# %%
def RMSE(y_actual, y_predicted):
    result = np.sqrt(np.mean(np.sum((y_actual - y_predicted) ** 2, axis=1)))
    return result

k_list = [2, 4, 6, 8, 10]
rmse_list = []

for k in k_list:
    predictions = np.array([KNN(trainX, trainY, testX_i, k) for testX_i in testX])
    rmse = RMSE(testY, predictions)
    rmse_list.append(rmse)
    #print(f"RMSE for k={k}: {rmse}")


plt.figure(figsize=(12, 6))
plt.plot(k_list, rmse_list, marker='o', linestyle='-', linewidth=2)
plt.title('KNN Model Performance Evaluation')
plt.xlabel('Values of k')
plt.ylabel('RMSE')
plt.grid(True)
plt.show()

# %% [markdown]
# ### $\color{green}{\text{Explanation/Reasoning}}$
# 
# The graph displays a decreasing RMSE trend as the value of k increases from k=2 to k=10, which indicates that as the number of neighbours considered in the kNN algorithm grows, the accuracy of predictions of the model improves. <br> <br>
# When the graph starts with the highest RMSE value at k=2 (i.e. when the algorithm considers only the closest two neighbors for prediction) the average error is larger. As k increases, the RMSE decreases steeply from k=2 to k=4, then continues decreasing but at an increasingly slower rate. This decrease in rate of decrease in the RMSE value indicates that the model starts to reach an optimal level of generalization, where additional neighbours do not contribute to further reducing the RMSE significantly. <br> <br>
# Also, the trajectory plots in the next parts indicate that the values of k do not cause overfitting since the value of start_second and end_second do not affect the accuracy of the predicted path significantly.

# %% [markdown]
# ### Visualization of Actual and Predicted Path
# 
# Compare the actual trajectory of robot with the one predicted by your KNN model for a 6 second interval of your choice from the testing video to check the performance of your model. Do this by plotting both paths on a single graph. The following  $\color{pink}{\text{important}}$ points may help you with figuring how to do this:
# 
# * For the actual trajectory/path, consider how the test video maps to the txt files based on fps rate provided earlier. How many points should you choose for the required interval?
# 
# * Which KNN model from the above ones should you choose and why? (provide a one-liner description)
# 
# * Based on the chosen time interval (e.g. 9-15 seconds), what should be the input to the KNN model (remember that your model is using a lookback of 1) such that the output corresponds to the same timesteps as the actual trajectory?
# 

# %%
optimal_k = 10

start_second = 5
end_second = start_second + 6
start_i = ((start_second * 30) - 1)
end_i = ((end_second * 30) - 1)

predicted_path = np.array([KNN(trainX, trainY, testX[i], optimal_k) for i in range(start_i, end_i)])
actual_path = np.array(testY[start_i:end_i])

plt.figure(figsize=(12, 6))
plt.plot(actual_path[:, 0], actual_path[:, 1], 'ro-', label='Actual Path')
plt.plot(predicted_path[:, 0], predicted_path[:, 1], 'bo-', label='Predicted Path')
plt.title('Actual and Predicted Robot Paths')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.legend()
plt.grid(True)
plt.show()

# %% [markdown]
# # **Part 1B: KNN using scikit-learn [10 marks]**
# 
# In this part, you will implement KNN using scikit-learn (so you can independently check how well your KNN model performs compared to state-of-the-art instead of asking the TAs).
# 
# - Repeat the same process that you did in Part1A for plotting the actual and predicted paths. Use the same time interval from the testing data as before (kind of understood since it is a comparison, but doesn't hurt to clarify).
# 

# %%
from sklearn.neighbors import KNeighborsRegressor

knn_sklearn = KNeighborsRegressor(n_neighbors = optimal_k)
knn_sklearn.fit(trainX, trainY)
predicted_path_sklearn = knn_sklearn.predict(testX[start_i:end_i])

plt.figure(figsize=(10, 6))
plt.plot(actual_path[:, 0], actual_path[:, 1], 'ro-', label='Actual Path')
plt.plot(predicted_path[:, 0], predicted_path[:, 1], 'bo-', label='KNN Predicted Path implemented manually')
plt.plot(predicted_path_sklearn[:, 0], predicted_path_sklearn[:, 1], 'go-', label='KNN Predicted Path implemented using sckikit-learn')
plt.title('Actual and Predicted Robot Paths')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.legend()
plt.grid(True)
plt.show()


