# Module8-9-answers
Answers to the assessment of Modules 8, 9 and 14
1. Correlation:
   a) Which of the following correlation coefficients indicates the strongest linear relationship between two variables?
      i) 0.2
      ii) -0.8
      iii) 0
      iv) 0.6
   **Ans) ii) -0.8. Correlation coeffs have a range of (-1,1). Going towards either of the ends indicates a strong relationship.**

   b) If the correlation coefficient between two variables is -0.9, what does this indicate about their relationship?
      i) Strong positive correlation
      ii) No correlation
      iii) Strong negative correlation
      iv) Weak negative correlation
   **Ans) iii) Strong negative correlation. Closer to -1 indicates a strong -ve relationship**
=======================================================================================================================================
2. Basics of Regression Models:
   a) What is the primary goal of regression analysis?
      i) To predict categorical outcomes
      ii) To predict continuous outcomes
      iii) To classify data points into clusters
      iv) To analyze variance in a dataset
   **Ans) ii)To predict continuous outcomes. Regression is a supervised learning technique used for predicting continuous numerical values.**
   
   b) In simple linear regression, how many independent variables are used to predict the dependent variable?
      i) One
      ii) Two
      iii) Three
      iv) It depends on the dataset
   **Ans) i) One**
   
========================================================================================================================================
3. Ordinary Least Squares (OLS):
   a)What is the main principle behind Ordinary Least Squares regression?
      i) Minimizing the sum of squared errors
      ii) Maximizing the sum of squared errors
      iii) Minimizing the sum of absolute errors
      iv) Maximizing the sum of absolute errors
   **Ans) i) Minimizing the sum of squared errors. We have to minimize the sum of the squared differences between the observed values and the predicted values to get a better fit.**
   
    b) What does the intercept term represent in the OLS regression equation?
      i) The slope of the regression line
      ii) The value of the dependent variable when all independent variables are zero
      iii) The average value of the independent variable
      iv) The variance of the residuals
   **Ans) ii) The value of the dependent variable when all independent variables are zero**
   
   ========================================================================================================================================
4.  Simple Linear Regression:
   a) In simple linear regression, what is the role of the independent variable?
      i) It is the variable being predicted
      ii) It is the variable being predicted from
      iii) It is the variable being controlled
      iv) It is not used in simple linear regression
      **Ans) ii) It is the variable being predicted from**

  b) How is the best-fit line determined in simple linear regression?
      i) By minimizing the sum of squared residuals
      ii) By maximizing the sum of squared residuals
      iii) By minimizing the correlation coefficient
      iv) By maximizing the correlation coefficient
      **Ans) i) By minimizing the sum of squared residuals**
  
    =========================================================================================================================================
5. Random Forests:
   a) What is a key characteristic of random forests?
      i) They consist of a single decision tree
      ii) They rely on boosting techniques
      iii) They are an ensemble learning method
      iv) They are a type of unsupervised learning algorithm
      **Ans) iii) They are an ensemble learning method**

   b) How does a random forest model prevent overfitting?
      i) By using a single decision tree
      ii) By averaging predictions from multiple decision trees
      iii) By increasing the complexity of each decision tree
      iv) By decreasing the number of decision trees in the forest
      **Ans) ii) By averaging predictions from multiple decision trees**

   ===============================================================================================================================================
6. Model Diagnostics:
   a) What is a residual plot used for in regression analysis?
      i) To visualize the relationship between independent and dependent variables
      ii) To identify outliers and patterns in the residuals
      iii) To determine the correlation coefficient
      iv) To assess multicollinearity among independent variables
     **Ans) ii) To identify outliers and patterns in the residuals**

   b) In regression analysis, what does it mean if the residuals are normally distributed?
      i) The model is biased
      ii) The model is unbiased
      iii) The model is overfit
      iv) The model is underfit
      **Ans) ii) The model is unbiased**

   ==================================================================================================================================================
7. Logistic Regression:
   a) What type of outcome variable does logistic regression predict?
      i) Continuous
      ii) Categorical
      iii) Ordinal
      iv) Nominal
      **Ans) ii) Categorical**

   b) How is the logistic function used in logistic regression?
      i) To calculate probabilities of class membership
      ii) To calculate mean squared error
      iii) To calculate the slope of the regression line
      iv) To calculate the intercept of the regression line
      **Ans) i) To calculate probabilities of class membership**

   =========================================================================================================================================================
8. K Nearest Neighbors (KNN):
   a) What does the 'K' represent in K Nearest Neighbors algorithm?
      i) The number of clusters
      ii) The number of nearest neighbors to consider
      iii) The number of features
      iv) The number of iterations
      **Ans) ii) The number of nearest neighbors to consider**

   b) How does KNN classify a new data point?
      i) By calculating the average of its nearest neighbors
      ii) By assigning it to the most common class among its nearest neighbors
      iii) By using gradient descent
      iv) By minimizing the Euclidean distance between data points
      **Ans) ii) By assigning it to the most common class among its nearest neighbors**

   ========================================================================================================================================================
9. K-Means Clustering:
   a) What is the objective of K-Means clustering?
      i) To maximize intra-cluster similarity and minimize inter-cluster similarity
      ii) To maximize inter-cluster similarity and minimize intra-cluster similarity
      iii) To minimize the number of clusters
      iv) To maximize the number of clusters
      **Ans) i) To maximize intra-cluster similarity and minimize inter-cluster similarity**

   b) How is the initial centroid position chosen in K-Means clustering?
      i) Randomly
      ii) Based on the mean of all data points
      iii) Based on the median of all data points
      iv) Based on the mode of all data points
      **Ans) i) Randomly**

   =======================================================================================================================================================
10. PCA (Principal Component Analysis):
    a) What is the main goal of PCA?
       i) To reduce the dimensionality of the data
       ii) To increase the dimensionality of the data
       iii) To classify data points into clusters
       iv) To maximize the variance in the data
      **Ans) i) To reduce the dimensionality of the data**

    b) How are principal components determined in PCA?
       i) By maximizing the covariance between variables
       ii) By minimizing the covariance between variables
       iii) By using gradient descent
       iv) By randomly selecting variables
       **Ans) i) By maximizing the covariance between variables**

    =======================================================================================================================================================
    **MODULE 14**
    =======================================================================================================================================================

1. Padding:
   a) What is the purpose of padding in convolutional neural networks?
      i) To reduce the size of feature maps
      ii) To increase the size of feature maps
      iii) To speed up the training process
      iv) To decrease the number of parameters in the network
      **Ans) ii) To increase the size of feature maps**
   
   b) Which type of padding adds zeros around the input image or feature map?
      i) Same padding
      ii) Valid padding
      iii) Full padding
      iv) Zero padding
      **Ans) iv) Zero padding**

   ======================================================================================================================================================
2. Strided Convolutions:
   a) What does the stride of a convolutional layer determine?
      i) The size of the filter/kernel
      ii) The number of filters/kernels
      iii) The step size of the filter/kernel
      iv) The activation function used
      **Ans) iii) The step size of the filter/kernel**
   
   b) How does increasing the stride in a convolutional layer affect the output size?
      i) Increases the output size
      ii) Decreases the output size
      iii) Has no effect on the output size
      iv) Depends on the padding used
      **Ans) ii) Decreases the output size**

   ==========================================================================================================================================================
3. Convolutions Over Volume:
   a) In a convolutional neural network, what does the depth of a filter/kernel represent?
      i) The size of the filter
      ii) The number of filters
      iii) The number of input channels
      iv) The number of output channels
      **Ans) iii) The number of input channels**
   
   b) How are convolutions applied over volume in a CNN?
      i) By using 3D filters/kernels
      ii) By applying 2D filters/kernels independently to each channel
      iii) By summing the convolutions across channels
      iv) By using pooling layers
      **Ans) ii) By applying 2D filters/kernels independently to each channel**

   ============================================================================================================================================================
   4. One Layer of a Convolutional Network:
   a) In a convolutional layer, what is the purpose of the activation function?
      i) To normalize the output values
      ii) To introduce non-linearity
      iii) To reduce the dimensionality of the input
      iv) To increase the interpretability of the model
      **Ans) ii) To introduce non-linearity**
      
   b) What are the learnable parameters in a convolutional layer?
      i) Filter/kernel weights and biases
      ii) Activation function parameters
      iii) Input data
      iv) Output feature maps
      **Ans) i) Filter/kernel weights and biases**

   ================================================================================================================================================================
   5. Simple Convolutional Network Example:
   a) What is the typical architecture of a simple convolutional neural network?
      i) Multiple convolutional layers followed by pooling layers and fully connected layers
      ii) Only convolutional layers with no pooling layers
      iii) Only pooling layers with no convolutional layers
      iv) Only fully connected layers with no convolutional layers
      **Ans) i) Multiple convolutional layers followed by pooling layers and fully connected layers**
      
   b) What is the purpose of the final fully connected layers in a CNN?
      i) To reduce the dimensionality of the output
      ii) To increase the interpretability of the output
      iii) To make predictions based on the features learned by convolutional layers
      iv) To apply regularization to the model
      **Ans) iii) To make predictions based on the features learned by convolutional layers**

   ==================================================================================================================================================================
   6. Pooling Layers:
   a) What is the main purpose of pooling layers in CNNs?
      i) To increase the number of parameters in the network
      ii) To reduce the size of feature maps
      iii) To introduce non-linearity
      iv) To increase the depth of feature maps
      **Ans) ii) To reduce the size of feature maps**
      
   b) Which of the following pooling operations takes the maximum value from each window?
      i) Max pooling
      ii) Average pooling
      iii) Global pooling
      iv) Min pooling
      **Ans) i) Max pooling**

   ====================================================================================================================================================================
   7. CNN Example:
   a) Which of the following tasks is typically performed using convolutional neural networks?
      i) Time series forecasting
      ii) Image classification
      iii) Text generation
      iv) Sentiment analysis
      **Ans) ii) Image classification**
      
   b) In a CNN architecture, what is the role of the feature extraction layers?
      i) To preprocess the input data
      ii) To extract relevant features from the input data
      iii) To make final predictions
      iv) To apply regularization to the model
      **Ans) ii) To extract relevant features from the input data**
   





