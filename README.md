# Predicting-El-Nino-or-La-Nina

1. Problem introduction
This problem is about using 1593 features to make predictions. For the regression problem, I need to predict the difference of temperatures betIen normal and other two conditions. And for the classification problem, I need to classify the three Iather conditions of El Nino, Normal, and La Nina, which I use 1, 0 and -1 indicate each of them. I will use some of the 480 training data to fit a model, and rest data as test data.
2. Data pre-processing and checking assumptions
By plotting the response difference of temperatures against time, I can see these data have time trend. So different predictor may result in different response value.  So linear regression model may not suitable for this problem. 
 
First of all, I separated the dataset into training data (2/3) and testing data (1/3) by using random sample. Then, as I can see from the data, there are 1953 features and only 480 observations, which means it is a large p small n problem. So I decided to reduce the dimension by using PCA at first and use some of principles to make further analysis. From the plot of proportion and plot of cumulative proportion, I decided to choose 150 principle components.
 
3. Model selection for regression problem
For the regression part, I tried 8 models. Because the responses are correlated, so I decided not to use best model method. Although I have reduced the dimension of the data, it still have high dimension, which means I’d better not use spline methods. In the end, I chose some shrinkage methods, such as PCR, PLS, Tree model and Neural Network.
Comparing with other models, the PLS and PCR models have relative small errors. And the PLS model have the best performance.
Table of MSE for all the fitted models
Model	Ridge Model	Lasso Model	PCR Model	PLS Model	CART with Bagging	Random Forest
Model	MARS Model	Neural Network
Test Error	101.29	94.09	86.65	78.25	115.35	111.20	108.04	104.86

4. Final model description for regression problem
I fit the PLS model by using cross-validation. Then, I checked the CV value and plot, and found that the model will have the smallest CV value when components equal to 4. After fitting the model, I got the MSE 78.25.
 
5. Model selection for classification problem
For the regression part, I tried 8 models. First, I fitted two tree models, which didn’t get a good result. Then, I used neural network methods. When I used the neural network model to predict the test data, I get some missing values, so I gave up this model. After that, I tried SVM methods with different kernel functions. Some of these models have smaller error rate compared with tree models. However, the SVM model with polynomial kernel function could only predict the condition when temperature is at normal level, which means the response equal to 0, so I have to drop this model. By comparing all the models, I found that the SVM model with radial kernel function have the smallest error rate.
Table of error rate for all the fitted models
Model	CART with Bagging	Random Forest	Neural Network	Deep Learning	SVM (Linear)	SVM (polynomial)	SVM (radial)	SVM (sigmoid)
Error Rate	0.45625	0.50625	N/A	0.4125	0.40625	0.54375	0.39375	0.41875

6. Final model description for classification problem
As I mentioned in the previous part, I chose SVM model with radial kernel function as our final classification model.  I used grid search to choose the cost and gamma parameters, which lead us get cost as 5 and gamma as 0.01. Then, I get the confusion matrix as follow. Ant the final error rate is 0.3938.
Table of error rate for all the fitted models

7. Conclusion
Although I have the same predict variables, I get different models for regression and classification models. In the regression problem, I got a PLS model that could get the smallest MSE. And in the classification problem, I got a SVM model with radial kernel function as cost function. I thought that when choosing different training data, I may get quite different results. 
