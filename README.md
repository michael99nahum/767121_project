# 767121
Air transportation fare prediction: Michael Nahum, Arianna Caiffa, Angel Marfiuc

## INTRODUCTION:

The purpose of the project is to use the dataset provided to predict flight prices for different airlines and routes in India, which can assist both customers and airlines in making informed decisions about booking flights and understanding the factors that affect flight prices.

The dataset includes information such as the airline's name, the flight number, source and destination cities, departure and arrival times, the number of stops, the class of the ticket, the flight duration, the number of days between the date of data collection and the date of departure, and ticket price in local currency.

To begin, we first cleaned the data, handling missing values and outliers. After cleaning the data, we performed exploratory data analysis (EDA) using visualization to evaluate the entire dataset. Then we examined correlations and data distribution, and we determined the metric that is most correlated with price.

Next, in the process of feature engineering we selected suitable features and transformed the data as necessary to prepare it for modeling. 
Following that, we divided the data into training and testing sets in order to evaluate three regression models using appropriate metrics to assess their performance. 
After that, we tuned the hyperparameters of our best-performing model, which is the Random Forest Regressor, to achieve the best possible performances.
We concluded our work with an interpretation of the results and by providing inisights based on our findings.


## METHODS:

### Environment used:

The environment that we have used for our project is python.exe in anaconda3. You can find in the main.ipynb file all the commented code that has been used for the results of this project.

### EDA and Feature Engineering:

With a first overview of our dataset, we can see that we have three numerical variables (duration, days_left, price), eight categorical variables (airline, flight, source_city, destination_city, stops, arrival_time, class) and one useless column (Unnamed: 0), which we have dropped. Moreover, the dataset does not contain neither missing nor duplicated values.

We can notice that the categorical variable stops, which represents the number of stops, has 3 possible values: “zero”, “one”, “two or more”. We can replace those value with their numerical value treating this category as a numerical variable. Encoding “two and more” as 2 shouldn’t be a problem if having more than two stops is rare. Otherwise, there is the risk to limit the capability of our model to predict correctly by introducing a cap for this variable. We transformed the stops variable using the pandas replace function. The same thing could have been obtained using the Ordinal Encoder, however we decided to use that function since we are not introducing a hierarchy, but we are simply replacing a string with its integer value.

Since for the project it is asked to interpret the results of the analysis and provide insights and recommendations based on the findings, it is useful to make a preliminary analysis to understand more about our dataset. We will do this by studying the dataset description, the histograms, and the correlation matrix for our numerical variables, and with a pair plot for understanding more about the class category, which intuitively seems to be a categorical variable which can influence the price of a flight.
The dataset description shows that our numerical ranges are very different: probably it will be necessary to scale the variables, for example using the Standard Scaler. Moreover, the max value of price and duration are much higher than the 75% percentile. This means that we will have some outliers in our dataset we should take care of. We considered removing outliers from the dataset a bias, therefore we decided to take care of them by using the Mean Absolute Error (MAE) as the index to evaluate the model’s performances: MAE is more robust to outliers since MSE and RMSE square the distances, therefore higher distances have more weight.


The histogram shows that the number of stops is rarely equal to the category "two or more". For this reason, we can assume that introducing a cap for our new numerical variable will not introduce a significant bias to our dataset, since it is very rare that a flight will have more than two stops. Besides, from the histogram we can confirm our intuitions regarding duration and price columns, which are skewed.
From the correlation matrix, we can notice that:

•	The correlation coefficient of 0.2 indicates a positive but moderate correlation between "duration" and "price". This suggests that flights with longer durations tend to have slightly higher prices than those with shorter durations. This could be due to the higher costs associated with longer flights. Duration is the variable with highest correlation with price.

•	Similar considerations can be done for the correlation between “stops” and “price” (which have a correlation of 0.12). This is due to the fact that “stops” and “duration” are highly correlated (coefficient of 0.47), suggesting that flights with longer durations tend to have higher number of stops (which, again, makes sense).
All the other variables have small correlation coefficients, so we cannot conclude much from them.

The last thing we did in our preliminary analysis is to evaluate the class category, through a pairplot. The pairplot, as could be imagined, shows that the Business class is associated with higher prices with respect to the Economy class.

### Train and Test split:

In order not to introduce a significant sampling bias, we have done stratified sampling to ensure that our test set will be representative of the whole dataset, with respect to the duration variable, which is the one most correlated with our target variable, the price. It is important to have a sufficient number of instances for each stratum, therefore we have used pd.cut() to create a duration variable with 5 categories. Our code shows that we achieved our purpose of having similar proportions for the whole dataset and the test set.

### Feature Engineering:

Since the flight columns is not useful for our prediction, we decided to drop it. As previously mentioned, since we have very different ranges for our numerical variables, it is necessary to scale them. This has been done using the Standard Scaler. The categorical variables, instead, have been encoded using the One Hot Encoder. Since we have just two categories for the class variable, we could have used just one binary variable by dropping the first category. However, we decided not to drop it in order for our model to work correctly also if another class (for example, first class) would be added in the future in Indian flights. To avoid data leakage, the feature engineering transformations have been fit only with the training set.

### Algorithms:

We have decided to implement three different models for our model: Decision Tree Regressor, very fast and easy to interpret, an Ensemble method (Random Forest Regressor) and finally an artificial Neural Network. The performances of each of those methods will be discussed in the next session.

## EXPERIMENTAL DESIGN:

### Evaluation metrics:

As analysed in the previous section, in the price variable there are some outliers, therefore we will use the Mean Absolute Error to evaluate the model’s performances. MAE is more robust to outliers because it does not square the distances. We will also show the R-squared value, which, differently from MAE, lies between 0 and 1 regardless of the output variable and therefore it is easier to interpret.

### Baseline:

The benchmark we have used to compare our models with is the Linear Regression, which is the simplest method: it is a linear approach to model the dependency between input and output and its coefficients are simply interpretable. Our baseline model has MAE of 4500 and R-squared of 0.91. Therefore, models with lower performances will be automatically discarded.

### Decision Tree Regressor:

The first model implemented is a Decision Tree Regressor, which is a very simple and interpretable regression model. Our main purpose for this first model is to improve the performances of our baseline model. Our new model has MAE of 1166 and R-squared higher than 0.97, therefore our performances are much better than our baseline.

### Random Forest Regressor:

The next step we did to improve the performances of our model is to use an Ensemble method, a Random Forest Regressor. This ensemble method, based on bootstrap aggregation, usually improves the performances by decreasing the degree of overfitting. Since we managed to obtain good results with a single Decision Tree Regressor, we thought that using a Random Forest would be the best method to improve our performances. This method has a MAE of 1075 and an R-squared higher than 98%.

### Artificial neural network:

The last method tried is an artificial neural network. ANN are not the best method for not very complex models because they are very prone to overfitting, which occurs when a network learns spurious patterns in the training data, which can lead to poor performance on new, unseen data.
We decided to try a sequential model with three dense layers. The first two have ReLU activation functions with respectively 64 and 32 neurons and the last layer is a single neuron, which gives a scalar output. We used the “mean squared error” loss function and the Adam optimizer for the training of the model. Moreover, we have set the input shape parameter of the first layer with the number of features of the training data.
Also in this case, we have increased the performances of our baseline model, with a MAE of 2380 and a R-squared of nearly 0.97. However, our performances are lower than the Ensemble method, confirming that maybe our problem is not complex enough to use an artificial neural network. Moreover, this model is very slow and “black-box”, therefore it would not be easy to interpret the results obtained, which is a specific requirement of this project.

### Hyperparameters tuning:

The last step for our project is to tune the hyperparameters of our best performing model, which is the Random Forest Regressor. Since the performance of our model are already quite good, we will just tune the hyperparameter related to the number of estimators, trying with 50, 100 and 150.
To achieve this goal, we used the Grid Search, obtaining a MAE of 1073 and an R-squared near to 0.99, improving the already good results previously shown. The best results are obtained with 150 estimators.

## RESULTS:

The last specific requirement we were asked for our task was the interpretation of the results and to provide insights and recommendations based on our findings. To interpret the results, we used the feature importance of our best performing model, which has an R-squared of nearly 0.99, as previously shown. We decided to give an interpretation based on our understanding of the task following the book’s procedure rather than using specific packages as SHAP library. The higher the feature importance, the more important the feature is for our model. The importance of a feature is computed as the normalised total reduction of the criterion brought by that feature. Therefore, this should be a message of connection with the variable we are trying to predict. To be sure of what obtained, we will compare our results with our intuition and our preliminary analysis that can be found in section c.
This placeholder table sums up the four attributes with highest feature importance. Their feature importance contains nearly the 95% of the total sum of all the feature importance.

                   
Attribute             Feature Importance

Economy	             49%

Business	           39%

duration	           6%

days left	           2%



This is coherent with what achieved in the preliminary analysis: the class is the only categorical variable which seem to have an impact on the price. In particular, Economy class has 10%  more impact in the price than the Business class. The pairplot used in the preliminary analysis showed that Business class was associated with a higher price with respect to the Economy. Also the numerical variable duration and days left seem to have a little impact on our model. The correlation matrix showed that duration was the variable most correlated to price. Other variables seem not to have a significant impact on the price of flights among Indian cities, therefore price seem not to be particularly influenced by source and destination cities, the airline or departure and arrival time.
If we trust the order of the feature importance for all the attributes, we can conclude that Air_India is the airline with the highest impact on the price, and Delhi and Mumbai are the cities with highest impact on the price. The reason might be because those cities are two of the most important in India, but we do not have enough elements for being certain about that, and further analysis should be required.

## CONCLUSIONS:

Our code trains and evaluates three different machine learning models for predicting the price of a flight. The models are Decision Tree Regressor, a Random Forest Regressor, and Artificial Neural Network. The performance index chosen is the MAE, but we will also show the R-squared value, which does not depend on the output variable. The results of our model will be compared with one of the easiest regression models, the Linear Regression, which we chose as our baseline model and are shown in the table below.

Method	                     MAE             R-squared

Baseline Model	             4500	               0.91

Decision Tree Regressor	     1165	               0.97

Neural Network	             2380	               0.96

Random Forest Regres.	       1075	               0.98

All the machine learning models improve the baseline performances. As we can see, the Random Forest Regressor model has the lowest mean absolute error and the highest R-squared value, which indicates that it is the best performing model among the three chosen. 

It is important to highlight that the performance of the Neural Network depend on the architecture chosen, so its performances could have been improved by finding the optimal architecture (using for example auto-sklearn package, which finds the optimal regression model). 

Moreover, the Decision Tree Regressor obtains good performances, and it is quite fast with respect to the ensemble method, which is the one with the highest performances. However, we are interested in finding an accurate method and therefore training speed is not the most important property of our algorithm.
By analysing histograms, correlation matrix, pairplot and feature importance we can conclude that class and duration are the variables which have the highest impact on the price. In particular, flights with longer durations seem to be associated with higher prices and the Business class seem to be associated with higher price with respect to the Economy class. Besides, the number of stops is positively correlated with the duration of the flights, which makes sense: longer flights tend to have more stops than short flights. 
This is probably the reason why also stops has a positive correlation with price. The fourth variable which impacts the most our model is the days left variable: this might be because the price changes depending on when the flight is booked, which again makes sense.

Other variables do not have a significant impact on our best performing model, but we can notice that two of the most important Indian cities (as Mumbai and Delhi) have a greater impact on the model than other cities.

The main takeaway point from this code is that machine learning models can be used to predict the price of flights among Indian cities with a reasonable degree of accuracy: by tuning the hyperparameters we obtained a R-squared value of nearly 99%. 



### Questions not fully answered:

Our model achieves the best performances with a Random Forest Regressor, with a 99% R-squared value and a MAE of 1073, which seems to be a good result. However, we could have tried other Ensemble methods or different Neural Network architectures that could be more suitable for this specific task to improve our performances.
Moreover, the results provided are based on the comparison of what obtained and what analysed in the preliminary analysis through histograms, pairplot and correlation matrix. A good way of moving forward would be to search for the flights’ prices, to understand how the changing of a variable would impact the price, with an empiric confirmation of what imagined.

### Next steps for this direction of future work

•	search for the flights’ prices to have an empiric confirmation of the impact of a variable in our model (in particular, the variables that have a higher feature importance).

•	Collect more data to train the models.

•	Explore other machine learning models that may be better suited for predicting the price of flights.

These are just a few ideas for future work in this area. We believe that machine learning has the potential to revolutionize the way customers book flights. By developing more accurate and reliable models, we can help people save money on their travel expenses and the company which has hired us to optimise their flight sales revenues!
