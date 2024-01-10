# Predicting_Real_Oil_Prices
1. This project was for my software engineering class. Where the client ( a business professor ) gave us two datasets consisted of historical macroeconomic data and nominal oil prices.
2. My group was tasked to train a machine learning model to predict future oil prices with the given data.




# XGBoost guide

A chill guide to get started with XGboost


## models to test for performance
-   due to high volatility nature of oil price, non-linear time series based forecasting provide the best forecasting
- use past oil prices and macroeconomic features 
- oil prices are non linear and complex
## Model Ensembles
Combining two or more machine learning models to maximize performance is known as model ensemble. Ensembling can often lead to better predictive power and more robust results compared to individual models. There are various ensemble techniques you can use. Here are a few common approaches:

1. **Voting Ensembles**:
   - **Hard Voting**: In this approach, you train multiple models independently and make predictions with each. The final prediction is the majority vote of the individual model predictions.
   - **Soft Voting**: Instead of counting votes, you can assign weights to each model's prediction probabilities and calculate a weighted average.

2. **Bagging**:
   - **Random Forest**: This is a well-known ensemble of decision trees. It combines the predictions of multiple decision trees trained on random subsets of the data.
   - **Bagged Ensembles**: You can apply bagging to various base models like decision trees, support vector machines, or any other model.

3. **Boosting**:
   - **AdaBoost**: Iteratively trains models on the same data, assigning higher weights to misclassified examples in each iteration.
   - **Gradient Boosting (e.g., XGBoost, LightGBM)**: Builds models sequentially, focusing on the errors of previous models.
   - **Stacking**: In stacking, you combine multiple models by training a "meta-model" on their predictions. The predictions of the base models become inputs for the meta-model.

4. **Neural Network Ensembles**:
   - Ensembling multiple neural networks with different architectures or initializations and averaging their predictions.

5. **Hybrid Approaches**:
   - Combining both regression and time series models, as discussed earlier, can be an example of a hybrid ensemble.

Steps to Ensemble Models:

1. **Train Individual Models**: Train multiple models using different algorithms, hyperparameters, or subsets of features.

2. **Generate Predictions**: Make predictions on your validation or test dataset using each of the trained models.

3. **Combine Predictions**:
   - For a voting ensemble, tally up votes or average probabilities.
   - For bagging or boosting, combine predictions according to their respective algorithms.

4. **Evaluate**: Evaluate the ensemble's performance using appropriate metrics on your validation or test dataset.

5. **Tune Ensemble**:
   - You can adjust weights, hyperparameters, or other settings to fine-tune the ensemble's performance.

It's important to note that ensemble techniques don't guarantee performance improvement for all datasets. It's recommended to test different ensemble strategies and compare them against the individual models. Be cautious of overfitting, as ensembles can lead to overfitting if not done properly. Ensembling can be computationally expensive, so consider the trade-off between performance gain and computation time.
## Feature Importance
Helps determine which features has a high corrlelation with the target values 

This is calculated from a trained model 
## Learning Curve 
Determines if the model is over, under, or good fit. 

Can also help determine if the dataset is unrepresented. 

reference:
 I.https://machinelearningmastery.com/tune-xgboost-performance-with-learning-curves/
 II.https://machinelearningmastery.com/learning-curves-for-diagnosing-machine-learning-model-performance/
III.https://machinelearningmastery.com/tune-xgboost-performance-with-learning-curves/


## Time series based XGBoost VS Regression based XGBoost
Both traditional regression-based XGBoost and time series-based XGBoost models have their merits and can be effective for predicting oil prices, depending on the specific characteristics of your data and your modeling goals. Let's explore the advantages of each approach:

**Regression XGBoost**:

1. **Flexibility**: Regression-based XGBoost can handle a wide range of predictive modeling tasks, not limited to time series data. If your dataset contains features beyond time-related variables, regression XGBoost can leverage them effectively.

2. **Feature Importance**: XGBoost provides feature importance scores, which can help you understand which features contribute the most to the prediction. This can be valuable for interpreting the model's behavior and gaining insights into the relationship between features and oil prices.

3. **Complex Relationships**: If the relationships between macroeconomic features and oil prices are complex and involve interactions that go beyond time dependencies, regression XGBoost might capture these relationships well.

**Time Series XGBoost**:

1. **Temporal Dependencies**: Time series-based XGBoost models, such as using lagged variables and previous oil prices as features, are explicitly designed to capture temporal dependencies. This can be crucial if the primary factors affecting oil prices are related to past prices and seasonality.

2. **Seasonality**: Time series XGBoost models can handle seasonality patterns and other temporal variations more effectively. If oil prices have strong cyclical patterns, a time series approach can be more appropriate.

3. **Interpolation**: Time series models can interpolate missing values in a time series, which is useful if you have gaps in your historical data.

In practice, the best approach might involve combining both. You can use a time series XGBoost model to capture temporal dependencies, seasonality, and patterns specific to oil prices. Then, you can incorporate macroeconomic features using regression XGBoost as additional inputs to enhance the model's predictive power.

Before deciding, it's recommended to experiment with both approaches, validate them on hold-out datasets, and compare their performance using appropriate evaluation metrics. The choice between regression XGBoost and time series XGBoost should be based on how well each approach addresses the unique characteristics of your data and the accuracy of predictions you achieve.
## Finding Accuracy 
- MSE and RMSE
# Loose steps to follow when starting:
## 1. Data Preprocessing
- Clean the data by handling missing values, outliers, and any inconsistencies.
- Normalize or standardize the features to ensure they are on similar scales.
- If your data includes time series features (e.g., historical employment rate), consider creating lag features to capture temporal dependencies.
## 2. Feature Selection /Extraction
- Given the large number of features, you might want to perform feature selection or dimensionality reduction techniques to identify the most relevant features. Techniques like Recursive Feature Elimination (RFE), feature importance from tree-based models, or Principal Component Analysis (PCA) could be helpful.
## 3. Time series Analysis
- Analyze the temporal patterns in your oil price data. Look for trends, seasonality, and any other patterns that could help inform your modeling approach.
## 4. Modeling
- Try different algorithms and models to see which performs best with your dataset. Some options to consider:
    - LSTM/GRU Networks: Neural networks with sequences as inputs can capture temporal patterns effectively.
    - Time Series Models: ARIMA, SARIMA, or other time series models that handle seasonality and trends.
    - Random Forests or Gradient Boosting: These ensemble methods can capture complex interactions between features.
    - XGBoost/LightGBM: These gradient boosting frameworks can handle large datasets and complex relationships.
    - Prophet: Especially useful if you have strong seasonality and multiple seasonal effects.
## 5. Cross-Validation and Evaluation:
- Split your data into training and testing sets (or consider time-based cross-validation for time series data).
- Evaluate your models using appropriate metrics for time series data, such as Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and potentially metrics that account for direction like Mean Absolute Percentage Error (MAPE).
- fold cross validation
## 6. Hyperparameter Tuning
- Fine-tune hyperparameters of your chosen models to improve performance.
## 7. Ensemble Techniques
- You can also consider ensembling multiple models together to leverage the strengths of different algorithms.
## 8. Monitor and Refine
- Periodically evaluate your model's performance as new data becomes available. Update and retrain your model as necessary to keep it accurate.
