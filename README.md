# Project 2

## Overview

Welcome to Are You a Fraud! This project involves an in-depth analysis utilizing machine learning models to predict if a credit card transaction is fraudulent or authentic. Below, you'll find an overview of the analysis, the dataset, the models used, and the key results. We also focused on the feature importance of our K-Nearest model as well as our Gradient Boosting model to better understand what variables are looked at most when determining fraud.

## Analysis Overview

### Purpose of the Analysis

The primary objective of this analysis is to read in credit card transactions and train our machine learning model to more accurately predict credit card fraud. We aim to leverage machine learning models to predict when a charge is fraud based on financial information in the provided dataset.

### Financial Information and Prediction Target

#### Financial Data

The dataset contains eight variables of data that include distance from home, distance from last transaction, ration to median purchase price, repeat retailer, used chip, used pin number, online order, and fraud. These variable can provide valuable insights into the validity of the credit card transactions on our provided dataset.


### Stages of Machine Learning Process

#### Stages of Machine Learning Process

The machine learning process encompassed data preprocessing, feature engineering, model selection, and evaluation. Each stage was carefully executed to ensure optimal model performance. We utilized the XGBoost model to make sure the Gradient Boosting model did not over fit. We visualized the importance features in the K-Nearest model as well as the Gradient Boosting model to see what variables had the greatest impact on fraudulent charges.

### Methods Used

We employed various machine learning models, including:
- K-Nearest Neighbors
- Gradient Boosting
- XGBoost


## Results

### Model Performances

We achieved the following classification report results for each model:



#### K-Nearest Neighbors
![image](https://github.com/ZekeH43/Project-2/assets/143846311/559013a0-e6ac-4d62-aa7a-1317df6bf599)
![image](https://github.com/ZekeH43/Project-2/assets/143846311/a97b2477-4124-469d-9c88-41786119a268)



#### Gradient Boosting
![image](https://github.com/ZekeH43/Project-2/assets/143846311/3195cc2a-2b41-44a6-bb75-d3b1de274243)
![image](https://github.com/ZekeH43/Project-2/assets/143846311/57bf5d7a-300d-473c-8847-c96347545a1f)


#### XGBoost
![image](https://github.com/ZekeH43/Project-2/assets/143846311/daf20b6a-058e-410e-9078-3ad7078d6634)
![image](https://github.com/ZekeH43/Project-2/assets/143846311/35ca0747-f02f-4d29-90db-3fcf1348fb21)


### Data Visualization

We created a correlation matrix heatmap to visually explore the relationships between selected features and the target variable.

![image](https://github.com/ZekeH43/Project-2/assets/143846311/f19fa54c-ff82-4f8d-b87d-16b289544e7f)


From our Gradient Boosting model we were able to visualize which variables were more important when making a decision if the transaction was fraudulent.

![image](https://github.com/ZekeH43/Project-2/assets/143846311/cdd9b6e3-b2ac-49a8-9b36-1f10bbf92120)




### Conclusion

In conclusion, the analysis provided valuable insights into predicting credit card fraud with machine learning models. Although our results may not have yielded exactly what we hoped for, we learned of valauable resources and practices that guided along the way, including feature importance. Overall Gradient Boosting was the most accurate at predicting false and non-false charges. We still made sure to use XGBoost to make Gradient Boosting did not over fit the model. As we noticed in out heat map median purchase price had the highest correlation at determining if a charge was fraudulent or not.      



