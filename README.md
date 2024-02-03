# Sampling Assignment     [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Samarjeet09/Assignment-2-Sampling/blob/main/Assignement_2_Sampling.ipynb) 
## Introduction to Sampling Assignment
This assignment focuses on exploring various sampling techniques to handle imbalanced datasets in the context of credit card fraud detection. The dataset used for this assignment is the [Credit Card Fraud Detection dataset](https://github.com/AnjulaMehto/Sampling_Assignment/blob/main/Creditcard_data.csv), and the objective is to assess the impact of different sampling techniques on model performance.

## Sampling Techniques
### 1. Simple Random Sampling
Simple Random Sampling involves randomly selecting a subset of data points from the entire dataset without any specific criteria. In this assignment, a fixed number of samples were randomly chosen, and the impact on model performance was evaluated.

### 2. Systematic Sampling
Systematic Sampling involves selecting every k-th item from a list after an initial random start. In this assignment, a systematic sampling approach was applied to create a subset of the dataset, and its impact on model performance was assessed.

### 3. Cluster Sampling
Cluster Sampling involves dividing the population into clusters, randomly selecting some clusters, and then sampling all members from the chosen clusters. In this assignment, K-Means clustering was used, and the impact on model performance was evaluated.

### 4. Stratified Sampling
Stratified Sampling involves dividing the population into strata based on certain characteristics and then randomly sampling from each stratum. In this assignment, stratification was done based on the 'Class' variable, and the impact on model performance was assessed.

### 5. Bootstrap Sampling
Bootstrap Sampling involves repeatedly sampling with replacement from the dataset to create multiple subsets. In this assignment, a fixed number of bootstrap samples were generated, and their impact on model performance was evaluated.

## Models Used
Several machine learning models were employed to assess the performance of each sampling technique:

### Logistic Regression:
A linear model used for binary classification.
### Random Forest Classifier: 
An ensemble learning method combining multiple decision trees.
### K-Nearest Neighbors (KNN) Classifier: 
A classification algorithm based on the distance to the k-nearest neighbors.
### Decision Tree Classifier: 
A tree-structured model for classification.
### XGBoost Classifier:
An optimized gradient boosting algorithm.
Each model was trained and tested on the different subsets created by the sampling techniques. 
## Evaluation Metric
The performance of each model was evaluated using the accuracy metric using this function for each sampling Technique.

```python
  
def trainModels(df,name):
    models = {
        'LogisticRegression':LogisticRegression(random_state=42),
        'RandomForestClassifier':RandomForestClassifier(random_state=42),
        'KNeighborsClassifier':KNeighborsClassifier(),
        'DecisionTreeClassifier':DecisionTreeClassifier(random_state=42),
        'XGBClassifier':XGBClassifier(random_state=42),

    }
    X,y = df.iloc[:,:-1] , df['Class']
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    row = dict()
    row['sampling_Technique'] = name
    for  model_name,model in models.items():
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)

        print(f'{model_name} - Accuracy: {accuracy}')
        row[model_name] = accuracy

    return row

```

## Conclusion
The assignment provides insights into the impact of various sampling techniques on the performance of different machine learning models in the context of an imbalanced dataset. The results are summarized in a CSV file, `result.csv,` .

|                      |Simple Random Sampling|Systematic Sampling|Cluster Sampling  |Stratified Sampling|Bootstrap Sampling|
|----------------------|----------------------|-------------------|------------------|-------------------|------------------|
|LogisticRegression    |0.853448275862069     |0.9170305676855895 |0.9533527696793003|0.9051724137931034 |0.9416666666666667|
|RandomForestClassifier|1.0                   |1.0                |1.0               |1.0                |1.0               |
|KNeighborsClassifier  |0.9224137931034483    |0.9716157205240175 |0.9737609329446064|0.9224137931034483 |0.9416666666666667|
|DecisionTreeClassifier|0.9741379310344828    |0.9912663755458515 |0.9970845481049563|0.9655172413793104 |1.0               |
|XGBClassifier         |0.9568965517241379    |0.9978165938864629 |0.9970845481049563|0.9913793103448276 |0.9833333333333333|

# We can Conclude that `Random Forest Classifier` provides us the best result followed by XGB classifier.
