# Predicting Heart Failure

## Introduction
According to the [Centers for Disease Control](https://www.cdc.gov/heartdisease/heart_failure.htm), 6.2 million adults have heart disease. It cost the United States an estimated 30.8 billion dollars a year in medical costs and lost work. Heart disease is America's number one killer, with close to 380,000 deaths a year. Addressing rates of heart disease could have a major impact on public health.

According to a study published in [Health Services Management Research](https://pubmed.ncbi.nlm.nih.gov/17958972/), physician workload has been increasing, leading to physician burnout and negatively impacts the delivery of care. Furthermore, the amount of diagnostic data a physician has to analyze on the fly is quite large. Perhaps it is possible to decrease diagnostic workload in heart failure cases, allowing doctors more time to focus on their quality of care. Which raises the question, is it possible to leverage patient data to predict cases of heart failure that result in death?

With machine learning technology, it is completely possible. In this project, a Random Tree Classifier model was developed based off of industry standard biomarkers that could predict cases of heart failure that resulted in death. Physicians could use such a tool to better predict the heart failure cases that result in death so that they can more efficiently deliver care. Not only would it provide better outcomes for the patients, but it would help physicians decrease their workload and improve their quality of life. 

## Methods
The dataset was taken from [kaggle.com](https://www.kaggle.com/andrewmvd/heart-failure-clinical-data), originally from a research paper by [Chicco and Jurman, 2020.](https://bmcmedinformdecismak.biomedcentral.com/articles/10.1186/s12911-020-1023-5). 

The Python libraries used were pandas and numpy for dataframe manipulation and cleaning, seaborn and matplotlib for visualization, scipy for statistical analysis, and sklearn and sklearn-imbalance for model preparation and implementation. 

The variables ```ejection_fraction```, ```age```, ```serum_sodium```, and ```serum_creatinine``` were chosen as features through statistical analysis and visualization. 

A K-Nearest Neighbors, Decision Tree, Logistic Regression, and Random Forest Ensemble models were developed. The Random Forest model perfomed the best out of the four. Ccp_alpha and class weight were determined through visualization, while the max tree depth and minimum leaf samples were determined through heuristic analysis. 

## Data Table

| Target                   | Datatype | Description                                                                        |
|--------------------------|----------|----------------------------------------------------------------------------------|
| ```died```               | int      | Whether or not the patient died from heart failure

| Feature                  | Datatype | Description                                                                        |
|--------------------------|----------|------------------------------------------------------------------------------------|
| ```age```                      | int      | Age of the patient in years                                                        |
| ```ejection_fraction```        | float64  | Percent of blood volume expelled from the heart with each heartbeat                |
| ```serum_sodium```             | float64  | Blood concentration of sodium in mEq/L                                             |
| ```serum_creatine```           | float64  | Blood concentration of creatine in mg/dL                                           |
| ```anaemia```                  | int      | 1 = Diagnosed anaemic, 0 = not anaemic                                             |
| ```creatinine_phosphokinase``` | float64  | Blood concentration of creatinine phosphokinase in IU/L                            |
| ```diabetes```                 | int      | 1 = diabetic, 0 = not diabetic                                                     |
| ```high_blood_pressure```      | int      | 1 = has high blood pressure, 0 = does not have high blood pressure                 |
| ```platelets```                | int      | Number of platelets per $\mu$/L of blood                                           |
| ```sex```                      | int      | 1 = Male, 0 = Female                                                               |
| ```smoking```                  | int      | 1 = Smokes tobacco, 0 = non-smoker                                                 |
| ````time````                   | int      | Number of days from collection of biomarkers to when the target data was collected |

## Results
The Random Forest model outperformed baseline by 2% overall accuracy, but was able to predict cases of death close to 95% of the time while only missing 5% of heart attack cases. It incorrectly identified non-fatal cases as fatal around 40% of the time. 

## Discussion
Since the model is predicting death events as the positive case, it is imperative to minimize false negatives and maximize true positives, that is, identify true cases of death from heart failure and reduce the amount of times it incorrectly classifies fatal events as non-fatal. Incorrectly classifying a non-fatal case as fatal is less costly than missing a case, or informally, a needless doctor's visit costs less than a patient death. 

The Random Forest model outperformed all the other models on those specific metrics, despite not having the highest accuracy. ```time``` was not used as a feature because it represented the time from initially collecting the data to then collecting the target variable. Using it would be using "future" unknown data to predict an event. It would skew the results and overall weaken the model in real-world implementation despite an increase of overall accuracy. A survival analysis must be conducted on ```time``` if it will be used as a feature. 

Surprisingly, diabetes, smoker status, high blood pressure, and anemia were not predicative in this dataset. This would probably change with more data collection, but this dataset and model is weakened by it's small sample size. Despite the shortcomings, it is very effective in predicting who will die from heart failure.

## Conclusion
Machine learning technology can be leveraged to predict fatal cases of heart failure. Using this technology could reduce physician workload and increase the quality of care delivered. A reduction in heart failure deaths could save millions of dollars yearly, while also having a positive influence on public health. 

The Random Forest Ensemble model was effective in predicting fatal cases, but could be improved on by reducing its false positive rate. ```time``` should be further processed and used in the model in an effort to increase the model's precision and reduce the false positive rate. 

 Implementation of a predictive heart failure fatality model would have major impacts on how healthcare is delivered. 

 ## Reproducability
 In order to reproduce this project, you will need to clone the repo and download the dataset from [kaggle.com](https://www.kaggle.com/andrewmvd/heart-failure-clinical-data). All random seeds = 123, and you will need to import the modules used in the notebook. You will need to be able to code in Python3, operate Jupyter Lab, and know how to use scikit-learn and scikit-imblearn, along with Pandas and Numpy. 