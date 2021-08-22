# Overview

In this exercise, supervised machine learning models are used to predict credit risk for customers given a variety of attributes. The output is binary and can take one of the two values:
* High Risk
* Low Risk

Six different machine learning models are used to predict the credit risk:
* Logistic Regression with Oversampling
* Logistic Regression with Smote Sampling
* Logistic Regression with Understampling
* Logistic Regression with Smoteen Sampling
* Balanced Random Forrest Classifier with a Maximum of 128 Estimators
* Ensemble Classifier with a Maximum of 100 Estimators

# Approach

In each model, we split the data into the **training set** and the **test set**. The data in the training set is used to train the model, and the test set data is used to predict the results. The predictions are compared against actual credit risk values to determine the model's **precision** and **recall** characteristics.

## Data

There are a total of 68,817 total data points. However, the distribution is extremely skewed with respect to the target:

* Low Risk: 68,470
* High Risk: 347

This immediately shows that any model we pick to train on this data will perform poorly when predicting the high risk target. In particular, the model will be biased towards predicting many customers as high risk (this decreasing the precision for the high risk target) to balance against keeping the recall metric of the high risk customers within range.

To combat this problem, we will try both oversampling, undersampling and also couple them with the smoteen sampling methods. However, our results show that none of these methods work in reducting the precision for the high risk category. This means the model will predict many customers as being high risk, when in reality, they are not. In this scenario, false positives are acceptable, since we do want to accurately predict the actual high risk customers.

However, the best Logistic Regression models do not achieve a higher than 70% recall for the high risk target prediction. This means even if the precision is sacrificed all the way down to 1% (as shown in the results below), the model is only able to predict at most 70% of true high risk customers accurately.

This issue is fixed by emplying balanced random forrest and ensemble classifier models. The ensemble classifier model the best recall percentage for high risk target (92%) while simultaneously also boosting the precision for the high risk customers to 9% (still low, but a vast 900% improvement over the logistic regression models).

# Results

The table below shows results for all six models. The output and analysis can be seen in the Jupyter notebooks uploaded in this repository. The result clearly shows we must use the **Ensemble Classifier** model to predict the credit risk.

| Model | Total Precision | Total Recall | Balanced Accuracy | High Risk Precision | High Risk Recall | Comment |
| ----- | --------------- | ------------ | ----------------- | ------------------- | ---------------- | ------- |
| Logistic Oversampling | 99% | 67% | 65.3% | 1% | 63% | Although the total precision is 99%, it's only 1% for high risk. Overall recall is under 70% |
| Logistic Smote | 99% | 66% | 65.12% | 1% | 64% | Similar results as above |
| Logistic Undersampling | 99% | 45% | 52.3% | 1% | 61% | The worst model of the set. Undersampling reduced the total recall by lowering the recall of the low risk target to a very low value (under 50%). Clearly, reducing the number of samples for *low risk* target had an adverse effect. |
| Logistic Smoteen | 99% | 55% | 62.3% | 1% | 70% | Better than above but still worse off than both oversampling models. Reducing the samples for the *low risk* category is generally not helping. |
| Balanced Random Forrest | 99% | 88% | 79.26% | 3% | 70% | Vastly improved over all logistic regression models by improving the total recall to near 90% and the *high risk* recall to about 70%. The high risk precision has increased by 300% as well to 3% |
| Ensemble | 99% | 94% | 93.17% | 9% | 92% | The best model by far. Achieves a total recall of 94% and *high risk* target recall of 92%. The *high risk* precision is 900% increased to 9%. |

