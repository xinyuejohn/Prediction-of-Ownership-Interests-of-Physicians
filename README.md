# Prediction-of-Ownership-Interests-of-Physicians
Classified ownership interests of physicians in pharmaceutical companies based on 1.4 million transactions from US Medicare and Medicaid Services.

## Data
There are three datasets: physicians.csv, payments.csv and companies.csv

## Challenge
The data is about licensed physicians in the United States, as well as payments or other transfers of value that they have received from pharmaceutical companies. These payments have been reported to government agencies according to regulatory requirements. It is real-world data. Transactions include things like speaker- or consulting fees, training seminars (e.g. learning to use a medical device), conference travel, food and beverages at business meetings, etc. They may also include payments that are made in connection to some ownership interest that the physician may hold in the pharmaceutical company, e.g. dividend payouts, payments in the form of stock or stock options, etc. The dataset includes 1400000 instances of transactions which have been received by 6000 physicians in the years 2013 to 2019 (transactions.csv), as well as additional data about these doctors (physicians.csv) and companies (companies.csv). For 5000 of these doctors, transactions related to ownership interest (if any) are included, for the other 1000 (the test set), these kinds of transactions have been removed from the transaction data. For each of these 1000 physicians, the model made a prediction about whether they have had such ownership interests (prediction=1) or not (prediction=0).

## Definition of Ownership Interest
‘Ownership interest’ in the context of this challenge is defined as follows: A physician has an ownership interest if he or she has received one or more transactions from a pharmaceutical company in the relevant time frame which has been reported to the government as being related to an ownership interest. In the training data, such transactions are marked by the corresponding indicator variable `ownership_indicator`, but for the 1000 doctors in the test set, these transactions are not appearing in training data.

## Model
Used XGBoost and ensemble methods.

## Evaluation
The model is evaluated based on the performance measure of balanced accuracy.
And the model reached 78% of BAC. 

