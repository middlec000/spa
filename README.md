# Project Overview
## Objective
Predict the likelihood an individual will experience first-time homelessness based on their utility customer data.  

## Background
This project was the pilot study for Spokane Predictive Analytics (SPA), a collaboration between Avista Utilities, the City of Spokane, Urbanova, and Eastern Washington University. Data was provided by Avista Utilities and the City of Spokane. Matching and de-identifying (the process of replacing identifying information such as names and addresses with internally generated ID numbers) was performed by the data analysis team at Avista Utilities.   

This GitHub repository contains the code used to 1) investigate the data, 2) preprocess the data from the de-identified state into a useful format for model fitting, and 3) train models to predict if an individual will experience first-time homelessness based on their utility billing behavior.  

I used this project as my thesis for my MS in Applied Mathematics degree though Eastern Washington University. My thesis paper was accepted and I graduated in June, 2021.

## Data Description
The de-identified data was provided in three groups: geographical (GeoData_Anon.csv - unused), Avista service agreements (ServiceAgreements_Anon.csv), and billing data (SpaData_YYYY_Anon.csv).  
A complete list of all variables provided in all three groups can be found in the [data dictionary](supporting_docs/data_dictionary.md).  

The Geographical data was not used for this project because of an unfortunate tradeoff in granularity - the number of levels of (categorical) geographical identifiers was too large to predict on, but if these levels were aggregated they became unhelpful.  

The Billing data has Composite Key of (`SPA_ACCT_ID`, `SPA_PREM_ID`, `MONTH`) = (account id, location id, month) and consists of information related to customers missing payments and Avista's activity in seeking payment.  

The Service Agreements data has Composite Key (`SPA_ACCT_ID`, `SPA_PREM_ID`) and consists of information related to the types of service agreements the utility company has with each customer.  

After investigation the features deemed useful and used for model fitting were:  
| Feature           | Description                                                                                     | Type    | Source(s)                                                  |
|-------------------|-------------------------------------------------------------------------------------------------|---------|------------------------------------------------------------|
| CMIS_MATCH        | Does the listed customer match an individual in the CMIS database? Matched using last 4 of SSN. | boolean | Avista                                                     |
| MONTH             | Month of arrears snapshot.                                                                      | integer | Avista/City of Spokane - utility payment                   |
| TOTAL_CUR_BALANCE | Total balance owed in gas, electric, water, sewer, and garbage utilities                        | float   | Avista - utility payment/City of Spokane – utility payment |
| BREAK_ARRANGEMENT | Start Severance: Break Arrangement                                                              | integer | Avista - collections activity                              |
| PAST_DUE          | Past Due Notice                                                                                 | integer | Avista - collections activity                              |
| SPA_PREM_ID       | Anonymized id mapping to an Avista premise.                                                     | integer | Avista                                                     |  

<span style="color:red">**OUTDATED</span>

## Challenges
### Problem Framing
It was unclear which problem framing would be most suitable for the given objective and available data, so several were investigated:
* Predict number of months until an individual will experience homelessness (continuous variable).  
* Predict if an individual is within six months of experiencing homelessness (binary variable). The current research indicates that positive cases begin to become distinguishable from negative cases around six months previous of experiencing homelessness.
* Predict if an individual is within one month of experiencing homelessness (binary variable).
* Predict if an individual will ever experience homelessness (binary variable).  

The data was most correlated with the last outcome measure so that problem statement was adopted.
### Billing Record ID Structure
The original billing information from the utilities organizations tracks account numbers, not individuals. The outcomes were recorded on an individual level, causing a discrepancy in data keys.  

There were instances where multiple people (two or more of: main, cotenant, landlord, family member, and third party agency) were associated with a single (account, location, month). Ideally all individuals would be retained, but individuals other than the main account holder were not financially responsible for the account so their homelessness outcomes were not related to the account billing activity. Only the main account holder was associated with the account's billing history.  

There were also instances where a single person was associated with multiple (location, month)s. A random choice was made so that (person, location, month) could be used as a composite key for the data.
### Data Imbalance
After preprocessing there were 357 positive cases and 91,234 negative cases. This made the prediction task of identifying the few positive cases difficult. Oversampling the positive class was employed, but it did not significantly improved model performance.

# File Descriptions in `code/`
## Data_Exploration
The data is explored using visual and numerical tools to answer several important questions about the data and investigate relationships within.
### Which Outcome Measure to Use?
Determine which of several potential outcome measures is the most correlated with the provided data.
### Which Years to Use?
Determine if there is a difference in relationships between predictors and outcome in different years.
### Which Billing Attributes to Keep?
Determine which level of aggregation of customer billing attributes is the most correlated with the outcome measure.
### Which Other Attributes to Use?
Determine which attributes besides the billing to include in model fitting.
### Data Imbalance
Determine the degree of imbalance in the data.
### Time and Events
Analyze when events occur over time and for what months we have data on them.
### Compare P and N
Look at the attributes most correlated with the outcome and assess how different the distributions of positives and negatives appear on each attribute.
### Geographical
Determine if there are any useful groupings of positives or negatives based on geographical attributes.

### Billing
* Data is combined from multiple files.
* New feature created for combined amount owed in all utilities bills each month by a single customer account.
* Only relevant features retained.
* Dates are reformatted to number of months since December, 2015 (earliest month in dataset).
* All null values in amounts owed are dropped.
* Null values for `BREAK_ARRANGEMENT` and `PAST_DUE` are replaced with `0`s.
* There are a few duplicate (`SPA_ACCT_ID`, `SPA_PREM_ID`, `MONTH`)'s. Of these duplicates just take the last and drop the others.
### Service Agreements
* Data is loaded and features are renamed to match naming in billing data.
* Extraneous features are dropped.
* Some accounts have multiple people associated with them at one time, some only have one. Associate only the main account holder with each account and remove the other people. The main account holder is financially responsible for the account so the account activity is an indicator of the main account holder only.
* Replace null `CMIS_MATCH` values with `False`s.
* Update data so that if one instance of a person has `CMIS_MATCH` == `True`, then all instances of that person have `CMIS_MATCH` == `True`.
* Reformat dates to match billing data format (number of months since December, 2015).
* Find the earliest month a person was recorded experiencing homelessness and store it in `ENROLL_DATE`.
* Inner join to billing data on (`SPA_ACCT_ID`, `SPA_PREM_ID`).
### Combined
* There are duplicate (`SPA_PER_ID`, `SPA_PREM_ID`, `MONTH`)'s. Of these duplicates just take the last and drop the others.
* Drop all data that occurs after an individual's `ENROLL_DATE`. This ensures we are predicting first-time (in this data) homelessness.
* Create feature `NUM_PREM_FOR_PER`, the cumulative number of premises a person has paid bills at each month.
* Create feature `NUM_PER_FOR_PREM`, the cumulative number of people a premises has seen for each month.

## [Model Fitting](code/06_Logistic_Statsmodels.md)
The method of K-Folds is employed with k=10 and the folds based on randomly choosing from the `SPA_PER_ID`s. For each of the 10 train/test data splits corresponding to the 10 folds, random oversampling was performed on the training set to balance the data. For each fold a logistic model is fit to the training data and predictions are made on the test data. A prediction is made for each (`SPA_PER_ID`, `SPA_PREM_ID`, `MONTH`). A single prediction for each person (`SPA_PER_ID`) is desired so the maximum prediction over all locations (`SPA_PREM_ID`s) and times (`MONTH`s) is retained as the final prediction for each person.  

Various performance metrics are calculated for the predictions: tp, fp, tn, fn, tnr, ppv, npv, f-1 score, accuracy, balanced accuracy, area under the curve.  
Definitions and descriptions at [https://en.wikipedia.org/wiki/Sensitivity_and_specificity](https://en.wikipedia.org/wiki/Sensitivity_and_specificity).

## [Performance Plot](code/results_images/roc_log.png)
The performance of the folds models are displayed using a Receiver Operator Characteristic (ROC) plot.
