# Data Sources
## Avista Utilities
* Service Agreements
* Utility Payment
* Collections Activity
## City of Spokane
* Homelessness Data: [Spokane Community Management Information System (CMIS)](https://my.spokanecity.org/chhs/cmis/)
* Utility Payment
## [United States Census (Not Used)](https://data.census.gov/cedsci/)
* Location Data
* Block Group Median Income
* Block Group Vulnerable Population Flag

# Data Dictionary
| File Name                  | Column Name                   | Description                                                                                                                                                                      | Type             | Source                                   |
|----------------------------|-------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------|------------------------------------------|
| GeoData_Anon.csv           | SPA_PREM_ID                   | Anonymized id mapping to an Avista premise.                                                                                                                                      | varchar          | Avista                                   |
| GeoData_Anon.csv           | CENSUS_TRACTID                | Maps to GEOID in census data set.                                                                                                                                                | varchar          | Avista/Census                            |
| GeoData_Anon.csv           | ACSDT5YR2018MedIncome         | Medium income for tract from the census.                                                                                                                                         | double           | Census                                   |
| GeoData_Anon.csv           | POSTAL                        | Zip code                                                                                                                                                                         | varchar          | Avista                                   |
| GeoData_Anon.csv           | VulnerablePopulationFlag      | Flag if census tract has a vulnerable population.  This was an Avista analysis performed by our GIS group.                                                                       | integer          | Avista                                   |
| GeoData_Anon.csv           | VulnerableCustomerAnalysis_Yr | Year that vulnerable population analysis was created.                                                                                                                            | varchar          | Avista                                   |
| ServiceAgreements_Anon.csv | SPA_PREM_ID                   | Anonymized id mapping to an Avista premise.                                                                                                                                      | varchar          | Avista                                   |
| ServiceAgreements_Anon.csv | SPA_ACCT_ID                   | Anonymized ID mapping to an Avista account. An account may include multiple service agreements for multiple services and multiple persons may be named on the account.           | varchar          | Avista                                   |
| ServiceAgreements_Anon.csv | SPA_SA_ID                     | Anonymized id mapping to an Avista service agreement. A service agreement is unique to one service at one service point, but may be related via the account to multiple persons. | varchar          | Avista                                   |
| ServiceAgreements_Anon.csv | SPA_PER_ID                    | Anonymized id mapping to an Avista customer.                                                                                                                                     | varchar          | Avista                                   |
| ServiceAgreements_Anon.csv | ACCT_REL_TYPE_CD              | Account/customer relationship.                                                                                                                                                   | varchar          | Avista                                   |
| ServiceAgreements_Anon.csv | CMIS_MATCH                    | Does the listed customer match an individual in the CMIS database? Matched using last 4 of SSN.                                                                                  | boolean          | Avista                                   |
| ServiceAgreements_Anon.csv | START_DT                      | Start date of service agreement.                                                                                                                                                 | date             | Avista                                   |
| ServiceAgreements_Anon.csv | END_DT                        | End date of service agreement.                                                                                                                                                   | date             | Avista                                   |
| ServiceAgreements_Anon.csv | SA_TYPE_DESCR                 | Service agreement type (electric or natural gas).                                                                                                                                | varchar          | Avista                                   |
| ServiceAgreements_Anon.csv | Class                         | Customer class (always residential).                                                                                                                                             | varchar          | Avista                                   |
| SpaData_YYYY_Anon.csv      | MONTH                         | Month of arrears snapshot.                                                                                                                                                       | varchar (YYYYMM) | Avista/City of Spokane - utility payment |
| SpaData_YYYY_Anon.csv      | RES_EL_CUR120_DAYS            | Residential electric balance owed (90-120 days)                                                                                                                                  | double           | Avista - utility payment                 |
| SpaData_YYYY_Anon.csv      | RES_EL_CUR22_DAYS             | Residential electric balance owed (0-22 days)                                                                                                                                    | double           | Avista - utility payment                 |
| SpaData_YYYY_Anon.csv      | RES_EL_CUR30_DAYS             | Residential electric balance owed (23-30 days)                                                                                                                                   | double           | Avista - utility payment                 |
| SpaData_YYYY_Anon.csv      | RES_EL_CUR60_DAYS             | Residential electric balance owed (31-60 days)                                                                                                                                   | double           | Avista - utility payment                 |
| SpaData_YYYY_Anon.csv      | RES_EL_CUR90_DAYS             | Residential electric balance owed (61-90 days)                                                                                                                                   | double           | Avista - utility payment                 |
| SpaData_YYYY_Anon.csv      | RES_EL_CUR_BAL_AMT            | Residential electric total balance owed                                                                                                                                          | double           | Avista - utility payment                 |
| SpaData_YYYY_Anon.csv      | RES_EL_OVER_120_DAYS          | Residential electric balance owed (120+ days)                                                                                                                                    | double           | Avista - utility payment                 |
| SpaData_YYYY_Anon.csv      | RES_GAS_CUR120_DAYS           | Residential gas balance owed (90-120 days)                                                                                                                                       | double           | Avista - utility payment                 |
| SpaData_YYYY_Anon.csv      | RES_GAS_CUR22_DAYS            | Residential gas balance owed (0-22 days)                                                                                                                                         | double           | Avista - utility payment                 |
| SpaData_YYYY_Anon.csv      | RES_GAS_CUR30_DAYS            | Residential gas balance owed (23-30 days)                                                                                                                                        | double           | Avista - utility payment                 |
| SpaData_YYYY_Anon.csv      | RES_GAS_CUR60_DAYS            | Residential gas balance owed (31-60 days)                                                                                                                                        | double           | Avista - utility payment                 |
| SpaData_YYYY_Anon.csv      | RES_GAS_CUR90_DAYS            | Residential gas balance owed (61-90 days)                                                                                                                                        | double           | Avista - utility payment                 |
| SpaData_YYYY_Anon.csv      | RES_GAS_CUR_BAL_AMT           | Residential gas total balance owed                                                                                                                                               | double           | Avista - utility payment                 |
| SpaData_YYYY_Anon.csv      | RES_GAS_OVER_120_DAYS         | Residential gas balance owed (120+ days)                                                                                                                                         | double           | Avista - utility payment                 |
| SpaData_YYYY_Anon.csv      | BREAK_ARRANGEMENT             | Start Severance: Break Arrangement                                                                                                                                               | integer          | Avista - collections activity            |
| SpaData_YYYY_Anon.csv      | BREAK_PAY_PLAN                | Break Pay Plan                                                                                                                                                                   | integer          | Avista - collections activity            |
| SpaData_YYYY_Anon.csv      | CALL_OUT                      | Collection Callout                                                                                                                                                               | integer          | Avista - collections activity            |
| SpaData_YYYY_Anon.csv      | CALL_OUT_MANUAL               | Collections Callout Manual                                                                                                                                                       | integer          | Avista - collections activity            |
| SpaData_YYYY_Anon.csv      | DUE_DATE                      | Due Date                                                                                                                                                                         | integer          | Avista - collections activity            |
| SpaData_YYYY_Anon.csv      | FINAL_NOTICE                  | Final Notice                                                                                                                                                                     | integer          | Avista - collections activity            |
| SpaData_YYYY_Anon.csv      | PAST_DUE                      | Past Due Notice                                                                                                                                                                  | integer          | Avista - collections activity            |
| SpaData_YYYY_Anon.csv      | SEVERANCE_ELECTRIC            | Start Severance: Nominate Electric then Gas                                                                                                                                      | integer          | Avista - collections activity            |
| SpaData_YYYY_Anon.csv      | SEVERANCE_GAS                 | Start Severance: Nominate Gas Only                                                                                                                                               | integer          | Avista - collections activity            |
| SpaData_YYYY_Anon.csv      | CITY_TOT_DUE                  | City total due in current month                                                                                                                                                  | double           | Avista - utility payment                 |
| SpaData_YYYY_Anon.csv      | CITY_30_DAYS_PAST_DUE_AMT     | City balance owed (30-59 days)                                                                                                                                                   | double           | Avista - utility payment                 |
| SpaData_YYYY_Anon.csv      | CITY_60_DAYS_PAST_DUE_AMT     | City balance owed (60-89 days)                                                                                                                                                   | double           | Avista - utility payment                 |
| SpaData_YYYY_Anon.csv      | CITY_90_DAYS_PAST_DUE_AMT     | City balance owed (90+ days)                                                                                                                                                     | double           | Avista - utility payment                 |
| SpaData_YYYY_Anon.csv      | SPA_PREM_ID                   | Anonymized id mapping to an Avista premise.                                                                                                                                      | varchar          | Avista                                   |
| SpaData_YYYY_Anon.csv      | SPA_ACCT_ID                   | Anonymized ID mapping to an Avista account. An account may include multiple service agreements for multiple services and multiple persons may be named on the account.           | varchar          | Avista                                   |  