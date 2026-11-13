<!-- PROVIDED: copy this file to docs/api_design.md and fill in every placeholder.
     The check fails while the word in capitals below is still anywhere in the file. -->

# API design (Week 5)

Team: Group 4 (Krish Iyer, Minh Nguyen, Calvin Zheng)

## 1. Where do the 164 features come from?

A request to /predict arrives for one applicant. The model needs 164 features, and most of
them are aggregates over many rows of the six child tables (Week 2).

The caller cannot realistically send all 164 features because many of them are calculated from historical information stored across multiple tables, such as previous credit applications, bureau records, payment history, and credit card balances. The caller would need access to all the original data and would have to calculate the features correctly before making the request. This would make the API more complicated and increase the possibility of incorrect or missing values.

There are two main ways the service could obtain these features. The first option is to calculate them whenever a prediction request arrives. The service would query the original tables, join the records, and calculate all the necessary aggregates. Although this would provide updated information, it would be slower because the calculations would need to happen for every request. It would also require more complicated database queries and additional processing.

The second option is to precompute the features and store them in a feature store. This would allow the service to retrieve an applicant's features using their unique SK_ID_CURR identifier. This approach is much faster because the features have already been calculated. It also reduces the possibility of training and serving using different feature calculations, provided both use the same feature engineering pipeline.

## 2. What we chose, and why

Our project uses a precomputed feature store, which contains one row per applicant and is keyed by SK_ID_CURR. The feature store is saved as models/feature_store.parquet and is generated using the same Week 2 feature engineering pipeline and cleaning process used when training the model.

When a caller makes a POST request to /predict, the request only needs to include the applicant's SK_ID_CURR. The API retrieves that applicant's existing feature row from the feature store and arranges the features in the exact order expected by the trained model.

The request can also contain three optional updated amounts:

AMT_INCOME_TOTAL: Updated yearly income.

AMT_CREDIT: Updated loan amount.

AMT_ANNUITY: Updated yearly repayment.

These fields are optional because the API can use the original stored values when no updates are provided. If the caller supplies a new amount, the service replaces the corresponding stored value with the updated amount.

When an amount changes, the service also recalculates the four ratio features using the updated values. These include CREDIT_INCOME_RATIO, ANNUITY_INCOME_RATIO, CREDIT_GOODS_RATIO, and ANNUITY_CREDIT_RATIO. This is important because changing an income or loan amount without updating its related ratios would result in inconsistent feature values. The recompute_ratios function handles these calculations and prevents division by zero from producing infinite values.

We chose this approach because it provides faster predictions, avoids repeatedly processing the six child tables, and helps maintain consistency between the features used during training and those used when making predictions.

## 3. What can go wrong

There are several potential problems that could affect the accuracy and reliability of the API.

1. Applicant is not in the feature store

A caller might submit an applicant ID that does not exist in the feature store. In this situation, the service cannot retrieve the applicant's features and cannot generate a prediction. Our implementation checks whether the SK_ID_CURR exists in the feature store. If it does not, the UnknownApplicant exception is raised and the API returns an HTTP 404 error explaining that the applicant was not found.

2. Feature store and model use different data or code

Another potential issue occurs if the feature store is generated using different data, cleaning procedures, or feature engineering code than the model used during training. This could cause the model to receive feature values that do not represent the same information it learned during training. This could lead to inaccurate predictions. Our project reduces this risk by generating the feature store using the same Week 2 pipeline and exporting it together with the registered model. However, the current API does not independently verify feature-store version compatibility at runtime, so the correct model and feature-store bundle must be exported and deployed together.

3. Applicant's history changes after the feature store is created

An applicant's financial information or payment history might change after the feature store has been generated. For example, an applicant could receive a new loan or make additional payments. If the feature store is not updated, the API will retrieve the older information and could produce a prediction based on outdated data. The solution is to regenerate the feature store using the updated source data and export a new model bundle when appropriate. The service should then be restarted or redeployed with the updated files.

## 4. Example requests

The following examples demonstrate the three possible outcomes when making a request to the /predict endpoint. These are real answers from our exported model (version 1, threshold 0.08), taken with the
service running on a laptop. Applicant 100021 is the first id in the feature store.

Example 1: Valid request (HTTP 200)

A valid request includes an applicant ID that exists in the feature store. Updated amounts are optional.

Request:

```json
{"SK_ID_CURR": 100021}
```

Response:

```json
{
  "SK_ID_CURR": 100021,
  "probability": 0.1282051282051282,
  "decision": "reject",
  "threshold": 0.08,
  "model_version": "1"
}
```

The probability of default, 0.128, is at least the threshold of 0.08, so the decision is reject.

Example 2: Invalid request (HTTP 422)

A request containing an invalid amount, such as negative yearly income, is rejected automatically by FastAPI and Pydantic before the prediction function executes.

Request:

```json
{"SK_ID_CURR": 100021, "AMT_INCOME_TOTAL": -1000}
```

Response:

```json
{
  "detail": [
    {
      "type": "greater_than",
      "loc": ["body", "AMT_INCOME_TOTAL"],
      "msg": "Input should be greater than 0",
      "input": -1000,
      "ctx": {"gt": 0.0}
    }
  ]
}
```

The HTTP 422 response indicates that the request failed validation because the income amount must be greater than zero.

Example 3: Unknown applicant (HTTP 404)

A request can have valid formatting but still fail if the applicant does not exist in the feature store.

Request:

```json
{"SK_ID_CURR": 1}
```

Response:

```json
{"detail": "Applicant with SK_ID_CURR 1 was not found."}
```

The HTTP 404 response indicates that the applicant could not be found in the feature store. Unlike the 422 response, this request passed input validation but could not be processed because the requested applicant does not exist.
