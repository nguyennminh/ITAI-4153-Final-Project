<!-- PROVIDED: copy this file to docs/api_design.md and fill in every placeholder.
     The check fails while the word in capitals below is still anywhere in the file. -->

# API design (Week 5)

Team: TODO

## 1. Where do the 164 features come from?

A request to /predict arrives for one applicant. The model needs 164 features, and most of
them are aggregates over many rows of the six child tables (Week 2).

TODO: Explain the problem in your own words: why can't the caller just send all 164 values?
Name at least two ways a service could get them, and what each would cost (speed,
complexity, the risk that training and serving compute features differently).

## 2. What we chose, and why

TODO: How our service gets them (the feature store, keyed by SK_ID_CURR), what the request
contains, and why that is the right choice for this project. What happens when the request
updates an amount (the ratios)?

## 3. What can go wrong

TODO: At least two, with what the service does: an applicant who is not in the feature store;
a feature store built from different data or code than the model; an applicant whose history
changed after the store was built.

## 4. Example requests

TODO: Paste one valid request and its answer, one invalid request and its 422, and one unknown
applicant and its 404 (from http://127.0.0.1:8000/docs or curl).
