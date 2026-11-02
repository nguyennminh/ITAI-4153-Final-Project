<!-- PROVIDED: copy this file to docs/test_findings.md and fill in every placeholder.
     The check fails while the word in capitals below is still anywhere in the file. -->

# Test findings (Week 4)

Team: TODO

Run `python run.py testreport` and copy the counts from `reports/test_report.md`.

| Kind | Tests | Passed | Failed | Known failures (xfail) | Skipped |
|---|---|---|---|---|---|
| Unit | TODO | TODO | TODO | TODO | TODO |
| Data | TODO | TODO | TODO | TODO | TODO |
| Model | TODO | TODO | TODO | TODO | TODO |

## Limits we chose, and why

TODO: For each data test and model test, the limit you picked (for example "default rate
between 6% and 10%", "no score moves by more than 0.001") and why that number: what you saw
in the data, and what a change past the limit would mean.

## Findings

Write one section per test that failed. Name the test exactly as in the report (for example
`test_higher_income_does_not_raise_risk`). If no test failed, write "No test failed" and
say which test you expected to be the most likely to fail, and why.

### TODO: test name

- **What failed:** TODO (the assert message: how many applicants, by how much)
- **Why:** TODO (what in the model or the data causes it; check it, don't guess)
- **Does it matter for the lender?** TODO
- **Decision:** TODO (fixed the model, how; or kept as a known failure with
  `@pytest.mark.xfail(strict=True, ...)`, and what would make you fix it)
