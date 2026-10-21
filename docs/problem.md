# Problem statement

Team: Krish Iyer, Minh Nguyen

## 1. Who uses the model, and for what decision?

The lender's credit underwriting team uses the model when a loan application arrives,
before any money is paid out. For each applicant, the model gives the probability that
they will have trouble repaying, and the underwriters use that score to approve, refer or
reject the application. Without a model, the decision relies on fixed rules and manual
review. That is slow and inconsistent, and many Home Credit applicants have little credit
history for the rules to work with.

## 2. The options

Applicants with a low predicted risk are approved automatically. Those with a high risk
are rejected. The uncertain middle band goes to a credit analyst, who reviews documents
and income and may contact the applicant. A referral costs analyst time and slows the
decision for the customer. It should therefore cost much less than a bad approval but
more than nothing, so we keep the middle band narrow and use it where a second look is
most likely to change the outcome.

## 3. The cost of approving someone who defaults

Approving a borrower who then does not repay costs **10 units**. This is the
lender's loss on the unpaid loan plus collection costs, and it is the most expensive
mistake the model can make.

## 4. The cost of rejecting someone who would have repaid

Rejecting a good customer costs **1 unit**: the profit lost on a loan that would have
been repaid. A missed default therefore costs ten times as much as a missed good customer,
so the lender should be strict. If the predicted probabilities are accurate, rejecting is
the cheaper choice whenever the chance of default is above 1/11, about **9%**. That is
barely above the 8% average default rate, so the cutoff should sit far below the default
of 0.5. The lender should accept turning away some good customers to avoid defaults.

## 5. Success metrics, and why

We report ROC-AUC, PR-AUC and KS on the fixed 1,000-applicant validation set. ROC-AUC
measures how well the model ranks defaulters above non-defaulters. PR-AUC measures how
precise the high-risk flags are; random guessing scores about 0.08 on it. KS is the
largest gap between the score distributions of the two groups, a standard measure in
credit scoring. Later we will add expected cost per applicant using the 10:1 costs, since
that is what the business actually pays. Accuracy does not work here: predicting "nobody
defaults" is about 92% accurate and catches no defaulters. A model is useful if it beats
approving everyone, which costs about 0.8 units per applicant, and our baseline should
reach ROC-AUC 0.65–0.75 (it reaches 0.71).

## 6. Risks and sensitive features

`CODE_GENDER` and `DAYS_BIRTH` (age) are protected characteristics, and
`NAME_FAMILY_STATUS` and `CNT_CHILDREN` are also sensitive. Other columns can stand in for
them. `NAME_INCOME_TYPE` and our `DAYS_EMPLOYED_ANOM` flag mark pensioners, so they are a
proxy for age. `OCCUPATION_TYPE` and the region ratings can be linked to gender, income
and ethnicity. In our sample, men default more often than women (10.6% vs 6.8%), so a
model would learn to score men as riskier because of their gender. A wrong rejection
denies credit to someone who would have repaid, and it can fall unfairly on whole groups.
That may be illegal under fair-lending rules. We will compare error rates across these
groups before using any model.

## 7. Assumptions

We assume the 5,000-applicant sample represents future applicants and that `TARGET`
correctly records payment difficulties. We also assume that the 10:1 costs are fixed and
the same for every loan, whatever its size, and that the model supports human decisions
rather than replacing them. The cost assumption matters most. If losses grow with the
loan amount, or if referrals turn out to be expensive, the best thresholds change and we
would need to optimize expected cost per loan rather than a single cutoff.
