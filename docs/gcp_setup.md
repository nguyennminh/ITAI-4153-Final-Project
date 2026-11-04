# Google Cloud setup for CI (one page)

<!-- PROVIDED -->

**Who:** ONE member of the team, the one who starts the free trial and owns the project.
Everyone else is added in step 5. Google Cloud's menus move from time to time; if a name
below differs slightly, look for the closest match.

## 1. Start the free trial (5 minutes)

1. Go to https://cloud.google.com/free and click **Get started for free**. Use a personal
   Google account that has **never** used the Google Cloud free trial and never paid for
   Google Cloud.
2. Enter a card. It is only to verify your identity: you are **not charged** during the
   trial ($300 of credit for 90 days). The trial ends when either runs out.
3. Create a project: top bar → project picker → **New project**. Name it
   `credit-risk-<team name>`. Write down its **project ID**.

## 2. A budget alert (2 minutes)

**Billing → Budgets & alerts → Create budget**: scope this project, amount **$25**, alerts at
50%, 90%, and 100%. You get an email long before the credit is at risk. (A whole week of CI
for one team uses a few cents; Cloud Build includes free build minutes every month.)

## 3. Turn on the services (2 minutes)

**APIs & Services → Enable APIs and services**, search for and enable:
**Cloud Build API**, **Artifact Registry API** (Week 5), and **Secret Manager API** (used by
the GitHub connection).

## 4. Connect GitHub and create the trigger (10 minutes)

1. **Cloud Build → Triggers** (region **global** is fine) → **Create trigger**.
2. Name: `pr-checks`. Event: **Pull request**.
3. Source → Repository → **Connect new repository** → **GitHub (Cloud Build GitHub App)**
   → sign in to GitHub → install the app **only on your team's repository** → select it.
4. Base branch: `^main$`. Comment control: **Not required**.
5. Configuration: **Cloud Build configuration file**, location **Repository**,
   file `cloudbuild.yaml`.
6. Service account: a new project usually shows **"No items to display"** here. Then, in a
   new tab, open **IAM & Admin → Service accounts → Create service account**: name
   `cloud-build-ci` → **Create and continue** → role **Logs Writer** → **Done**. Back in the
   trigger tab, reopen the field (or reload the page) and choose
   `cloud-build-ci@<project-id>.iam.gserviceaccount.com`. Click **Create**.
7. If a build fails with a permission error about logs: **IAM & Admin → IAM**, find the
   trigger's service account, **Edit** → add the role **Logs Writer**, save, and rerun.

## 5. Add your teammates (2 minutes)

**IAM & Admin → IAM → Grant access**: each teammate's Google account, role **Cloud Build
Editor** (and **Viewer**). They can then see and rerun builds.

## 6. Branch protection on GitHub (5 minutes)

On GitHub, open your **repository** page and click its **⚙ Settings** tab (the last tab under
the repository name, not the Settings under your profile picture) → **Branches** → **Add
classic branch protection rule** (or **Add branch ruleset**) for `main`:

- **Require a pull request before merging** (untick "Require approvals" unless your team
  wants a review on every pull request)
- **Require status checks to pass** → search for and add the Cloud Build check (it appears
  after the trigger has run once on a pull request; its name contains `pr-checks`)

This works on a **public** repository. On a private one, branch protection needs GitHub Pro
(free for students through the GitHub Student Developer Pack).

## Never

- Never commit a key, a token, or a service-account JSON file. Cloud Build needs none: the
  GitHub app connects it.
- Never make a Cloud Storage bucket with data public (Week 5).
