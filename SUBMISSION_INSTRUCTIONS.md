## Technical Assessment: Data Science & AI Specialist (P1)

Estimated time: 2 hours Submission: Git repo + short write-up. Everything is submitted in one Word document.

Objective
Build a binary classifier that predicts whether a short text describes a real disaster, wrapped in a simple local web UI that anyone can run on their laptop with a single command. The use of AI to help you in coding and development is encouraged.


## Task
Dataset: Kaggle "Natural Language Processing with Disaster Tweets" (nlp-getting-started). Download train.csv and test.csv from the competition page. The training set has ~7,600 hand-labeled tweets with a binary target column (1 = real disaster, 0 = not).

1. Build a classifier. Train or configure any model you like — pre-trained transformers, TF-IDF + logistic regression, zero-shot LLM, or rule-based + ML hybrid. We care about working code and sensible choices, not state-of-the-art F1.
2. Build a local UI. Wrap your classifier in a web interface that lets a non-technical user paste in a tweet and get a prediction. Streamlit and Gradio are both good defaults; Flask + a simple HTML page is also fine. At minimum the UI must:
●	Accept text input and display the prediction (label + confidence)
●	Handle empty input and obvious error cases gracefully
●	Run on localhost and open in a browser
Beyond that, design and features are up to you. Anything that makes the tool more useful, legible, or honest about model behavior is fair game.
3. Make it one command to run. A reviewer should be able to:
git clone <your-repo>
cd <your-repo>
<one or two setup commands>
<one command to launch the app>
…and have a working UI in the browser within a minute or two on a fresh machine. Pin dependencies. Pin the Python version. Test it yourself in a clean virtual environment before submitting.
4. Include a batch prediction script. A small CLI like:
python predict.py --input tweets.csv --output predictions.csv
that reads a CSV with a text column and writes a CSV with text, label, score. This lets reviewers run your model against a held-out test set without clicking through the UI 500 times.
Submission
Your submission must include:
1.	Anonymous repo link
○	A public GitHub or GitLab repo. The hosting account, repo name, commit history, README, and any LICENSE/AUTHORS files must not contain your real name, photo, email address, or other identifying details. A throwaway account or pseudonym is fine—we use anonymous review to reduce bias.
○	If you’d rather not create a new account, you can submit an anonymized mirror of an existing repo via Anonymous GitHub (https://anonymous.4open.science).
○	The repo should contain: README, source code, pinned requirements.txt (or pyproject.toml), pinned Python version, the batch predict.py script, and either a committed model artifact or a training script that produces one in under 5 minutes on CPU.
2.	Screenshot of the UI running locally
○	One PNG or JPG showing your UI in a browser at localhost, with a sample input and prediction (label + confidence) visible. The browser address bar should be in shot.
3.	One-page write-up (PDF or markdown, attached)
○	What you built and why this approach
○	AI tools used and what you validated or changed
○	Limitations and what you’d improve with more time
Before you submit, please confirm:
○	The repo is cloned into a fresh virtual environment and the setup is verified and run commands work end-to-end
○	The UI opens in a browser and returns a prediction for a non-empty input within a minute or two of cloning
○	predict.py runs against a sample CSV and writes a CSV with text, label, score columns
○	My repo, commit history, and any embedded metadata do not reveal my real name
Partial submissions are welcome, if something isn’t finished, submit what you have with a brief note on what’s missing.
