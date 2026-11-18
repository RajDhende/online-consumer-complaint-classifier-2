# Viva Preparation — 40 Questions and Answers

All answers match the actual implementation in this repository. Numbers are from
the current trained artefacts (`model/metrics.json`).

---

## A. Concepts (1–10)

**1. What is text classification?**
The task of assigning a predefined label to a piece of text based on its content.
Here: assign one of nine financial product families to a complaint narrative.
It is a supervised learning problem — the model learns from labelled examples.

**2. What is the difference between supervised and unsupervised learning?**
Supervised learning trains on labelled examples (text + correct category) to
predict labels for new text. Unsupervised learning finds structure without labels
(clustering, topic modelling). This project is supervised because the CFPB data
already carries the correct `product` label.

**3. What is TF-IDF and why use it instead of raw word counts?**
TF-IDF = term frequency × inverse document frequency. A word gets a high weight
when it is frequent in one document but rare across the corpus, so discriminative
words ("escrow", "collector") dominate while ubiquitous words ("the", "account")
are down-weighted. Raw counts would let common words dominate every document
equally.

**4. What is an n-gram and why use bigrams?**
An n-gram is a sequence of n consecutive tokens. Bigrams ("credit report",
"late payment") capture short phrases whose meaning differs from the individual
words. This project uses `ngram_range=(1,2)`, i.e. unigrams plus bigrams.

**5. What does `sublinear_tf=True` do?**
It replaces raw term frequency `tf` with `1 + log(tf)`. Repeating a word ten times
should not make it ten times more important; the log dampens that effect.

**6. Why is the data sparse, and which data structure handles it?**
Most documents contain only a small fraction of the 60,000 vocabulary terms, so
each row is mostly zeros. scikit-learn stores the matrix as a CSR sparse matrix,
keeping only non-zero values — memory and compute scale with non-zeros, not with
rows × columns.

**7. Explain Logistic Regression in one paragraph.**
A linear model that computes a weighted sum of input features per class, converts
the scores to probabilities with softmax, and learns the weights by minimising
cross-entropy loss. In text classification each feature weight says how strongly
a term pushes toward a class, which makes it interpretable.

**8. Explain Multinomial Naive Bayes.**
A probabilistic model that applies Bayes' theorem, assuming features are
conditionally independent given the class. It estimates P(word | class) from
counts, with Laplace smoothing (`alpha`) to avoid zero probabilities. Training is
a single pass over the data, so it is extremely fast but its independence
assumption is unrealistic for natural language.

**9. What is class imbalance and why does it matter?**
When some classes have far more examples than others. A model can reach high
accuracy by always predicting the majority class while being useless for the
rest. Here credit reporting dominates, so accuracy alone would be misleading.

**10. What is overfitting and how is it prevented here?**
Overfitting is memorising training data instead of generalising. Prevented by:
a held-out test split, regularised Logistic Regression (default L2), limitations
on feature count, and removing duplicate narratives so near-identical texts
cannot appear in both train and test.

---

## B. Dataset (11–16)

**11. Which dataset did you use, and why?**
The CFPB Consumer Complaint Database — real complaints about financial products,
published by the US Consumer Financial Protection Bureau. It is public, large,
genuinely noisy, and its `product` field provides natural labels, which makes it
ideal for supervised text classification.

**12. How many rows and categories?**
40,000 narrative-bearing complaints sampled from the full database; 11 raw
product families; 9 categories used for modelling after cleaning.

**13. Why did you cap the dominant class at 4,000?**
Credit reporting made up roughly 68% of the sample. Without a cap, the model
would optimise mostly for that class and neglect the rest. Capping equalises
influence and improves macro-level performance; the trade-off is discarding
training rows.

**14. Why were some categories dropped?**
A minimum of 150 rows was required. *Prepaid card* (77 rows) and *Other
financial service* (2 rows) cannot support a classifier — the model cannot learn
a reliable pattern from so few examples, and the test split would have almost no
rows for them.

**15. Why did you merge historical product names?**
The CFPB renamed products over the years (e.g. several credit-reporting variants,
"Bank account or service" vs "Checking or savings account"). Merging them into
canonical labels prevents the model from learning artefacts of naming history and
keeps classes consistent across 2015–2024.

**16. Is the sample a representative survey of US consumers?**
No. It is a uniform random sample of the complaints that were published, but
complaints themselves are not a statistically representative sample of the
population — people self-select to complain. Counts describe the sample, not the
country.

---

## C. Preprocessing (17–22)

**17. List the preprocessing steps in order.**
Lower-casing → redaction placeholder removal (`XXXX`) → URL removal →
contraction expansion → punctuation/digit stripping → tokenisation →
stop-word removal → re-join into a cleaned string.

**18. Why expand contractions?**
So "can't" becomes "cannot" and "don't" becomes "do not", keeping negation
meaningful and making the token vocabulary consistent.

**19. Why remove stop-words?**
Words like "the", "and", "would" appear in every class and add dimensionality
without signal. Removing them shrinks the feature space and reduces noise.

**20. Why remove the CFPB redaction placeholders?**
Narratives contain `XXXX` where the CFPB masked personal data. Left in, they
become a high-frequency token with no semantic meaning.

**21. Is the same preprocessing used at prediction time?**
Yes — the exact same `clean_text()` function is imported by both the training
script and the app. Mismatched preprocessing between training and inference is a
classic silent bug; sharing the function prevents it.

**22. What happens if a user enters text that cleans to nothing?**
The app detects that the cleaned string is empty and shows an error message
instead of passing an empty vector to the model. The batch classifier marks such
rows as unclassified.

---

## D. Features and modelling (23–30)

**23. How were features built?**
`TfidfVectorizer(max_features=60_000, ngram_range=(1,2), min_df=3, max_df=0.9,
sublinear_tf=True)`, fitted on the training split and applied to the test split.

**24. Why fit the vectorizer on the training set only?**
Fitting on all data would let the test set influence the vocabulary and IDF
weights — data leakage that inflates reported scores. The vectorizer must never
see test text during fitting.

**25. What do `min_df` and `max_df` mean?**
`min_df=3`: ignore terms appearing in fewer than 3 documents (typos, one-offs).
`max_df=0.9`: ignore terms appearing in more than 90% of documents (no
discriminative power).

**26. Why compare Logistic Regression and Naive Bayes specifically?**
They are the two canonical baselines for sparse text: one discriminative and
regularised, one generative and extremely fast. Comparing them demonstrates both
that the pipeline works and that model choice matters.

**27. What hyperparameters were used?**
TF-IDF: 60,000 features, (1,2)-grams, min_df 3, max_df 0.9, sublinear TF.
Logistic Regression: `max_iter=2000`, `C=1.0` (default L2), `class_weight="balanced"`.
Naive Bayes: `alpha=0.1`.

**28. What does `class_weight="balanced"` do?**
It scales each class's contribution to the loss inversely to its frequency, so
errors on rare classes cost more. It counters the residual imbalance after
capping.

**29. How was the best model chosen?**
By weighted F1 on the held-out test set. Weighted F1 accounts for class support,
unlike plain accuracy, and avoids letting one large class decide the winner.

**30. Which model won and by how much?**
Logistic Regression: 0.7942 accuracy / 0.7949 weighted F1 versus Naive Bayes
0.7585 / 0.7464. The macro-F1 gap is larger (0.761 vs 0.652), showing Naive Bayes
is disproportionately weak on smaller classes.

---

## E. Evaluation (31–35)

**31. Why not report accuracy alone?**
Under imbalance, accuracy hides poor minority-class performance. A model could
score 68% by always predicting "Credit reporting" and never be useful. Per-class
precision/recall/F1 and macro averages expose that.

**32. Explain precision and recall with an example.**
For Mortgage with precision 0.882 and recall 0.896: of all complaints the model
*labelled* Mortgage, 88.2% truly were; of all complaints that *truly were*
Mortgage, the model found 89.6%.

**33. What does the confusion matrix show you?**
Every actual→predicted count. The diagonal is correct predictions. Off-diagonal
cells quantify specific confusions — here the largest is Debt collection →
Credit reporting (83), caused by shared vocabulary ("collection account on my
credit report").

**34. What is leakage and how is it prevented?**
Leakage is test information influencing training, which inflates scores. Prevented
by: fitting TF-IDF on train only, deduplicating narratives before splitting, and
never tuning on the test set.

**35. What is the weakest class and why?**
Payday or personal loan: F1 0.512 on 59 test rows. Only 294 training examples
exist and its vocabulary ("loan", "interest", "payment") is shared with debt
collection and vehicle loans.

---

## F. Implementation and app (36–39)

**36. How is the trained model served?**
`train_model.py` persists the fitted vectorizer and model with joblib plus a
JSON metadata file. The app loads them once per process with
`st.cache_resource`, cleans input with the same `preprocess` module, transforms
it with the saved vectorizer, and calls `predict` / `predict_proba`.

**37. How does the app handle a missing dataset or model?**
It renders a clear setup message with the exact command to run (`fetch_data.py`
or `train_model.py`) and keeps other pages navigable. The sidebar shows dataset
and model status and offers a "Reload model & data" button that clears the
caches.

**38. How does the CSV upload work?**
The user uploads any CSV; the app normalises known column aliases, validates that
a text column exists, lets the user pick the text column, cleans and vectorises
all rows in one batch, adds `predicted_category`, `predicted_display` and
`confidence` columns, shows the distribution, and offers the results as a
downloadable CSV. Limits (max 10,000 rows) protect memory.

**39. How did you test the app?**
`tests/test_smoke.py` uses Streamlit's `AppTest` harness: it renders all six
pages and asserts no exception, performs a full single-text prediction and
validates the returned category/confidence, and exercises the batch classifier.
Result: 8 tests pass. The live app was additionally verified in Chrome with
screenshots of every page.

---

## G. Deployment and project judgement (40–42)

**40. How would you deploy this to the cloud?**
Push to GitHub, then create a Streamlit Community Cloud app pointing at `app.py`
with `requirements.txt`. The dataset (gzip, 12 MB) and model artefacts are
committed, so no training is required on the server. Details and troubleshooting
are in `docs/DEPLOYMENT.md`. The project itself has not been deployed, and the
report says so explicitly.

**41. What would you improve with more time?**
Fine-tune a transformer (DistilBERT) for comparison; hierarchical classification
(product group → product → issue); cost-sensitive learning instead of capping;
active learning from low-confidence cases; drift monitoring.

**42. What was the hardest part, and what did you learn?**
Discovering that the official bulk export no longer contains narrative text. It
forced a provenance decision: find a faithful mirror, verify it, and document the
limitation instead of hiding it. The lesson: validate assumptions about a data
source by inspecting the actual bytes before building on top of it.

---

## Quick facts to memorise

| Question | Answer |
| --- | --- |
| Sample size | 40,000 narratives |
| Rows modelled | 16,662 (after dedup, min class 150, cap 4,000) |
| Categories | 9 |
| Split | 80/20 stratified, random_state 42 |
| Features | 60,000 TF-IDF (1–2 grams) |
| Best model | Logistic Regression |
| Accuracy | 0.7942 |
| Weighted F1 | 0.7949 |
| Macro F1 | 0.7611 |
| Weakest class | Payday / personal loan (F1 0.512) |
| Strongest class | Mortgage (F1 0.889) |
| Test rows | 3,333 |
| Largest confusion | Debt collection → Credit reporting (83) |
| Tests | 8 passed (AppTest) |
