# Understand and explain this project

## Follow one run

1. `python -m fraud_detection` enters `fraud_detection/__main__.py`. It reads options such as sample size and output filename.
2. `load_data()` reads the CSV. `validate_data()` checks the schema and values, and returns a numeric DataFrame with a predictable column order.
3. `split_data()` takes a stratified random sample and then splits it into training and holdout rows. A fixed seed makes the same input and settings repeatable. The `fraud` column becomes `y`; the seven remaining columns become `X`.
4. `build_models()` creates four independent estimators. KNN combines a `StandardScaler` and classifier in a `Pipeline`. Calling its `fit()` method learns the scaling statistics from training rows only.
5. `run_experiment()` fits each model on the same training rows, then gets predictions and fraud scores for the same holdout rows.
6. `classification_metrics()` calculates metrics using actual labels first. The CLI saves the results with the dataset hash, split settings, and library versions so another person can identify the experiment.

Importing the modules does not load data or train models. The `if __name__ == "__main__"` guard runs the command-line entry point only when requested.

## Why the metric fix matters

Suppose a model flags three transactions as fraud. Only one is actually fraud, and there are two actual fraud cases in total.

- Precision is **1 / 3**: one correct flag out of three flags.
- Recall is **1 / 2**: one detected case out of two actual cases.

Passing predictions as the first argument to `classification_report` reverses those interpretations and makes support count predictions instead of actual examples. Accuracy can stay unchanged, which makes the mistake easy to miss. `test_precision_recall_and_support_use_actual_labels_first` tests this unequal-precision-and-recall case directly.

Confusion matrices use `[[true negatives, false positives], [false negatives, true positives]]`. The notebook labels actual and predicted axes to make the layout explicit.

## Why include the baseline?

Most rows are not fraud. Predicting “not fraud” for every transaction scores 91.28% accuracy on the default holdout, yet finds zero fraud cases. Compare fraud recall and false alarms alongside overall accuracy.

Precision and recall use the model's default class decision. Average precision and ROC AUC use its probability scores across thresholds. Average precision is the metric reported here; it is not interchangeable with a trapezoidal precision-recall area.

## Why these models?

- **KNN** classifies using nearby training examples. Scaling matters because a large numeric range can otherwise dominate distance. This version retains three neighbors from the original analysis.
- **Gradient Boosting** combines successive small decision trees. The scikit-learn classifier uses its default hyperparameters with an explicit random seed.
- **XGBoost** is another boosting implementation. Here it uses 100 trees, maximum depth 3, learning rate 0.1, histogram-based splits, one CPU worker, and an explicit seed. Its success does not validate a different model's generalization.

These are fixed comparisons, not tuned models. The saved holdout scores do not estimate uncertainty across seeds or prove performance on another data source.

## Interview practice

Use the code and your own words to answer these questions:

1. What happens to fraud recall if every prediction is “not fraud”?
2. Why should the scaler learn its mean and standard deviation from training rows only?
3. What does stratification preserve, and what does it fail to protect against?
4. How would false positives affect a cardholder? What about false negatives?
5. Why do we keep a dataset hash and random seed in the result file?
6. Which findings can you defend from this experiment, and which need more evidence?

Describe this as an educational project you can run, inspect, and explain. Credit the original project work and distinguish the later cleanup, corrected evaluation, and automated tests from the historical presentation results. Avoid claiming deployment or business impact that has not been measured.

## Small extensions to implement yourself

1. Add a logistic-regression pipeline with scaling and compare it using the same split.
2. Use stratified cross-validation **within the training data** to choose KNN's neighbor count. Reserve a fresh final test set if you use the existing holdout while developing the change.
3. Explore fraud decision thresholds on validation data and graph the precision/recall tradeoff. Do not tune the threshold on the final test set.
4. Verify and document the dataset's original source and license. If customer IDs or timestamps become available, design grouped or chronological validation.

Run the tests after each change and explain what behavior your new test protects.
