# Provenance Tracking for Skrub Data Ops Pipelines

## Overview
This project was developed as part of the Responsible Data Engineering Project at the DEEM Lab, TU Berlin. The main goal is to introduce **provenance tracking** into **skrub DataOps pipelines**, enabling better transparency, debugging, and analysis of data transformations and machine learning workflows.

The project extends existing skrub functionality by tracking how data flows through complex pipelines built on top of pandas and scikit-learn.

---

## Motivation
Data provenance is essential for:
- Understanding how results are produced
- Debugging complex data pipelines
- Auditing and reproducibility
- Performance and optimization analysis

This project explores how provenance tracking can be integrated with minimal overhead into real-world data processing pipelines.

---

## Provenance Tracking in skrub
We introduce provenance tracking at the **DataOp** level in skrub pipelines. The tracking mechanism records relationships between input data, intermediate transformations, and final outputs.

Key design goals:
- Minimal runtime overhead
- Compatibility with existing skrub pipelines
- Support for complex pandas and scikit-learn operations
- Clear and inspectable provenance metadata

---

## Implementation Details


---

## Supported Operations

### ASPJ Pandas Operations
We provide provenance support for **ASPJ-style pandas operations**, including:
- **Aggregation**
- **Selection**
- **Projection**
- **Join**

These operations are commonly used in data preparation pipelines and represent basic building blocks for numerous relational operators.

Implementation highlights:
- 
- Eager Tracking of the row-level provenance
- Why provenance

---

### scikit-learn Estimators
Provenance tracking is also supported for **scikit-learn estimators**, allowing us to trace how transformed features and model outputs depend on the original data.

Supported components include:
- Transformers (e.g., scaling, encoding)
- Estimators used within pipelines
- Fit / transform / predict stages

This enables end-to-end provenance from raw data up until model predictions.

---

## Complex Pipelines
To validate the robustness of our approach, we implemented **10 complex pipelines** that combine:
These pipelines were designed to reflect realistic data science workflows and stress-test the provenance tracking mechanism.

Each pipeline demonstrates:
- Correct provenance propagation
- Compatibility with existing skrub APIs
- Reasonable performance overhead


Basic Data Analaysis - This pipeline performs an end-to-end data analysis, focusing on understanding delivery performance and customer satisfaction. It starts by loading multiple datasets—customers, orders, order items, payments, and reviews. The datasets are merged into a single df DataFrame, with date columns converted to datetime objects. The pipeline filters only delivered orders and computes new features, such as delivery_delay (difference between actual and estimated delivery dates), order_value (based on product price), and a binary is_delayed indicator. It cleans the data by removing NaNs and extreme outliers, then visualizes the data using skrub’s graphing tools. Finally, it prepares a subset of relevant features for predictive modeling, marking them appropriately as predictors (X) and target (y) for training a machine learning model, such as logistic regression, to predict whether an order will be delayed.
This combines data preprocessing, feature engineering, visualization, and machine learning preparation into a single reproducible pipeline.

Customer Churn - This pipeline performs customer churn prediction users using recent order history and transactional data. It begins by loading datasets for customers, orders, order items, payments, and reviews. The orders are filtered to only include delivered orders, and order timestamps are converted to datetime. For each customer, features are engineered, including the total number of orders (order_count), average order price, average payment value, and a binary churn indicator defined as whether the customer’s most recent order occurred more than 180 days before the latest recorded order. The data is aggregated at the customer level, cleaned, and scaled using StandardScaler. An XGBClassifier is then applied within the skrub framework: the features are wrapped as X, the target is marked as y, and the pipeline uses skrub’s built-in methods to apply the model, split into training and test sets, fit a learner, and make predictions. Finally, the predictions are evaluated using a classification report.

Fuzzy Joiner - This pipeline is focused on exploring and joining transactional e-commerce data using skrub and a fuzzy matching approach. First, it loads multiple datasets—customers, orders, order items, payments, reviews, order payments, and geolocation Each dataset is wrapped with skrub.var() so that operations can track the origin of data. After confirming the loaded columns for order_items and payments, the pipeline performs a fuzzy join using fuzzy_join(). It attempts to join order_items and payments based on the columns product_id (from order items) and order_id (from payments), while also adding metadata about how the match was made (add_match_info=True). The resulting augmented_df contains combined information from both tables, allowing you to analyze relationships between products sold and payments received, including any fuzzy or approximate matches.

Joiners - This pipeline performs a predictive modeling workflow to identify orders likely to receive bad reviews, using skrub. It begins by sampling all relevant datasets (customers, orders, order items, payments, reviews, order payments, geolocation, and products) and wrapping them with skrub.var() for tracking. Order-level aggregates are computed, including total items, total price, total freight, and total payments, which are merged into a master orders_full DataFrame. A binary target variable bad_review is created based on review scores (≤2). The pipeline further joins the first product per order with its category via skrub Joiner and merges geolocation features from the customer zip codes. Features are split into numeric and categorical types, with preprocessing applied via StandardScaler for numeric features and OneHotEncoder for categorical features. The preprocessed data is wrapped as X for modeling, while y represents the target. A HistGradientBoostingClassifier is then applied within the skrub framework. The pipeline uses skrub’s methods to apply the model, optionally split into training and test sets, fit a learner, evaluate performance, and generate prediction probabilities. Finally, the top predicted probabilities of bad reviews are extracted, enabling analysis of which orders are most at risk for poor customer satisfaction.

Spatial Join - This pipeline implements a predictive modeling workflow for identifying late deliveries orders. It starts by loading sampled datasets (customers, orders, order items, payments, reviews, geolocation, products) and wraps them in skrub.var().
Geolocation data is aggregated to compute centroids (mean latitude and longitude) for each zip code, city, and state combination. Two skrub.Joiner objects are created: one to join customer information to orders, and another to join geolocation centroids to customers, allowing enriched feature representation. The orders DataFrame is augmented with a binary target variable late, indicating whether the order was delivered after the estimated delivery date. The features (X) and target (y) are marked for skrub tracking. A TableVectorizer is applied to automatically encode categorical and numeric features in a machine-learning-ready format.Finally, a HistGradientBoostingClassifier is applied within the skrub pipeline, and 5-fold cross-validation is performed to evaluate predictive performance. The pipeline prints the transformed features, the cross-validation scores, and the mean accuracy.

Squashing Scaler - This pipeline implements a customer-level revenue prediction workflow, with a key focus on handling outliers and extreme purchasing behavior using SquashingScaler from skrub. It starts by loading all relevant e-commerce datasets—customers, orders, order items, payments, reviews, order payments, and geolocation.The first step aggregates order-level data for each order (total_items, total_price, total_freight, total_payment) and merges it into a master orders_full DataFrame. NaN values in the aggregates are filled with zeros to handle missing payments or items. Then, customer-level features are computed by aggregating over all their orders, including metrics such as the number of orders, mean and sum of payments, mean items per order, mean order price, and mean freight. These features are enriched with geolocation information (latitude, longitude, city, and state) by merging with the customer and geolocation datasets.mThe pipeline splits features into numeric and categorical types. Numeric features are first “squashed” using SquashingScaler, which compresses extreme values to reduce the effect of outliers while preserving order, and then standardized using StandardScaler. Categorical features are one-hot encoded.
The target variable y is set as the total sum of payments per customer (sum_payment), representing future revenue potential. The preprocessed features (X) are wrapped in a skrub pipeline and passed to a HistGradientBoostingRegressor. The pipeline uses skrub’s make_learner and train_test_split to train the model, evaluate its performance, and generate predictions for each customer.

Various String Encoders - This pipeline addresses the problem of predicting order return risk (or delivery lateness) in e-commerce using multiple types of textual and categorical features and encoders, all wrapped in skrub. It starts by loading and sampling the main datasets—customers, orders, order items, payments, reviews, order payments, geolocation, and products—and wraps them as skrub.var() objects to track data lineage. The orders are augmented with a binary target variable is_late indicating whether delivery occurred after the estimated date. Order-level aggregates are created for items (total items, price, freight), payments (payment types, maximum installments, total value), and products (concatenated product categories, mean weight, maximum number of photos). Customer-level features (city and state) are also merged, producing a final df table with both numeric, categorical, and textual features.Three different encoding strategies are applied using skrub’s TableVectorizer:

    1. GapEncoder for high-cardinality categorical fields (vectorizerGap), capturing latent embeddings.

    2. MinHashEncoder for approximate similarity of sets (vectorizerHash).

    3. TextEncoder based on sentence-transformers for product and customer textual fields (vectorizerText).

Each encoded version of X is then used to train a HistGradientBoostingClassifier to predict is_late. The pipeline evaluates performance via cross-validation (cv=2) and stores results for each encoder type.

---

## Testing
We implemented a comprehensive test suite to ensure correctness and stability:
- Unit tests for individual provenance-enabled operations
- Integration tests for full pipelines
- Edge case testing for joins, aggregations, and transformations

test_DeterminismCheck - This script is designed to check the determinism of a skrub-based pipeline by running it twice and comparing the outputs. Here's a breakdown of what it does:

Setup:
Adds the project’s parent directory to sys.path so modules can be imported reliably.
Parses a --track-provenance argument to optionally enable skrub provenance tracking.
Helper Function (replace_nan_with_minus_one):
Converts NaN values to -1 recursively in lists or arrays.This is important because NaN != NaN in pandas/numpy, which would make deterministic comparisons fail.
Script Execution (run_script):
Runs the target script (BasicDataAnalysisCase.py) using subprocess.
Expects the script to save its output (likely a skrub DataFrame or a wrapper) to a pickle file (output.pkl).
Loads the pickle for further inspection.

Determinism Check:
Runs the target script twice, capturing outputs out1 and out2.
Converts skrub outputs to pandas DataFrames with .skb.preview().
Normalizes NaN and provenance columns to -1 to ensure a fair comparison.
Compares the two DataFrames with .equals().

Result: -> Prints ✅ if the outputs are identical, ❌ if they differ.
This test might not be functional if the sampling is still turned on when reading from the file. When turned off, this test works
---

## Benchmarks
To evaluate performance, we designed benchmarks comparing:
- Pipelines with provenance tracking enabled
- Equivalent pipelines without provenance tracking

Metrics include:
- Execution time overhead
- Memory usage overhead

The results help quantify the trade-offs between transparency and performance.

---

## Experiments
We conducted experiments to analyze:
- Provenance graph size growth
- Impact of pipeline complexity on overhead
- Scalability with increasing data size

These experiments provide insight into how the system behaves under realistic workloads.

---

## Project Structure
```text
.
├── src/                # Source code
├── tests/              # Test suite
├── benchmarks/         # Benchmark scripts
├── experiments/        # Experimental analysis
├── pipelines/          # Complex pipeline implementations
└── README.md
```

---

## How to Run
(Add instructions here for setting up the environment, running pipelines, tests, benchmarks, and experiments.)

---

## Results and Discussion
(Summarize key findings, performance results, and lessons learned.)

---

## Limitations and Future Work
(Describe known limitations and possible extensions, such as broader pandas support, visualization of provenance graphs, or optimization strategies.)

---

## Authors
- Teodor
- Jeanne
- Yigit
- Eduard Chalovski

---

