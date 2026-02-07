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
- Multiple ASPJ pandas operations
- Feature engineering steps
- scikit-learn transformers and estimators
- Nested and chained DataOps

These pipelines were designed to reflect realistic data science workflows and stress-test the provenance tracking mechanism.

Each pipeline demonstrates:
- Correct provenance propagation
- Compatibility with existing skrub APIs
- Reasonable performance overhead

---

## Testing
We implemented a comprehensive test suite to ensure correctness and stability:
- Unit tests for individual provenance-enabled operations
- Integration tests for full pipelines
- Edge case testing for joins, aggregations, and transformations

Tests are written to validate both:
??
- Functional correctness
- Correctness of provenance metadata

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

