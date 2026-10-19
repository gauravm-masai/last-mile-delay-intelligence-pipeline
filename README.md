# Capstone Project - Delhivery Last-Mile Delay Intelligence Pipeline

## Project Overview

At Delhivery, one of India's largest logistics networks, every parcel is booked with a promised delivery time (Express 36 hours, Standard 60 hours, Economy 96 hours). There are three open problems and no pipeline that answers any of them:

- **How long will a parcel really take?** Planners need a delivery-hours estimate for each parcel at dispatch time.
- **Which parcels will miss their promise?** Ops can place a proactive customer call or escalate to a hub manager, but only for parcels flagged before they are late. A missed delayed parcel costs more than an unnecessary call.
- **Which hubs need which playbook?** Each of the 24 hubs is labelled Tier 1, 2 or 3, but nobody has checked whether the label matches how the hub really behaves.

To answer the above questions, we need to create a pipeline in one connected repository that has below steps.
- **Part A - Data Foundation**: Clean the raw shipment data, engineer features, and perform exploratory data analysis to identify delay patterns.
- **Part B - Regression**: Build and evaluate models to predict the actual delivery time in hours for each shipment.
- **Part C - Classification**: Build a model to predict shipment delays, select relevant features, tune the model, and determine an alert threshold.
- **Part D - Hub Segmentation**: Cluster hubs by operational performance, identify mismatches with official tiers, and package the trained model.
- **Final Script**: Score new shipments, estimate delay probabilities, assign hub segments, and generate recommended operational actions.

Note - The data is synthetic. Delhivery's real shipment records are never used. Every row in this brief's dataset is produced by the generator script below; the brand is only the setting for the decisions.


## Repository Structure

- `generate_data.py` - provided data-generation script.
- `src/shipment_prep.py` - shared data cleaning and feature engineering.
- `part_a_clean_eda.py` - data cleaning and exploratory analysis.
- `part_b_regression.py` - regression analysis.
- `part_c_classification.py` - classification modelling.
- `part_d_segments.py` - segmentation analysis.
- `package_model.py` - model packaging.
- `predict_delay.py` - delay prediction.
- `data/` - generated datasets and modelling table.
- `models/` - saved model artifacts.
- `outputs/` - analysis and modelling outputs.

## Setup and Execution

Instructions will be added after the assignment pipeline is implemented.

## Part A: Data Cleaning and Exploratory Data Analysis

### Objectives
Part A prepares the shipment dataset for downstream analysis and modelling. The main objectives are to clean and standardize shipment data, handle missing and invalid values, combine shipment records with hub information, engineer the required features, and define whether each shipment was delayed based on its promised delivery time.

### High-level approach
1. **Data cleaning:** Remove duplicate shipment records, standardize service types, handle missing payment modes, convert declared values to numeric format, treat invalid parcel weights as missing, and impute missing numeric values using historical medians.
2. **Feature engineering and hub integration:** Merge shipment records with hub information and create the required features, including promised delivery hours, log-transformed declared value, weekend and COD indicators, and dispatch hour.
3. **Delay labelling:** Compare actual delivery hours against the promised delivery hours for each service type to create the binary `is_delayed` target.
4. **Validation and quality checks:** Calculate data-quality metrics and compare them against reference values to verify that the cleaning and feature engineering steps produce the expected results.
5. **Exploratory analysis:** Generate delay-rate summaries by service type, hub tier, and distance band, along with a numeric correlation matrix to examine relationships between features and delivery hours.

The outputs include the cleaned modelling table, validation reports, delay summaries, and correlation matrix. These outputs provide a consistent, validated dataset and an initial understanding of the factors associated with delivery delays before proceeding to subsequent modelling tasks.

### Data findings

- **Delay rate by service type:** Express shipments had the highest delay rate at **48.97%**, compared with 20.56% for Standard and 8.57% for Economy shipments.
- **Delay rate by hub tier:** The delay rate increased from **4.00% at Tier 1** to **21.50% at Tier 2** and **44.00% at Tier 3**, indicating substantially higher delays at higher-tier hubs in this dataset.
- **Delay rate by distance band:** The delay rate increased with distance: **2.19%** for 0-200 km, **13.68%** for 201-400 km, **43.64%** for 401-600 km, and **85.23%** for distances above 600 km.
- **Correlation with delivery hours:** The three numeric columns most strongly correlated with `delivery_hours`, ranked by absolute Pearson correlation and excluding `delivery_hours` itself, were **promised_hours (+0.8372)**, **[distance_km] (+0.4013)**, and **[n_stops] (+0.3247)**. These correlations describe linear associations and do not establish causation.
- **Log transformation of declared value:** In the filled modelling table, `declared_value_inr` had a mean of **₹1,533.05** and a median of **₹1,121.50**, while `log_declared_value` had a mean of **7.028** and a median of **7.022**. The log transformation reduces the influence of large values and makes the distribution less right-skewed.

## Results

To be completed after running the analyses and evaluating the models. These will be added later.