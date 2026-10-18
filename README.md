# Shipment Delay Prediction and Customer Segmentation

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

- `generate_data.py` — provided data-generation script.
- `src/shipment_prep.py` — shared data cleaning and feature engineering.
- `part_a_clean_eda.py` — data cleaning and exploratory analysis.
- `part_b_regression.py` — regression analysis.
- `part_c_classification.py` — classification modelling.
- `part_d_segments.py` — segmentation analysis.
- `package_model.py` — model packaging.
- `predict_delay.py` — delay prediction.
- `data/` — generated datasets and modelling table.
- `models/` — saved model artifacts.
- `outputs/` — analysis and modelling outputs.

## Setup and Execution

Instructions will be added after the assignment pipeline is implemented.

## Results

To be completed after running the analyses and evaluating the models.