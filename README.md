# ICU Mortality Risk Prediction Model

This repository contains a minimal inference pipeline 
for mortality risk prediction on Covid and non-Covid patients in the intensive care unit (ICU) using an XGBoost model.
Data collection and model development was a collaboration between Unetiq GmbH and Universitätsklinikum Regensburg (UKR) 
as part of the NUM CODEX+ project.

## Background

Mortality prediction is a crucial part of the risk assessment of ICU patients to enable rapid decision-making.
Conventional risk scores such as SOFA or APACHE II often have limited predictive power, 
especially on Covid patients that increasingly needed ICU treatment during the Covid pandemic.
For this model, top predictive ICU measurement features have been identified through feature selection and importance analysis
on both Covid and non-Covid patient cohorts. 
The model significantly outperforms APACHE II (AUC 0.64) and SOFA (0.6) scores on Covid patients.


## Tech Stack

| Part     | Technology       |
|----------|------------------|
| Backend | Python + xgboost | 

Notes on additional required packages:```scikit-learn``` is a dependency of xgboost; 
```pandas``` is used for Python-specific, convenient data processing.

<!---| Deployment |                  |
-->

## Usage

#### Prepare Environment

Install required Python dependencies with

```shell
pip install -r requirements.txt
```

#### Data

Aggregated ICU measurement data is processed from JSON files located in ```data/```. 
Example data from 10 Covid and 10 non-Covid patients is included for demonstration.

#### Model

The XGBoost model is loaded from a JSON file in ```models/```. The model is still under development and is currently based on data collected up until May 2021.

#### Prediction

Inference example:
```python
python inference.py -m icu-model.json -d example_data_covid.json
```
The script outputs the predicted per-case mortality risk as probability scores. 
Predictions are saved in ```results/```.

#### Notes on Individual Measurement Features

To avoid feature redundancy and impute missing values, 
the relations within some feature groups have been researched and analyzed. 
The results were integrated into the model and the example data in the following ways:

The model assumes **body temperature** to have been monitored invasively (through the bladder). 
For cases where monitored temperature was not available, 
body temperature measured manually was converted to monitored temperature by simple replacement (_Temp *mean_) and by adding 0.55°C to the manual temperature value (_Temp *max_).