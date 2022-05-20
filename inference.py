import argparse
import config
import xgboost as xgb
import pandas as pd

SELECTED_FEATURES = config.ICU_FEATURES

# Define default model and data paths
DEFAULT_MODEL_PATH = "models/icu-model.json"
COVID_DATA_PATH = "data/example_data_covid.json"
NONCOVID_DATA_PATH = "data/example_data_noncovid.json"


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", dest="model_path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("-d", dest="data_path", type=str, default=NONCOVID_DATA_PATH)
    args = parser.parse_args()

    # Load XGBoost model
    model = xgb.XGBClassifier()
    model.load_model(args.model_path)

    # Get predictions
    preds = read_and_predict(args.data_path, model)

    # Show and save results
    save_results(preds, file_name="results_" + args.data_path.split("/")[1])

    return


def read_and_predict(data_path, model):

    """
    Read data from json file and predict from the data using the provided model.
    Args:
        data_path (str): Path to json file with data.
        model (XGBoost object): XGBoost model.
    """

    # Read data
    data = pd.read_json(data_path)
    x_data = data[SELECTED_FEATURES]

    # Predict
    preds = model.predict_proba(x_data)[:, 1]

    return preds


def save_results(preds, file_name="results.json"):

    """Print and save a dataframe in JSON with predictions in ./results/{name}."""

    print(f"Saving results to {file_name}")
    result_df = pd.DataFrame({"Predictions": preds})
    print(result_df)
    result_df.to_json(f"results/{file_name}")

    return


if __name__ == "__main__":
    main()
