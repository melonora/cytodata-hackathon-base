from pathlib import Path
import pandas as pd

def save_predictions_classifier(preds, output_dir):
    """
    TODO: make this better? maybe use vol predictor code?
    TODO: drop unnecessary index
    """
    records = []
    for pred in preds:
        record = dict()
        for col in ["id", "y", "yhat"]:
            record[col] = pred[col].squeeze().numpy()
        record["loss"] = [pred["loss"].item()] * len(pred["id"])
        records.append(pd.DataFrame(record))

    pd.concat(records).reset_index().drop(columns="index").to_csv(
        Path(output_dir) / "model_predictions.csv", index_label=False
    )
    
def save_predictions_cell_stage(preds, output_dir):
#     """
#     TODO: DOESNT WORK PROPERLY! get 5 class predictions, need max from probability outputs
#     """
#     records = []
#     for pred in preds:
#         record = dict()
#         for col in ["id", "y", "yhat"]:
#             record[col] = pred[col].argmax()
#         record["loss"] = [pred["loss"].item()] * len(pred["id"])
#         records.append(pd.DataFrame(record))
    
#     pd.concat(records).reset_index().drop(columns="index").to_csv(
#         Path(output_dir) / "model_predictions.csv", index_label=False
#     )
    """
    TODO: make this better? maybe use vol predictor code?
    TODO: drop unnecessary index
    """
    records = []
    for pred in preds:
        record = dict()
        for col in ["id", "y"]:
            record[col] = pred[col].squeeze().numpy()
        for col in ["yhat"]:
            record[col] = np.argmax(pred[col].numpy(), 1)
        record["loss"] = [pred["loss"].item()] * len(pred["id"])
        records.append(pd.DataFrame(record))
    pd.concat(records).reset_index().drop(columns="index").to_csv(
        Path(output_dir) / "model_predictions_multiclass.csv", index_label=False
    )
    
def save_predictions_classifier_multiclass(preds, output_dir):
    """
    TODO: make this better? maybe use vol predictor code?
    TODO: drop unnecessary index
    """
    records = []
    for pred in preds:
        record = dict()
        for col in ["id", "y"]:
            record[col] = pred[col].squeeze().numpy()
        for col in ["yhat"]:
            record[col] = np.argmax(pred[col].numpy(), 1)
        record["loss"] = [pred["loss"].item()] * len(pred["id"])
        records.append(pd.DataFrame(record))
    pd.concat(records).reset_index().drop(columns="index").to_csv(
        Path(output_dir) / "model_predictions_multiclass.csv", index_label=False
    )