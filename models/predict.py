from models.models_list import get_models
from simpletransformers.classification import ClassificationModel
from dataset.utils import get_data, labels
import pandas as pd
import numpy as np
from scipy.special import softmax


def predict_for_single_model(model_info):
    print(f'Generating predictions using {model_info.description()}')
    predictions, raw_outputs = predict(model_info)
    generate_file(model_info.simple_name(), predictions)


def predict_for_average_ensemble(best_models):
    _, y1 = predict(best_models[0])
    y1 = softmax(y1, axis=1)
    _, y2 = predict(best_models[1])
    y2 = softmax(y2, axis=1)

    predictions = np.argmax((y1 + y2) / 2, axis=1)
    generate_file('Ensemble', predictions)


def predict(model_info):
    model = ClassificationModel(
        model_info.model_type,
        model_info.model_name,
        num_labels=len(labels),
    )

    predictions, raw_outputs = model.predict(list(test_data["text"].values))
    return predictions, raw_outputs


def generate_file(run_name, predictions):
    result = []
    for pid, prediction in zip(list(test_data["pid"].values), predictions):
        result.append([pid, labels[prediction]])
    pd.DataFrame(result, columns=["pid", "class_label"]).to_csv(f"{run_name}.tsv", sep='\t', index=False)


if __name__ == "__main__":
    test_data = get_data("test", without_label=True)

    best_models = get_models('best')
    for model_info in best_models:
        predict_for_single_model(model_info)

    predict_for_average_ensemble(best_models)
