import os
import pandas as pd
from models.config import global_config, get_fine_tuning_args
from models.models_list import get_models
from utils import find_best_checkpoint, norm
from simpletransformers.classification import ClassificationModel
from dataset.utils import get_preprocessed_data, labels
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class ResultAggregator:

    def __init__(self, model_name):
        self.model_name = model_name
        self.results = pd.DataFrame(columns=['acc', 'precision', 'recall', 'f1', 'model_version'])

    def add(self, model_version, result):
        if result is not None:
            result['model_version'] = model_version
            self.results = self.results.append(result, ignore_index=True)

    def get_result(self, mode='best', main_metric='f1'):
        values = []
        if mode == 'best':
            values = self._get_best_result_by_main_metric(main_metric)
        elif mode == 'mean':
            values = self._get_mean_result()

        return f"{self.model_name} & {' & '.join([str(norm(v)) for v in values if 'v' not in str(v)])} \\\\"

    def _get_best_result_by_main_metric(self, main_metric):
        return self.results.iloc[self.results[main_metric].idxmax()]

    def _get_mean_result(self):
        self.results.drop('model_version', axis=1, inplace=True)
        return self.results.mean()


def eval_model(model_info, model_args):
    eval_data = get_preprocessed_data(data_split)

    best_checkpoint = find_best_checkpoint(model_args.output_dir)
    if best_checkpoint is None:
        return None

    model = ClassificationModel(
        model_info.model_type,
        os.path.join(model_args.output_dir, best_checkpoint),
        num_labels=len(labels),
        args=model_args
    )

    print(f'Evaluating {model_info.description()}')
    predictions, raw_outputs = model.predict(list(eval_data['text'].values))
    y_true = eval_data['labels'].values

    return {
        'acc': norm(accuracy_score(y_true, predictions)),
        'precision': norm(precision_score(y_true, predictions, average='macro')),
        'recall': norm(recall_score(y_true, predictions, average='macro')),
        'f1': norm(f1_score(y_true, predictions, average='macro'))
    }


def print_results():
    for line in results:
        print(line)


if __name__ == '__main__':
    results = []
    data_split = 'dev'

    for model_info in get_models('basic') + get_models('DepRoBERTa'):
        agg = ResultAggregator(model_info.model_name)
        for version in range(global_config.runs):
            model_info.model_version = f'v{version + 1}'
            model_args, dropout = get_fine_tuning_args(model_info)
            model_args.eval_batch_size = 50

            agg.add(model_info.model_version, eval_model(model_info, model_args))
        results.append(agg.get_result())

    print_results()
