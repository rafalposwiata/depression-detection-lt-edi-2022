import os
from config import get_lm_pretraining_args
from simpletransformers.language_modeling import LanguageModelingModel
import logging

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)


class DepRoBERTa:

    def __init__(self,
                 model_type="roberta",
                 model_name="roberta-large",
                 model_version="v1",
                 pretrain_model=None,
                 data_dir="../data/reddit_depression_corpora",
                 data_name='reddit_depression_corpora'):
        self.model_type = model_type
        self.model_name = model_name
        self.model_version = model_version
        self.pretrain_model = pretrain_model
        self.data_dir = data_dir
        self.data_name = data_name
        self.lm_args = get_lm_pretraining_args()
        self.lm_args.output_dir = f"trained_models/deproberta_{model_version}"
        self.lm_args.cache_dir = f"trained_models/deproberta_{model_version}/cache"

        self.model = self._get_model()

    def train(self):
        train_data, val_data = self._get_data()
        self.model.train_model(train_data, eval_file=val_data)

    def eval(self):
        _, val_data = self._get_data()
        self.model.eval_model(val_data)

    def _get_model(self):
        model_path = self.pretrain_model if self.pretrain_model is not None else self.model_name
        return LanguageModelingModel(self.model_type, model_path, args=self.lm_args)

    def _get_data(self):
        train_data = os.path.join(self.data_dir, f'{self.data_name}_train.txt')
        val_data = os.path.join(self.data_dir, f'{self.data_name}_val.txt')
        return train_data, val_data


if __name__ == "__main__":
    deproberta = DepRoBERTa()
    deproberta.train()
    deproberta.eval()
