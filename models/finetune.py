import argparse
from metrics import macro_f1_score
from utils import find_best_checkpoint, clean
from transformers_models import MODEL_CLASSES
from simpletransformers.classification import ClassificationModel
from dataset.utils import get_preprocessed_data, labels
from models_list import get_models
from config import get_fine_tuning_args, global_config


def fine_tune():
    print(f'Fine-tuning\t{model_info.description()}')

    train_data = get_preprocessed_data("train", use_shuffle=True)
    eval_data = get_preprocessed_data("dev")

    model = ClassificationModel(
        model_info.model_type,
        model_info.get_model_path(),
        num_labels=len(labels),
        args=model_args,
    )

    set_dropout(model)
    model.train_model(train_data, eval_df=eval_data, f1=macro_f1_score)

    best_checkpoint = find_best_checkpoint(model_args.output_dir)
    clean(model_args.output_dir, best_checkpoint)


def set_dropout(model: ClassificationModel):
    config = model.config
    config.attention_probs_dropout_prob = dropout.att_dropout
    config.hidden_dropout_prob = dropout.h_dropout
    config.classifier_dropout = dropout.c_dropout

    config_class, model_class, tokenizer_class = MODEL_CLASSES[model_info.model_type]
    model.model = model_class.from_pretrained(model_info.get_model_path(), config=config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=str, default="basic")
    args = parser.parse_args()

    for model_info in get_models(args.models):
        for version in range(global_config.runs):
            model_info.model_version = f'v{version + 1}'
            model_args, dropout = get_fine_tuning_args(model_info)

            fine_tune()
