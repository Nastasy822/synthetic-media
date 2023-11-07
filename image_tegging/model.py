from transformers import ViTImageProcessor, BertTokenizer, VisionEncoderDecoderModel
import os
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = str(0.0)
import evaluate
from transformers import Trainer, TrainingArguments
import torch
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
# from aac_metrics.functional import spice

class ImageToTextModel:
    def __init__(self):
        self.training_args = TrainingArguments(
                                output_dir='./results',
                                num_train_epochs=1,
                                per_device_train_batch_size=2,
                                per_device_eval_batch_size=2,
                                warmup_steps=2,
                                weight_decay=0.01,
                                logging_dir='./logs',
                                logging_steps=2,
                                report_to=["tensorboard"],
                                logging_strategy="steps",
                                evaluation_strategy = "epoch", #To calculate metrics per epoch
                                )

    def init_model(self, name_of_encoder, name_of_decoder, name_tokenizer):

        self.model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
                                                                                name_of_encoder,
                                                                                name_of_decoder,
                                                                                )
        # self.device = torch.device('mps')
        # self.model = self.model.to(device=self.device)

        encoder = ViTImageProcessor.from_pretrained(name_of_encoder)
        self.tokenizer = BertTokenizer.from_pretrained(name_tokenizer)

        self.model.config.decoder_start_token_id = self.tokenizer.cls_token_id
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

        return encoder, self.tokenizer

    def get_data_for_calculate_metrics(self, eval_preds):
        real_label = []
        pred_label = []
        logits, labels = eval_preds
        for i in range(len(labels)):
            pred_label.append(np.argmax(logits[0][i], axis=-1))
            real_label.append(labels[i])

        return real_label, pred_label

    # The BLEU score is typically represented as a value between 0 and 1,
    # 1 - perfect match
    # 0 - being a perfect mismatch to the reference translations
    def blue(self, real_label, pred_label):
        decoded_real_label = [self.tokenizer.decode(i) for i in real_label]
        decoded_pred_label = [self.tokenizer.decode(i) for i in pred_label]

        return corpus_bleu(decoded_real_label, decoded_pred_label)

    # The rouge - score is
    def rouge(self, real_label, pred_label):
        rouge = evaluate.load('rouge')
        decoded_real_label = [self.tokenizer.decode(i) for i in real_label]
        decoded_pred_label = [self.tokenizer.decode(i) for i in pred_label]

        return rouge.compute(predictions=decoded_pred_label, references=decoded_real_label)

    def spice(self, real_label, pred_label):
        decoded_real_label = [self.tokenizer.decode(i) for i in real_label]
        decoded_pred_label = [self.tokenizer.decode(i) for i in pred_label]

        # return spice(decoded_pred_label, decoded_real_label)

    def compute_metrics(self, eval_preds):
        real_label, pred_label = self.get_data_for_calculate_metrics(eval_preds)
        return {'blue': self.blue(real_label, pred_label),
                'rouge': self.rouge(real_label, pred_label)['rougeL'],
                # 'spice':  self.spice(real_label, pred_label)['rougeL'],
                }


    def train(self, datsets_train, dataset_val):

        self.trainer = Trainer(
                model=self.model,
                args=self.training_args,
                train_dataset=datsets_train,
                eval_dataset=dataset_val,
                compute_metrics=self.compute_metrics,
                )

        self.trainer.train()

    def predict(self, dataset_test):
        # tokenizer.decode(encoded['input_ids'])
        return self.tokenizer.decode(self.trainer.predict(dataset_test)[1][0])

