from transformers import BertForSequenceClassification
from transformers import AdamW, get_linear_schedule_with_warmup
import random
import numpy as np
import torch
from evaluating import Evaluator
from tqdm import tqdm

class BertModel(object):
    def __init__(self, label_dict, dataloader_train, dataloader_val, epochs=3):
        self.epochs = epochs
        self.num_labels = len(label_dict)
        self.dataloader_train = dataloader_train
        self.dataloader_val = dataloader_val
        self.evaluator = Evaluator(label_dict)
        self._load()
        self._initialize_seeds()
        self._initialize_device()

    def _load(self):
        self._load_model()
        self._load_optimizer()
        self._load_scheduler()

    def _initialize_seeds(self):
        # seeding our random values:
        seed_val = 17
        random.seed(seed_val)
        np.random.seed(seed_val)
        torch.manual_seed(seed_val)
        torch.cuda.manual_seed_all(seed_val)

    def _initialize_device(self):
        # make sure we're using the right device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        # sanity check for which device is in use
        print(f'Device in use: {self.device}')

    def _load_model(self):
        self.model = BertForSequenceClassification.from_pretrained(
            'bert-base-uncased',
            num_labels=self.num_labels,
            output_attentions=False,
            output_hidden_states=False
        )

    def _load_pretrained(self):
        #load the model from an already finetuned version
        self.model = BertForSequenceClassification.from_pretrained("bert-base-uncased",
                                                              num_labels=self.num_labels,
                                                              output_attentions=False,
                                                              output_hidden_states=False)
        self.model.load_state_dict(torch.load(
            'Models/<<INSERT MODEL NAME HERE>>.model',
            map_location=torch.device('cpu')
        ))

    def _load_optimizer(self):
        # optimizer defining learning rate and optimizing weights
        # scheduler definingand how learning rate changes through time
        # AdamW = adam with weight decay
        # linear schedule with warmup builtin from transformers
        # TODO: lookup AdamW
        # original bert paper recs something 2e-5 - 5e-5
        # what works well is 1e-5
        # epsilon value default is 1e-8
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=1e-5,
            eps=1e-8
        )

    def _load_scheduler(self):
        epochs = self.epochs  # 10 works well

        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=0,  # default value
            num_training_steps=len(self.dataloader_train) * epochs
            # this is how many times we want the learning rate to change
        )

    def evaluate(self):
        # performs similarly to training, but does so in
        # evaluation mode, which means it does not update weights, etc
        # and you ignore the gradients
        # and we use our logits for our predictions
        self.model.eval()

        loss_val_total = 0
        predictions, true_vals = [], []

        for batch in self.dataloader_val:
            batch = tuple(b.to(self.device) for b in batch)

            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'labels': batch[2],
                      }

            with torch.no_grad():
                outputs = self.model(**inputs)
            loss = outputs[0]
            logits = outputs[1]
            loss_val_total += loss.item()

            # if using gpu, you will need to pull the values onto the cpu
            # in order to use with numpy
            logits = logits.detach().cpu().numpy()
            label_ids = inputs['labels'].cpu().numpy()
            predictions.append(logits)
            true_vals.append(label_ids)

        loss_val_avg = loss_val_total / len(self.dataloader_val)

        predictions = np.concatenate(predictions, axis=0)
        true_vals = np.concatenate(true_vals, axis=0)

        return loss_val_avg, predictions, true_vals

    def train(self):
        for epoch in tqdm(range(1, self.epochs + 1)):

            self.model.train()

            loss_train_total = 0

            progress_bar = tqdm(self.dataloader_train, desc='Epoch {:1d}'.format(epoch), leave=False, disable=False)
            for batch in progress_bar:
                self.model.zero_grad()

                batch = tuple(b.to(self.device) for b in batch)

                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'labels': batch[2],
                          }

                outputs = self.model(**inputs)

                loss = outputs[0]
                loss_train_total += loss.item()
                loss.backward()
                #clipping the gradient and normalizing to prevent extreme highs and lows
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                #incrementing the optimizer and scheduler
                self.optimizer.step()
                self.scheduler.step()

                #updating the progress bar
                progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item() / len(batch))})

            torch.save(self.model.state_dict(), f'finetuned_BERT_epoch_{epoch}.model')

            tqdm.write(f'\nEpoch {epoch}')

            loss_train_avg = loss_train_total / len(self.dataloader_train)
            tqdm.write(f'Training loss: {loss_train_avg}')

            val_loss, predictions, true_vals = self.evaluate()
            val_f1 = self.evaluator.f1_score_func(predictions, true_vals)
            tqdm.write(f'Validation loss: {val_loss}')
            tqdm.write(f'F1 Score (Weighted): {val_f1}')

    def validation_accuracy(self):
        _, predictions, true_vals = self.evaluate()
        self.evaluator.accuracy_per_class(predictions, true_vals)