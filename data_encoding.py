import torch
from transformers import BertTokenizer
from torch.utils.data import TensorDataset

class DataEncoder(object):
    def __init__(self):
        pass

class SmileDataEncoder(DataEncoder):
    def __init__(self, df):
        self.df = df
        self.load_tokenizer()
        self.encode_data()

    def load_tokenizer(self):
        self.tokenizer = BertTokenizer.from_pretrained(
            'bert-base-uncased',
            do_lower_case=True
        )

    def encode_data(self):
        encoded_data_train = self.tokenizer.batch_encode_plus(
            self.df[self.df.data_type == 'train'].text.values,
            add_special_tokens=True,
            return_attention_mask=True,
            pad_to_max_length=True,
            max_length=256,
            return_tensors='pt'
        )

        encoded_data_val = self.tokenizer.batch_encode_plus(
            self.df[self.df.data_type == 'val'].text.values,
            add_special_tokens=True,
            return_attention_mask=True,
            pad_to_max_length=True,
            max_length=256,
            return_tensors='pt'
        )

        # encoded data val input_ids represents each input as a number
        # input_ids represent each token with a single number (an index)
        # attention_masks show which tokens are padding
        # labels_train provides a label for each training example (is a tensor containing a single value)
        input_ids_train = encoded_data_train['input_ids']
        attention_masks_train = encoded_data_train['attention_mask']
        labels_train = torch.tensor(self.df[self.df.data_type == 'train'].column_label.values)

        input_ids_val = encoded_data_val['input_ids']
        attention_masks_val = encoded_data_val['attention_mask']
        labels_val = torch.tensor(self.df[self.df.data_type == 'val'].column_label.values)

        self.dataset_train = TensorDataset(input_ids_train,
                                      attention_masks_train, labels_train)
        self.dataset_val = TensorDataset(input_ids_val,
                                    attention_masks_val, labels_val)



    def get_encoded_datasets(self):
        return self.dataset_train, self.dataset_val