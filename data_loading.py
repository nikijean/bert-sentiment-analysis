import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

class DataLoader(object):
    def load_data(self):
        pass

class SmileDataLoader(DataLoader):
    def __init__(self, data_path):
        self.data_path = data_path
        self.load_data()
        self.train_test_split()
        self.encode_data()
        self.load_torch_dataloaders()

    def encode_data(self):
        data_encoder = SmileDataEncoder(self.df)
        self.dataset_train, self.dataset_val = data_encoder.get_encoded_datasets()

    def load_data(self):
        # read in the smile dataset into a df with columns as indicated in the names list
        df = pd.read_csv(
            self.data_path,
            names=['id', 'text', 'category'])
        # set id column to also be the index
        df.set_index('id', inplace=True)
        # look at a tweet
        # df.text.iloc[0]
        # look at the label distribution
        #df.category.value_counts()
        # we only want the single labels for this use case.
        # remove all tweets with the nocode category
        df = df[df.category != 'nocode']
        #look at value counts again:
        #df.category.value_counts()

        # remove all tweets with multiple categories
        df = df[df.category.str.contains("\|") == False]
        #df.category.value_counts()

        # note there is a class imbalance
        # create a dictionary that gives an id to each category

        id_dict = {}
        possible_labels = df.category.unique()

        label_dict = {}
        for i, label in enumerate(possible_labels):
            label_dict[label] = i

        #Add the numerical label into the df
        def get_value(row):
            value = row["category"]
            return label_dict[value]

        df["column_label"] = df.apply(lambda row: get_value(row), axis=1)
        # alternative:
        # df["column_label"]= df.category.replace(label_dict)

        self.df = df

    def train_test_split(self):
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            self.df.index.values,
            self.df.column_label.values,
            test_size=0.15,
            random_state=17,
            stratify=self.df.column_label.values
        )
        self.df['data_type'] = ['not_set'] * self.df.shape[0]
        self.df.loc[self.X_train, 'data_type'] = 'train'
        self.df.loc[self.X_val, 'data_type'] = 'val'
        #df.groupby(['category', 'column_label', 'data_type']).count()

    def load_torch_dataloaders(self):
        #dataloader is an iterable. each item in the iterable is a
        #a batch of {batch_size}. Each batch consists of three parallel
        # lists. List at index [0] is the training sample representation (a set of indices)
        # List at index [1] is the attention_masks, which tells the model which words to
        # ignore because theyt are padding. List at index [2] provides the ground truth
        # labels for the corresponding training samples.
        #TODO: set batch_size externally
        batch_size = 4
        self.dataloader_train = DataLoader(
            self.dataset_train,
            sampler=RandomSampler(self.dataset_train),  # prevents model from learning sequence based info
            batch_size=batch_size
        )

        self.dataloader_val = DataLoader(
            self.dataset_val,
            sampler=RandomSampler(self.dataset_val),  # prevents model from learning sequence based info
            batch_size=32  # not many computations, so we can do 32
        )

    def get_torch_dataloaders(self):
        return self.dataloader_train, self.dataloader_val

    def get_df(self):
        return self.df

    def get_train_validation_set(self):
        return self.X_train, self.X_val, self.y_train, self.y_val


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

if __name__ == '__main__':
    data_loader = SmileDataLoader('notebooks/Data/smile-annotations-final.csv')