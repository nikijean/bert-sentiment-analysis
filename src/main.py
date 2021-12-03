
from data_loading import SmileDataLoader
from data_encoding import SmileDataEncoder
from modeling import BertModel

if __name__ == '__main__':
    data_loader = SmileDataLoader('data/smile-annotations-final.csv')
    df = data_loader.get_df()
    X_train, X_val, y_train, y_val = data_loader.get_train_validation_set()

    dataloader_train, dataloader_val = data_loader.get_torch_dataloaders()
    bert_model = BertModel(data_loader.label_dict, dataloader_train, dataloader_val)
    bert_model.train()

    
