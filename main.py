# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from data_loading import SmileDataLoader
from data_encoding import SmileDataEncoder
def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    data_loader = SmileDataLoader('notebooks/Data/smile-annotations-final.csv')
    df = data_loader.get_df()
    X_train, X_val, y_train, y_val = data_loader.get_train_validation_set()

    dataloader_train, dataloader_val = data_loader.get_torch_dataloaders()

    
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
