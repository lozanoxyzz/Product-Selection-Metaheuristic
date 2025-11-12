import pandas as pd
import os


FILES = {
        '50': "dataset_PIA_50.csv",
        '100': "dataset_PIA_100.csv",
        '150': "dataset_PIA_150.csv"
        }

##### PHASE 0: DATA LOADING #####

def load_datasets():

    # DATASET Options
    print('1- 50 products')
    print('2- 100 products')
    print('3- 150 products')
    choice = input('Choose an option: ')

    if choice == "1":
        filename = FILES['50']
    elif choice == "2":
        filename = FILES["100"]
    elif choice == "3":
        filename = FILES["150"]
    else:
        print("Invalid option. Exiting")
        return None

    df = pd.read_csv(filename)
    df.columns = [c.strip().lower() for c in df.columns]

    return df

if __name__ == '__main__':
    print(load_datasets())


