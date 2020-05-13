import os
import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_pth, categories_pth):
    """ Loads the  disaster messages and categories into a dataframe.

    Arguments:
        messages_pth: Is a String,a csv file containing disaster messages.
        categories_pth: Is a string , a csv file containing disaster categories of each message.

    Returns:
       A pandas dataframe
    """
    # loading the  messages dataset
    if os.path.exists(messages_pth):
        messages = pd.read_csv(messages_pth)
    else:
        messages = pd.read_csv(messages_pth + '.gz', compression='gzip')

    # loading the categories dataset
    if os.path.exists(categories_pth):
        categories = pd.read_csv(categories_pth)
    else:
        categories = pd.read_csv(categories_pth + '.gz', compression='gzip')

    # merging both the datasets on id
    df = pd.merge(messages, categories, on='id', how='outer')

    return df

def clean_data(df):
    """Cleaning the data.

    Arguments:
        df : the pandas dataframe containing the disaster messages and the categories.

    Return:
        A new dataframe with cleaned data
    """
    
    categories = df['categories'].str.split(';', expand=True)
    ro_s = categories[0:1]

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything
    # up to the second to last character of each string with slicing
    cat_names = [(v.split('-'))[0] for v in ro_s.values[0]]

    # renaming the columns of categories
    categories.columns = cat_names

    for col in categories:

        categories[col] = categories[col].str[-1]
    
        # from str to numeric
        categories[col] = categories[col].astype(int)

    # binary conversion
    categories = (categories > 0).astype(int)

    # dropping categories column from the dataframe
    df.drop('categories', axis=1, inplace=True)

    # new categories df concatenated
    df = pd.concat([df, categories], axis=1)

    # dropping duplicates with inplace as True hence no assignment to df again
    df.drop_duplicates(inplace=True)

    return df

def save_data(df, db_filename, table_name='CleanedDataTable'):
    """Save data into database.

    Arguments:
        df: the pandas dataframe which contains the disaster messages and the categories that were cleaned.
        db_filename: Is a String, the df is saved into this database file.
        table_name:Is a String, the is saved into this table on the database.
    """
    engine = create_engine('sqlite:///' + db_filename)
    df.to_sql(table_name, engine, index=False, if_exists='replace', chunksize=600)

def main():
    if len(sys.argv) == 4:

        messages_pth, categories_pth, database_pth = sys.argv[1:]

        print('Loading the data now..... \n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_pth, categories_pth))
        df = load_data(messages_pth, categories_pth)

        print('Cleaning the data now.....')
        df = clean_data(df)

        print('Saving the final data.....\n    DATABASE: {}'.format(database_pth))
        save_data(df, database_pth)

        print('Cleaned data was saved to the  database')

    else:
        print('Please provide the path of the messages and categories '\
              'datasets as the  arguments in order, as '\
              'well as the path to the database for saving the cleaned data '\
              'as the third argument.')


if __name__ == '__main__':
    main()
