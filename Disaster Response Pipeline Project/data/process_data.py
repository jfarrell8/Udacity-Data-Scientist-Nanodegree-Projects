import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Load data given a messages file and a categories file.
    
    Args:
       messages_filepath: messages file
       categories_filepath: categories file
    
    Returns:
        df: DataFrame containing the messages and categories data
    """
    
    # load messages and categories data
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    # merge data into one df
    df = messages.merge(categories, on="id")
    return df


def clean_data(df):
    """
    Clean dataframe for input into a model in a future step.
    
    Args:
        df: dataframe to be cleaned
    
    Returns:
        df: cleaned dataframe
    """
    
    # split categories column in df into separate columns
    categories = pd.Series(df['categories']).str.split(";", expand=True)

    # select the first row of the categories dataframe
    row = categories.iloc[0,:]

    # use this row to extract a list of new column names for categories.
    category_colnames = row.apply(lambda x: x[:-2])

    # rename the columns of `categories`
    categories.columns = category_colnames  
    
    # convert category values to just numbers 0 or 1
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = pd.Series(categories[column]).apply(lambda x: x[-1])

        # convert column from string to numeric
        categories[column] = categories[column].apply(lambda x: int(x))

    # drop the original categories column from 'df'
    df.drop(columns='categories', inplace=True)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)

    # remove duplicates
    df.drop_duplicates(inplace=True)

    # remove 'related' category rows with value of 2
    indices = df[df['related'] == 2].index
    df.drop(indices, inplace=True)

    return df


def save_data(df, database_filename):
    """
    Save the dataframe table into a SQLite database. A database and a table
    will both be created in this function.
    
    Args:
        df: dataframe to save
        database_filename: database filename to create and save df to
    """
    
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('etl_disaster_pipeline', engine, index=False, if_exists='replace')  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()