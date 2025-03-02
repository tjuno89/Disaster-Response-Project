import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data( messages_filepath ,categories_filepath  ):
    # Load messages dataset
    messages = pd.read_csv(messages_filepath)
    # Load categories dataset
    categories = pd.read_csv(categories_filepath )
    # Merge datasets
    df = pd.merge(messages, categories, on='id', how='inner')
    return df


def clean_data(df):
    # Split the values in the 'categories' column on the ';' character
    categories = df['categories'].str.split(';', expand=True)
    # Use the first row of categories dataframe to create column names for the categories data
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x.split('-')[0])
    # Rename columns of 'categories' with new column names
    categories.columns = category_colnames
    # Convert category values to just numbers 0 or 1
    for column in categories:
        categories[column] = categories[column].apply(lambda x: x[-1])
        categories[column] = pd.to_numeric(categories[column])
    # Drop the original categories column from df
    df = df.drop('categories', axis=1)
    # Concatenate the original dataframe with the new categories dataframe
    df = pd.concat([df, categories], axis=1)
    # Drop duplicates
    df = df.drop_duplicates()
    return df



def save_data(df, DisasterResponse):
    engine = create_engine(f'sqlite:///{DisasterResponse}')
    df.to_sql('Messages', engine, index=False, if_exists='replace')  

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