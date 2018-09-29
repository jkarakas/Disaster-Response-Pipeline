#! /usr/bin/env python3
# coding=utf-8

# The Data processing script


import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """Load  two csv files into pandas dataframes and
    merge them into one.

    Parameters
    ----------
    messages_filepath : string
        location of the messages csv file
    categories_filepath : string
        location of the categories csv file

    Returns
    -------
    pandas.DataFrame
        The merged dataframe
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    return pd.merge(messages, categories, on='id')


def clean_data(df):
    """
    Process a dataframe

    Parameters
    ----------
    df: pandas.DataFrame
        The pandas.Dataframe to be processed

    Returns
    -------
    pandas.DataFrame
        The processed dataframe
    """

    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(pat=';', n=None, expand=True)

    # use the first row to extract a list of new column names for categories.
    category_colnames = categories.iloc[0].str[:-2]

    # rename the columns of `categories`
    categories.columns = category_colnames

    for column in categories:
        # set each value to be the last character of the string
        # and cast to int
        categories[column] = categories[column].str[-1].astype(int)

    # drop the original categories column from `df`
    df = df.drop(columns='categories')

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)

    # drop duplicates
    df = df.drop_duplicates(keep='first')

    return df


def save_data(df, database_filename):
    """
    Writes a dataframe to a Sql-lite Database

    Parameters
    ----------
    df: pandas.DataFrame
        The pandas.Dataframe to be written
    database_filename: string
        The filename path for the database

    Returns
    -------
    None
    """
    print('Writing {} to {} database: '.format(df, database_filename))
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('messages_categories', engine, index=False)


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
        print('Please provide the filepaths of the messages and categories '
              'datasets as the first and second argument respectively, as '
              'well as the filepath of the database to save the cleaned data '
              'to as the third argument. \n\nExample: python process_data.py '
              'disaster_messages.csv disaster_categories.csv '
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
