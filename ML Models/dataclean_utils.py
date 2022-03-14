import pandas as pd


def get_column_dtypes(df):
    '''
    :param: df = pandas dataframe
    returns list of datatypes, one for each column in df
    '''
    dtypes = []
    for col in df.columns:
        dtypes.append(df[col].dtype)
    return dtypes


def set_column_dtypes(df, dtypes):
    '''
    sets the datatypes of items in columns of df, according to dtypes
    :param: df = pandas dataframe
    :param: dtypes = list of datatype objects, one for each column in df
    '''
    assert (len(df.columns) == len(dtypes))
    col_dtype = {df.columns[i]: dtypes[i] for i in range(len(df.columns))}
    for col in col_dtype:
        df[col].astype(col_dtype[col])
    return df


def show_unique_objects(df):
    '''
    if column in df is dtype "Object", print the columns unique values
    useful to see all the values present in a dataframe
    '''
    for col in df.columns:
        if df[col].dtype == "O":
            print(col + ': ', df[col].unique())
    return None


def explode(df, remove_unknown=False):
    '''
    Turns any column in df that is categorical (ie: has dtype Object) into multiple columns, with one column for
    each unique value. Populates these new columns with 1, or 0, depending on which categorical variable the original
    column contained.
    :param: df = pandas dataframe
    :param: remove_unknown = True if we want to disguard any "unknown" values
    '''

    exploded = pd.DataFrame()
    for col in df.columns:
        if df[col].dtype == 'int':
            exploded = pd.concat([exploded, df[col]], axis=1)
        elif df[col].dtype == "O":
            unique = df[col].unique()
            if remove_unknown: # remove "Unknown" if specified
                unique = unique[unique != "Unknown"]
            for item in unique:
                expcol = (df[col] == item).astype(int)
                expcol.name = str(df[col].name) + '_' + str(item)  # create new column name
                exploded = pd.concat([exploded, expcol], axis=1)
    return exploded