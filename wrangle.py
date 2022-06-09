import pandas as pd
import numpy as np
import os
from env import get_db_url
from sklearn.model_selection import train_test_split
import sklearn.preprocessing

def get_zillow_data():
    '''Returns a dataframe of all single family residential properties from 2017. Initial 
    query is from the Codeup database. File saved as CSV and called upon after initial query.'''
    filename = "zillow.csv"
    if os.path.isfile(filename):
        return pd.read_csv(filename)
    else:
        sql_query = '''
        SELECT properties_2017.bedroomcnt AS Number_of_Bedrooms,
        properties_2017.bathroomcnt AS Number_of_Bathrooms,
        properties_2017.calculatedfinishedsquarefeet AS Square_Feet, 
        properties_2017.taxvaluedollarcnt AS Tax_Appraised_Value, 
        properties_2017.yearbuilt AS Year_Built, 
        properties_2017.taxamount AS Tax_Assessed, properties_2017.fips AS County_Code,
        properties_2017.lotsizesquarefeet AS Lot_Size
        FROM properties_2017
        JOIN propertylandusetype USING (propertylandusetypeid)
        WHERE propertylandusedesc = 'Single Family Residential';
        '''
        df = pd.read_sql(sql_query, get_db_url('zillow'))
        # save as .csv
        df.to_csv('zillow.csv')
    return df


def prepare_zillow_data(df):
    ''' Prepares zillow data'''
    #drop null values
    df = df.dropna()
    # change fips codes to actual county name
    df['County_Code'].mask(df['County_Code'] == 6037, 'LA', inplace=True)
    df['County_Code'].mask(df['County_Code'] == 6111, 'Ventura', inplace=True)
    df['County_Code'].mask(df['County_Code'] == 6059, 'Orange', inplace=True)
    df.rename(columns = {'County_Code':'County'}, inplace = True)
    # one-hot encode County and concat to df
    dummy_df = pd.get_dummies(df[['County']],dummy_na=False, drop_first=False)
    df = pd.concat([df, dummy_df], axis=1)
    #limit homes to 1 bed , .5 bath, and at least 120sq ft     
    df = df[df.Square_Feet> 120]
    df = df[df.Number_of_Bedrooms > 0]
    df = df[df.Number_of_Bathrooms > .5]
    # convert floats to int except taxes and bedrooms
    df['Year_Built'] = df['Year_Built'].astype(int)
    df['Square_Feet'] = df['Square_Feet'].astype(int)
    df['Number_of_Bedrooms'] = df['Number_of_Bedrooms'].astype(int)
    df['Tax_Appraised_Value'] = df['Tax_Appraised_Value'].astype(int)
    df['Lot_Size'] = df['Lot_Size'].astype(int)
    # create a column for Tax Rates
    df['Tax_Rate'] = round((df.Tax_Assessed / df.Tax_Appraised_Value) * 100,2)
    return df
   
   
def handle_outliers(df):
    # handle outliers: square footage less than 10,000 and 7 beds and 7.5 baths or less
    df = df[df.Number_of_Bedrooms <=7]
    df = df[df.Number_of_Bathrooms <=7.5]
    df = df[df.Square_Feet <=10_000]
    df = df[df.Lot_Size <=20_000]
    df = df[df.Tax_Appraised_Value <=3_500_000]
    df = df[df.Tax_Appraised_Value >=50_000]
    return df

def split_zillow_data(df):
    ''' This function splits the cleaned dataframe into train, validate, and test 
    datasets and statrifies based on the target - Tax_Appraised_Value.'''

    train_validate, test = train_test_split(df, test_size=.2, 
                                        random_state=123)
    train, validate = train_test_split(train_validate, test_size=.3, 
                                   random_state=123)
    
    return train, validate, test


def scale_data(train,
              validate,
              test,
              columns_to_scale=['Number_of_Bedrooms','Number_of_Bathrooms', 'Square_Feet', 'Lot_Size']):
    '''
    Scales the split data.
    Takes in train, validate and test data and returns the scaled data.
    '''
    train_scaled = train.copy()
    validate_scaled = validate.copy()
    test_scaled = test.copy()
    
    #using MinMaxScaler (best showing distribution once scaled)
    scaler = sklearn.preprocessing.MinMaxScaler()
    scaler.fit(train[columns_to_scale])
    
    #creating a df that puts MinMaxScaler to work on the wanted columns and returns the split datasets and counterparts
    train_scaled[columns_to_scale] = pd.DataFrame(scaler.transform(train[columns_to_scale]),
                                                 columns=train[columns_to_scale].columns.values).set_index([train.index.values])
    
    validate_scaled[columns_to_scale] = pd.DataFrame(scaler.transform(validate[columns_to_scale]),
                                                 columns=validate[columns_to_scale].columns.values).set_index([validate.index.values])
    
    test_scaled[columns_to_scale] = pd.DataFrame(scaler.transform(test[columns_to_scale]),
                                                 columns=test[columns_to_scale].columns.values).set_index([test.index.values])
    
    
    return train_scaled, validate_scaled, test_scaled

def create_county_db(df):
    LA = df[df.County == 'LA']
    Orange = df[df.County == 'Orange']
    Ventura = df[df.County == 'Ventura']   
    return LA, Orange, Ventura 

# create functions that splits train_scaled, validate_scaled, test_scaled data by County
def train_county(train_scaled):
    LA_scaled_TR = train_scaled[train_scaled.County == 'LA']
    Orange_scaled_TR = train_scaled[train_scaled.County == 'Orange']
    Ventura_scaled_TR = train_scaled[train_scaled.County == 'Ventura']   
    return LA_scaled_TR, Orange_scaled_TR, Ventura_scaled_TR 

def validate_county(validate_scaled):
    LA_scaled_V = validate_scaled[validate_scaled.County == 'LA']
    Orange_scaled_V = validate_scaled[validate_scaled.County == 'Orange']
    Ventura_scaled_V = validate_scaled[validate_scaled.County == 'Ventura']   
    return LA_scaled_V, Orange_scaled_V, Ventura_scaled_V 

def test_county(test_scaled):
    LA_scaled_T = test_scaled[test_scaled.County == 'LA']
    Orange_scaled_T = test_scaled[test_scaled.County == 'Orange']
    Ventura_scaled_T = test_scaled[test_scaled.County == 'Ventura']   
    return LA_scaled_T, Orange_scaled_T, Ventura_scaled_T 