
import numpy as np
import pandas as pd

def bools(df):
    """
    public_meeting: we will fill the nulls as 'False'
    permit: we will fill the nulls as 'False
    """
    z = ['public_meeting', 'permit']
    for i in z:
        df[i].fillna(False, inplace = True)
        df[i] = df[i].apply(lambda x: float(x))
    return df

def locs(df, trans = ['longitude', 'latitude', 'gps_height', 'population']):
    """
    fill in the nulls for ['longitude', 'latitude', 'gps_height', 'population'] by using medians from 
    ['subvillage', 'district_code', 'basin'], and lastly the overall median
    """
    df.loc[df.longitude == 0, 'latitude'] = 0
    for z in trans:
        df[z].replace(0., np.NaN, inplace = True)
        df[z].replace(1., np.NaN, inplace = True)
        
        for j in ['district_code', 'basin']:
        
            df['median'] = df.groupby([j])[z].transform('median')
            df[z] = df[z].fillna(df['median'])
        
        df[z] = df[z].fillna(df[z].median())
        del df['median']
    return df

def construction(df):
    """
    A lot of null values for construction year. Of course, this is a missing value (a placeholder).
    For modeling purposes, this is actually fine, but we'll have trouble with visualizations if we
    compare the results for different years, so we'll set the value to something closer to
    the other values that aren't placeholders. Let's look at the unique years and set the null
    values to 50 years sooner.
    Let's set it to 1910 since the lowest "good" value is 1960.
    """
    df.loc[df['construction_year'] < 1950, 'construction_year'] = 1910
    return df

# Alright, now let's drop a few columns
# Needed to drop quite a few categorical columns so that the data would fit in memory in Azure
# Tested the model before and after (from 6388 columns to 278) in Colab and only had a ~0.03% reduction in performance

def removal(df):
  # id: we drop the id column because it is not a useful predictor.
  # amount_tsh: is mostly blank - delete
  # wpt_name: not useful, delete (too many values)
  # subvillage: too many values, delete
  # scheme_name: this is almost 50% nulls, so we will delete this column
  # num_private: we will delete this column because ~99% of the values are zeros.
  features_to_drop = ['id','amount_tsh',  'num_private', 
          'quantity', 'quality_group', 'source_type', 'payment', 
          'waterpoint_type_group', 'extraction_type_group', 'wpt_name', 
          'subvillage', 'scheme_name', 'funder', 'installer', 'recorded_by',
          'ward']
  df = df.drop(features_to_drop, axis=1)

  return df

def dummy(df):
    dummy_cols = ['basin', 'lga', 'public_meeting',
       'scheme_management', 'permit', 'extraction_type',
       'extraction_type_class', 'management', 'management_group',
       'payment_type', 'water_quality', 'quantity_group', 'source',
       'source_class', 'waterpoint_type', 'region']

    df = pd.get_dummies(df, columns=dummy_cols)

    return df

def dates(df):
    """
    date_recorded: this might be a useful variable for this analysis, although the year itself would be useless in a practical scenario moving into the future. We will convert this column into a datetime, and we will also create 'year_recorded' and 'month_recorded' columns just in case those levels prove to be useful. A visual inspection of both casts significant doubt on that possibility, but we'll proceed for now. We will delete date_recorded itself, since random forest cannot accept datetime
    """
    df['date_recorded'] = pd.to_datetime(df['date_recorded'])
    df['year_recorded'] = df['date_recorded'].apply(lambda x: x.year)
    df['month_recorded'] = df['date_recorded'].apply(lambda x: x.month)
    df['date_recorded'] = (pd.to_datetime(df['date_recorded'])).apply(lambda x: x.toordinal())
    return df

def dates2(df):
    """
    Turn year_recorded and month_recorded into dummy variables
    """
    for z in ['month_recorded', 'year_recorded']:
        df[z] = df[z].apply(lambda x: str(x))
        good_cols = [z+'_'+i for i in df[z].unique()]
        df = pd.concat((df, pd.get_dummies(df[z], prefix = z)[good_cols]), axis = 1)
        del df[z]
    return df

def small_n(df):
    "Collapsing small categorical value counts into 'other'"
    cols = [i for i in df.columns if type(df[i].iloc[0]) == str]
    df[cols] = df[cols].where(df[cols].apply(lambda x: x.map(x.value_counts())) > 100, "other")
    return df