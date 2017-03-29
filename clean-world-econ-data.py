import pandas as pd
import numpy as np
import itertools
import os
from fancyimpute import KNN

pd.set_option('display.max_columns', 2500)
pd.set_option('max_rows', 2500)

# the indicators file can be found here: https://www.kaggle.com/worldbank/world-development-indicators

# get a dataframe filtered by the dates of with the maximum number
# of countries all with consecutive data for min_consecutive_years
def get_countries_with_overlapping_consecutive_years(input_df, min_consecutive_years):
    country_years = pd.DataFrame(input_df, columns=['CountryName', 'Year'])
    country_years = country_years.groupby('Year')
    min_year = input_df['Year'].min()
    country_count_consec_years = {}
    best_interval_countries = set()
    best_interval_start = 0
    best_interval_end = 0
    year_group_dict = dict(list(country_years))

    for idx, (year, group) in enumerate(country_years):
        lower_year = year - min_consecutive_years
        if lower_year >= min_year:
            country_years_for_low_year = year_group_dict[lower_year]
            for (index, data) in country_years_for_low_year.iterrows():
                cur_c_name = data['CountryName']
                country_count_consec_years[cur_c_name] -= 1

        for (index, data) in group.iterrows():
            prev_count = country_count_consec_years.get(data['CountryName'], 0)
            country_count_consec_years[data['CountryName']] = prev_count + 1

        countries_in_interval = set()
        for country_name, count in country_count_consec_years.items():
            if count >= min_consecutive_years:
                countries_in_interval.add(country_name)

        if len(countries_in_interval) > len(best_interval_countries):
            best_interval_start = lower_year
            best_interval_end = year
            best_interval_countries = countries_in_interval

    if len(best_interval_countries) < len(input_df['CountryName'].unique()):
        print('Unable to find window including all countries, using best interval ' + str(best_interval_countries))

    input_df = input_df[input_df['Year'] >= best_interval_start]
    input_df = input_df[input_df['Year'] <= best_interval_end]
    input_df = input_df[input_df['CountryName'].isin(best_interval_countries)]
    return input_df


def get_raw_data_frame(f_name):
    with open(f_name) as indicators_file:
        indicators_df = pd.read_csv(indicators_file)

    grouped_indicators = pd.groupby(indicators_df, ['CountryName', 'Year'])
    print('Num country year groups = ' + str(len(grouped_indicators)))
    rows_list = []
    for idx, (name, group) in enumerate(grouped_indicators):
        cur_row = {}
        cur_row['CountryName'] = name[0]
        cur_row['Year'] = name[1]
        for (index, data) in group.iterrows():
            cur_indicator_name = data['IndicatorName']
            cur_indicator_value = data['Value']
            cur_row[cur_indicator_name] = cur_indicator_value

        rows_list.append(cur_row)

    return pd.DataFrame(rows_list)


def prune_results_by_target(input_df, output_col_name, output_next_suffix, min_observations, drop_nulls):
    input_df = input_df[input_df[output_col_name].notnull()]

    if drop_nulls:
        for col in input_df:
            non_nuls_in_col = input_df[input_df[col].notnull()]
            if len(non_nuls_in_col) < min_observations:
                input_df = input_df.drop(col, 1)
            else:
                input_df = non_nuls_in_col

    input_df = input_df.sort_values(by=['CountryName', 'Year'])
    input_df['next_gni_per_capita'] = np.NaN
    result_cpy = input_df.copy()
    cur_row, next_row = itertools.tee(result_cpy.iterrows())
    next(next_row, None)
    zip_iter = zip(cur_row, next_row)

    next_output_col_name = output_col_name + output_next_suffix

    for (idx, row_1_data), (_, row_2_data) in zip_iter:
        row_1_year = row_1_data['Year']
        row_2_year = row_2_data['Year']

        def get_slim(df_to_slim):
            return df_to_slim['CountryName'], df_to_slim['Year']

        if (row_1_data['CountryName'] == row_2_data['CountryName']) and (row_1_year + 1 == row_2_year):
            input_df.set_value(idx, next_output_col_name, row_2_data[output_col_name])
        else:
            print('Skipping pair (%s, %s)' % (get_slim(row_1_data), get_slim(row_2_data)))

    if drop_nulls:
        input_df = input_df[input_df[next_output_col_name].notnull()]

    return input_df


#Use KNN to impute missing values
def impute_missing_data(input_df):
    # Drop all null columns up front
    for col in input_df:
        non_nuls_in_col = input_df[input_df[col].notnull()]
        if len(non_nuls_in_col) == 0:
            input_df = input_df.drop(col, 1)

    df_mat = input_df.drop(['CountryName', 'Year'], 1).as_matrix()
    filled_mat = KNN(k=10).complete(df_mat)
    retval_cols = input_df.columns.values
    retval_cols = list(filter(lambda x: x != 'CountryName' and x != 'Year', retval_cols))
    ret_df = pd.DataFrame(filled_mat, columns=retval_cols)
    ret_df['CountryName'] = input_df['CountryName'].values
    ret_df['Year'] = input_df['Year'].values
    return ret_df


param_file_name = 'data/econ-indicators.csv'
param_output_col_name = 'GNI per capita (constant 2005 US$)'
param_output_next_suffix = '_next'
param_min_observations = 10
param_desired_consecutive_years = 10
param_null_handling_mode = 'IMPUTE' #one of DROP, IMPUTE, IGNORE
param_output_file_name = 'cleaned-econ-data.csv'

result_df = get_raw_data_frame(param_file_name)
result_df = prune_results_by_target(result_df, param_output_col_name, param_output_next_suffix, param_min_observations,
                                    param_null_handling_mode == 'DROP')

gnis_by_country_year = pd.DataFrame(result_df, columns=['CountryName', 'Year', param_output_col_name,
                                                        param_output_col_name + param_output_next_suffix])
print(gnis_by_country_year)

result_df = get_countries_with_overlapping_consecutive_years(result_df, param_desired_consecutive_years)

if param_null_handling_mode == 'IMPUTE':
    result_df = impute_missing_data(result_df)

result_df.to_csv(os.path.join('data', param_output_file_name), index=False)
