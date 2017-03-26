import pandas as pd
import numpy as np
import itertools

pd.set_option('display.max_columns', 2500)
pd.set_option('max_rows', 2500)
output_col_name = 'GNI per capita (constant 2005 US$)'
min_observations = 1000

with open('data/econ-indicators.csv') as indicators_file:
    indicators_df = pd.read_csv(indicators_file)
    grouped_indicators = pd.groupby(indicators_df, ['CountryName', 'Year'])
    print('num groups = ' + str(len(grouped_indicators)))
    rows_list = []
    for idx, (name, group) in enumerate(grouped_indicators):
        print(name)
        cur_row = {}
        cur_row['CountryName'] = name[0]
        cur_row['Year'] = name[1]
        for (index, data) in group.iterrows():
            cur_indicator_name = data['IndicatorName']
            cur_indicator_value = data['Value']
            cur_row[cur_indicator_name] = cur_indicator_value

        rows_list.append(cur_row)

    result_df = pd.DataFrame(rows_list)
    result_df = result_df[result_df[output_col_name].notnull()]

    for col in result_df:
        non_nuls_in_col = result_df[result_df[col].notnull()]
        if len(non_nuls_in_col) < min_observations:
            result_df = result_df.drop(col, 1)
        else:
            result_df = non_nuls_in_col

    result_df.sort_values(by=['CountryName', 'Year'], inplace=True)
    result_df['next_gni_per_capita'] = np.NaN
    result_cpy = result_df.copy()
    cur_row, next_row = itertools.tee(result_cpy.iterrows())
    next(next_row, None)
    zip_iter = zip(cur_row, next_row)

    for (idx, row_1_data), (_, row_2_data) in zip_iter:
        row_1_year = row_1_data['Year']
        row_2_year = row_2_data['Year']

        def get_slim(df_to_slim):
            return df_to_slim['CountryName'], df_to_slim['Year']

        if (row_1_data['CountryName'] == row_2_data['CountryName']) and (row_1_year + 1 == row_2_year):
            result_df.set_value(idx, 'next_gni_per_capita', row_2_data[output_col_name])
        else:
            print('Skipping pair (%s, %s)' % (get_slim(row_1_data), get_slim(row_2_data)))

    result_df = result_df[result_df['next_gni_per_capita'].notnull()]
    gnis_by_country_year = pd.DataFrame(result_df, columns=['CountryName', 'Year', output_col_name,
                                                            'next_gni_per_capita'])
    print(gnis_by_country_year)
    result_df.to_csv('data/cleaned-econ-data.csv', index=False)
