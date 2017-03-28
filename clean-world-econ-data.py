import pandas as pd
import numpy as np
import itertools

pd.set_option('display.max_columns', 2500)
pd.set_option('max_rows', 2500)

# the indicators file can be found here: https://www.kaggle.com/worldbank/world-development-indicators


# Marks beginning and end of consecutive years of data for a country
# returns list of tuples of format (country, year, event one of 'begin' 'end')
def compute_country_year_intervals(input_df):
    country_year_events = []
    distinct_countries = result_df['CountryName'].unique()
    for cur_country_name in distinct_countries:
        country_rows = input_df[input_df['CountryName'] == cur_country_name]
        years_for_country = country_rows['Year']
        years_for_country = years_for_country.sort_values()
        begin_year = years_for_country.min()
        last_year = years_for_country.min()

        def mark_interval():
            nonlocal country_year_events
            nonlocal begin_year
            nonlocal last_year
            country_year_events.append((cur_country_name, begin_year, 'begin'))
            country_year_events.append((cur_country_name, last_year, 'end'))

        for year in years_for_country:
            if year > last_year + 1:
                mark_interval()
                begin_year = year
            last_year = year

        if begin_year != last_year:
            mark_interval()

    country_year_events.sort(key=lambda row: row[1])
    return country_year_events


# get a dataframe filtered by the dates of with the maximum number
# of countries all with consecutive data for min_consecutive_years
def get_countries_with_overlapping_consecutive_years_v2(input_df, min_consecutive_years):
    country_years = pd.DataFrame(input_df, columns=['CountryName', 'Year'])
    country_years = country_years.groupby('Year')
    min_year = input_df['Year'].min()
    country_count_consec_years = {}
    best_interval_countries = set()
    best_interval_start = 0
    best_interval_end = 0
    for idx, (year, group) in enumerate(country_years):
        lower_year = year - min_consecutive_years
        if lower_year >= min_year:
            country_years_for_low_year = country_years.filter(lambda x: x['Year'] == lower_year)
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

    input_df = input_df[input_df['Year'] >= best_interval_start]
    input_df = input_df[input_df['Year'] <= best_interval_end]
    input_df = input_df[input_df['CountryName'].isin(best_interval_countries)]
    return input_df


def get_countries_with_overlapping_consecutive_years(input_df, min_consecutive_years):
    country_intervals = compute_country_year_intervals(input_df)
    active_countries = set()
    countries_per_year = {}

    for (country, year, event) in country_intervals:
        if event == 'begin':
            active_countries.add(country)
        elif event != 'end':
            raise ValueError('Unknown event value ' + event)

        countries_per_year[year] = active_countries.copy()
        if event == 'end':
            active_countries.remove(country)

    min_year = input_df['Year'].min()
    max_year = input_df['Year'].max()
    best_interval_size = 0
    best_interval_begin_year = 0
    best_interval_end_year = 0
    countries_in_best_interval = set()

    print('Country intervals = ' + str(country_intervals))
    print('Countries per year = ' + str(countries_per_year))

    num_overlapping_countries = 0
    active_countries.clear()
    for cur_year in range(min_year, max_year + 1):
        interval_for_year = country_intervals[cur_year]
        lower_year = cur_year - min_consecutive_years
        interval_to_drop = country_intervals[lower_year]
        for country in interval_to_drop:
            if country in active_countries:
                num_overlapping_countries -= 1
                active_countries.remove(country)


        if lower_year >= min_year and len(active_countries) > best_interval_size:
            best_interval_begin_year = lower_year
            best_interval_end_year = cur_year
            best_interval_size = len(active_countries)
            countries_in_best_interval = active_countries

    if best_interval_size == 0:
        raise ValueError('Unable to find %d consecutive years with at least 1 country with data' %
                         min_consecutive_years)

    input_df = input_df[input_df['Year'] >= best_interval_begin_year]
    input_df = input_df[input_df['Year'] <= best_interval_end_year]
    input_df = input_df[input_df['CountryName'].isin(countries_in_best_interval)]
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
        if idx >= 150:
            break

    return pd.DataFrame(rows_list)


def prune_results_by_target(input_df, output_col_name, output_next_suffix, min_observations):
    input_df = input_df[input_df[output_col_name].notnull()]
    for col in input_df:
        non_nuls_in_col = input_df[input_df[col].notnull()]
        if len(non_nuls_in_col) < min_observations:
            input_df = input_df.drop(col, 1)
        else:
            input_df = non_nuls_in_col

    input_df.sort_values(by=['CountryName', 'Year'], inplace=True)
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

    input_df = input_df[input_df[next_output_col_name].notnull()]
    return input_df

param_file_name = 'data/econ-indicators.csv'
param_output_col_name = 'GNI per capita (constant 2005 US$)'
param_output_next_suffix = '_next'
param_min_observations = 10
param_desired_consecutive_years = 6

result_df = get_raw_data_frame(param_file_name)
result_df = prune_results_by_target(result_df, param_output_col_name, param_output_next_suffix, param_min_observations)

gnis_by_country_year = pd.DataFrame(result_df, columns=['CountryName', 'Year', param_output_col_name,
                                                        param_output_col_name + param_output_next_suffix])
print(gnis_by_country_year)

result_df = get_countries_with_overlapping_consecutive_years_v2(result_df, param_desired_consecutive_years)
result_df.to_csv('data/cleaned-econ-data.csv', index=False)
