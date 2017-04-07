from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt


def get_normalizer(data):
    normalizer = MinMaxScaler(feature_range=(0, 1))
    return normalizer.fit(data)


def get_predictors(input_df, output_col, normalizer=None):
    result_df = input_df.drop(['Year', 'CountryName', output_col], 1)
    col_count = len(result_df.columns)
    if normalizer is not None:
        return normalizer.fit_transform(result_df), col_count
    return result_df, col_count


def get_outcomes(input_df, output_col, normalizer=None):
    if normalizer is not None:
        return normalizer.fit_transform(input_df[output_col])
    return input_df[output_col].values.astype('float32')


def format_data_as_time_series(input_df, output_col, num_steps, input_normalizer, output_normalizer):
    result_x = []
    result_y = []
    predictors, pred_count = get_predictors(input_df, output_col, input_normalizer)
    outcomes = get_outcomes(input_df, output_col, output_normalizer)
    for idx in range(len(input_df)-num_steps-1):
        cur_rows = predictors[idx:idx + num_steps]
        result_x.append(cur_rows)
        # outcomes are already time-shifted + 1
        result_y.append(outcomes[idx + num_steps - 1])

    # keras requires format [observations, num_steps, predictors]. Observations will be a bit less than rows in input_df
    # due to length of window num_steps
    def reshape_data(val, pred_count):
        return np.reshape(val, (val.shape[0], num_steps, pred_count))

    return reshape_data(np.array(result_x), pred_count), np.array(result_y), pred_count


def get_train_test_val_data(filename):
    data_no_nulls = pd.read_csv(filename)
    data_no_nulls = data_no_nulls.sort_values(by=['CountryName', 'Year'])
    model_train_data = data_no_nulls[data_no_nulls['Validation'] == False]
    model_validation_data = data_no_nulls[data_no_nulls['Validation'] == True]
    model_data_midpoint = int(len(model_train_data) / 2)
    train_data = model_train_data[:model_data_midpoint]
    test_data = model_train_data[model_data_midpoint:]
    return train_data, test_data, model_validation_data


def build_model(hidden_height, num_layers, num_steps, num_predictors, dropout_rate):
    model = Sequential()
    model.add(LSTM(hidden_height, input_shape=(num_steps, num_predictors), return_sequences=num_layers != 1))
    for idx in range(num_layers - 1):
        model.add(LSTM(hidden_height, return_sequences=idx != num_layers - 2))
        if num_layers != 2:
            model.add(Dropout(dropout_rate))

    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


class GranularSet:
    def __init__(self, lower_bound, upper_bound, granularity):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.granularity = granularity

    def convert_to_set(self):
        cur_val = self.lower_bound
        result_list = []
        while cur_val < self.upper_bound:
            result_list.append(cur_val)
            cur_val += self.granularity
        return result_list


class ModelConfig:
    hidden_height = GranularSet(800, 801, 2)
    num_layers = GranularSet(3, 4, 2)
    num_steps = 10
    batch_size = None #None defaults to number of countries
    output_col = 'GNI per capita (constant 2005 US$)_next'
    epoch_count = GranularSet(30, 31, 2)
    dropout = GranularSet(0.05, 0.6, 0.05)
    load_existing_model = False


def build_model_with_hyperparams(model_conf, data_frame, input_normalizer, output_normalizer,
                                 cv=False, n_splits=3, n_jobs=4):
    train_x, train_y, num_preds = format_data_as_time_series(data_frame, model_conf.output_col,
                                                             model_conf.num_steps, input_normalizer,
                                                             output_normalizer)
    param_grid = dict(hidden_height=model_conf.hidden_height.convert_to_set(),
                      num_layers=model_conf.num_layers.convert_to_set(),
                      num_steps=[model_conf.num_steps], num_predictors=[num_preds],
                      dropout_rate=model_conf.dropout.convert_to_set(),
                      epochs=model_conf.epoch_count.convert_to_set(),
                      batch_size=[model_conf.batch_size])
    best_params = {}
    if cv:
        rnn_model = KerasRegressor(build_fn=build_model, verbose=2)

        grid = GridSearchCV(estimator=rnn_model, param_grid=param_grid, cv=TimeSeriesSplit(n_splits=n_splits),
                            n_jobs=n_jobs)
        grid_fit = grid.fit(train_x, train_y)
        best_params = grid_fit.cv_results_['params'][0]
        print('Params selected via cross validation: %s' % str(best_params))
    else:
        for k, v in param_grid.items():
            best_params[k] = v[0]

    rnn_model = build_model(best_params['hidden_height'], best_params['num_layers'], best_params['num_steps'],
                            best_params['num_predictors'], best_params['dropout_rate'])
    return rnn_model, best_params


#If cv is not specified, granular sets in model_conf should only yield one value
def create_and_train_model(model_conf, data_frame, input_normalizer, output_normalizer,
                           cv=False, n_splits=3, n_jobs=4, write_to_file=None):
    rnn_model, best_params = build_model_with_hyperparams(model_conf,
                                                          data_frame,
                                                          input_normalizer,
                                                          output_normalizer,
                                                          cv=cv,
                                                          n_splits=n_splits,
                                                          n_jobs=n_jobs)
    train_x, train_y, num_preds = format_data_as_time_series(data_frame, model_conf.output_col,
                                                             model_conf.num_steps,
                                                             input_normalizer,
                                                             output_normalizer)
    rnn_model.fit(train_x, train_y, epochs=best_params['epochs'], batch_size=best_params['batch_size'], verbose=2)
    predicted_train = rnn_model.predict(train_x, batch_size=model_conf.batch_size)
    scaled_predicted = output_normalizer.inverse_transform(predicted_train)
    scaled_real_outs = output_normalizer.inverse_transform(train_y)
    train_rmse = math.sqrt(mean_squared_error(scaled_real_outs, scaled_predicted))

    if write_to_file is not None:
        rnn_model.save_weights(write_to_file)

    return rnn_model, train_rmse


def load_model_from_file(model_conf, data_frame, weights_file, input_normalizer, output_normalizer):
    rnn_model, best_params = build_model_with_hyperparams(model_conf,
                                                          data_frame,
                                                          input_normalizer,
                                                          output_normalizer)
    rnn_model.load_weights(weights_file)
    return rnn_model


def evaluate_model(rnn_model, model_conf, test_data, input_normalizer, output_normalizer):
    input_x, input_y, _ = format_data_as_time_series(test_data, model_conf.output_col,
                                                     model_conf.num_steps, input_normalizer,
                                                     output_normalizer)

    predicted_train = rnn_model.predict(input_x, batch_size=model_conf.batch_size)
    scaled_predicted = output_normalizer.inverse_transform(predicted_train)
    scaled_real_outs = output_normalizer.inverse_transform(input_y)
    eval_rmse = math.sqrt(mean_squared_error(scaled_real_outs, scaled_predicted))

    eval_years = test_data['Year'].unique()
    eval_years = eval_years[model_conf.num_steps + 1:]

    return eval_years, scaled_predicted, scaled_real_outs, eval_rmse


def eval_individual_country(model, model_conf, data_set, country_name, input_normalizer, output_normalizer):
    data_set_for_country = data_set[data_set['CountryName'] == country_name]
    eval_years, scaled_predicted, scaled_real, rmse = evaluate_model(model, model_conf, data_set_for_country,
                                                                     input_normalizer, output_normalizer)
    actual_plt, = plt.plot(eval_years, scaled_real, color='black', label='Actual')
    predicted_plt, = plt.plot(eval_years, scaled_predicted, color='blue', alpha=0.8, label='Expected')
    min_year = data_set_for_country['Year'].min()
    max_year = data_set_for_country['Year'].max()
    max_gni_per_capita = max(scaled_predicted.max(), scaled_real.max())
    plt.title(country_name)
    plt.xlabel("Year")
    plt.ylabel("Forecast GNI Per Capita")
    plt.legend(handles=[actual_plt, predicted_plt])
    plt.xticks(range(min_year, max_year))
    plt.yticks(range(0, int(max_gni_per_capita) + 100, 1000))
    plt.show(block=True)


if __name__ == '__main__':
    # Data is generated with 1971 - 2000 years for training data. A safe number num_steps value is 10.
    config = ModelConfig()
    config.hidden_height = GranularSet(1000, 1001, 200)
    config.num_layers = GranularSet(2, 3, 2)
    config.epoch_count = GranularSet(100, 101, 5)
    config.dropout = GranularSet(0.0, 0.1, 0.2)
    config.load_existing_model = True

    outer_train_data, outer_test_data, outer_validation_data = get_train_test_val_data('data/cleaned_no_nulls.csv')
    if config.batch_size is None:
        config.batch_size = len(outer_train_data['Year'].unique())

    train_preds, _ = get_predictors(outer_train_data, config.output_col)
    train_outs = get_outcomes(outer_train_data, config.output_col)
    outer_input_normalizer = get_normalizer(train_preds)
    outer_output_normalizer = get_normalizer(train_outs)

    if not config.load_existing_model:
        out_model, train_rmse = create_and_train_model(config,
                                                       outer_train_data,
                                                       outer_input_normalizer,
                                                       outer_output_normalizer,
                                                       write_to_file='models/rnn/trained_model.hdf5')
        print('Train RMSE: %d' % int(train_rmse))
    else:
        out_model = load_model_from_file(config,
                                         outer_train_data,
                                         'models/rnn/trained_model.hdf5',
                                         outer_input_normalizer,
                                         outer_output_normalizer)

    print('Train std dev: %d' % np.std(outer_train_data[config.output_col]))
    _, _, _, test_rmse = evaluate_model(out_model, config, outer_test_data, outer_input_normalizer,
                                        outer_output_normalizer)
    print('Test RMSE: %d' % int(test_rmse))
    print('Test std dev: %d' % np.std(outer_test_data[config.output_col]))

    def eval_countries_in_dataset(dataset):
        distinct_countries = dataset['CountryName'].unique()
        for country in distinct_countries:
            eval_individual_country(out_model, config, dataset, country, outer_input_normalizer,
                                    outer_output_normalizer)

    eval_countries_in_dataset(outer_test_data)

