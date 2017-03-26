from sklearn import linear_model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

min_observations = 6000
with open('data/cleaned-econ-data.csv') as predictors:
    output_col_name = 'GNI per capita (constant 2005 US$)'
    pred_df = pd.read_csv(predictors)
    y_vals = pd.DataFrame(pred_df, columns=[output_col_name, 'Year'])
    x_vals = pred_df.drop(output_col_name, 1)
    median_year = x_vals['Year'].median()
    print(median_year)
    unused_pred_cols = ['CountryName', 'Year']
    x_training = x_vals[x_vals['Year'] < median_year].drop(unused_pred_cols, 1)
    x_test = x_vals[x_vals['Year'] >= median_year].drop(unused_pred_cols, 1)
    y_training = y_vals[y_vals['Year'] < median_year].drop(unused_pred_cols[1], 1)
    y_test = y_vals[y_vals['Year'] >= median_year].drop(unused_pred_cols[1], 1)
    regr = linear_model.LinearRegression(n_jobs=8)
    regr.fit(x_training, y_training)
    predicted_gni = regr.predict(x_test)

    example_index = range(0, len(x_test))
    plt.scatter(example_index, y_test, color='black')
    plt.scatter(example_index, predicted_gni, color='blue', alpha=0.5)
    plt.xticks(())
    plt.yticks(())
    plt.show()

    mse = np.average(np.square(predicted_gni - y_test))
    print('Variance score: %.2f' % regr.score(x_test, y_test))
    print('mse = ' + str(mse))
