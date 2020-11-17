import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Import data
df = pd.read_csv("medical_examination.csv")

# Add 'overweight' column
df['overweight'] = np.where(df['weight'] / ((df['height'] / 100) ** 2) > 25, 1, 0)

# Normalize data by making 0 always good and 1 always bad. If the value of 'cholestorol' or 'gluc' is 1, make the value 0. If the value is more than 1, make the value 1.

df['cholesterol'] = df['cholesterol'].agg(lambda x: 1 if x > 1 else 0)

df['gluc'] = df['gluc'].agg(lambda x: 1 if x > 1 else 0)


# Draw Categorical Plot
def draw_cat_plot():
    # Create DataFrame for cat plot using `pd.melt` using just the values from 'cholesterol', 'gluc', 'smoke', 'alco', 'active', and 'overweight'.
    # create DataFrame with 'cardio', 'variable' and 'value' columns
    df_cat = pd.melt(df, id_vars=['cardio'], value_vars=['active', 'alco', 'cholesterol', 'gluc', 'overweight', 'smoke'])
    # df_cat structure:
    #         cardio variable  value
    # 0            0   active      1
    # 1            1   active      1
    # 2            1   active      0
    # 3            1   active      1
    # 4            0   active      0
    # ...        ...      ...    ...
    # 419995       0    smoke      1
    # 419996       1    smoke      0
    # 419997       1    smoke      0
    # 419998       1    smoke      0
    # 419999       0    smoke      0

    # condense DataFrame to indexes only (no columns) with 'value' row counts defining the structure
    df_cat = df_cat.groupby(['cardio', 'variable', 'value'])['value'].count()
    # df_cat structure:
    # cardio  variable     value
    # 0       active       0         6378
    #                      1        28643
    #         alco         0        33080
    #                      1         1941
    #         cholesterol  0        29330
    #                      1         5691
    #         gluc         0        30894
    #                      1         4127
    #         overweight   0        15915
    #                      1        19106
    #         smoke        0        31781
    #                      1         3240
    # 1       active       0         7361
    #                      1        27618
    #         alco         0        33156
    #                      1         1823
    #         cholesterol  0        23055
    #                      1        11924
    #         gluc         0        28585
    #                      1         6394
    #         overweight   0        10539
    #                      1        24440
    #         smoke        0        32050
    #                      1         2929

    # df_cat.index is a MultiIndex with names 'cardio', 'variable' and 'value'

    # add 'value' column with the values of the 'value' index
    df_cat = pd.DataFrame(df_cat)
    # df_cat structure:
    #                           value
    # cardio variable    value       
    # 0      active      0       6378
    #                    1      28643
    #        alco        0      33080
    #                    1       1941
    #        cholesterol 0      29330
    #                    1       5691
    #        gluc        0      30894
    #                    1       4127
    #        overweight  0      15915
    #                    1      19106
    #        smoke       0      31781
    #                    1       3240
    # 1      active      0       7361
    #                    1      27618
    #        alco        0      33156
    #                    1       1823
    #        cholesterol 0      23055
    #                    1      11924
    #        gluc        0      28585
    #                    1       6394
    #        overweight  0      10539
    #                    1      24440
    #        smoke       0      32050
    #                    1       2929
    
    # rename the 'value' column to 'total' and then convert all indexes ('cardio', 'variable' and 'value') to columns by resetting the index
    df_cat = df_cat.rename(columns={ 'value': 'total' }).reset_index()
    # df_cat structure:
    #     cardio     variable  value  total
    # 0        0       active      0   6378
    # 1        0       active      1  28643
    # 2        0         alco      0  33080
    # 3        0         alco      1   1941
    # 4        0  cholesterol      0  29330
    # 5        0  cholesterol      1   5691
    # 6        0         gluc      0  30894
    # 7        0         gluc      1   4127
    # 8        0   overweight      0  15915
    # 9        0   overweight      1  19106
    # 10       0        smoke      0  31781
    # 11       0        smoke      1   3240
    # 12       1       active      0   7361
    # 13       1       active      1  27618
    # 14       1         alco      0  33156
    # 15       1         alco      1   1823
    # 16       1  cholesterol      0  23055
    # 17       1  cholesterol      1  11924
    # 18       1         gluc      0  28585
    # 19       1         gluc      1   6394
    # 20       1   overweight      0  10539
    # 21       1   overweight      1  24440
    # 22       1        smoke      0  32050
    # 23       1        smoke      1   2929

    # Draw the catplot with 'sns.catplot()'
    # make two bar charts with the two 'cardio' values (0 and 1) and with the 'variable' column as the x axis and the 'total' column as the y axis
    # each 'variable' will have two bars based on the 'value' column (0 and 1)
    fig = sns.catplot(data=df_cat, col="cardio", x="variable", y="total", hue="value", kind="bar").fig


    # Do not modify the next two lines
    fig.savefig('catplot.png')
    return fig


# Draw Heat Map
def draw_heat_map():
    # Clean the data
    df_heat = df[
      # remove diastolic pressure higher than systolic
      (df['ap_lo'] <= df['ap_hi']) &
      # remove height less than the 2.5th percentile
      (df['height'] >= df['height'].quantile(0.025)) &
      # remove height more than the 97.5th percentile
      (df['height'] <= df['height'].quantile(0.975)) &
      # remove weight less than the 2.5th percentile
      (df['weight'] >= df['weight'].quantile(0.025)) &
      # remove weight more than the 97.5th percentile
      (df['weight'] <= df['weight'].quantile(0.975))
    ]
    # Calculate the correlation matrix
    corr = df_heat.corr()

    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr)
    # Set up the mask[np.triu_indices_from(mask)] = True figure
    mask[np.triu_indices_from(mask)] = True

    fig, ax = plt.subplots(figsize=(8,7))

    # Draw the heatmap with 'sns.heatmap()'
    sns.heatmap(corr, mask=mask, center=0, annot=True, fmt='.1f', linewidths=.5, square=True, cbar_kws={ 'shrink': 0.5 })

    # Do not modify the next two lines
    fig.savefig('heatmap.png')
    return fig
