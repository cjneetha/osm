import pandas as pd
from datetime import datetime

import sys
import os

sys.path.append('../')

import snippets.amazon_constants_file_paths as paths






if __name__ == '__main__':

    folder = paths.DIR_AMAZON
    print("Read the clothing, shoes and jewelry dataset")
    clothing = pd.read_pickle(os.path.join(folder, "reviews_Clothing_Shoes_and_Jewelry_5.pkl.gzip"))

    # ------------------------ construct data stream ----------------------------
    #data = pd.concat([beauty, phones, clothing, electronics, home, apps, cds, health, kindle])
    data = pd.concat([clothing])

    data.reset_index(inplace=True, drop=False)
    data.set_index("date", inplace=True)
    data.sort_index(inplace=True)

    # considering data only from 2011
    data = data.loc["2011":, :]

    # setting the categories to be chosen for each quarter in the year
    print(data.head())


    # setting the categories to be chosen for each quarter in the year
    product_categories = {
        datetime(2011, 1, 31): ['clothing, shoes and jewelry'],
        datetime(2011, 4, 30): ['clothing, shoes and jewelry'],
        datetime(2011, 7, 31): ['clothing, shoes and jewelry'],
        datetime(2011, 10, 31): ['clothing, shoes and jewelry'],
        datetime(2012, 1, 31): ['clothing, shoes and jewelry'],
        datetime(2012, 4, 30): ['clothing, shoes and jewelry'],
        datetime(2012, 7, 31): ['clothing, shoes and jewelry'],
        datetime(2012, 10, 31): ['clothing, shoes and jewelry'],
        datetime(2013, 1, 31): ['clothing, shoes and jewelry'],
        datetime(2013, 4, 30): ['clothing, shoes and jewelry'],
        datetime(2013, 7, 31): ['clothing, shoes and jewelry'],
        datetime(2013, 10, 31): ['clothing, shoes and jewelry'],
        datetime(2014, 1, 31): ['clothing, shoes and jewelry'],
        datetime(2014, 4, 30): ['clothing, shoes and jewelry'],
        datetime(2014, 7, 31): ['clothing, shoes and jewelry']
    }

    # create three month groups
    grouped = data.groupby(pd.Grouper(freq='3M'))

    filtered_data = []
    for name, group in grouped:
        # get the categories
        categories = group.loc[:, "category"].unique()

        # randomly select 3 categories
        selected = product_categories[name]

        # filter for selected categories
        selected_data = group[group['category'].isin(selected)]

        # add to result list
        filtered_data.append(selected_data)

    filtered_data = pd.concat(filtered_data)

    # remove duplicates
    filtered_data = filtered_data.reset_index(drop=False).drop_duplicates(["date", "review_id"]).set_index(
        ["date", "review_id"])

    # save
    filtered_data.reset_index(drop=False, inplace=True)
    filtered_data.set_index(["date", "review_id"], inplace=True)
    filtered_data.sort_index(inplace=True)
    filtered_data.to_pickle(os.path.join(folder, "filtered", "reviews.pkl.gzip"))


