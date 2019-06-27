import pandas


def post_pandas_data():
    data = pandas_df.to_json(orient='split')