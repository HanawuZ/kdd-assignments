import pandas as pd
import pathlib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Define cereals csv file path.
cereals_path = pathlib.Path(__file__).parent / "cereal.csv"
 

def dataPreprocessing(cereals_df):
    # print(cereals_df.info())

    # Drop attributes `name`, `mfr` and `type`.
    # These attributes aren't numerical type.
    cereals_df = cereals_df.loc[:, ~cereals_df.columns.isin(["name", "mfr", "type"])]

    # Replace value -1 in attributes `potass`,`carbo` and `sugars` with average values of each attribute.
    # Value -1 is missing value according to this context.
    for attr in ["carbo", "sugars", "potass"]:
        cereals_df[attr] = cereals_df[attr].replace(to_replace=-1, value=cereals_df[attr].mean())

    # return cereals_df
    return cereals_df

if __name__ == "__main__":
    # Read cereals csv file.
    cereals_df = pd.read_csv(cereals_path)

    # # Print cereals dataframe first 5 records and information.
    # print(cereals_df.head())
    # print(cereals_df.info())

    # Do data preprocessing
    # 1. Drop attributes `name`, `mfr` and `type`.  
    # 2. Fill missing value (-1) with average value of attribute.
    cereals_df = dataPreprocessing(cereals_df)
    print(cereals_df.info())
    # Initialize linear regression model.
    linear_regression_model = LinearRegression()

    