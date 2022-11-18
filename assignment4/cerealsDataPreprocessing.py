from sklearn.model_selection import train_test_split

# * 
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

# * Fucntion for split validation.
def splitValidation(cereals_df):

    # Split dataframe to train set and test set.
    cereals_train_df, cereals_test_df = train_test_split(cereals_df, test_size=0.2, random_state=0)

    # Split feature labels & class label(s).
    cereals_feature_train = cereals_train_df.drop(columns = ["rating"])
    cereals_feature_test = cereals_test_df.drop(columns = ["rating"])

    cereals_rating_train = cereals_train_df["rating"]
    cereals_rating_test = cereals_test_df["rating"]

    return cereals_feature_train, cereals_feature_test, cereals_rating_train, cereals_rating_test