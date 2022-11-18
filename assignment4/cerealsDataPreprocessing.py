
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
    return