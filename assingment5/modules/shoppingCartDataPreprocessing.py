import numpy as np
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder

#* Function for remove missing value, which is 0.
def dataPreprocessing(shoppingCartData):
    """
    ### Function for prepare datas
    - Remove missing value, which is 0 and NaN.
    - Transform all records into list of transaction.
    """
    transaction = []
    # Iterate rows.
    for row in range(shoppingCartData.shape[0]):
        # Declare temp list for storing each transaction data.
        temp = []

        # Iterate columns.
        for col in range(shoppingCartData.shape[1]):

            # Append data value to list `temp` if value is not 0(' ') and is not NaN.
            if (str(shoppingCartData.loc[row,col]) != ' ') and (shoppingCartData.loc[row,col] is not np.nan):
                temp.append(str(shoppingCartData.loc[row,col])) 

        # Append each transaction to transactions list.
        transaction.append(temp)
    return transaction

#* Function for encoding transaction.
def encodeTransaction(shopping_transaction):
    te = TransactionEncoder()
    encoded_shopping_transaction = pd.DataFrame(
        te.fit(shopping_transaction).transform(shopping_transaction), 
        columns=te.columns_
    )
    return encoded_shopping_transaction