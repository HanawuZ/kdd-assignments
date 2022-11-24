import numpy as np
import pandas as pd
import seaborn as sns
import pathlib
from modules.shoppingCartDataPreprocessing import *
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

# Define Random shopping cart file's path.
SHOPPING_CART_PATH = pathlib.Path(__file__).parent / "datasets" / "Random_shopping_cart.csv"

shoppingCartData = pd.read_csv(SHOPPING_CART_PATH, header=None)

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
    # Print out samples of transaction, if preparation is done, each transaction should have only string values. 
    # print(transaction[8])

    # Encode transaction using TransactionEncoder()


#* Function for applying association rule.
def applyAssociationRule(shopping_transaction):
    """
    Function for applying association rule by passing transaction list to appiori algorithm.
    """

    te = TransactionEncoder()
    te_ary = te.fit(shopping_transaction).transform(shopping_transaction)
    df = pd.DataFrame(te_ary, columns=te.columns_)

    # Define min support.
    min_support = 0.1
    # Define min confidence.
    min_confidence = 0.6
    
    # Excute appiori algorithm.
    frequent_itemsets=apriori(df, min_support=min_support, use_colnames=True)
    
    # Generate association rule.
    frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))

    print(frequent_itemsets)
    
shopping_transaction = dataPreprocessing(shoppingCartData)
applyAssociationRule(shopping_transaction)