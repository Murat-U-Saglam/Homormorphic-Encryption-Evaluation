import tenseal as ts
import pandas as pd
import sqlite3
from key_manager import context
import numpy as np
import random


random.seed(73)


class dataHandler:
    def __init__(self, URL):
        """Constructor

        Args:
            URL (String): URL to locate the database
        """
        self.URL = URL

    def data_to_df(self) -> pd.DataFrame:

        """converts database to pandas dataframe.

        Returns:
            pd.DataFrame: Requested Dataframe
        """

        with sqlite3.connect(self.URL) as conn:
            df = pd.read_sql_query("SELECT * from Salaries", conn)
            return df

    def clean_df(self, df) -> pd.DataFrame:
        numeric_feature_names = [
            "BasePay",
            "OvertimePay",
            "OtherPay",
            "TotalPay",
            "Year",
        ]
        numeric_features = df[numeric_feature_names]

        # Replace empty values to NA
        numeric_features = numeric_features.replace("", np.nan)
        numeric_features = numeric_features.replace("Not Provided", np.nan)
        numeric_features = numeric_features.dropna()
        Y = numeric_features["Year"]
        numeric_features = numeric_features.drop(["Year"],axis = 1)
        return numeric_features, Y

    def standardise_df(self, df) -> pd.DataFrame:
        Y = df["Year"]
        data = (df - df.mean()) / df.std()
        return data, Y

    def encrypt_df(self, df) -> pd.DataFrame:
        for heading in df:
            vectorised_df = df[heading].to_numpy().tolist()
            he_vectorised_df = ts.ckks_vector(context, vectorised_df)
            df[heading] = he_vectorised_df
        return df


# https://towardsdatascience.com/linear-regression-with-pytorch-eb6dedead817
class linearRegression():
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def encrypt_df(self) -> pd.DataFrame:
        for heading in self.X:
            vectorised_df = self.X[heading].to_numpy().tolist()
            he_vectorised_df = ts.ckks_vector(context, vectorised_df)
            self.X[heading] = he_vectorised_df
        return self.X
    
    def train_test_split(self, test_ratio = 0.2) -> tuple:
        """
        Define test train split

        Args:
            test_ratio (float, optional): _description_. Defaults to 0.2.

        Returns:
            tuple: x testing data, y testing data, x training data and y training data.
        """
        tmp = pd.concat([self.X, self.Y], axis = 1)
        tmp = tmp.sample(frac=1).reset_index(drop=True)
        self.test_X, self.test_Y = tmp[(self.X.columns)].sample(int(len(tmp)*test_ratio)), tmp[(self.Y.name)].sample(int(len(tmp)*test_ratio))
        self.train_X, self.train_Y = tmp[(self.X.columns)], tmp[(self.Y.name)]
        return (self.test_X, self.test_Y, self.train_X, self.train_Y)
    
    def forward(self):
        for heading in self.train_X:
            print(self.train_X[heading])
        
        
    def iterate(self):
        pass



def main():
    print("Running as main")
    URL = "./DATA/sf_salaries/database.db"
    conn = dataHandler(URL)
    df = conn.data_to_df()
    cleaned_df, Y = conn.clean_df(df)
    #standardised_df, Y = conn.standardise_df(cleaned_df)
    
    model = linearRegression(cleaned_df, Y)
    model.encrypt_df()
    model.train_test_split()
    model.forward()



if __name__ == "__main__":
    main()
else:
    print("this is library")
