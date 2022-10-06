"""This file will be used to load data from csv"""

import pandas as pd
import numpy as np
import config
from typing import TypeVar

PandasDataFrame = TypeVar('pd.core.frame.DataFrame')

class LoadData:
    
    def __init__(
        self,path:str,
        ) -> None:
        self.path = config.TRAINING_FILE
    
    def read_data(self,file_type:str)->PandasDataFrame:
        """
        This method reads a csv file from input path

        Attributes:
            file_type (str): csv input path
        Returns:
            df (DataFrame): loaded dataframe

        """
        try:
            df = pd.read_csv(self.path,sep="\t")
            df.rename(
                columns={
                    "#1 String":"text_1","#2 String":"text_2"},inplace = True)
            if file_type !="train":
                df = df[["text_1","text_2","Quality"]]
            else:
                df = df[["text_1","text_2"]]
        except Exception as e:
            print("Error occured while loading file -->\n",e)
        return df
        

if __name__=="__main__":
    train_path = r"C:\New_Project\SNHU\glue_data\MRPC\train - train.tsv"
    load = LoadData(train_path)
    df = load.read_data("train")
    print(df.head(1))
        
