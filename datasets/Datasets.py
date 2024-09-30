"""
Data Access Object: PARROT.datasets.Datasets() is responsible for fetching the raw data and returning it in a structured form, typically as a Pandas.DataFrame.
"""
import pandas as pd
import logging
import os
class Datasets():
    """
    A class that serves as a Data Access Object is responsible for fetching the raw data and returning it in a structured form, typically as a Pandas.DataFrame.
    
    
    A class used to load data from a file and return a pandas DataFrame.

    Attributes
    ----------
    file_path : str
        The path to the file from which data will be loaded.

    Methods
    -------
    get_data_frame()
        Loads the data from the specified file and returns it as a Pandas.DataFrame.
    
    Example
    -------
    To use this class, you would initialize it with the file path and then call load_data:
    
    \>>> loader = Datasets("data.csv")
    \>>> df = loader.load_data()
    \>>> print(df.head())
    """
    
    def __init__(self, dataset: str, sample_size: int = None) -> None:
        """
        Initializes the dataframe for the requested dataset.
        """
        try:
            logging.basicConfig( 
                filename=None,
                level=logging.INFO,          
                format='%(asctime)s - %(levelname)s - %(message)s',  #our custom log format
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            self.__cwd = os.getcwd() #get the current working directory.
            
            self.__file__ = self.__cwd +"/datasets/"+ dataset + ".csv"
            if sample_size is not None:
                logging.info(f"[Datasets] Loading dataset with sample_size: {sample_size}.")
                self.__data = pd.read_csv(self.__file__)
                self.__data = self.__data.sample(sample_size)
            else:
                logging.info(f"[Datasets] Loading dataset: {dataset}.")
                self.__data = pd.read_csv(self.__file__)
            logging.info(f"[Datasets] Dataset loaded successfully.")
        except FileNotFoundError as e:
            logging.exception(e)

    def get_data_frame(self) -> pd.DataFrame:
        """
        This method returns the dataset loaded into memory.
        """
        return self.__data