"""
Data Access Object: PARROT.datasets.Datasets() is responsible for fetching the raw data and returning it in a structured form, typically as a Pandas.DataFrame.
"""
import pandas as pd
import logging
import os
class Datasets():
    """
    A Data Access Object responsible for loading raw data from disk and returning it in a structured format as a Pandas.DataFrame. This class provides methods to handle datasets efficiently and supports sampling a subset of the data if needed.

    Attributes
    ----------
        current_dataset (str):
            The name of the dataset to be loaded (without the file extension).

        __cwd (str):
            The current working directory where the datasets folder resides.

        __FILE__ (str):
            The full file path of the dataset CSV file, constructed using the current_dataset and working directory.

        __data (pd.DataFrame):
            The loaded dataset in memory, stored as a Pandas.DataFrame. This can either be the entire dataset or a random sample, based on the provided parameters.

    Methods
    -------
        get_data_frame()
            Loads the data from the specified file and returns it as a Pandas.DataFrame.
        
        set_data_frame()
            Updates the data_frame object in memory.
    """
    
    def __init__(self, dataset: str, sample_size: int = None) -> None:
        """
        Initializes the Datasets class by loading a dataset from the specified path. If a sample_size is provided, a random sample of that size is returned instead of the entire dataset.

        Arguments:
        ----------
            dataset (str): The name of the dataset file (without the .csv extension) to be loaded.
            sample_size (int, optional): The number of random samples to load from the dataset. If None, the entire dataset is loaded.
        
        Returns:
        --------
            None.

        Raises:
        -------
            FileNotFoundError: Raised if the specified dataset file is not found in the expected location on disk.
        """
        try:
            logging.basicConfig( 
                filename=None,
                level=logging.INFO,          
                format='%(asctime)s - %(levelname)s - %(message)s',  #our custom log format
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            self.current_dataset = dataset
            self.__cwd = os.getcwd() #get the current working directory.
            
            self.__FILE__ = os.path.join(os.path.dirname(__file__), self.current_dataset + ".csv")
            if sample_size is not None:
                logging.info(f"[Datasets] Loading dataset with sample_size: {sample_size}.")
                self.__data = pd.read_csv(self.__FILE__)
                self.__data = self.__data.sample(sample_size)
            else:
                logging.info(f"[Datasets] Loading dataset: {dataset}.")
                self.__data = pd.read_csv(self.__FILE__)
            logging.info(f"[Datasets] Dataset loaded successfully.")
        except FileNotFoundError as e:
            logging.exception(e)

    def get_data_frame(self) -> pd.DataFrame:
        """
        Returns the dataset that has been loaded into memory as a Pandas.DataFrame. This method allows other classes or functions to access the data for further analysis or processing.

        Arguments:
        ---------
            None.

        Returns:
        --------
            pd.DataFrame: The dataset that was loaded, either the full dataset or a sample, depending on the initialization parameters.
        """
        return self.__data

    def set_data_frame(self, data_frame:pd.DataFrame) -> None:
        """
        Updates the dataset that has been loaded into memory as a Pandas.DataFrame. This method allows other classes or functions to access the data for further analysis or processing.

        Arguments:
        ---------
            data_frame (pd.Dataframe):
                The new data_frame that has to be updated in memory.

        Returns:
        --------
            None.
        """
        self.__data = data_frame
