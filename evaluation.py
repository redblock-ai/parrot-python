"""
parrot.evaluation encompases the metrics which enable evaluation of candidate responses.
"""
from datasets.datasets import Datasets
import logging

#config logging
logging.basicConfig( 
                filename=None,
                level=logging.INFO,          
                format='%(asctime)s - %(levelname)s - %(message)s',  #our custom log format
                datefmt='%Y-%m-%d %H:%M:%S'
            )

class MillionaireException(Exception):
    """
    Base class for custom Millionaire exceptions.
    """
    pass

class MillionaireExceptionGroup:
    class InvalidDataset(MillionaireException):
        """
        Exception raised for invalid dataset type.
        """
        def __init__(self, message="Invalid dataset object received"):
            self.message = message
            super().__init__(self.message)
    
    class EmptyDataFrame(MillionaireException):
        """
        Exception raised if the data_frame is empty.
        """
        def __init__(self, message="Data frame cannot be empty! Please check the dataset passed for evaluation") -> None:
            self.message = message
            super().__init__(self.message)

class MillionaireMetric:
    """
    Enables evaluation of Dataset() containing candidate-llm responses for millionaire-set. 
    """
    def __init__(self, dataset: Datasets) -> None:
        try:
            logging.info("[MillionaireMetric] Initializing MillionaireMetric for evaluation of candidate responses...")
            if not isinstance(dataset, Datasets):
                raise MillionaireExceptionGroup.InvalidDataset(message="Dataset passed cannot be of type None.")
            if dataset.current_dataset != "millionaire":
                raise MillionaireExceptionGroup.InvalidDataset(message="Dataset passed does not correspond to Millionaire-set.")
            
            self.__data_frame = dataset.get_data_frame() #get the millionaire candidate response, dataframe.
            if len(self.__data_frame) == 0:
                raise MillionaireExceptionGroup.EmptyDataFrame()
            logging.info("[MillionaireMetric] Initialization was successful!")
        except Exception as e:
            logging.error(str(e))
            raise e

class JeopardyMetric:
    pass

class Evaluate:
    pass