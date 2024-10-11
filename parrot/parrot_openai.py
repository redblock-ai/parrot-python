"""
A class which serves as a Adapter for OpenAI GPT model inference over PARROT testset.
"""
from .datasets.datasets import Datasets
import logging
from openai import OpenAI
import pandas as pd

class OpenAdapterException(Exception):
    """
    Base class for Custom Exceptions within Ollama Adapter.
    """
class ExceptionGroup:
    class InvalidDataset(OpenAdapterException):
        """Exception raised for invalid dataset access object"""
        def __init__(self, message="Invalid Dataset Object passed."):
            self.message = message
            super().__init__(self.message)
    
    class InvalidDataframe(OpenAdapterException):
        """Exception raised for invalid dataframe"""
        def __init__(self, message="Invalid Dataset Object passed."):
            self.message = message
            super().__init__(self.message)

    class ModelNameCannotBeNone(OpenAdapterException):
        def __init__(self, message="Invalid model passed for inference."):
            self.message = message
            super().__init__(self.message)
    
    class ModelNameCannotBeEmpty(OpenAdapterException):
        def __init__(self, message="Paramter 'model_name' cannot be empty."):
            self.message = message
            super().__init__(self.message)

    class PromptCannotBeNone(OpenAdapterException):
        def __init__(self, message="Paramter 'prompt' passed cannot be of type None."):
            self.message = message
            super().__init__(self.message)
    
    class PromptCannotBeEmpty(OpenAdapterException):
        def __init__(self, message="Paramter 'prompt' cannot be empty."):
            self.message = message
            super().__init__(self.message)

    class InsufficientBatchSize(OpenAdapterException):
        def __init__(self, message="Paramter 'prompt' cannot be empty."):
            self.message = message
            super().__init__(self.message)

class OpenAdapter:
    """
    A custom-class, that servers as a adapter for model inference over PARROT-datasets using OpenAI endpoint on top of langchain.
    """
    def __init__(self, 
    dataset:Datasets = None,
    model_name: str = None,
    prompt: str = None,
    temperature: float = 0,
    top_k: int = 10,
    api_key: str = None,
    top_n: float = 0.2) -> None:
        try:
            logging.basicConfig( 
                filename=None,
                level=logging.INFO,          
                format='%(asctime)s - %(levelname)s - %(message)s',  #our custom log format
                datefmt='%Y-%m-%d %H:%M:%S'
            )

            self.__dataset_handler = dataset #the dataset object is now used to handle that particular instance.
            if not isinstance(self.__dataset_handler, Datasets) or self.__dataset_handler is None:
                raise ExceptionGroup.InvalidDataset(message=f"Invalid Dataset Object passed of type: {type(self.__dataset_handler)}")
            data_frame = self.__dataset_handler.get_data_frame()

            #perform parameter checks.
            if data_frame is None or not isinstance(data_frame, pd.DataFrame):
                raise ExceptionGroup.InvalidDataframe("[OpenAdapter] - Invalid parameter 'data_frame' passed for inference.")
            self.__data = data_frame #init data
            logging.info("[OpenAdapter] - Data initialized") 
            
            #perform checks for parameter 'model_name'
            if model_name is None:
                raise ExceptionGroup.ModelNameCannotBeNone(message=f"The model name {model_name}, cannot be of type None.")
            elif model_name.strip() == "":
                raise ExceptionGroup.ModelNameCannotBeEmpty()
            
            self.__client = OpenAI()
            self.__model = model_name, 
            self.__temperature = temperature
            self.__top_k = top_k
            self.__top_n = top_n
            
            if prompt is None:
                raise ExceptionGroup.PromptCannotBeNone()
            elif prompt.strip() == "":
                raise ExceptionGroup.PromptCannotBeEmpty()
            self.__prompt = prompt

            self.__key = api_key
            try:
                if self.__dataset_handler.current_dataset == 'jeopardy':
                    self.__invoke({"question": "how are you?", "category":"Casual Greetings"}) #testing to check if the chain is functional. This should not raise an exception.
                else:
                    self.__invoke({"question": "how are you?"}) #testing to check if the chain is functional. This should not raise an exception.
            except Exception as e:
                logging.error(str(e))
                raise Exception(f"Model not found, download it using the CLI command 'ollama run <model-name>'")
            logging.info("[OpenAdapter] Chain built successfully.")
            logging.info(f"[OpenAdapter] {model_name} is ready for benchmarking!")
        except Exception as e:
            logging.exception(e)
            raise e
    
    def __invoke(self, question: str) -> str:
        """ 
        This private method calls the OpenAI endpoint on the key provided for chat.completion. By default max_tokens are set to limit the output tokens of the model. 
        """
        try:
            self.__messages= [
                {
                    "role": "user",
                    "content": question
                }
            ]
            self.__response = self.__client.chat.completions.create(
                model = self.__model,
                messages = self.__messages,
                max_tokens=15
            )
            return self.__response.choices[0].message.content
        except Exception as e:
            pass
