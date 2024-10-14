"""
A class which serves as a Adapter for OpenAI GPT model inference over PARROT testset.
"""
from .datasets.datasets import Datasets
import logging
from openai import OpenAI
from openai.error import RateLimitError
import pandas as pd
import time 
import os

class OpenAdapterException(Exception):
    """
    Base class for Custom Exceptions within Open Adapter.
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

    class ExceptionFromOpenApi(OpenAdapterException):
        def __init__(self, message="Something went wrong, API call failed."):
            self.message = message
            super().__init__(self.message)

    class InvalidCredentialsPassed(OpenAdapterException):
        def __init__(self, message="Please check the API credentials passed!"):
            self.message = message
            super().__init__(self.message)
    
    class RateLimitError(OpenAdapterException):
        def __init__(self, message="Rate limit reached for the minute:"):
            self.message = message
            super().__init__(self.message)

class OpenAdapter:
    """
    A custom-class, that servers as a adapter for model inference over PARROT-datasets using OpenAI endpoint.
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
            
            
            self.__model = model_name, 
            self.__temperature = temperature
            self.__top_k = top_k
            self.__top_n = top_n
            
            if prompt is None:
                raise ExceptionGroup.PromptCannotBeNone()
            elif prompt.strip() == "":
                raise ExceptionGroup.PromptCannotBeEmpty()
            self.__prompt = prompt

            if not isinstance(api_key, str):
                raise ExceptionGroup.InvalidCredentialsPassed(message= "<api_key> must of be of type 'str', received credentials of type: "+str(type(api_key)))
            elif api_key is None or api_key.strip() == "":
                raise ExceptionGroup.InvalidCredentialsPassed(message="Authentication failed: <api_key> cannot be empty.")
            

                
            os.environ["OPENAI_API_KEY"] = api_key
            logging.info("[OpenAdapter] API_KEY initialized as an OS environment variable.")

            self.__client = OpenAI()
            logging.info(f"{model_name} says, "+ str(self.__invoke(question="Hey how are you?"))) #handshake with API, this must not raise an exception.
            logging.info(f"[OpenAdapter] {model_name} is ready for benchmarking!")
        except Exception as e:
            logging.exception(e)
            raise e
    
    def __invoke(self, question: str) -> str:
        """ 
        This private method calls the OpenAI endpoint on the key provided for chat.completion. By default max_tokens are set to limit the output tokens of the model. 
        """
        try:
            question = self.__prompt + question #append system prompt to the question being asked.
            try:
                response = self.__client.chat.completions.create(
                        messages=[{
                            "role": "user",
                            "content": question,
                        }],
                        model="gpt-4o-mini",
                        max_tokens = 20
                    )
                return response.choices[0].message.content
            except RateLimitError as e:
                raise ExceptionGroup.RateLimitError(message=("Rate limit reached for the minute"))
            except Exception as e:
                raise ExceptionGroup.ExceptionFromOpenApi(message=str(e))
        except ExceptionGroup.ExceptionFromOpenApi as e:
            logging.error("[OpenAdapter] The following error occured while making an API call: "+str(e))
            raise e
            

    def process_questions(self) -> list:
            """
            Processes a dataset containing questions, invokes the API-endpoint on each question, and returns a list of answers. Logs progress throughout the process.

            Arguments:
            ---------
                None.

            Returns:
            --------
                answers (list): A list of answers generated by the LLM for each question in the dataset.

            Raises:
            -------
                InvalidDataset: Raised when the dataset object is invalid.
                InvalidDataframe: Raised when the dataframe inside the dataset object is invalid or None.
                InsufficientBatchSize: Raised when the dataset size is too small for the inference process.
                Exception: Raised if any unexpected error occurs during the question-processing loop.
            """
            try:
                logging.info("[OpenAdapter] Generating candidate...")
                answers = list()
                data_frame = self.__dataset_handler.get_data_frame()
                total_samples = len(data_frame)
                inputs = dict()
                if total_samples<100:
                    raise ExceptionGroup.InsufficientBatchSize("[OpenAdapter] Insufficient sample size! Please make sure you provide 100+ samples..")
                progress_interval = total_samples // 100  

                logging.info(f"[OllamaAdapter] Starting inference for {self.__dataset_handler.current_dataset} dataset...")
                if self.__dataset_handler.current_dataset == "jeopardy":
                    
                    for i, row in enumerate(data_frame.iterrows(), start=1):
                        question = str(row[1]['question'])
                        category = str(row[1]['category'])
                        got_valid_response = False
                        question = question + "\n Category: " +category
                        while got_valid_response is False:
                            try:
                                answer = self.__invoke(question=question)
                                got_valid_response = True
                            except ExceptionGroup.RateLimitError as e:
                                logging.error("Rate limit reached for the minute, re-trying in 60 seconds..")
                                time.sleep(60)
                                continue
                        if i == 1:
                            logging.info("[OpenAdapter] Here's a sample inference:\n Question: %s \n Output: %s"%(question, answer))
                        answers.append(answer)


                        if i % progress_interval == 0 or i == total_samples:
                            progress_percent = (i / total_samples) * 100
                            logging.info(f"\r[OpenAdapter] Progress: {progress_percent:.0f}% ({i}/{total_samples})")

                    logging.info(f"[OpenAdapter] Inference on dataset: {self.__dataset_handler.current_dataset} is now complete.")

                elif self.__dataset_handler.current_dataset == "millionaire":
                    for i, row in enumerate(data_frame.iterrows(), start=1):
                        question = str(row[1]['question'])
                        got_valid_response = False
                        while got_valid_response is False:
                            try:
                                answer = self.__invoke(question=question)
                                got_valid_response = True
                            except ExceptionGroup.RateLimitError as e:
                                logging.error("Rate limit reached for the minute, re-trying in 60 seconds..")
                                time.sleep(60)
                                continue
                        if i == 1:
                            logging.info("[OpenAdapter] Here's a sample inference:\n Question: %s \n Output: %s"%(question, answer))
                        answers.append(answer)


                        if i % progress_interval == 0 or i == total_samples:
                            progress_percent = (i / total_samples) * 100
                            logging.info(f"\r[OpenAdapter] Progress: {progress_percent:.0f}% ({i}/{total_samples})")

                    logging.info(f"[OpenAdapter] Inference on dataset: {self.__dataset_handler.current_dataset} is now complete.")
                return answers
            except ExceptionGroup.ExceptionFromOpenApi as e:
                logging.error("[ERR] The following error occured while trying to answer the questions: "+str(e))
                raise e

    def __to_lower(self, string: str) -> str:
        """
        Returns a string converted into lowercase for case consistency.
        
        Arguments:
            ---------
                string (str): A string that has to be converted.
        
        Returns:
            --------
                string (str): Target string in lowercase.
        """
        string = str(string)
        return string.lower()

    def perform_inference(self) -> Datasets:
        """
            Updates the current dataframe by inserting two new columns, namely 'raw_output' and 'output', and returns the updated dataframe. Logs progress throughout the process.

            Arguments:
            ---------
                None.

            Returns:
            --------
                candidate_output (pd.Dataframe): A dataframe with answers generated by the LLM for each question in the dataset.

            Raises:
            -------
                InvalidDataset: Raised when the dataset object is invalid.
                InvalidDataframe: Raised when the dataframe inside the dataset object is invalid or None.
                InsufficientBatchSize: Raised when the dataset size is too small for the inference process.
                Exception: Raised if any unexpected error occurs during the question-processing loop.
        """
        try:
            logging.info("[OpenAdapter] Starting LLM inference now.")
            __answers__ = self.process_questions()
            logging.info("[OpenAdapter] Saving these answers within the data_frame.")
            self.__data["raw_output"] = __answers__
            #the only normalization needed is converstion of text into lowercase.
            self.__data["normalized_output"] = self.__data["raw_output"].apply(self.__to_lower)
            self.__dataset_handler.set_data_frame(self.__data) #updates the data_frame with latest changes made locally.
            return self.__dataset_handler
        except Exception as e:
            logging.error("[OpenAdapter] The following error occured while trying processing the candidate responses: "+str(e))
