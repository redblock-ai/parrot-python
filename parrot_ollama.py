"""
A class that servers as a Ollama adapter for model inference over PARROT-datasets.
"""

import pandas as pd
import logging
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
import subprocess
from datasets.datasets import Datasets
from langchain_ollama.llms import OllamaLLM

class OllamaAdapterExceptions(Exception):
    """
    Base class for Custom Exceptions within Ollama Adapter.
    """
    pass

class ExceptionGroup:
    class InvalidDataset(OllamaAdapterExceptions):
        """Exception raised for invalid dataset access object"""
        def __init__(self, message="Invalid Dataset Object passed."):
            self.message = message
            super().__init__(self.message)
    
    class InvalidDataframe(OllamaAdapterExceptions):
        """Exception raised for invalid dataframe"""
        def __init__(self, message="Invalid Dataset Object passed."):
            self.message = message
            super().__init__(self.message)

    class ModelNameCannotBeNone(OllamaAdapterExceptions):
        def __init__(self, message="Invalid model passed for inference."):
            self.message = message
            super().__init__(self.message)
    
    class ModelNameCannotBeEmpty(OllamaAdapterExceptions):
        def __init__(self, message="Paramter 'model_name' cannot be empty."):
            self.message = message
            super().__init__(self.message)

    class PromptCannotBeNone(OllamaAdapterExceptions):
        def __init__(self, message="Paramter 'prompt' passed cannot be of type None."):
            self.message = message
            super().__init__(self.message)
    
    class PromptCannotBeEmpty(OllamaAdapterExceptions):
        def __init__(self, message="Paramter 'prompt' cannot be empty."):
            self.message = message
            super().__init__(self.message)

    class InsufficientBatchSize(OllamaAdapterExceptions):
        def __init__(self, message="Paramter 'prompt' cannot be empty."):
            self.message = message
            super().__init__(self.message)

class OllamaAdapter:
    """
    A custom-class, that servers as a adapter for model inference over PARROT-datasets using Ollama.
    """
    
    def __init__(self, 
    dataset:Datasets = None,
    model_name: str = None,
    prompt: str = None,
    temperature: float = 0,
    top_k: int = 10,
    top_n: float = 0.2) -> None:
        """
        Initializes the inference by building a chain.
        """
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
                raise ExceptionGroup.InvalidDataframe("[OllamaAdapter] - Invalid parameter 'data_frame' passed for inference.")
            self.__data = data_frame #init data
            logging.info("[OllamaAdapter] - Data initialized") 
            
            #perform checks for parameter 'model_name'
            if model_name is None:
                raise ExceptionGroup.ModelNameCannotBeNone(message=f"The model name {model_name}, cannot be of type None.")
            elif model_name.strip() == "":
                raise ExceptionGroup.ModelNameCannotBeEmpty()
            
            self.__model = OllamaLLM( model=model_name, 
                temperature=temperature, 
                top_k=top_k, 
                top_p=top_n)
            logging.info(f"[OllamaAdapter] New instance created for {model_name}.")
            
            #perform checks for paramter 'prompt'
            if prompt is None:
                raise ExceptionGroup.PromptCannotBeNone()
            elif prompt.strip() == "":
                raise ExceptionGroup.PromptCannotBeEmpty()
                
            logging.info("[OllamaAdapter] Building promp_template object for the user prompt provided.")
            self.__prompt = ChatPromptTemplate.from_template(prompt)
            logging.info("[OllamaAdapter] PromptTemplate object initialized successfully.")

            logging.info("[OllamaAdapter] Building a chain for inference...")
            self.__chain = self.__prompt | self.__model
            logging.debug("[OllamaAdapter] Checking if requested LLM is available on Ollama locally...")
            try:
                self.__chain.invoke({"question": "how are you?"}) #testing to check if the chain is functional. This should not raise an exception.
            except:
                raise Exception(f"Model not found, download it using the CLI command 'ollama run <model-name>'")
            logging.info("[OllamaAdapter] Chain built successfully.")
            logging.info(f"[OllamaAdapter] {model_name} is ready for benchmarking!")
        except Exception as e:
            logging.exception(e)
            raise e


    def get_answer(self, inputs:dict) -> str:
        """
        Helper method, which accepts a dictionary of inputs to the LLM as query and returns the response received from the LLM.
        """
        try:
            response = self.__chain.invoke(inputs)
            return response
        except Exception as e:
            logging.error(f"\r[OllamaAdapter] LLM failed to process inputs: {inputs}, due to error: {str(e)}")

    
    def process_questions(self) -> list:
            """
            Helper Method, which accepts the dataframe and returns the list of candidate answers.
            """
            try:
                logging.info("[OllamaAdapter] Generating candidate answers for the WWTBAM dataset...")
                answers = list()
                data_frame = self.__dataset_handler.get_data_frame()
                total_samples = len(data_frame)
                inputs = dict()
                if total_samples<100:
                    raise ExceptionGroup.InsufficientBatchSize("[OllamaAdapter] Insufficient sample size! Please make sure you provide 100+ samples..")
                progress_interval = total_samples // 100  

                logging.info(f"[OllamaAdapter] Starting inference for {self.__dataset_handler.current_dataset} dataset...")
                if self.__dataset_handler.current_dataset == "millionaire":
                    
                    for i, row in enumerate(data_frame.iterrows(), start=1):
                        inputs.clear() #clear the inputs.
                        inputs['question'] = row[1]['question']
                        answer = self.get_answer(inputs = inputs)
                        if i == 1:
                            logging.info("[OllamaAdapter] Here's a sample output: %s"%(answer))
                        answers.append(answer)


                        if i % progress_interval == 0 or i == total_samples:
                            progress_percent = (i / total_samples) * 100
                            logging.info(f"\r[OllamaAdapter] Progress: {progress_percent:.0f}% ({i}/{total_samples})")

                    logging.info("[OllamaAdapter] Inference is now complete.")
                elif self.__dataset_handler.current_dataset == "jeopardy":
                    for i, row in enumerate(data_frame.iterrows(), start=1):
                        inputs.clear()
                        inputs['question'] = row[1]['question']
                        inputs['category'] = row[1]['category']
                        answer = self.get_answer(inputs= inputs)
                        if i == 1:
                            logging.info("[OllamaAdapter] Here's a sample output: %s"%(answer))
                        answers.append(answer)


                        if i % progress_interval == 0 or i == total_samples:
                            progress_percent = (i / total_samples) * 100
                            logging.info(f"\r[OllamaAdapter] Progress: {progress_percent:.0f}% ({i}/{total_samples})")
                return answers
                
            except Exception as e:
                logging.error("[ERR] The following error occured while trying to answer the questions: "+str(e))
