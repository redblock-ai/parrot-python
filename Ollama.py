"""
A class that servers as a Ollama adapter for model inference over PARROT-datasets.
"""

import pandas as pd
import logging
import os
import Ollama
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain.chains import LLMChain

class OllamaAdapter:
    """
    A custom-class, that servers as a adapter for model inference over PARROT-datasets using Ollama.
    """
    __data: pd.DataFrame = None
    __prompt: str = None
    __model: OllamaLLM = None
    __prompt: ChatPromptTemplate = None
    __chain: object = None


    def __init__(self, 
    data_frame:pd.DataFrame = None,
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
            
            #perform parameter checks.
            if data_frame is None or not isinstance(data_frame, pd.DataFrame):
                raise Exception("[OllamaAdapter] - Invalid parameter 'data_frame' passed for inference.")
            self.__data = data_frame #init data
            logging.info("[OllamaAdapter] - Data initialized") 
            
            
            if model_name is None or model_name.strip() == "":
                raise Exception("[OllamaAdapter] - Invalid parameter: 'model_name' passed for inference.")
            
            self.__model = OllamaLLM( model="phi-3.5", 
                temperature=temperature, 
                top_k=top_k, 
                top_p=top_n)
            logging.info(f"[OllamaAdapter] New instance created for {model_name}.")
            
            if prompt is None or prompt.strip() == "":
                raise Exception("[OllamaAdapter] Prompt cannot be empty!")

            logging.info("[OllamaAdapter] Building promp_template object for the user prompt provided.")
            self.__prompt = ChatPromptTemplate.from_template(prompt)
            logging.info("[OllamaAdapter] PromptTemplate object initialized successfully.")

            logging.info("[OllamaAdapter] Building a chain for inference...")
            self.__chain = self.__prompt | self.__model
            logging.info("[OllamaAdapter] Chain built successfully.")
        except Exception as e:
            logging.exception(e)