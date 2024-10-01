"""
A class that servers as a Ollama adapter for model inference over PARROT-datasets.
"""

import pandas as pd
import logging
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
import subprocess
from langchain_ollama.llms import OllamaLLM

class OllamaAdapter:
    """
    A custom-class, that servers as a adapter for model inference over PARROT-datasets using Ollama.
    """
    
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
            
            self.__model = OllamaLLM( model=model_name, 
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


    def get_answer(self, question:str) -> str:
        """
        Helper method, which accepts a string/user query and returns the response received from the LLM.
        """
        try:
            response = self.__chain.invoke({"question": question})
            return response
        except Exception as e:
            logging.error(f"\r[OllamaAdapter] Failed to answer the question: {question}, due to error: {str(e)}")

    
    def process_questions(self, data_frame: pd.DataFrame) -> list:
            """
            Helper Method, which accepts the dataframe and returns the list of candidate answers.
            """
            try:
                logging.info("[OllamaAdapter] Generating candidate answers for the WWTBAM dataset...")
                answers = list()
                total_samples = len(data_frame)

                if total_samples<100:
                    raise Exception("[OllamaAdapter] Insufficient sample size! Please make sure you provide 100+ samples..")
                progress_interval = total_samples // 100  

                for i, row in enumerate(data_frame.iterrows(), start=1):
                    
                    question = row[1]['question']
                    answer = self.get_answer(question)
                    if i == 1:
                        logging.info("[OllamaAdapter] Here's a sample output: %s"%(answer))
                    answers.append(answer)


                    if i % progress_interval == 0 or i == total_samples:
                        progress_percent = (i / total_samples) * 100
                        logging.info(f"\r[OllamaAdapter] Progress: {progress_percent:.0f}% ({i}/{total_samples})")

                logging.info("[OllamaAdapter] Inference is now complete.")
                return answers
                
            except Exception as e:
                logging.error("[ERR] The following error occured while trying to answer the questions: "+str(e))
