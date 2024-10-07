"""
parrot.evaluation encompases the metrics which enable evaluation of candidate responses.
"""
from datasets.datasets import Datasets
import logging
from qa_metrics.f1 import f1_match,f1_score_with_precision_recall
from qa_metrics.pedant import PEDANT
from qa_metrics.transformerMatcher import TransformerMatcher
from statistics import mean
import pandas as pd

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
    
    class SampleEvaluationFailed(MillionaireException):
        """
        Exception raised if the sample evaluation fails, or is insuccessful.
        """
        def __init__(self, message= "Failed to evaluate sample!") -> None:
            self.message = message
            super().__init__(self.message)

class MillionaireMetric:
    """
    Enables evaluation of Dataset() containing candidate-llm responses for millionaire-set. 
    """
    def __init__(self, dataset: Datasets) -> None:
        try:
            logging.info("[MillionaireMetric] Initializing MillionaireMetric for evaluation of candidate responses...")
            logging.info("[MillionaireMetric] Performing necessary checks..")
            if not isinstance(dataset, Datasets):
                raise MillionaireExceptionGroup.InvalidDataset(message="Dataset passed cannot be of type None.")
            if dataset.current_dataset != "millionaire":
                raise MillionaireExceptionGroup.InvalidDataset(message="Dataset passed does not correspond to Millionaire-set.")
            
            self.__data_frame = dataset.get_data_frame() #get the millionaire candidate response, dataframe.
            if len(self.__data_frame) == 0:
                raise MillionaireExceptionGroup.EmptyDataFrame()

            logging.info("[MillionaireMetric] Loading weights required to compute performance per sample...")
            #loading the weights:
            self.__WEIGHTS__ = {"Level_1": 0.025821596244131457, 
                                "Level_2": 0.007042253521126761, 
                                "Level_3": 0.007042253521126761, 
                                "Level_4": 0.011737089201877935, 
                                "Level_5": 0.011737089201877935, 
                                "Level_6": 0.018779342723004695, 
                                "Level_7": 0.025821596244131457, 
                                "Level_8": 0.051643192488262914, 
                                "Level_9": 0.07042253521126761, 
                                "Level_10": 0.06807511737089202, 
                                "Level_11": 0.14788732394366197, 
                                "Level_12": 0.13145539906103287, 
                                "Level_13": 0.03286384976525822, 
                                "Level_14": 0.16431924882629106, 
                                "Level_15": 0.22535211267605634}
            logging.info("[MillionaireMetric] Weights have been initialized.")

            #init requirements:
            logging.info("[MillionaireMetric] Mapping device for logits...")
            self.__pedant = PEDANT()
            self.__transformer_matcher = TransformerMatcher("tiny-bert") 

            logging.info("[MillionaireMetric] All checks are complete. Initialization was successful!")
        except Exception as e:
            logging.error(str(e))

            raise e

    def __get_pedant_score(self, question:str, answer:str, candidate_answer:str) -> float:
        """
        This method employs a judge LLM (tiny-bert) metric that produces a score based of the 7 AC rules towards PEDANT wrapped around exception handling.
        """
        try:
            return self.__pedant.get_score(reference=answer, candidate=candidate_answer, question=question)
        except Exception as e: #__pedant.get_score raises its own exception.
            logging.error(str(e))
            raise e

    def __get_judge_match_score(self, answer: str, candidate: str, question: str) -> tuple:
        """
        This method employs a judge LLM (tiny-bert) which returns a score for candidate and labeled answer match. 
        This method returns a tuple of match score and boolean match result.
        """
        try:
            score = self.__transformer_matcher.get_score(answer, candidate, question)
            match_result = self.__transformer_matcher.transformer_match(reference = answer, candidate = candidate, question = question)
            return (score, match_result)
        except Exception as e:
            logging.error(str(e))
            raise e

    def evaluate_candidate_responses(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        The evaluate method makes use of the helper method to perform regression over a dataframe by iteration.
        """
        try:
            logging.info("[MillionaireMetric] Determining answer correctness for each sample...")
            scores = list() #local variable to store individual pedant scores.
            judge_match_score = list() #local variable to store JudgeLLM score.
            judge_match = list()
            for index, row in df.iterrows():
                question = row["question"]
                answer = row["normalized_correct_opt"]
                candidate = row["normalized_output"]
                score = self.__get_pedant_score(question = question, answer=answer, candidate_answer=candidate)
                judge_results = self.__get_judge_match_score(answer= answer, question= question, candidate= candidate)  
                judge_match_score.append(judge_results[0])
                judge_match.append(judge_results[1])
                scores.append(score)
            logging.info("[MillionaireMetric] Done.")
            logging.info("[MillionaireMetric] Injecting values into the dataframe.")
            df["pedant_score"] = scores
            df["match_score"] = judge_match_score
            df["match"] = judge_match
            logging.info("[MillionaireMetric] Done.")
            return df
        except RuntimeError as e:
            logging.error("[MillionaireMetric] The following error ocurred while evaluating candidate samples: "+str(e))
            raise e

class JeopardyMetric:
    pass

class Evaluate:
    pass