"""
parrot.evaluation encompases the metrics which enable evaluation of candidate responses.
"""
from datasets.datasets import Datasets
import logging
from qa_metrics.pedant import PEDANT
from statistics import mean
import pandas as pd
import re

#config logging
logging.basicConfig( 
                filename=None,
                level=logging.INFO,          
                format='%(asctime)s - %(levelname)s - %(message)s',  #our custom log format
                datefmt='%Y-%m-%d %H:%M:%S'
            )

class EvaluationException(Exception):
    """
    Base class for custom exceptions.
    """
    pass

class EvaluationExceptionGroup:
    class InvalidDataset(EvaluationException):
        """
        Exception raised for invalid dataset type.
        """
        def __init__(self, message="Invalid dataset object received"):
            self.message = message
            super().__init__(self.message)
    
    class EmptyDataFrame(EvaluationException):
        """
        Exception raised if the data_frame is empty.
        """
        def __init__(self, message="Data frame cannot be empty! Please check the dataset passed for evaluation") -> None:
            self.message = message
            super().__init__(self.message)
    
    class SampleEvaluationFailed(EvaluationException):
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
                raise EvaluationExceptionGroup.InvalidDataset(message="Dataset passed cannot be of type None.")
            if dataset.current_dataset != "millionaire":
                raise EvaluationExceptionGroup.InvalidDataset(message="Dataset passed does not correspond to Millionaire-set.")
            
            self.__data_frame = dataset.get_data_frame() #get the millionaire candidate response, dataframe.
            if len(self.__data_frame) == 0:
                raise EvaluationExceptionGroup.EmptyDataFrame()

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
            logging.info("[MillionaireMetric] All checks are complete. Initialization was successful!")
        except Exception as e:
            logging.error(str(e))

            raise e

    def __get_level_number(self, string:str) -> int:
        """
        This private member function extracts the level of the question being posed out of the 15 levels in place for Millionaire.
        """
        try:
            return int(re.findall(pattern="\((.[0-9]*)", string=string)[0])#can raise array out of bound index error when there is no such number.)
        except:
            return 1 #returns question of level 1 

    def __get_pedant_score(self, question:str, answer:str, candidate_answer:str) -> float:
        """
        This method employs a judge LLM (tiny-bert) metric that produces a score based of the 7 AC rules towards PEDANT wrapped around exception handling.
        """
        try:
            return self.__pedant.get_score(reference=answer, candidate=candidate_answer, question=question)
        except Exception as e: #__pedant.get_score raises its own exception.
            logging.error(str(e))
            raise e

    def evaluate_candidate_responses(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        The evaluate method makes use of the helper method to perform regression over a dataframe by iteration.
        """
        try:
            logging.info("[MillionaireMetric] Determining answer correctness for each sample...")
            scores = list() #local variable to store individual pedant scores.
            for index, row in df.iterrows():
                question = row["question"]
                answer = row["normalized_correct_opt"]
                candidate = row["normalized_output"]
                score = self.__get_pedant_score(question = question, answer=answer, candidate_answer=candidate)
                scores.append(score)
            logging.info("[MillionaireMetric] Done.")
            logging.info("[MillionaireMetric] Injecting values into the dataframe.")
            df["pedant_score"] = scores
            logging.info("[MillionaireMetric] Done.")
            return df
        except EvaluationExceptionGroup.SampleEvaluationFailed as e:
            logging.error("[MillionaireMetric] The following error ocurred while evaluating candidate samples: "+str(e))
            raise e

    def compute_millionaire_score(self) -> dict:
        """
        This method computes the millionaire score by using the latent variable weight, per level. 
        """
        try:
            logging.info("[MillionaireMetric] Determining PARROT-Millionaire score.")
            self.__data_frame = self.evaluate_candidate_responses(self.__data_frame) #perform samplewise evaluation.

            logging.info("[MillionaireMetric] Extracting level no. from question info per-sample.")
            self.__data_frame["level"] = self.__data_frame["question_info"].apply(self.__get_level_number)
            logging.info("[MillionaireMetric] New column added with name 'level' to the data-frame.")
            score_at_level = list()
            millionaire_score = dict() #results object.

            logging.info("[MillionaireMetric] Calculating final results..")
            #determine the performance per-level. Summation of performance at each level yields the final parrot-milllionaire score.
            for level_number in self.__data_frame["level"].unique():
                pedant_scores = self.__data_frame["pedant_score"].loc[self.__data_frame["level"] == level_number]
                avg_pedant_score_per_question = mean(pedant_scores)
                key = f"Level_{level_number}"
                ac_score = self.__WEIGHTS__[key] * avg_pedant_score_per_question
                performance_at_level = f"performance_at_level_{level_number}"
                millionaire_score[performance_at_level] = round(ac_score, 2)
                score_at_level.append(ac_score)
            logging.info("[MillionaireScore] PARROT-Millionaire results are now ready!")
            
            millionaire_score["millionaire_score"] = round(sum(score_at_level), 2) 
            
            return millionaire_score
        except EvaluationExceptionGroup.SampleEvaluationFailed as e:
            logging.error("[MillionaireMetric] The following error ocurred while calculating the Millionaire score: "+str(e))
            raise e

class JeopardyMetric:
    pass

class Evaluate(MillionaireMetric, JeopardyMetric):
    """
    Class that inherits from JeopardyMetric and MillionaireMetric to serve as a composite evaluation for determining the PARROT score.
    """

    def __init__(self, dataset: Datasets) -> None:
        try:
            logging.info("[Evaluate] Determining Metric to apply for the outputs..")
            # Directly call the appropriate base class initializer based on dataset type
            if dataset.current_dataset == "millionaire":
                MillionaireMetric.__init__(self, dataset)
            elif dataset.current_dataset == "jeopardy":
                JeopardyMetric.__init__(self, dataset)
            else:
                raise EvaluationExceptionGroup.InvalidDataset(
                    "[Evaluate] Invalid dataset passed! Dataset must be of type Jeopardy or Millionaire."
                )
        except Exception as e:
            logging.error("[Evaluate] The following error occurred while initializing evaluation module: " + str(e))
            raise e

    # Wrapper method that computes the millionaire score.
    def get_millionaire_report(self):
        return self.compute_millionaire_score()

    def get_parrot_score(self):
        millionaire_report = self.get_millionaire_report()
        millionaire_score = millionaire_report["millionaire_score"]
        parrot_score = millionaire_score 
        return parrot_score

