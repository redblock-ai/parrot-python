"""
To run the test_script, first install pytest using <pip install pytest>
using CLI run the script: <pytest -v --disable-warnings test_cases.py>
"""

import pytest
from .datasets.datasets import Datasets
from .parrot_ollama import OllamaAdapter
from .evaluation import MillionaireMetric, JeopardyMetric
from qa_metrics.pedant import PEDANT
import pandas as pd
from .parrot_openai import OpenAdapter
import json


with open('Key.json', 'r') as file:
    data = json.load(file)

# Print the data
OPEN_AI_KEY = data["OpenAI_key"]

#Datasets TESTCASES:
@pytest.mark.Datasets
@pytest.mark.DatasetsSampleSizeSuccess
def test_datasets_sample_size_success():
    """
    Test if the Datasets object is able to return a data_frame with size n.
    """
    sample = 50
    obj = Datasets(dataset = "millionaire", sample_size=sample)
    data = obj.get_data_frame()

    assert len(data) == sample

@pytest.mark.Datasets
@pytest.mark.DatasetsSampleSizeFailure
def test_datasets_sample_size_failure():
    """
    Test if the Datasets object is raising an exception when an invalid sample size is provided.
    """
    with pytest.raises(ValueError):
        sample = -1203 #invalid sample size, this cannot be negative.
        obj = Datasets(dataset = "millionaire", sample_size=sample) #this call should raise an ValueError.
        data = obj.get_data_frame()
        assert data is None #data should be None.

@pytest.mark.Datasets
@pytest.mark.DatasetsSampleSizeFailure
def test_datasets_sample_size_not_provided_success():
    """
    Test if the Datatsets object is returning the entire dataset without expecting a sample_size param.
    """
    obj = Datasets(dataset = "millionaire")
    data = obj.get_data_frame()
    assert data is not None #pd.Dataframe should not be None.
    assert len(data.columns) != 0 #colomns should not be empty.
    assert len(data) !=0 #Dataframe cannot be empty!

#OllamaAdapter TESTCASES:
@pytest.mark.OllamaAdapter
@pytest.mark.OllamaAdapterInitSuccess
def test_ollama_adapter_initialization_success():
    """
    Test successful initialization of OllamaAdapter
    """
    obj = Datasets(dataset = "millionaire")
    model_name = "llama3.2:1b"
    prompt = """Answer following question directly without any additional text, Question: {question}"""
    
    adapter = OllamaAdapter(
        dataset=obj, 
        model_name=model_name, 
        prompt=prompt
    )
    
    assert adapter._OllamaAdapter__data is not None
    assert adapter._OllamaAdapter__model is not None
    assert adapter._OllamaAdapter__prompt is not None
    assert adapter._OllamaAdapter__chain is not None

@pytest.mark.OllamaAdapter
@pytest.mark.OllamaAdapterInitFailure
def test_ollama_adapter_initialization_failure_invalid_dataframe():
    """
    Test that OllamaAdapter raises an Exception when an invalid or None data_frame is passed
    """
    model_name = "test_model"
    prompt = "This is a test prompt."

    #we are expecting an exception to be raised here:
    with pytest.raises(Exception) as exc_info:
        OllamaAdapter(
            data_frame=None,  #data_frame is NaN
            model_name=model_name, 
            prompt=prompt
        )

@pytest.mark.OllamaAdapter
@pytest.mark.OllamaAdapterInitFailure
def test_ollama_adapter_initialization_failure_invalid_model():
    """
    Test that OllamaAdapter raises an Exception when the requested model isn't available.
    """
    obj = Datasets(dataset = "millionaire")
    model_name = "ph"
    prompt = """Answer following question directly without any additional text, Question: {question}"""
    

    #we are expecting an exception to be raised here:
    with pytest.raises(Exception):
        
        OllamaAdapter(
                dataset=obj,  
                model_name=model_name, #Model isn't available locally.
                prompt=prompt
            )

@pytest.mark.OllamaAdapter        
@pytest.mark.OllamaAdapterMillionaireTestSuccess
def test_ollama_adapter_millionaire_test_success():
    """
    Test if OllamaAdapter is able to generate candidate answers for Millionaire set.
    """
    sample_size = 100
    dataset = Datasets(dataset= "millionaire", sample_size=sample_size)
    model_name = "llama3.2:1b"
    prompt = """Answer following question directly without any additional text, Question: {question}"""
    ollama_handler = OllamaAdapter(
            dataset = dataset,  
            model_name=model_name, #Model isn't available locally.
            prompt=prompt
        )

    answers = ollama_handler.process_questions()

    assert answers is not None #response must not be None.
    assert isinstance(answers, list) #check if response is list of answers.
    assert len(dataset.get_data_frame()) == len(answers) #No empty responses.

@pytest.mark.OllamaAdapter
@pytest.mark.OllamaAdapterJeopardyTestSuccess
def test_ollama_adapter_jeopardy_test_success():
    """
    Test if OllamaAdapter is able to generate candidate answers for Jeopardy set.
    """
    sample_size = 100
    dataset = Datasets(dataset= "jeopardy", sample_size=sample_size)
    model_name = "llama3.2:1b"
    prompt = """Answer following question directly without any additional text, Question: {question}"""
    ollama_handler = OllamaAdapter(
            dataset = dataset,  
            model_name=model_name, #Model isn't available locally.
            prompt=prompt
        )

    answers = ollama_handler.process_questions()

    assert answers is not None #response must not be None.
    assert isinstance(answers, list) #check if response is list of answers.
    assert len(dataset.get_data_frame()) == len(answers) #No empty responses.

#Test_cases for: MillionaireMetric
@pytest.mark.MillionaireMetric
@pytest.mark.MillionaireMetricInitSuccess
def test_millionaire_metric_initialization_success():
    """
    Test if MillionaireMetric is instantiated successfully on passing a Millionaire-set DAO.
    """
    obj = Datasets(dataset = "millionaire", sample_size=100)

    model_name = "llama3.2:1b"
    prompt = """Answer following question directly without any additional text, Question: {question}"""
        
    ollama_handler = OllamaAdapter(
                dataset = obj,  
                model_name=model_name, 
                prompt=prompt
            )
    obj = ollama_handler.perform_inference() #get the updated data_frame with answers.

    assert obj.current_dataset is "millionaire"
    milm = MillionaireMetric(dataset= obj)
    assert isinstance(milm.__MILLIONAIRE_WEIGHTS__, dict)

@pytest.mark.MillionaireMetric
@pytest.mark.MillionaireMetricInitFailure
def test_millionaire_metric_init_failure_invalid_dataset():
    """
    Test if MillionaireMetric raises an Exception when mis-matching dataset is passed for evaluation.
    """
    obj = Datasets(dataset = "jeopardy", sample_size=100)

    model_name = "llama3.2:1b"
    prompt = """Answer following question directly without any additional text \nQuestion: {question}"""
        
    ollama_handler = OllamaAdapter(
                dataset = obj,  
                model_name=model_name, 
                prompt=prompt
            )
    obj = ollama_handler.perform_inference() #get the updated data_frame with answers.

    with pytest.raises(Exception):
        milm = MillionaireMetric(dataset= obj) #dataset passed is if type "jeopardy" which should raise an invalid dataset type.

@pytest.mark.MillionaireMetric
@pytest.mark.MillionaireMetricScoreGeneration
def test_millionaire_metric_score_generation():
    """
    Test if MillionaireMetric is able to gauge the answer correctness of a sample label and candidate response.
    """
    obj = Datasets(dataset = "millionaire", sample_size=100)
    size = len(obj.get_data_frame())
    
    model_name = "llama3.2:1b"
    prompt = """Answer following question directly without any additional text \nQuestion: {question}"""
        
    ollama_handler = OllamaAdapter(
                dataset = obj,  
                model_name=model_name, 
                prompt=prompt
            )
    obj = ollama_handler.perform_inference() #get the updated data_frame with answers.
    milm = MillionaireMetric(dataset= obj)
    data = milm.evaluate_candidate_responses(df=obj.get_data_frame())
    report = milm.compute_millionaire_score()
    assert isinstance(data, pd.DataFrame) #check if the data_frame is not corrupt 
    assert len(data) !=0 and len(data) == size #evaluation must not lead to sample size reduction. 
    assert "pedant_score" in data.columns #pedant score is the new column that is added.
    assert isinstance(report, dict)
    assert report.get("millionaire_score", None) is not None #the report MUST contain the millionaire score!
    assert isinstance(report["millionaire_score"], float) and report["millionaire_score"]> 0.00
    assert len(report.keys()) > 1 #report should also contain individual performance scores per level!

        
#Test_cases for: JeopardyMetric
@pytest.mark.JeopardyMetric
@pytest.mark.JeopardyMetricInitSuccess
def test_jeopardy_metric_initialization_success():
    """
    Test if JeopardyMetric is instantiated successfully on passing a Jeopardy-set DAO.
    """
    obj = Datasets(dataset = "jeopardy", sample_size=100)

    model_name = "llama3.2:1b"
    prompt = """Answer following question directly without any additional text, \n Question: {question} \n Category: {category}"""
        
    ollama_handler = OllamaAdapter(
                dataset = obj,  
                model_name=model_name, 
                prompt=prompt
            )
    obj = ollama_handler.perform_inference() #get the updated data_frame with answers.

    assert obj.current_dataset is "jeopardy"
    milm = JeopardyMetric(dataset= obj)
    assert isinstance(milm.__JEOPARDY_WEIGHTS__, dict)

@pytest.mark.JeopardyMetric
@pytest.mark.JeopardyMetricInitFailure
def test_jeopardy_metric_init_failure_invalid_dataset():
    """
    Test if JeoaprdyMetric raises an Exception when mis-matching dataset is passed for evaluation.
    """
    obj = Datasets(dataset = "millionaire", sample_size=100)

    model_name = "llama3.2:1b"
    prompt = """Answer following question directly without any additional text \nQuestion: {question}"""
        
    ollama_handler = OllamaAdapter(
                dataset = obj,  
                model_name=model_name, 
                prompt=prompt
            )
    obj = ollama_handler.perform_inference() #get the updated data_frame with answers.

    with pytest.raises(Exception):
        jeop = JeopardyMetric(dataset= obj) #dataset passed is if type "millionaire" which should raise an invalid dataset type.

@pytest.mark.JeopardyMetric
@pytest.mark.JeopardyMetricScoreGeneration
def test_jeopardy_metric_score_generation():
    """
    Test if JeopardyMetric is able to gauge the answer correctness of a sample label and candidate response.
    """
    obj = Datasets(dataset = "jeopardy", sample_size=100)
    size = len(obj.get_data_frame())
    
    model_name = "llama3.2:1b"
    prompt = """Answer following question directly without any additional text \nQuestion: {question} \n Category: {category}"""
        
    ollama_handler = OllamaAdapter(
                dataset = obj,  
                model_name=model_name, 
                prompt=prompt
            )
    obj = ollama_handler.perform_inference() #get the updated data_frame with answers.
    jeop = JeopardyMetric(dataset= obj)
    data = jeop.evaluate_candidate_responses(df=obj.get_data_frame())
    report = jeop.compute_jeopardy_score()
    assert isinstance(data, pd.DataFrame) #check if the data_frame is not corrupt 
    assert len(data) !=0 and len(data) == size #evaluation must not lead to sample size reduction. 
    assert "pedant_score" in data.columns #pedant score is the new column that is added.
    assert isinstance(report, dict)
    assert report.get("jeopardy_score", None) is not None #the report MUST contain the millionaire score!
    assert isinstance(report["jeopardy_score"], float) and report["jeopardy_score"]> 0.00
    assert len(report.keys()) > 1 #report should also contain individual performance scores per level!


#Testcases for OpenAI adapter!
@pytest.mark.OpenAdapter
@pytest.mark.OpenAdapterInitFailure
def test_api_key_not_found():
    """
    Test if the user has passed an api_key, if not should raise an exception.
    """
    obj = Datasets(dataset = "millionaire", sample_size=100)
    model_name = "gpt-4o-mini"
    prompt = """Answer following question directly without any additional text \nQuestion:"""
    with pytest.raises(Exception):
        open_handler = OpenAdapter(
                dataset = obj,  
                model_name=model_name, 
                prompt=prompt
            ) #no api_key was passed, should raise an excetion: OpenAdapterException.InvalidCredentials().

@pytest.mark.OpenAdapter
@pytest.mark.OpenAdapterInitSuccess
def test_valid_credentials_passed():
    """
    Test if the user has passed an api_key, if not should raise an exception.
    """
    obj = Datasets(dataset = "millionaire", sample_size=100)
    model_name = "gpt-4o-mini"
    prompt = """Answer following question directly without any additional text \nQuestion:"""
   
    open_handler = OpenAdapter(
                dataset = obj,  
                model_name=model_name, 
                prompt=prompt,
                api_key= OPEN_AI_KEY
            ) #Valid API key passed, this should not raise an exception

