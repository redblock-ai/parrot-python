"""
To run the test_script, first install pytest using <pip install pytest>
using CLI run the script: <pytest -v --disable-warnings test_cases.py>
"""

import pytest
import pandas as pd
from datasets.datasets import Datasets
from parrot_ollama import OllamaAdapter

#Datasets TESTCASES:
@pytest.mark.DatasetsSampleSizeSuccess
def test_Datasets_sample_size_success():
    """
    Test if the Datasets object is able to return a data_frame with size n.
    """
    sample = 50
    obj = Datasets(dataset = "millionaire", sample_size=sample)
    data = obj.get_data_frame()

    assert len(data) == sample

@pytest.mark.DatasetsSampleSizeFailure
def test_Datasets_sample_size_failure():
    """
    Test if the Datasets object is raising an exception when an invalid sample size is provided.
    """
    with pytest.raises(ValueError):
        sample = -1203
        obj = Datasets(dataset = "millionaire", sample_size=sample)
        data = obj.get_data_frame()

        assert data is None

@pytest.mark.DatasetsSampleSizeNotProvidedSuccess
def test_Datasets_sample_size_not_provided_success():
    """
    Test if the Datatsets object is returning the entire dataset without expecting a sample_size param.
    """
    obj = Datasets(dataset = "millionaire")
    data = obj.get_data_frame()
    assert data is not None
    assert len(data.columns) != 0
    assert len(data) !=0


#OllamaAdapter TESTCASES:
@pytest.mark.OllamaAdapterInitSuccess
def test_OllamaAdapter_initialization_success():
    """
    Test successful initialization of OllamaAdapter
    """
    obj = Datasets(dataset = "millionaire")
    data = obj.get_data_frame()
    model_name = "phi3.5"
    prompt = """Answer following question in directly without any additional text, Question: {question}"""
    
    adapter = OllamaAdapter(
        data_frame=data, 
        model_name=model_name, 
        prompt=prompt
    )
    

    assert adapter._OllamaAdapter__data is not None
    assert adapter._OllamaAdapter__model is not None
    assert adapter._OllamaAdapter__prompt is not None
    assert adapter._OllamaAdapter__chain is not None

@pytest.mark.OllamaAdapterInitFailure
def test_OllamaAdapter_initialization_failure_invalid_dataframe():
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

    
