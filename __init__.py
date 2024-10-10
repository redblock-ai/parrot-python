from .datasets.datasets import Datasets
from .parrot_ollama import OllamaAdapter
from .evaluation import MillionaireMetric, JeopardyMetric
from statistics import mean

def evaluate_model(model_name:str, generate_reports:bool = False, millionaire_sample_size:int = None, jeopardy_sample_size:int = None, temperature: float= 0) -> dict:
    if millionaire_sample_size is None:
        millionaire = Datasets(dataset = "millionaire")
    else:
        millionaire = Datasets(dataset = "millionaire", sample_size = millionaire_sample_size)
    if jeopardy_sample_size is None:
        jeopardy = Datasets(dataset = "jeopardy")
    else:
        jeopardy = Datasets(dataset = "jeopardy", sample_size = jeopardy_sample_size)

    ollama_adapter = OllamaAdapter(dataset= millionaire, model_name=model_name,temperature=0, prompt="Answer the following question directly without additional text\n Question: {question}")
    ollama_adapter_j = OllamaAdapter(dataset= jeopardy, model_name=model_name,temperature=0, prompt="Answer the following question directly without additional text based on the category\n Question: {question}\n Category: {category}")

    millionaire = ollama_adapter.perform_inference()
    jeopardy = ollama_adapter_j.perform_inference() 

    millionaire_eval = MillionaireMetric(dataset=millionaire)
    millionaire_report = millionaire_eval.compute_millionaire_score()
    jeopardy_eval = JeopardyMetric(dataset=jeopardy)
    jeopardy_report = jeopardy_eval.compute_jeopardy_score()

    result = dict()
    result["model_name"] = model_name
    result["parrot_score"] = mean([millionaire_report["millionaire_score"], jeopardy_report["jeopardy_score"]])
    if generate_reports is False:
        return result
    else:
        result["parrot_millionaire"] = millionaire_report
        result["parrot_jeopardy"]= jeopardy_report
        return result

