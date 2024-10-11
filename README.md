
# PARROT Benchmark for Large Language Model (LLM) Evaluation

**PARROT (Performance Assessment of Reasoning and Responses On Trivia)** is a novel benchmarking framework designed to evaluate Large Language Models (LLMs) on real-world, complex, and ambiguous QA tasks. PARROT is based on two major datasets derived from popular game shows: *Who Wants to Be a Millionaire?* and *Jeopardy*. It introduces unique metrics that capture the complexity and difficulty of real-world question-answering tasks, providing a more rigorous assessment of LLMs' reasoning and decision-making capabilities.

## Features

- **Dual-Dataset Approach**: 
  - **PARROT-Millionaire Dataset**: Simulates high-stakes decision-making under pressure, focusing on progressively more difficult fact-based questions.
  - **PARROT-Jeopardy Dataset**: Focuses on deeper reasoning and ambiguity-handling, assessing LLMs' ability to deal with nuanced trivia.
  
- **Metrics**:
  - **Millionaire Metric**: Weights questions based on their difficulty, giving more importance to answering complex questions correctly.
  - **Jeopardy Metric**: Evaluates the model’s ability to handle multi-layered, ambiguous questions with a structured difficulty gradient.
  
- **Comprehensive Scoring**: The framework calculates the **PARROTscore**, a composite metric representing the model’s performance across the two datasets, providing a holistic evaluation.
---
## Getting Started

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/HarshaLLM/parrot.git
   cd parrot
   ```
2. **Install parrot on Machine**:
   ```bash
   pip install -e .
   ```

3. **Initializing Ollama (Optional)**:

   Start by [downloading](https://ollama.com/download) Ollama (an open-source project that is a powerful and user-friendly platform for running LLMs on your local machine) if it has not already been installed locally for benchmarking an open-source Large Language Model (LLM).

   Download your choice of LLM supported by [Ollama](https://ollama.com/library) using the command:
   ```bash
   ollama run <model-name>
   ```
  
5. **Running Benchmark**:
   Run the benchmark by loading your LLMs and using the provided API.
   ```python
   from parrot import evaluate_model
   evaluate_model(model)
   ```

6. **Dataset Access**:
   The datasets, PARROT-Millionaire and PARROT-Jeopardy, are included in the repository and can be used to benchmark different LLMs.

## Usage

PARROT is a versatile framework for benchmarking LLMs in the context of trivia-based question answering. The framework can be adapted to evaluate various models, from small-scale to state-of-the-art LLMs like GPT-4, Claude-3.5-Sonnet, and more.

To assess your model’s performance using `evaluate_model`:

```python
from parrot import evaluate_model #import method from parrot

model = "llama3.2:1b" #[Required] Name of the Large Language Model that has to be benchmarked. 
reports = True #[Optional] Generate detailed reports of evaluation. 
mill_sample_size = 100 #[Optional] Sample size for parrot-millionaire set evaluation.  
jeop_sample_size = 100 #[Optional] Sample size for parrot-jeopardy set evaluation.

"""
NOTE: By default, evaluation is carried out for entire test sets (80k+ samples).
If sample size is not explicitly provided for individual sets.
"""

#run benchmark
results = evaluate_model(
                            model_name = model,
                            generate_reports = reports, 
                            millionaire_sample_size = mill_sample_size, 
                            jeopardy_sample_size = jeop_sample_size
)

print(results) #print the results.
```

Results:
```json
{
"model_name": "llama3.2:1b",
"parrot_score": 0.435,
"parrot_millionaire": {
                        "performance_at_millionaire_level_1": 0.01,
                        "performance_at_millionaire_level_2": 0.01,
                        "performance_at_millionaire_level_3": 0.01,
                        "performance_at_millionaire_level_4": 0.01,
                        "performance_at_millionaire_level_5": 0.01,
                        "performance_at_millionaire_level_6": 0.01,
                        "performance_at_millionaire_level_7": 0.01,
                        "performance_at_millionaire_level_8": 0.01,
                        "performance_at_millionaire_level_9": 0.01,
                        "performance_at_millionaire_level_10": 0.03,
                        "performance_at_millionaire_level_11": 0.04,
                        "performance_at_millionaire_level_12": 0.05,
                        "performance_at_millionaire_level_13": 0.01,
                        "performance_at_millionaire_level_14": 0.33,
                        "performance_at_millionaire_level_15": 0.33,
                        "millionaire_score": 0.23},
"parrot_jeopardy": {
                      "performance_at_jeopardy_level_1": 0.04,
                      "performance_at_jeopardy_level_2": 0.05,
                      "performance_at_jeopardy_level_3": 0.02,
                      "performance_at_jeopardy_level_4": 0.02,
                      "performance_at_jeopardy_level_5": 0.02,
                      "performance_at_jeopardy_level_6": 0.03,
                      "performance_at_jeopardy_level_7": 0.01,
                      "performance_at_jeopardy_level_8": 0.04,
                      "performance_at_jeopardy_level_9": 0.02,
                      "performance_at_jeopardy_level_10": 0.02,
                      "performance_at_jeopardy_level_11": 0.37,
                      "jeopardy_score": 0.64}
}
```

## Citing PARROT Framework

If you use the PARROT framework in your research, please cite us:

```bibtex
@article{parrot2025,
  title={PARROT: Performance Assessment of Reasoning and Responses On Trivia for LLM Benchmarking},
  author={Anonymous},
  journal={COLING 2025},
  year={2025},
  note={https://github.com/HarshaLLM/parrot}
}
```

We would appreciate it if you could include this citation in any publications or projects that leverage our framework.

## License

This project is licensed under the License cc-by-4.

## Contributions

Contributions are welcome! Please open an issue or submit a pull request for any improvements, bug fixes, or features you’d like to add.
