
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

## Getting Started

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/parrot-framework.git
   cd parrot-framework
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Running Benchmark**:
   Run the benchmark by loading your LLMs and using the provided API.
   ```python
   from parrot import evaluate_model
   evaluate_model(model)
   ```

4. **Dataset Access**:
   The datasets, PARROT-Millionaire and PARROT-Jeopardy, are included in the repository and can be used to benchmark different LLMs.

## Usage

PARROT is a versatile framework for benchmarking LLMs in the context of trivia-based question answering. The framework can be adapted to evaluate a wide range of models, from small-scale to state-of-the-art LLMs like GPT-4, Claude-3.5-Sonnet, and more.

To assess your model’s performance:

```python
from parrot import MillionaireMetric, JeopardyMetric

# Evaluate on the Millionaire dataset
millionaire_score = MillionaireMetric.evaluate(model)

# Evaluate on the Jeopardy dataset
jeopardy_score = JeopardyMetric.evaluate(model)

# Calculate the overall PARROTscore
parrot_score = (millionaire_score + jeopardy_score) / 2
```

## Citing PARROT Framework

If you use the PARROT framework in your research, please cite us:

```bibtex
@article{parrot2025,
  title={PARROT: Performance Assessment of Reasoning and Responses On Trivia for LLM Benchmarking},
  author={Anonymous},
  journal={COLING 2025},
  year={2025},
  note={https://github.com/yourusername/parrot-framework}
}
```

We would appreciate it if you could include this citation in any publications or projects that leverage our framework.

## License

This project is licensed under the License cc-by-4.

## Contributions

Contributions are welcome! Please open an issue or submit a pull request for any improvements, bug fixes, or features you’d like to add.
