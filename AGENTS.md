# AGENTS.md

## Python environment

- Use the Conda environment `Federated-Learning-PyTorch` for Python commands in this repository.
- Prefer `conda run -n Federated-Learning-PyTorch <command>` over bare `python`, `python3`, `pytest`, or `pip`.
- This is more reliable than relying on `conda activate` because Codex shell commands may run in separate shell sessions.
- Examples:
  - `conda run -n Federated-Learning-PyTorch python src/main.py`
  - `conda run -n Federated-Learning-PyTorch pytest`
  - `conda run -n Federated-Learning-PyTorch pip install -r requirements.txt`

## Change summaries

- Whenever modifying code in this project, include a copy-ready Conventional Commits style message, for example: `fix(scope): concise description`.
- Pick the commit type and scope based on the actual change, following the Conventional Commits convention.
- Include a suggested branch name for the change using the project style, for example: `2026-04-22-tmp-improve-fed-cifar-acc`.
