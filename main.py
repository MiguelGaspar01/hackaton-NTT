from pathlib import Path
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor


NOTEBOOKS = [
    "EDA.ipynb",
    "Preprocessment.ipynb",
    "FeatureSelection.ipynb",
    "Modelling.ipynb",
    "Evaluation.ipynb",
    "Deployment.ipynb",
]


def run_notebook(notebook_path: Path, timeout: int = 1800, kernel_name: str = "python3"):
    print(f"\nRunning: {notebook_path.name}")

    with notebook_path.open("r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    ep = ExecutePreprocessor(timeout=timeout, kernel_name=kernel_name)

    ep.preprocess(nb, {"metadata": {"path": str(notebook_path.parent)}})

    with notebook_path.open("w", encoding="utf-8") as f:
        nbformat.write(nb, f)

    print(f"Finished: {notebook_path.name}")


def main():
    base_dir = Path(__file__).resolve().parent

    for notebook_name in NOTEBOOKS:
        notebook_path = base_dir / notebook_name

        if not notebook_path.exists():
            raise FileNotFoundError(f"Notebook not found: {notebook_path}")

        run_notebook(notebook_path)

    print("\nAll notebooks executed successfully.")


if __name__ == "__main__":
    main()