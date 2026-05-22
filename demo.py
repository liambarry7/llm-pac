"""
    Turn into demo.py

    Things to demo:
    - LLM pipelines (test on little data)
    - Demo best ML models - use pickle models (MLP)

"""
from feature_preprocessing import get_feature_dataset, get_llm_dataset

def demo_mlp():
    pass

if __name__ == "__main__":
    # best ml = mlp

    model_type = input("Enter model type ('llm' or 'ml')")
    if model_type == "llm":
        x_train, x_test, y_train, y_test = get_llm_dataset()

    elif model_type == "ml":
        x_train, x_test, y_train, y_test = get_feature_dataset()
