import joblib
from tensorflow.keras.models import load_model

def load_ml_model(model_path):
    """
    Load a machine learning model from a file.

    Args:
        model_path (str): The path to the model file.

    Returns:
        The loaded model object.
    """
    if model_path.endswith('.h5'):
        # Load Keras model from .h5 file
        model = load_model(model_path)
    elif model_path.endswith('.pkl'):
        # Load Scikit-learn model from .pkl file
        model = joblib.load(model_path)
    else:
        raise ValueError("Invalid model file type. Must be either .h5 or .pkl.")

    return model
