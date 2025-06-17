import os

from ai_edge_litert.interpreter import Interpreter

from src.config import BLAZEFACE_MODEL_PATH, MOBILEFACENET_MODEL_PATH


def inspect_tflite_model(model_path):
    """
    Inspects a TensorFlow Lite model and prints its input and output details.
    """
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at '{model_path}'")
        print("Please ensure the model path is correct and the file exists.")
        return

    try:
        interpreter = Interpreter(model_path=model_path)
        interpreter.allocate_tensors()  # Allocate tensors to get detailed info

        print(f"--- Model: {os.path.basename(model_path)} ---")

        # Get input details
        input_details = interpreter.get_input_details()
        print("\n--- Input Details ---")
        for i, detail in enumerate(input_details):
            print(f"Input {i}:")
            print(f"  Name: {detail['name']}")
            print(f"  Shape: {detail['shape']}")
            print(f"  Dtype: {detail['dtype']}")
            print(f"  Quantization: {detail['quantization']}")
            print(f"  Index: {detail['index']}")
            print("-" * 20)

        # Get output details
        output_details = interpreter.get_output_details()
        print("\n--- Output Details ---")
        for i, detail in enumerate(output_details):
            print(f"Output {i}:")
            print(f"  Name: {detail['name']}")
            print(f"  Shape: {detail['shape']}")
            print(f"  Dtype: {detail['dtype']}")
            print(f"  Quantization: {detail['quantization']}")
            print(f"  Index: {detail['index']}")
            print("-" * 20)

        print("--------------------------------------------------\n")

    except Exception as e:
        print(f"An error occurred while inspecting {model_path}: {e}")
        print(
            "This might happen if the .tflite file is corrupted or not a valid TFLite model."
        )


if __name__ == "__main__":
    # Call the inspection function for each model
    print("Starting model inspection...\n")
    inspect_tflite_model(BLAZEFACE_MODEL_PATH)
    inspect_tflite_model(MOBILEFACENET_MODEL_PATH)
    print("Model inspection complete.")
