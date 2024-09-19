import gradio as gr
import os
import torch
from model import create_effnetb2_feature_extractor
from timeit import default_timer as timer
from typing import Tuple, Dict

#1. Setup class names
with open("class_names.txt", "r") as f:
    class_names = [food101_class_names.strip() for food101_class_names in f.readlines()]

#2. Model and transforms preparation
effnetb2, effnetb2_transforms = create_effnetb2_feature_extractor(num_classes=101)
#Load save weights
effnetb2.load_state_dict(torch.load(f="09_pretrained_effnetb2_food101_model.pth",
                                    map_location=torch.device("cpu"))) # load the model to the CPU

#3.Predict function
def predict(img) -> Tuple[Dict, float]:
    """Transforms and performs a prediction on img and returns prediction and time taken.
    """
    # Start the timer
    start_time = timer()

    # Transform the target image and add a batch dimension
    img = effnetb2_transforms(img).unsqueeze(0)

    # Put model into evaluation mode and turn on inference mode
    effnetb2.eval()
    with torch.inference_mode():
        # Pass the transformed image through the model and turn the prediction logits into prediction probabilities
        pred_probs = torch.softmax(effnetb2(img), dim=1)

    # Create a prediction label and prediction probability dictionary for each prediction class (this is the required format for Gradio's output parameter)
    pred_labels_and_probs = {class_names[i]: float(pred_probs[0][i]) for i in range(len(class_names))}

    # Calculate the prediction time
    pred_time = round(timer() - start_time, 5)

    # Return the prediction dictionary and prediction time
    return pred_labels_and_probs, pred_time

#4. Gradio Interface
#Building a gradio Interface
#Use 'gr.interface'
#Create the interface
title = "FoodVision Big"
description = "An EfficientNetB2 feature extractor computer vision model to classify food images into 101 classes of food from the Food101 dataset."
article = "--"
#Create example list
example_list = [["examples/"+ example] for example in os.listdir("examples")]
demo = gr.Interface(fn=predict,
                    inputs = gr.Image(type="pil"),
                    outputs = [gr.Label(num_top_classes=5, label='predictions'),
                               gr.Number(label="Prediction Time (s)")],
                    examples = example_list,
                    title=title,
                    description = description)
demo.launch(debug=False)
