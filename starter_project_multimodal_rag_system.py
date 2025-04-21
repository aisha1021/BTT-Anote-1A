## Stage 1: Environment Setup

!sudo apt update && sudo apt install ffmpeg

from PIL import Image
import numpy as np
import cv2
import torch
from torchvision import models, transforms
import numpy as np
import whisper #Used to convert audio files into text using Whisper AI
import sounddevice as sd #Used to collect audio from microphone
import wave #Helps convert audio into wave format
import keyboard #Helps control the keyboard
import threading

## Stage 2: Audio Processing and Conversion to Text (Whisper)
## Objective: Capture live audio input, convert it to text using Whisper, and prepare the text data for integration into the RAG pipeline.

recording = False
keyboard.hook(None, suppress=True, on_remove=None)

def record_audio(filename="recording.wav", samplerate=44100, channels=1):
    global recording
    print("Press SPACE to start recording...")
    keyboard.wait('space')
    recording = True
    print("Recording... Press SPACE to stop.")

    myrecording = []
    def callback(indata, frames, time, status):
        if recording:
            myrecording.append(indata.copy())

    with sd.InputStream(samplerate=samplerate, channels=channels, callback=callback):
        while recording:
            if keyboard.is_pressed('space'):
                recording = False
                print("Stopping recording...")
                break

    # Concatenate recorded chunks
    myrecording = np.concatenate(myrecording, axis=0)

    # Save the recording as a WAV file
    wavefile = wave.open(filename, 'wb')
    wavefile.setnchannels(channels)
    wavefile.setsampwidth(2)  # 16-bit audio
    wavefile.setframerate(samplerate)
    wavefile.writeframes(myrecording.tobytes())
    wavefile.close()
    print("Recording saved to", filename)

def transcribe_audio(audio_file): #A function to convert audio into text. Path to the WAV file to be transcribed.
  model = whisper.load_model("medium")
  result = model.transcribe(audio_file)
  text = result['text']
  return text

audio_file = record_audio()  # Record the audio and save it as liveRecording.wav
transcription = transcribe_audio(audio_file)  # Transcribe the recorded audio

print("Transcription:", transcription)

## Stage 3: Image Processing and Conversion to Text
## Objective: Capture or load images, perform object detection, and convert the detected objects into text descriptions for use in the RAG pipeline.

from PIL import Image
import numpy as np
import cv2
import torch

image = cv2.imread('image.jpg')

if image is None:
    print("Error: Could not load image. Please check the file path.")
else:
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

    results = model(image_rgb)

    labels, cords = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]

from google.colab import drive
drive.mount('/content/drive')

def get_object_description(results):
    descriptions = []
    for i in range(len(results.pandas().xyxy[0])):
        row = results.pandas().xyxy[0].iloc[i]
        label = row['name']
        confidence = row['confidence']
        descriptions.append(f"{label} with confidence {confidence:.2f}")
    return descriptions

descriptions = get_object_description(results)
print(descriptions)

## Stage 4: Video Processing and Conversion to Text
## Objective: Capture video, extract key frames, perform object detection on these frames, and convert the results into text descriptions.

def load_and_save_video(input_file, output_file="output.avi"):
    cap = cv2.VideoCapture(input_file)

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)

    cap.release()
    out.release()

def extract_key_frames(video_file, interval=30):
    cap = cv2.VideoCapture(video_file)
    frames = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % interval == 0:
            frames.append((frame_idx, frame))
        frame_idx += 1

    cap.release()
    return frames

def detect_objects_in_frames(frames):
    # Load pre-trained Faster R-CNN model
    model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()

    # Define the COCO class labels
    coco_names = [
        "__background__", "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
        "truck", "boat", "traffic light", "fire hydrant", "N/A", "stop sign", "parking meter", "bench",
        "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "N/A",
        "backpack", "umbrella", "N/A", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
        "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
        "bottle", "N/A", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "N/A", "dining table", "N/A", "toilet", "N/A", "tv", "laptop", "mouse",
        "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "N/A",
        "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
    ]

    # Transform to convert the image to the input format expected by the model
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    descriptions = []
    for idx, frame in frames:
        # Convert OpenCV frame to PIL image
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img_tensor = transform(pil_image)

        # Perform object detection
        with torch.no_grad():
            predictions = model([img_tensor])[0]

        frame_desc = f"Frame {idx}: "
        for i in range(len(predictions["boxes"])):
            confidence = predictions["scores"][i].item()
            if confidence > 0.5:  # Filter detections with low confidence
                label = coco_names[predictions["labels"][i].item()]
                frame_desc += f"Detected {label} (confidence: {confidence:.2f}), "

        descriptions.append(frame_desc.rstrip(', '))

    return descriptions

input_video = "input_video.mp4"  # Replace with your uploaded video file name
output_video = "output.avi"
load_and_save_video(input_file=input_video, output_file=output_video)

# Extract key frames from the video
frames = extract_key_frames(output_video, interval=30)

# Perform object detection and generate text descriptions
descriptions = detect_objects_in_frames(frames)

# Output the text descriptions
for desc in descriptions:
    print(desc)

## Stage 5: Text Data Integration and Retrieval
## Objective: Retrieve and process text data from all modalities (audio, image, video) and integrate it for use in the RAG pipeline.

# import re
# from transformers import (
#     DPRQuestionEncoderTokenizer,
#     DPRQuestionEncoder,
#     DPRContextEncoder,
#     DPRContextEncoderTokenizer
# )
# import torch

# # Initialize DPR models and tokenizers
# question_encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base") # https://huggingface.co/facebook/dpr-question_encoder-single-nq-base
# question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")

# context_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base") # https://huggingface.co/facebook/dpr-ctx_encoder-single-nq-base
# context_tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")

# # Function to encode query
# def encode_query(query):
#     inputs = question_tokenizer(query, return_tensors="pt")
#     embeddings = question_encoder(**inputs).pooler_output
#     return embeddings

# # Function to encode context
# def encode_context(context):
#     inputs = context_tokenizer(context, return_tensors="pt")
#     embeddings = context_encoder(**inputs).pooler_output
#     return embeddings.squeeze()

# # Function to retrieve the most relevant context
# def retrieve_text(query, contexts):
#     query_embedding = encode_query(query)
#     context_embeddings = torch.stack([encode_context(ctx) for ctx in contexts])
#     similarities = torch.matmul(query_embedding, context_embeddings.T)
#     print(similarities)
#     best_match_idx = torch.argmax(similarities).item()
#     return contexts[best_match_idx]

# # Function to integrate text data from different modalities
# def integrate_text_data(*args):
#     combined_text = " ".join(args)
#     return combined_text

# # Function to handle the query and retrieve relevant contexts
# def handle_query(query, audio_text, image_text, video_text):
#     # Initialize a list to store relevant contexts
#     relevant_texts = []

#     # Check for the presence of each modality in the query
#     if "audio" in query.lower():
#         relevant_texts.append(audio_text)
#     if "image" in query.lower():
#         relevant_texts.append(image_text)
#     if "video" in query.lower():
#         relevant_texts.append(video_text)

#     # If no specific modality is mentioned, include all contexts
#     if not relevant_texts:
#         relevant_texts = [audio_text, image_text, video_text]

#     # Retrieve the most relevant text for each context
#     combined_texts = [retrieve_text(query, [text]) for text in relevant_texts]

#     # Combine all relevant texts into a single response
#     combined_text = " ".join(combined_texts)

#     return combined_text

# # Example text data from different modalities --> This will be replace with the appropriate variables when everyone is finished!
# audio_text = "The audio describes the scene as a bustling street with cars honking and people talking."
# image_text = "Detected objects: street sign (confidence: 0.92), car (confidence: 0.97), pedestrian (confidence: 0.90)."
# video_text = ("Frame 1: Detected car (confidence: 0.97), pedestrian (confidence: 0.90), "
#               "crosswalk (confidence: 0.85). Frame 2: Detected car (confidence: 0.95), "
#               "moving across crosswalk. Frame 3: Detected street sign (confidence: 0.92), "
#               "pedestrian (confidence: 0.90), standing at the corner.")

# # Integrate all context data into a single combined context
# combined_text_context = integrate_text_data(audio_text, image_text, video_text)
# print(f"\nCombined Text Context: {combined_text_context}\n")

# # User input for the query
# user_query = input("Enter your query: ")

# # Handle the query and retrieve the most relevant context
# response = handle_query(user_query, audio_text, image_text, video_text)
# print(f"\nResponse: {response}")

import torch
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer, DPRContextEncoder, DPRContextEncoderTokenizer, pipeline

# Step 1: Text Retrieval using DPR (optional for context, but not used for final output in this version)
def retrieve_text(query, contexts):
    # Load DPR model and tokenizer
    question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
    question_encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")

    context_tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
    context_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")

    # Encode the query
    query_inputs = question_tokenizer(query, return_tensors='pt')
    query_embedding = question_encoder(**query_inputs).pooler_output

    # Encode the contexts
    context_embeddings = []
    for context in contexts:
        context_inputs = context_tokenizer(context, return_tensors='pt')
        context_embedding = context_encoder(**context_inputs).pooler_output
        context_embeddings.append(context_embedding)

    # Compute dot products between the query and context embeddings
    scores = torch.cat(context_embeddings, dim=0) @ query_embedding.T

    relevant_text = []
    # Find the most relevant context
    for i in range(len(scores)):
      if scores[i] > 0.5:
        relevant_text.append(contexts[i])
    #print(scores)
    return relevant_text

    ## Print out the corresponding Relevance Score for each sentence
    # relevant_text = []
    # # Find the most relevant context
    # for i in range(len(scores)):
    #     relevance_score = scores[i].item()
    #     print(f"Context: '{contexts[i]}'\nRelevance Score: {relevance_score:.4f}\n")
    #     if relevance_score > 0.62:
    #         relevant_text.append((contexts[i], relevance_score))

    # return relevant_text

# Step 2: Integration and Sentence Generation
def integrate_and_generate_sentences(query, audio_text, image_text, video_texts):
    # Combine the information into coherent sentences
    combined_description = ""
    contexts = ["In the audio: " + audio_text, "In the image: " + image_text] + ["In the video: " + f for f in video_texts]
    #print(contexts)

    # Retrieve the most relevant text based on the query
    relevant_texts = retrieve_text(query, contexts)
    print(relevant_texts)

    # Generate sentence for video text descriptions
    for relevant_text in relevant_texts:
      if "video" in relevant_text:
          video_sentence = "The video shows "
          items = []
          for text in relevant_text:
              #print(text)
              if "Frame" in text:
                  # Process each detected item and check confidence
                  deframe = text.index(": ")
                  #print(deframe)
                  detections = text[deframe+2:].split(", ")
                  #print(detections)
                  for detection in detections:
                      if " (confidence: " in detection:
                          item, confidence = detection.rsplit(" (confidence: ", 1)
                          if 'Detected' in item:
                            item = item[9:]
                          confidence = float(confidence.rstrip(")"))
                          #print(item, confidence)
                          if confidence > 0.7:
                              items.append(item)
                      else:
                        if 'Detected' in detection:
                            detection = detection[9:]
                        items.append(detection)
          #print('video items', items)
          if items:
              video_sentence += " and ".join(items) + ". "
              combined_description += video_sentence.strip()

      # Generate sentence for image text description
      if "image" in relevant_text:
          text = relevant_text
          deframe = text.index(": ")
          #print(deframe)
          detections = text[deframe+2:].split(", ")
          items = []
          for detection in detections:
              if " (confidence: " in detection:
                  item, confidence = detection.rsplit(" (confidence: ", 1)
                  if 'Detected objects' in item:
                      item = item[18:]
                  confidence = float(confidence.rstrip(")"))
                  if confidence > 0.7:
                      items.append(item)
          #print('image items', items)
          if items:
              image_sentence = f"The image shows " + " and ".join(items) + "."
              combined_description += " " + image_sentence.strip()

      # Generate sentence for audio text description
      if "audio" in relevant_text:
          audio_sentence = f"The audio describes the scene as {audio_text.split(' as ')[1].strip()}"
          combined_description += " " + audio_sentence.strip()

    return combined_description.strip()



# Test Case: Combine everything
if True:
    # Example input data
    query = "What objects are present in the image?"
    audio_text = "The audio describes the scene as a quiet room."
    image_text = "Detected objects: laptop (confidence: 0.95), coffee cup (confidence: 0.88)"
    video_texts = [
        "Frame 1: Detected person (confidence: 0.92), chair (confidence: 0.85), desk (confidence: 0.88)",
        "Frame 2: Detected person (confidence: 0.94), sitting on chair"
    ]

    # Generate the integrated text context with full sentences
    result = integrate_and_generate_sentences(query, audio_text, image_text, video_texts)

    # Output the final generated description
    print(result)



## Stage 6
## Basic RAG Pipeline Implementation
## Objective: Implement a simple RAG pipeline that queries the combined multimodal text data
## and generates contextually relevant responses.

import torch
import datasets
import faiss
# **Pretrained RAG model
from transformers import RagTokenizer, RagRetriever, RagTokenForGeneration

# Load tokenizer
tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-nq")

# Load retriever and set the index name to 'default' or your custom index
retriever = RagRetriever.from_pretrained("facebook/rag-token-nq", index_name="default")

# Load the RAG model with token-based input
rag_model = RagTokenForGeneration.from_pretrained("facebook/rag-token-nq", retriever=retriever)

# ** Query Processing

query = "What are the benefits of using multimodal data in machine learning?"

#Tokenize and Encode the Query:
input_ids = tokenizer(query, return_tensors="pt").input_ids

# **Retrieve Relevant Documents

# Retrieve documents related to the input query
retrieved_docs = retriever(input_ids.numpy(), return_tensors="pt")

# **Response generation

# Generate a response from the RAG model
outputs = rag_model.generate(input_ids, context_input_ids=retrieved_docs["context_input_ids"])

# Decode the generated response to text
response = tokenizer.batch_decode(outputs, skip_special_tokens=True)
print(response[0])
