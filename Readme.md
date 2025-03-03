# Mediapipe Real-Time Object Detection

This repository contains a real-time object detection project using [Mediapipe Tasks](https://developers.google.com/mediapipe/solutions) and OpenCV. It includes:

- **Models** (TFLite files) for object detection and classification.
- **Labels** for mapping model outputs to human-readable categories.
- **Utility scripts** for preprocessing, annotation, and other helper functions.
- **Example code** demonstrating both synchronous and asynchronous detection modes.

---


### Notable Files/Folders

1. **experiments_notebook.ipynb**  
   - A Jupyter Notebook with various experimental codes for object detection and classification, including examples of synchronous vs. asynchronous detection and classifier integration.

2. **mediapipe_real_time_object_detection.py**  
   - The main script to run real-time object detection from your webcam.  
   - Can be configured for synchronous (`IMAGE`) or asynchronous (`LIVE_STREAM`) modes.  
   - Press **q** to quit or **s** to save a snapshot of the annotated frame.

3. **utils/**  
   - Contains utility functions such as:
     - **convert_to_mp_image** for converting frames to `mp.Image`
     - **print_detection** for drawing bounding boxes and labels on frames
     - **setup_camera** for initializing the webcam
     - **detection_callback** for handling asynchronous results
   - These helpers keep the main script cleaner and more modular.

4. **models/**  
   - Contains TFLite models (e.g., `efficientdet_lite0_float16.tflite`, `efficientnet_lite0_float32_classifier.tflite`, etc.) used for detection and/or classification.

5. **labels/**  
   - Contains label files (as `.txt` files) mapping model output indices to category names for classifiers.

6. **results/**  
   - A directory where snapshots or other output files can be saved (optional).  
   - You can configure your code to store snapshots or logs here.

---

## How to Use

1. **Install Requirements**  
   
     ```bash
   pip install -r requirements.txt
     ```
   - Add any other required libraries (e.g., numpy, jupyter, etc.) as needed.

2. **Download models** 
   - You can find them here : [Mediapipe Object Detection Models](https://ai.google.dev/edge/mediapipe/solutions/vision/object_detector/index?hl=fr#models), or [Mediapipe Classification Models](https://ai.google.dev/edge/mediapipe/solutions/vision/image_classifier/index?hl=fr#models).

   For instance you can download the default model :
   ```bash 
   wget -P models/ https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float16/latest/efficientdet_lite0.tflite
   ``` 
2. **Run the Main Script**  
   - From the project root, run:
     ```bash
     python mediapipe_real_time_object_detection.py --model=models/efficientdet_lite0_float16.tflite --mode=image
     ```
   - Replace `--model` with your desired TFLite model and `--mode` with either `image` (synchronous) or `live_stream` (asynchronous).  
   - **Note:** The best performance is often achieved with the default parameters of the object detector. Adjusting thresholds or resolution may degrade results.

3. **Using the Experiments Notebook**  
   - Open `experiments/experiments_notebook.ipynb` in Jupyter or another compatible environment to see example codes and experimentation logs.  
   - This notebook showcases how to integrate classification, manipulate detection callbacks, etc.

4. **Capture Snapshots**  
   - While the script is running, press **s** to save the current annotated frame (which includes bounding boxes and labels) as `snapshot.jpg`.  
   - Press **q** to exit the program.

---

## Notes & Customization

- **Model & Labels:**  
  If you use a different model, ensure it matches the correct label file in `labels/` (if required).  

- **Resolution & Performance:**  
  Adjust camera resolution in `utils/setup_camera` or in the main script to balance speed and detection accuracy.  

- **Classifier Integration:**  
  If you want to perform classification on detected ROIs, refer to the examples in the `experiments_notebook.ipynb`.

