# Xencruit - Interview Confidence Meter

Xencruit is a real-time interview confidence assessment tool that analyzes facial expressions, posture, eye openness, and mobile phone usage to determine an interviewee's confidence level.

## Features
- **Face Mesh Detection**: Uses MediaPipe to track facial landmarks.
- **Posture Analysis**: Determines body posture with MediaPipe Pose estimation.
- **Blink Rate Monitoring**: Tracks blink frequency to assess alertness.
- **Mobile Phone Detection**: YOLOv8 model detects if a phone is in use.
- **Confidence Score Calculation**: Computes an overall confidence percentage based on multiple parameters.
- **Visual Indicators**: Displays real-time confidence level and warning messages.

## Installation

### Prerequisites
Ensure you have the following installed:
- Python 3.8+
- OpenCV
- MediaPipe
- NumPy
- Ultralytics YOLOv8

### Steps
```sh
# Clone the repository
git clone https://github.com/debojyoti10CC/Xencruit.git
cd Xencruit

# Install dependencies
pip install -r requirements.txt
```

## Usage
Run the following command to start Xencruit:
```sh
python xencruit.py
```
Press `q` to exit the application.

## Future Improvements
- **Speech Analysis**: Integrate voice tone analysis for deeper confidence evaluation.
- **ML Model Training**: Improve confidence scoring with AI-based behavioral analysis.
- **Web Interface**: Deploy Xencruit as a web-based application for wider accessibility.

## Contribution
Feel free to fork the repository and submit pull requests for enhancements.

## License
This project is open-source and available under the [MIT License](LICENSE).

