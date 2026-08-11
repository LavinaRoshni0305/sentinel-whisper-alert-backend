🚨 Zero-Touch SOS: Multimodal Emergency Response System

📌 Overview

Zero-Touch SOS is a multimodal emergency response system designed to trigger an SOS alert without requiring the user to physically interact with their device.

The system combines voice and motion-based detection to identify distress situations and automatically initiate an emergency response. It is designed to reduce response time during situations where manually accessing a phone may not be possible.

🎯 Objectives

- Enable emergency alerts without physical interaction.
- Detect potential distress situations using multiple input modalities.
- Reduce false alarms through multimodal signal verification.
- Automatically share the user's location with emergency contacts.
- Provide a fast and reliable emergency communication mechanism.

✨ Key Features

- 🎙️ Voice-based SOS detection using OpenAI Whisper.
- 📱 Motion detection using the device's accelerometer and gyroscope.
- 📍 Real-time location detection using Google Fused Location Provider.
- 📩 Emergency SMS alerts using Android SMS services.
- 🌐 Backend API built using Flask.
- 🔄 Multimodal detection and alert processing.
- 📱 Android application with an integrated WebView interface.
- 🚨 Automatic emergency alert triggering.

🏗️ System Architecture

                    ┌─────────────────────┐
                    │    Android Device   │
                    └──────────┬──────────┘
                               │
              ┌────────────────┴────────────────┐
              │                                 │
       Voice Detection                    Motion Detection
        (Whisper)                       (Accelerometer/
                                         Gyroscope)
              │                                 │
              └────────────────┬────────────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Multimodal SOS    │
                    │     Processing      │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │    Flask Backend    │
                    └──────────┬──────────┘
                               │
              ┌────────────────┴────────────────┐
              │                                 │
              ▼                                 ▼
       Location Retrieval                Emergency Alert
              │                                 │
              └────────────────┬────────────────┘
                               ▼
                    Emergency Contacts

🛠️ Technologies Used

Android

- Java
- Android Studio
- WebView
- AudioRecord API
- Android Sensor APIs
- SMSManager
- Google Fused Location Provider
- OkHttp

Machine Learning

- Python
- OpenAI Whisper
- PyTorch
- Torchaudio

Backend

- Flask
- Flask-CORS
- REST APIs
- Python

Development & Deployment

- Git
- GitHub
- Render

📂 Project Structure

Zero-Touch-SOS/
│
├── android-app/
│   └── Android Studio project files
│
├── backend/
│   ├── app.py
│   ├── requirements.txt
│   └── model/backend files
│
├── apk/
│   └── ZeroTouchSOS.apk
│
└── README.md

🚀 Getting Started

1. Clone the Repository

git clone YOUR_GITHUB_REPOSITORY_URL
cd Zero-Touch-SOS

2. Backend Setup

Create and activate a Python virtual environment:

python -m venv venv

Activate it:

Windows

venv\Scripts\activate

macOS/Linux

source venv/bin/activate

Install the required dependencies:

pip install -r requirements.txt

Start the Flask server:

python app.py

3. Android Application

Open the "android-app" folder in Android Studio.

Allow Gradle to sync and build the project.

Connect an Android device or use an Android emulator and run the application.

📱 APK

A pre-built APK is included in the repository for testing.

APK: "apk/ZeroTouchSOS.apk"

Download the APK, install it on an Android device, and grant the required permissions.

🔐 Required Permissions

The application may require permissions for:

- 🎤 Microphone
- 📍 Location
- 📩 SMS
- 📱 Sensors
- 🔔 Notifications

These permissions are required for the emergency detection and alert functionality.

📊 Performance

The implemented detection modules achieved the following results during testing:

Module| Accuracy
Motion Detection| 96%
Voice Detection| 90%

Performance may vary depending on device hardware, environmental conditions, background noise, and testing conditions.

🔄 Emergency Alert Workflow

User encounters an emergency
            ↓
Voice / Motion signal detected
            ↓
Signal processed
            ↓
SOS condition identified
            ↓
Current location retrieved
            ↓
Emergency alert generated
            ↓
Alert sent to emergency contacts

🔮 Future Enhancements

The next phase of the system can extend multimodal detection with:

- 👁️ Blink-based SOS detection using MediaPipe Face Mesh.
- ✋ Hand gesture-based emergency signals using MediaPipe Hands.
- 🔀 Improved multimodal fusion.
- 🧠 Advanced false-positive reduction.
- 📲 Push notifications and additional communication channels.
- ⚡ Improved real-time processing and battery efficiency.

👥 Project

Project: Zero-Touch SOS: Multimodal Emergency Response System

Domain: Android Development | Machine Learning | Emergency Response | Multimodal AI

📄 License

This project is developed for academic and demonstration purposes.
