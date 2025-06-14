import requests
import base64
import os

# Specify the path to your audio file
audio_file_path = "harvard.wav"

# Read the audio file in binary mode and encode it to base64
with open(audio_file_path, "rb") as audio_file:
    audio_base64 = base64.b64encode(audio_file.read()).decode('utf-8')

# Prepare the payload for the POST request
payload = {
    "instances": [
        {"b64": audio_base64}
    ]
}

# Send the POST request to your ASR server
url = "http://127.0.0.1:5001/asr"  # Assuming your server is running locally on port 5001
response = requests.post(url, json=payload)

# Print the response
if response.status_code == 200:
    print("Transcription result:", response.json())
else:
    print(f"Failed to get transcription. Status code: {response.status_code}")
