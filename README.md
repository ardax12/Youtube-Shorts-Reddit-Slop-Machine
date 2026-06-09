Automatically creates Youtube Shorts slop videos using:

- Reddit posts
- Microsoft Edge TTS voices
- Whisper word-level subtitles
- Gameplay footage
- Background music
- Automatic YouTube uploads

---

## Features

- Fetches top posts from multiple subreddits
- Converts stories to realistic AI narration
- Generates word-by-word subtitles
- Combines narration with gameplay footage
- Adds background music
- Uploads finished videos directly to YouTube
- Prevents reposting already processed stories

---

## Requirements

### Python

Python 3.10+ is recommended.

### FFmpeg

#### Windows

1. Download FFmpeg from https://ffmpeg.org/download.html
2. Extract the archive.
3. Add the `bin` folder to your system PATH.

Verify installation:

```bash
ffmpeg -version
```

---

## Installation

### 1. Clone or download the project

```bash
git clone https://github.com/yourusername/reddit-video-generator.git

cd reddit-video-generator
```

### 2. Create a virtual environment

#### Windows

```bash
python -m venv venv

venv\Scripts\activate
```

#### Linux / macOS

```bash
python3 -m venv venv

source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```
---

## Required Files

Your project folder should look like:

```text
project/
│
├── script.py
├── requirements.txt
├── gameplay.mp4
├── client_secret.json
├── music/
│   ├── track1.mp3
│   ├── track2.mp3
│   └── track3.mp3
│
├── output/
└── processed_posts.json
```

### gameplay.mp4

A gameplay (or anything you want) video used as background footage.

### music/

Place one or more `.mp3` files inside the music folder.

Background music is chosen randomly for each video.

---

## YouTube Setup

### 1. Create a Google Cloud Project

Go to:

https://console.cloud.google.com/

### 2. Enable YouTube Data API v3

Enable:

- YouTube Data API v3

### 3. Create OAuth Credentials

Create:

- OAuth Client ID
- Desktop Application

Download the credentials file and rename it:

```text
client_secret.json
```

Place it in the project root directory.

### 4. First Login

On the first run:

```bash
python script.py
```

A browser window will open asking you to authorize your YouTube account.

After approval:

```text
token.pickle
```

will be generated automatically and reused in future runs.

---

## Whisper Model Download

The first launch downloads the Whisper model:

```python
stable_whisper.load_model("medium")
```

The download can take several minutes depending on your internet speed.

---

## Running

Simply execute:

```bash
python script.py
```

The script will:

1. Fetch Reddit stories
2. Generate narration
3. Generate subtitles
4. Create a video
5. Upload to YouTube
6. Save processed post IDs

---

## Notes

### GPU Acceleration

The script automatically detects CUDA:

```python
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
```

NVIDIA GPUs are highly recommended.

### CPU Usage

The script will still work on CPU but:

- Whisper transcription will be much slower
- Video rendering will take significantly longer

---

## Troubleshooting

### FFmpeg not found

```text
ffmpeg is not recognized as an internal or external command
```

Solution:

- Install FFmpeg
- Add FFmpeg's `bin` directory to PATH

### CUDA not detected

Check:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

If it returns:

```text
False
```

Install the CUDA-enabled version of PyTorch.

### YouTube Authentication Error

Delete:

```text
token.pickle
```

Then run:

```bash
python script.py
```

and authorize again.

---
