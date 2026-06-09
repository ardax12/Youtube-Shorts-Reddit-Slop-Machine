import json
import torch
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from tqdm import tqdm
import pickle

# Check if CUDA is available
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on: {DEVICE} (GPU: {torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else "Running on CPU")
import os
import time
import random
import traceback
import asyncio
import numpy as np
from edge_tts import Communicate
from moviepy import (
    VideoFileClip,
    AudioFileClip,
    CompositeAudioClip,
)
from moviepy.audio.AudioClip import concatenate_audioclips
import stable_whisper
import requests
import subprocess

# ================= CONFIG =================

MAX_VIDEO_DURATION = 180
GAMEPLAY_VIDEO = "gameplay.mp4"
MUSIC_FOLDER = "music"
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
PROCESSED_FILE = "processed_posts.json"

SUBREDDITS = [
    "TrueOffMyChest",
    "tifu",
    "pettyrevenge",
    "MaliciousCompliance",
    "AmItheAsshole",
    "confession",
    "AITAH",
    "ProRevenge",
    "TalesFromRetai"
]

MALE_VOICES = [
    "en-AU-WilliamMultilingualNeural",
    "en-US-BrianMultilingualNeural",
    "en-US-AndrewMultilingualNeural"
]

FEMALE_VOICES = [
    "en-US-AriaNeural",
    "en-US-EmmaMultilingualNeural",
    "en-US-AvaMultilingualNeural"
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (RedditJSONBot/1.0)"
}

PROFANITY = [
    ""
]

WHISPER_MODEL = stable_whisper.load_model("medium", device=DEVICE)

SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]

def get_youtube_service():
    creds = None

    if os.path.exists("token.pickle"):
        with open("token.pickle", "rb") as token:
            creds = pickle.load(token)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                "client_secret.json", SCOPES
            )
            creds = flow.run_local_server(port=0)

        with open("token.pickle", "wb") as token:
            pickle.dump(creds, token)

    return build("youtube", "v3", credentials=creds)


def load_processed_posts():
    if os.path.exists(PROCESSED_FILE):
        try:
            with open(PROCESSED_FILE, "r", encoding="utf-8") as f:
                return set(json.load(f))
        except Exception:
            return set()
    return set()


def save_processed_posts(processed_posts):
    with open(PROCESSED_FILE, "w", encoding="utf-8") as f:
        json.dump(list(processed_posts), f, indent=2)


def censor_word(word):
    if len(word) <= 4:
        return word[0] + "*"
    return word[0] + "*" * (len(word) - 2) + word[-1]


def censor_text(text):
    for w in PROFANITY:
        text = text.replace(w, censor_word(w))
        text = text.replace(w.capitalize(), censor_word(w.capitalize()))
    return text

def upload_to_youtube(video_path, title, description, tags=None):
    print("[YOUTUBE] Uploading video...")

    youtube = get_youtube_service()

    body = {
        "snippet": {
            "title": title[:95],
            "description": description[:100],
            "tags": tags or ["reddit", "storytime", "shorts"],
            "categoryId": "24"  # Entertainment
        },
        "status": {
            "privacyStatus": "public",
            "selfDeclaredMadeForKids": False
        }
    }

    media = MediaFileUpload(
        video_path,
        chunksize=-1,
        resumable=True,
        mimetype="video/mp4"
    )

    request = youtube.videos().insert(
        part="snippet,status",
        body=body,
        media_body=media
    )

    response = None
    pbar = tqdm(total=100)

    while response is None:
        status, response = request.next_chunk()
        if status:
            pbar.update(int(status.progress() * 100) - pbar.n)

    pbar.close()
    print(f"[YOUTUBE] Uploaded: https://youtu.be/{response['id']}")



# ================= REDDIT JSON =================

def get_posts(subreddit, limit=5):
    print(f"\n[REDDIT] Fetching r/{subreddit}")
    url = f"https://www.reddit.com/r/{subreddit}/top.json"
    params = {"t": "week", "limit": limit}

    try:
        r = requests.get(url, headers=HEADERS, params=params, timeout=10)
        time.sleep(2)  # Rate limiting
    except Exception as e:
        print(f"[REDDIT] Request failed: {e}")
        return []

    if r.status_code != 200:
        print(f"[REDDIT] Non-200 response: {r.status_code}")
        return []

    posts = []
    try:
        data = r.json()
        for c in data["data"]["children"]:
            p = c["data"]
            if not p.get("selftext") or p.get("over_18") or len(p["selftext"]) < 300:
                continue

            posts.append({
                "id": p["id"],
                "title": p["title"],
                "text": p["selftext"]
            })
    except Exception as e:
        print(f"[REDDIT] JSON parsing error: {e}")
        return []

    return posts


# ================= TTS FUNCTION =================

async def generate_tts_edge(text, voice_name, output_path):
    print(f"[TTS] Generating audio with {voice_name}")

    try:
        communicate = Communicate(text, voice_name)
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                with open(output_path, "ab") as f:
                    f.write(chunk["data"])
        return True
    except Exception as e:
        print(f"[TTS] Error: {e}")
        return False


def generate_tts(text, voice_name, output_path):
    return asyncio.run(generate_tts_edge(text, voice_name, output_path))


# ================= WHISPER TRANSCRIPTION =================

def transcribe_with_whisper(audio_path):
    print(f"[WHISPER] Transcribing audio: {audio_path}")

    try:
        result = WHISPER_MODEL.transcribe(audio_path)

        # Extract word timings from stable-ts result
        word_timings = []

        for segment in result.segments:
            for word in segment.words:
                word_timings.append({
                    "word": word.word.strip(),
                    "start": word.start,
                    "end": word.end
                })

        print(f"[WHISPER] Extracted {len(word_timings)} words")
        return word_timings

    except Exception as e:
        print(f"[WHISPER] Error: {e}")
        traceback.print_exc()
        return []


# ================= SRT GENERATION =================

def format_srt_time(seconds):
    """Convert seconds to SRT timestamp format (HH:MM:SS,mmm)"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def create_srt_file(word_timings, srt_path):
    """Generate an SRT subtitle file from word timings"""
    print(f"[SRT] Creating subtitle file: {srt_path}")

    try:
        with open(srt_path, "w", encoding="utf-8") as f:
            for idx, word_data in enumerate(word_timings, start=1):
                start_time = format_srt_time(word_data["start"])
                end_time = format_srt_time(word_data["end"])
                word = word_data["word"]

                # Write SRT entry
                f.write(f"{idx}\n")
                f.write(f"{start_time} --> {end_time}\n")
                f.write(f"{word}\n")
                f.write("\n")

        print(f"[SRT] Successfully created {len(word_timings)} subtitle entries")
        return True

    except Exception as e:
        print(f"[SRT] Error creating file: {e}")
        traceback.print_exc()
        return False


# ================= VIDEO LOGIC =================

def get_random_music():
    if not os.path.exists(MUSIC_FOLDER):
        return None
    tracks = [
        os.path.join(MUSIC_FOLDER, f)
        for f in os.listdir(MUSIC_FOLDER)
        if f.lower().endswith(".mp3")
    ]
    return random.choice(tracks) if tracks else None


def crop_to_vertical(video):
    w, h = video.size
    crop_size = min(w, h)
    x1 = (w - crop_size) // 2
    y1 = (h - crop_size) // 2
    video = video.cropped(x1=x1, y1=y1, x2=x1 + crop_size, y2=y1 + crop_size)
    video = video.resized((1080, 1920))
    return video


def create_video(audio_path, word_timings, out_path):
    print(f"[VIDEO] Creating video: {out_path}")

    voice = None
    video = None
    music = None
    final_audio = None

    temp_video_path = out_path.replace(".mp4", "_temp.mp4")
    srt_path = out_path.replace(".mp4", ".srt")

    try:
        # Load voice audio
        voice = AudioFileClip(audio_path)
        voice_duration = voice.duration
        print(f"[VIDEO] Voice duration: {voice_duration:.2f}s")
        if voice_duration > MAX_VIDEO_DURATION:
            print(f"[SKIP] Voice is too long ({voice_duration:.1f}s > {MAX_VIDEO_DURATION}s)")
            voice.close()
            return False

        # Load and prepare video
        video = VideoFileClip(GAMEPLAY_VIDEO)
        max_start = max(0, video.duration - voice_duration - 60)
        random_start = random.uniform(0, max_start) if max_start > 0 else 0
        video = video.subclipped(random_start, random_start + voice_duration)

        # Prepare audio tracks
        audio_tracks = [voice]
        music_path = get_random_music()

        if music_path:
            print(f"[VIDEO] Adding background music")
            music = AudioFileClip(music_path)
            if music.duration < voice_duration:
                n_loops = int(np.ceil(voice_duration / music.duration))
                music = concatenate_audioclips([music] * n_loops)
            music = music.with_volume_scaled(0.08)
            audio_tracks.insert(0, music)

        final_audio = CompositeAudioClip(audio_tracks).with_duration(voice_duration)

        # Set audio to video
        video_with_audio = video.with_audio(final_audio)

        # Render video WITHOUT subtitles first
        print("[VIDEO] Rendering base video...")
        video_with_audio.write_videofile(
            temp_video_path,
            fps=30,
            codec="h264_nvenc",
            audio_codec="aac",
            threads=4,
            ffmpeg_params=[
                "-preset", "p5",
                "-rc", "constqp",
                "-qp", "19",
                "-profile:v", "high",
                "-pix_fmt", "yuv420p",
            ]
        )
        print("[VIDEO] Base video rendered!")

        # Close MoviePy resources before FFmpeg
        video_with_audio.close()
        if final_audio:
            final_audio.close()
        if video:
            video.close()
        if voice:
            voice.close()
        if music:
            music.close()

        # Generate SRT file
        if word_timings:
            if not create_srt_file(word_timings, srt_path):
                print("[ERROR] Failed to create SRT file")
                return False

            # Burn subtitles using FFmpeg
            print("[FFMPEG] Burning subtitles into video...")

            # Escape the SRT path for Windows
            srt_path_escaped = srt_path.replace("\\", "/").replace(":", "\\:")

            # Style parameters matching the original MoviePy specifications:
            # - FontSize=140 (matching font_size=140)
            # - PrimaryColour=&HFFFFFF (white, matching color="white")
            # - OutlineColour=&H000000 (black, matching stroke_color="black")
            # - Outline=20 (matching stroke_width=20)
            # - Alignment=5 (center, matching position=("center", "center"))
            # - MarginV=0 (center vertically)
            ffmpeg_cmd = [
                "ffmpeg",
                "-i", temp_video_path,
                "-vf",
                f"subtitles='{srt_path_escaped}':force_style='FontName=Montserrat Extra Bold,FontSize=28,PrimaryColour=&HFFFFFF,OutlineColour=&H000000,Outline=3,Alignment=10,MarginV=0'",
                "-c:v", "h264_nvenc",
                "-preset", "p5",
                "-rc", "constqp",
                "-qp", "19",
                "-c:a", "copy",
                "-movflags", "+faststart",
                out_path
            ]

            result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)

            if result.returncode != 0:
                print(f"[FFMPEG] Error: {result.stderr}")
                return False

            print("[FFMPEG] Subtitles burned successfully!")

            # Clean up temporary video
            if os.path.exists(temp_video_path):
                os.remove(temp_video_path)
                print("[CLEANUP] Removed temporary video")

            # Optionally keep or remove SRT file
            # os.remove(srt_path)  # Uncomment to delete SRT after burning

        else:
            print("[VIDEO] No subtitles - word timings missing")
            # Just rename temp to final
            if os.path.exists(temp_video_path):
                os.rename(temp_video_path, out_path)

        print("[VIDEO] Video creation complete!")
        return True

    except Exception as e:
        print(f"[VIDEO] Error: {e}")
        traceback.print_exc()

        # Cleanup on error
        if os.path.exists(temp_video_path):
            try:
                os.remove(temp_video_path)
            except:
                pass

        return False

    finally:
        # Cleanup resources
        if final_audio:
            final_audio.close()
        if video:
            video.close()
        if voice:
            voice.close()
        if music:
            music.close()


# ================= MAIN =================

def main():
    processed_posts = load_processed_posts()
    print(f"[INFO] Loaded {len(processed_posts)} previously processed posts")

    if not os.path.exists(GAMEPLAY_VIDEO):
        print(f"[ERROR] {GAMEPLAY_VIDEO} not found!")
        return

    male_turn = True

    for sub in SUBREDDITS:
        posts = get_posts(sub)

        if not posts:
            print(f"[INFO] No suitable posts found in r/{sub}")
            continue

        for post in posts:
            if post["id"] in processed_posts:
                print(f"[SKIP] Post {post['id']} already processed before")
                continue

            voice_name = random.choice(MALE_VOICES if male_turn else FEMALE_VOICES)
            male_turn = not male_turn

            print(f"\n{'=' * 60}")
            print(f"[POST] r/{sub} - {post['id']}")
            print(f"[TITLE] {post['title'][:80]}...")

            # Prepare text
            full_text = f"{post['title']}. {post['text']}"
            full_text = censor_text(full_text)

            # Generate unique filename
            timestamp = int(time.time())
            audio_path = f"{OUTPUT_DIR}/tts_{post['id']}_{timestamp}.mp3"
            video_path = f"{OUTPUT_DIR}/{post['id']}.mp4"

            # Generate TTS audio
            if not generate_tts(full_text, voice_name, audio_path):
                print("[ERROR] TTS generation failed")
                continue

            if not os.path.exists(audio_path):
                print("[ERROR] Audio file not created")
                continue

            # Transcribe with Whisper to get word timings
            word_timings = transcribe_with_whisper(audio_path)

            if not word_timings:
                print("[ERROR] No word timings available - skipping video")
                continue

            # Create video
            try:
                success = create_video(audio_path, word_timings, video_path)
                if not success:
                    print("[INFO] Video creation failed or skipped")
                    continue
                upload_to_youtube(
                    video_path=video_path,
                    title=post["title"],
                    description=post["text"][:4000],
                    tags=["reddit", sub, "storytime", "shorts"]
                )
                processed_posts.add(post["id"])
                save_processed_posts(processed_posts)
                print(f"[SAVE] Marked post {post['id']} as processed")
            except Exception as e:
                print(f"[ERROR] Video creation failed: {e}")
                continue

            # Cleanup temporary audio file
            if os.path.exists(audio_path):
                try:
                    time.sleep(1)
                    os.remove(audio_path)
                    print(f"[CLEANUP] Removed temporary audio")
                except Exception as e:
                    print(f"[CLEANUP] Could not remove audio: {e}")

            time.sleep(3)  # Rate limiting between posts

    print("\n[DONE] All videos processed!")


if __name__ == "__main__":
    main()