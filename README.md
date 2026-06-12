# ComVis with Spotify

Hand-tracking Spotify controller built with OpenCV, MediaPipe, Spotipy, and Pycaw. The app uses your webcam to detect hand gestures and map them to Spotify playback controls and system volume changes.

## Features

- Next track by moving your index finger into the green zone on the right
- Previous track by moving your index finger into the red zone on the left
- Pause or resume playback by moving your index finger into the blue zone at the top center
- Control system volume by pinching your thumb and index finger together

## Requirements

- Windows
- Python 3.12
- A working webcam
- A Spotify Premium account for playback control
- Spotify developer credentials

## Installation

1. Create and activate a virtual environment if you want to keep dependencies isolated.
2. Install the project dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the project root with your Spotify credentials:

   ```env
   SPOTIPY_CLIENT_ID=your_client_id
   SPOTIPY_CLIENT_SECRET=your_client_secret
   SPOTIPY_REDIRECT_URI=http://localhost:8888/callback
   ```

4. Make sure the redirect URI in your Spotify developer dashboard matches the value in `.env`.

## Running

Start the app with:

```bash
python SpotifyGestureControl.py
```

The webcam window opens automatically. Press `d` to quit.

## How It Works

The project uses `HandTrackingModule.py` to detect hand landmarks and read finger positions from the webcam frame. `SpotifyGestureControl.py` then checks the index fingertip location against colored screen zones and uses the Spotify Web API to switch tracks or toggle playback. When the thumb and index finger are held apart or brought together, the app maps that distance to the system master volume.

## Notes

- This project is tuned for a single detected hand.
- Volume control relies on Pycaw and is intended for Windows.
- If Spotify actions do not trigger, verify that your Spotify credentials, redirect URI, and playback device are set up correctly.
