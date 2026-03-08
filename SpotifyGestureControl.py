import cv2 as cv
import HandTrackingModule as htm
import numpy
import spotipy as sp
from spotipy.oauth2 import SpotifyOAuth
import os
from dotenv import load_dotenv

load_dotenv()

spotify = sp.Spotify(
    auth_manager=SpotifyOAuth(
        client_id=os.getenv("SPOTIPY_CLIENT_ID"),
        client_secret=os.getenv("SPOTIPY_CLIENT_SECRET"),
        redirect_uri=os.getenv("SPOTIPY_REDIRECT_URI"),
        scope="user-modify-playback-state user-read-playback-state",
        open_browser=True,
    )
)

camWidth, camHeight = 640, 480

capture = cv.VideoCapture(0)
capture.set(3, camWidth)
capture.set(4, camHeight)

detector = htm.HandDetector(max_num_hands=1)
current_zone = None

while True:
    ret, frame = capture.read()
    flipped_frame = cv.flip(frame, 1)

    flipped_frame = detector.findHands(flipped_frame)
    lmList = detector.findPosition(flipped_frame, draw=False)

    red_x1, red_x2 = 0, camWidth // 5
    green_x1, green_x2 = camWidth - camWidth // 5, camWidth
    blue_x1, blue_x2 = camWidth // 2 - 100, camWidth // 2 + 100
    blue_y1, blue_y2 = 0, 100

    zone = None

    if lmList and len(lmList) > 8:
        x1, y1 = int(lmList[8][1]), int(lmList[8][2])
        cv.circle(flipped_frame, (x1, y1), 20, (255, 0, 255), cv.FILLED)

        if red_x1 <= x1 <= red_x2:
            zone = "red"
        elif green_x1 <= x1 <= green_x2:
            zone = "green"
        elif blue_x1 <= x1 <= blue_x2 and blue_y1 <= y1 <= blue_y2:
            zone = "blue"

    if zone is not None and zone != current_zone:
        try:
            if zone == "green":
                spotify.next_track()
            elif zone == "red":
                spotify.previous_track()
            elif zone == "blue":
                playback = spotify.current_playback()
                if playback and playback.get("is_playing"):
                    spotify.pause_playback()
                else:
                    spotify.start_playback()
        except Exception:
            pass

    current_zone = zone

    cv.rectangle(flipped_frame, (green_x1, 0), (green_x2, camHeight), (0, 255, 0), 2)
    cv.rectangle(flipped_frame, (red_x1, 0), (red_x2, camHeight), (0, 0, 255), 2)
    cv.rectangle(flipped_frame, (blue_x1, blue_y1), (blue_x2, blue_y2), (255, 0, 0), 2)

    cv.imshow("Webcam", flipped_frame)

    if cv.waitKey(20) & 0xFF == ord('d') or not ret:
        break

capture.release()
cv.destroyAllWindows()