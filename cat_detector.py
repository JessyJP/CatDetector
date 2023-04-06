import argparse
import os
import tempfile
import shutil
import cv2
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
import youtube_dl


def download_youtube_video(url, tempdir):
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]',
        'outtmpl': os.path.join(tempdir, 'video.%(ext)s')
    }

    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    return os.path.join(tempdir, 'video.mp4')


def is_cat(frame):
    img = cv2.resize(frame, (224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)

    model = MobileNetV2(weights='imagenet', include_top=True)
    preds = model.predict(img)
    decoded_preds = tf.keras.applications.mobilenet_v2.decode_predictions(preds, top=1)

    if decoded_preds[0][0][1] == 'Egyptian_cat':
        return True
    return False


def cat_detector(video_path, skip_frames, output_dir):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    saved_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % skip_frames == 0:
            if is_cat(frame):
                output_path = os.path.join(output_dir, f"cat_{saved_count:04d}.png")
                cv2.imwrite(output_path, frame)
                saved_count += 1

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='Detect cats in a video and save screenshots')
    parser.add_argument('video', help='Local video file path or URL to a YouTube video')
    parser.add_argument('output_dir', help='Output directory for the screenshots')
    parser.add_argument('-f', '--frames', type=int, help='Minimum skip between screenshots in frames')
    parser.add_argument('-s', '--seconds', type=int, help='Minimum skip between screenshots in seconds')

    args = parser.parse_args()

    if args.frames and args.seconds:
        parser.error('Specify either frames or seconds, not both')

    if args.seconds:
        fps = 30
        skip_frames = args.seconds * fps
    else:
        skip_frames = args.frames if args.frames else 100

    if os.path.exists(args.output_dir):
        if not os.path.isdir(args.output_dir):
            parser.error('Output directory path is not a directory')
    else:
        os.makedirs(args.output_dir)

    if "youtube.com" in args.video or "youtu.be" in args.video:
        with tempfile.TemporaryDirectory() as tempdir:
            video_path = download_youtube_video(args.video, tempdir)
            cat_detector(video_path, skip_frames, args.output_dir)
            os.remove(video_path)
    else:
        if not os.path.isfile(args.video):
            parser.error('Video file does not exist')
        cat_detector(args.video, skip_frames, args.output_dir)


if __name__ == '__main__':
    main()
