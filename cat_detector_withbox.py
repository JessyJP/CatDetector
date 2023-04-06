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
import time


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
    decoded_preds = tf.keras.applications.mobilenet_v2.decode_predictions(preds, top=5)

    cat_classes = [
        'Egyptian_cat',
        'tiger_cat',
        'tabby',
        'Persian_cat',
        'Siamese_cat',
        'Maine_Coon',
    ]

    for pred in decoded_preds[0]:
        if pred[1] in cat_classes:
            return True

    return False

from yolov4.tf import YOLOv4

def is_cat_or_dog(frame, yolo):
    img = cv2.resize(frame, (yolo.input_size, yolo.input_size))
    img = img / 255.0

    boxes, _, classes = yolo.predict(img)

    cat_dog_classes = [16, 17]  # In COCO dataset, 16 = dog, 17 = cat

    for i, class_id in enumerate(classes):
        if class_id in cat_dog_classes:
            return boxes[i]

    return None

def add_overlay(frame, bbox, color=(0, 255, 0)):
    x, y, w, h = bbox
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

def cat_detector(video_path, skip_frames, output_dir):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    saved_count = 0

    yolo = YOLOv4()
    yolo.classes = "coco.names"
    yolo.make_model()
    yolo.load_weights("yolov4.weights", weights_type="yolo")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % skip_frames == 0:
            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) // 1000  # Get timestamp in seconds

            bbox = is_cat_or_dog(frame, yolo)
            if bbox:
                add_overlay(frame, bbox)
                output_path = os.path.join(output_dir, f"Detected-cat_or_dog-{saved_count:04d}-{timestamp}s.png")
                cv2.imwrite(output_path, frame)
                saved_count += 1

            checked_output_path = os.path.join(output_dir, f"Checked-{timestamp}s.png")
            cv2.imwrite(checked_output_path, frame)

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
