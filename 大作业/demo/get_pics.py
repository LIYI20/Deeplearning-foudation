import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os

def extract_frames(video_path, num_frames):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception("Error: Cannot open video file.")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_ids = [i*total_frames//num_frames for i in range(num_frames)]

    frames = []
    for frame_id in frame_ids:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()
    return frames

def arrange_frames(frames, rows, cols):
    if len(frames) != rows * cols:
        raise ValueError("Number of frames does not match the grid size.")

    frame_height, frame_width, _ = frames[0].shape
    grid_width = cols * frame_width
    grid_height = rows * frame_height
    result_image = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)

    for idx, frame in enumerate(frames):
        row = idx // cols
        col = idx % cols
        result_image[row * frame_height:(row + 1) * frame_height, col * frame_width:(col + 1) * frame_width] = frame

    return result_image

def add_label(image, text, position):
    img_pil = Image.fromarray(image)
    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.truetype("DejaVuSans.ttf", 20)
    draw.text(position, text, font=font, fill=(255, 255, 255))
    return np.array(img_pil)

def main():
    video_path = '/root/multi_model/try.mp4'
    num_frames = 12  # 指定抽取的关键帧数量
    rows, cols = 3, 4  # 排列的行数和列数
    position = (10, 10)  # 标签的位置
    text = 'Video Frames'  # 标签文本

    frames = extract_frames(video_path, num_frames)
    arranged_image = arrange_frames(frames, rows, cols)
    final_image = add_label(arranged_image, text, position)

    # cv2.imshow('Arranged Frames', final_image)
    # save to the output_dir
    output_dir = '/root/multi_model/output_dir'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    success = cv2.imwrite(os.path.join(output_dir, 'output.jpg'), final_image)
    if not success:
        raise Exception("Error: Cannot write image file.")
    

if __name__ == '__main__':
    main()