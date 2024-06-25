import os
from deepface import DeepFace
from collections import Counter
os.system('ffmpeg -version')

# mp4_path = './example.mp4'
mp4_path = '这么可爱真是抱歉.mp4'

def get_keyframes(mp4_file_path, output_dir='./output'):
    os.makedirs(output_dir, exist_ok=True)
    # os.system(f'ffmpeg -i {mp4_file_path} -vf "select=eq(pict_type\,I),showinfo" -vsync vfr {output_dir}/%03d.png')
    # os.system(f'bash ./ffmpeg.sh --file_path , --output_dir')
    os.system(f'ffmpeg -i {mp4_file_path} -vf "select=(gte(t\,10))*(isnan(prev_selected_t)+gte(t-prev_selected_t\,10))" -vsync vfr {output_dir}/%01d0.png')

output_dir='./output'
get_keyframes(mp4_path,output_dir)


def anyalize_video(folder_path):
    dominant_emotions = []
    for keyframe in os.listdir(folder_path):
        if keyframe.endswith('.png'): #做删选，有时候会有一些文件
            keyframe_path = os.path.join(folder_path, keyframe)
            time=os.path.splitext(keyframe)[0]
            result = DeepFace.analyze(
                img_path = keyframe_path,
                actions = ['emotion'],
                enforce_detection = False
            )
            #只抽取emotion项目
            result = result[0]['dominant_emotion']
            dominant_emotions.append({result,time})
    return dominant_emotions

video_results = anyalize_video(output_dir)
print(video_results)


