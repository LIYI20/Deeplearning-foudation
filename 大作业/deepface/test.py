# deep库的导入就一行代码
from deepface import DeepFace


img_path="deepface\images\\6.png"
result = DeepFace.analyze(
                img_path = img_path,
                actions = ['emotion'],
                enforce_detection = False
            )
print(result)