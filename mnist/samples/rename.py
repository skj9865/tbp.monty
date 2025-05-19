import os
import re

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', s)]

def rename_images_in_folder(folder_path):
    files = sorted(os.listdir(folder_path), key=natural_sort_key)
    image_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    for idx, filename in enumerate(image_files):
        ext = os.path.splitext(filename)[1]
        new_name = f"img_{idx}{ext}"
        src = os.path.join(folder_path, filename)
        dst = os.path.join(folder_path, new_name)
        os.rename(src, dst)
        print(f"Renamed: {filename} → {new_name}")

# 사용 예시
rename_images_in_folder("trainingSample/0")  # 또는 절대 경로를 입력: "C:/path/to/0"
rename_images_in_folder("trainingSample/1")  # 또는 절대 경로를 입력: "C:/path/to/0"
rename_images_in_folder("trainingSample/2")  # 또는 절대 경로를 입력: "C:/path/to/0"
rename_images_in_folder("trainingSample/3")  # 또는 절대 경로를 입력: "C:/path/to/0"
rename_images_in_folder("trainingSample/4")  # 또는 절대 경로를 입력: "C:/path/to/0"
rename_images_in_folder("trainingSample/5")  # 또는 절대 경로를 입력: "C:/path/to/0"
rename_images_in_folder("trainingSample/6")  # 또는 절대 경로를 입력: "C:/path/to/0"
rename_images_in_folder("trainingSample/7")  # 또는 절대 경로를 입력: "C:/path/to/0"
rename_images_in_folder("trainingSample/8")  # 또는 절대 경로를 입력: "C:/path/to/0"
rename_images_in_folder("trainingSample/9")  # 또는 절대 경로를 입력: "C:/path/to/0"
