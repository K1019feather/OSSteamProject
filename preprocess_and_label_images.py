import os
from PIL import Image, ImageEnhance

# 설정
data_dir= 'C:/Users/jmoa7/trashnet/data/dataset-resized'  # 클래스별 이미지 폴더가 있는 디렉토리
output_dir = './processed_data'
classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic']
resize_size = (224, 224)
brightness_factor = 1.2  # 1.0이면 그대로, 1.2면 20% 더 밝게

# 출력 폴더 생성
os.makedirs(output_dir, exist_ok=True)

image_paths = []
labels = []

for label_idx, class_name in enumerate(classes):
    class_dir = os.path.join(data_dir, class_name)
    if not os.path.exists(class_dir):
        continue

    output_class_dir = os.path.join(output_dir, class_name)
    os.makedirs(output_class_dir, exist_ok=True)

    for fname in os.listdir(class_dir):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(class_dir, fname)

            try:
                # 이미지 열기
                with Image.open(img_path) as img:
                    # RGB 변환 (팔레트나 L 등 대응)
                    img = img.convert('RGB')

                    # Resize
                    img = img.resize(resize_size)

                    # 밝기 조정
                    enhancer = ImageEnhance.Brightness(img)
                    img = enhancer.enhance(brightness_factor)

                    # 저장
                    save_path = os.path.join(output_class_dir, fname)
                    img.save(save_path)

                    # 라벨링
                    image_paths.append(save_path)
                    labels.append(label_idx)

            except Exception as e:
                print(f"❌ 이미지 처리 실패: {img_path} → {e}")

print(f"✅ 총 처리 이미지 수: {len(image_paths)}")

