import os
import random
from PIL import Image, ImageEnhance, ImageFilter

# 설정
input_dir = './processed_data'
output_dir = './augmented_data'
classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic']
augment_count = 200  # 클래스당 추가 이미지 수

os.makedirs(output_dir, exist_ok=True)

def augment_image(img):
    # 랜덤 회전
    angle = random.uniform(-15, 15)
    img = img.rotate(angle)

    # 좌우 반전
    if random.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)

    # 밝기 변화
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(random.uniform(0.8, 1.2))

    # 채도 변화
    enhancer = ImageEnhance.Color(img)
    img = enhancer.enhance(random.uniform(0.8, 1.2))

    # 블러
    if random.random() < 0.3:
        img = img.filter(ImageFilter.GaussianBlur(radius=1))

    return img

for class_name in classes:
    class_input_dir = os.path.join(input_dir, class_name)
    class_output_dir = os.path.join(output_dir, class_name)
    os.makedirs(class_output_dir, exist_ok=True)

    image_files = [f for f in os.listdir(class_input_dir) if f.endswith(('.jpg', '.png'))]
    print(f"📂 {class_name}: 원본 {len(image_files)}개 → 증강 {augment_count}개")

    for i in range(augment_count):
        src_file = random.choice(image_files)
        src_path = os.path.join(class_input_dir, src_file)

        try:
            with Image.open(src_path) as img:
                img = img.convert('RGB')
                aug_img = augment_image(img)
                aug_img.save(os.path.join(class_output_dir, f'aug_{i}_{src_file}'))
        except Exception as e:
            print(f"❌ 증강 실패: {src_file} → {e}")

print("✅ 증강 완료!")
