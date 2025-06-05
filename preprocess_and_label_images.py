import os
from PIL import Image, ImageEnhance

# 설정
VALID_EXTENSIONS = ['.jpg', '.jpeg', '.png']
RESIZE_SIZE = (224, 224)
BRIGHTNESS_FACTOR = 1.2

def preprocess_image(img_path, resize_size=RESIZE_SIZE, brightness=BRIGHTNESS_FACTOR):
    with Image.open(img_path) as img:
        img = img.convert('RGB')  # RGB 변환
        img = img.resize(resize_size)  # 리사이즈
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(brightness)  # 밝기 조정
        return img

def process_class_folder(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    image_count = 0

    for fname in os.listdir(input_dir):
        if fname.lower().endswith(tuple(VALID_EXTENSIONS)):
            try:
                img_path = os.path.join(input_dir, fname)
                processed_img = preprocess_image(img_path)
                processed_img.save(os.path.join(output_dir, fname))
                image_count += 1
            except Exception as e:
                print(f"❌ 실패: {fname} → {e}")

    print(f"✅ {output_dir}: {image_count}장 처리 완료")

def process_all_classes(data_dir, save_root, classes):
    for class_name in classes:
        input_path = os.path.join(data_dir, class_name)
        output_path = os.path.join(save_root, class_name)
        if os.path.isdir(input_path):
            process_class_folder(input_path, output_path)

# 예시 실행
if __name__ == "__main__":
    data_dir = r"C:\Users\jmoa7\trashnet\data\dataset-resized"
    save_dir = r"C:\Users\jmoa7\trashnet\processed_data"
    class_list = ['cardboard', 'glass', 'metal', 'paper', 'plastic']
    process_all_classes(data_dir, save_dir, class_list)
