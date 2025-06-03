import os
import shutil
import pandas as pd

processed_dir = './processed_data'
augmented_dir = './augmented_data'
final_dir = './final_data'
classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic']

image_paths = []
labels = []

os.makedirs(final_dir, exist_ok=True)

for label_idx, class_name in enumerate(classes):
    src_dirs = [
        os.path.join(processed_dir, class_name),
        os.path.join(augmented_dir, class_name)
    ]
    dst_dir = os.path.join(final_dir, class_name)
    os.makedirs(dst_dir, exist_ok=True)

    for src_dir in src_dirs:
        if not os.path.exists(src_dir):
            continue

        for fname in os.listdir(src_dir):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                src_path = os.path.join(src_dir, fname)
                new_name = f"{class_name}_{len(image_paths)}_{fname}"
                dst_path = os.path.join(dst_dir, new_name)

                try:
                    shutil.copy2(src_path, dst_path)
                    image_paths.append(dst_path)
                    labels.append(label_idx)
                except Exception as e:
                    print(f"❌ 복사 실패: {src_path} → {e}")

print(f"✅ 이미지 총 수: {len(image_paths)}")

# CSV 저장
df = pd.DataFrame({
    'image_path': image_paths,
    'label': labels
})
df.to_csv('final_labels.csv', index=False)
print("📄 라벨 CSV 저장 완료: final_labels.csv")
