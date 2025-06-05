from torchvision import transforms

# 학습용 transform
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),  # 전처리와 동일
    transforms.RandomHorizontalFlip(),  # 증강
    transforms.RandomRotation(10),      # 증강
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet 기준
        std=[0.229, 0.224, 0.225]
    )
])

# 테스트용 transform
transform_test = transforms.Compose([
    transforms.Resize((224, 224)),  # 전처리와 동일
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
