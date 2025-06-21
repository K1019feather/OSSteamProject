# 프로젝트 개요

약시인들의 분리수거 배출을 돕기 위해 일상에서 배출되는 분리수거 폐기물 이미지를 자동으로 분류하고 TTS로 알려주는 딥러닝 모델을 구축하는 것이 목표  
분류 대상 : 5가지 클래스  
- cardboard  
- glass  
- metal  
- paper  
- plastic

# 특징

- 실시간 분리수거 쓰레기 분류
- 약시인들을 고려한 실시간 음성 기능
- 비장애인들도 고려한 텍스트 기능
- 과정이 간단하고 명료함

# 사용한 모델

- 원본 모델 출처: https://github.com/tanishq-ctrl/waste-classification
- EfficientNetB0 : imageNet 데이터로 사전 학습된 모델 사용
- 출력층 수정 : 5가지 클래스로 모델 수정  
- fine-tuning :  
  trainable=False로 출력층만 학습 (baseline 모델)  
  이후 trainable=True로 전체 모델을 언프리즈 후 파인튜닝 진행  
  학습률 실험을 통해 가장 성능이 높은 모델을 선정하여 사용

# 데이터셋 출처

1) trashnet : github.com/garythung/trashnet
2) trashbox : github.com/nikhilvenkatkumsetty/TrashBox

# 데이터셋 개요

- 총 이미지 수: 4526
- 이미지 형식: JPG
- 크기: 224x224

| Class        | 총 이미지 수 | 비율 (%)    |
| ------------ | ------- | --------- |
| 📄 paper     | 1065    | 23.5% |
| 🧪 glass     | 955     | 21.1%     |
| 🧴 plastic   | 850     | 18.8%     |
| 🟫 cardboard | 845     | 18.7%     |
| ⚙️ metal     | 811     | 17.9%     |


# 프로젝트 구조

OSSteamProject/  
├── WASTE_MANAGEMENT.ipynb       // 실험·테스트용  
├── data_utils.py                // 전처리 함수를 모아놓는 파일  
├── WASTE_MANAGEMENT.py          // 전처리 리팩토링한 실질적 실행 파일  
├── dataset-resized/             // 전처리할 5가지 클래스  
│   ├── cardboard/  
│   ├── glass/  
│   ├── metal/  
│   ├── paper/  
│   └── plastic/  
└── models/                      // 학습한 모델 모아두는 파일

## 📌 프로젝트 개발 순서

1. 김민채 - EfficientNet-B0 기본 로드 및 출력층 교체  
   정모아 - 클래스별 이미지 데이터 수집 마무리  
2. 김민채 - Dummy Input 통과 확인  
   정모아 - 전처리 및 Augmentation 파이프라인 완성  
   정명환 - Baseline 모델 학습 완료  
3. 김민채 - 모델 구조 수정, freeze/unfreeze 실험  
   정모아 - 전처리 코드 리팩토링, 이미지 시각화 함수 추가  
   정명환 - 다양한 학습률 실험 및 학습 로그 정리  
4. 김민채 - freeze/unfreeze 전략 문서화  
5. 정명환 - 전체 언프리즈 후 Fine-tuning 완료  
   김민채 - 실시간 분류 스크립트 초안 작성  
   정모아 - 시연용 테스트셋 구성  
6. 정모아 - 오분류 샘플 자동 추출 기능 구현  
7. 정명환 - TTS 연동 통합 테스트  
8. 정명환 - 최종 성능 레포트 작성 및 제출 준비
