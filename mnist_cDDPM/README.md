# MNIST cDDPM (Row-shift artifact removal)

## 목적
MNIST 이미지에 row-shift artifact를 조건 입력(condition)으로 주고, 원본 digit 구조를 유지하면서 artifact를 완화/제거하여 복원하는 conditional DDPM을 학습하는 toy setting.

## 폴더 구조
- `ddpm/`  
  model(Conditional UNet), diffusion scheduler, 전처리/유틸 함수 등 구현

- `scripts_cDDPM0114/`  
  실행용 entry 스크립트  
  - `main_cDDPM0114.py`: 학습 실행  
  - `infer_cDDPM0114.py`: 체크포인트로 샘플 생성 및 평가(MSE, SSIM)

- `runs_cDDPM0114/`  
  학습/추론 결과 저장  
  - `train_full200000/`: 학습 체크포인트, 설정, 로그, 중간 샘플
    - `samples_step*.png`: 학습 중 일정 step마다 sampling한 결과
  - `train_full200000/infer/model_last/`: 학습 종료 후 최종 체크포인트로 생성한 샘플 및 평가 결과

