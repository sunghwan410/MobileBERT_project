# MobileBERT를 활용한 애니 리뷰 감정 분석 프로젝트  

<p align="center">
  <img src="https://ifh.cc/g/8hOMNb.jpg" alt="Anime Banner" width="600"/>
</p>

---

## 📖 프로젝트 개요

디지털 시대에 접어들며 애니메이션 리뷰는 콘텐츠 흥행의 핵심 지표가 되었다.  
본 프로젝트는 다양한 애니메이션 리뷰 데이터를 기반으로 감정 분석 모델을 구축했다. 프로젝트의 목적은 다음과 같다:

- 리뷰 감정의 경향성을 분석했다.  
- 인기 애니메이션 간 평판을 비교했다.  
- 추천 시스템 또는 큐레이션 기반 활용 가능성을 검토했다.  

리뷰 원문은 러시아어였고, 이를 영어로 번역한 후 **MobileBERT** 모델을 파인튜닝하여 감정 분류(Positive / Neutral / Negative)를 수행했다.

---

## 📊 데이터셋 구축 및 번역 과정

| 항목       | 내용 |
|------------|------|
| 📄 원본 파일 | Anime_reviews_RU.csv |
| 🔢 총 리뷰 수 | 약 76,000건 |
| 🧾 주요 컬럼 | anime, text, rate |
| 🌍 언어     | 러시아어 |
| 🌐 번역 도구 | Google Translator |
| ⚙️ 병렬 처리 | anime.py로 멀티프로세싱을 적용했다 (최대 8코어) |
| 🕒 번역 시간 | 약 2일 소요됐다. |
| 🧹 전처리    | rate가 유효하지 않거나 결측치인 데이터를 제거했고, 약 73,000건을 유지했다. |

<p align="center">
  <img src="https://ifh.cc/g/wsZgKG.png" alt="Translation Process 1" width="300" style="margin-right: 10px;"/>
  <img src="https://ifh.cc/g/dBlDx5.png" alt="Translation Process 2" width="300"/>
</p>

Google Translator API를 사용하여 리뷰 1,000건씩 74개 파일로 나눠 번역을 진행했고, 결과는 `translated.parts` 폴더에 저장했다.

---

## 🧠 학습 데이터 구성 방법

| 기준 | 내용 |
|------|------|
| 🔍 추출 비율 | 전체의 약 4% (약 3,000건)을 샘플링했다. |
| ⚖️ 클래스 균형 | stratify 샘플링으로 Positive/Negative 비율을 유지했다. |
| 🧪 분할 | 학습:검증 = 80:20으로 분할했다. |

전체 분포를 반영하여 제한된 자원에서도 일반화 성능을 확보했다.

---

## 🚀 MobileBERT Fine-tuning 결과

### 모델 설정

| 항목 | 설정 |
|------|------|
| Pretrained | google/mobilebert-uncased |
| Input length | 256 tokens |
| Class 수 | 3개 또는 2개 실험을 수행했다. |
| Optimizer | AdamW (lr=2e-5)를 사용했다. |
| Batch size | 8 |
| Epochs | 4 (3-class) / 10 (2-class) |

### 2진 분류 성능 예시

| Epoch | Train Loss | Val Accuracy |
|-------|------------|---------------|
|   1   | 0.6228     | 0.7733        |
|   2   | 0.3996     | 0.8400        |
|   3   | 0.2624     | **0.8533** ✅ |

---

## 🔍 Inference 및 분석

전체 리뷰셋에 대해 감정 예
