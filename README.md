# 📘 MobileBERT를 활용한 애니메이션 리뷰 감정 분석

<p align="center">
  <img src="https://ifh.cc/g/8hOMNb.jpg" alt="Anime Banner" width="600"/>
</p>

## 📖 프로젝트 개요

애니메이션 리뷰는 콘텐츠 흥행의 핵심 지표이다.  
본 프로젝트는 애니메이션 리뷰 데이터를 기반으로 감정 분석 모델을 구축하였으며, 주요 목표는 다음과 같다:

- 리뷰 감정의 전반적인 흐름 분석  
- 애니메이션 간 평판 비교  
- 추천 시스템 및 큐레이션에의 활용 가능성 탐색  

리뷰 원문은 러시아어이며, 영어로 번역 후 MobileBERT를 파인튜닝하여 감정 분류(Positive / Neutral / Negative)를 수행하였다.

---

## 📊 데이터셋 및 번역 과정

| 항목       | 내용 |
|------------|------|
| 원본 파일 | Anime_reviews_RU.csv |
| 총 리뷰 수 | 약 76,000건 |
| 주요 컬럼 | `anime`, `text`, `rate` |
| 언어 | 러시아어 |
| 번역 도구 | Google Translator |
| 병렬 처리 | `anime.py`로 멀티프로세싱 (최대 8코어) |
| 번역 시간 | 약 2일 소요 |
| 전처리 | 유효하지 않거나 결측된 `rate` 제거 (약 73,000건 유지) |

> 리뷰를 1,000건 단위로 분할하여 74개 파일로 번역하였으며, 결과는 `translated.parts` 폴더에 저장하였다.

---

## 🧠 학습 데이터 구성

| 기준 | 내용 |
|------|------|
| 샘플링 | 전체의 약 4% (약 3,000건) |
| 클래스 균형 | `stratify` 기반 샘플링 |
| 학습/검증 분할 | 80:20 비율로 분리 |

---

## 🚀 MobileBERT Fine-tuning

### 모델 설정

| 항목 | 설정 |
|------|------|
| Pretrained | `google/mobilebert-uncased` |
| Input length | 256 tokens |
| 클래스 수 | 3개 또는 2개 |
| Optimizer | AdamW (lr=2e-5) |
| Batch size | 8 |
| Epochs | 4 (3-class) / 10 (2-class) |

### 2-class 분류 결과 예시

| Epoch | Train Loss | Val Accuracy |
|-------|------------|--------------|
| 1     | 0.6228     | 0.7733       |
| 2     | 0.3996     | 0.8400       |
| 3     | 0.2624     | **0.8533** ✅ |

---

## 🔍 감정 예측 및 분석

- 전체 리뷰셋에 대해 감정 예측 수행  
- 애니메이션별 긍정/부정 비율 계산  
- 결과 파일: `top_bottom5_anime_reviews.csv`

---

## 📈 시각화 예시

### ✅ 긍정 비율 상위 Top 5

| 순위 | 애니메이션 |
|------|------------|
| 1 | Your Name (너의 이름은) |
| 2 | Spirited Away (센과 치히로의 행방불명) |
| 3 | Attack on Titan (진격의 거인) |
| 4 | Violet Evergarden (바이올렛 에버가든) |
| 5 | Mob Psycho 100 (모브 사이코 100) |

### ⚠️ 부정 비율 상위 Top 5

| 순위 | 애니메이션 |
|------|------------|
| 1 | Mars of Destruction |
| 2 | Pupa |
| 3 | School Days |
| 4 | Vampire Holmes |
| 5 | Boku no Pico |

---

## 💻 주요 코드 설명

| 파일명 | 설명 |
|--------|------|
| `anime.py` | 멀티프로세싱 기반 번역 자동화 |
| `finetune_mobilebert_anime.py` | 학습 전체 프로세스 처리 |
| `inference_mobilebert_anime.py` | 예측 및 결과 통계/시각화 처리 |

---

## 🔚 결론 및 향후 계획

- 번역 품질과 전처리 수준이 모델 성능에 큰 영향을 미쳤다.  
- 중립 감정은 해석이 모호하여 2-class 분류의 성능이 상대적으로 더 높았다.  
- 정확도 85% 이상으로 실무 활용 가능성을 확인하였다.

### 향후 방향

- 러시아어 기반 RuBERT 감정 분석 실험  
- 감정 기반 큐레이션/추천 시스템 개발  
- 장르 및 연도 기반 감정 트렌드 분석

---

## 🛠️ 개발 환경

| 항목 | 버전 |
|------|------|
| Python | 3.9 |
| PyTorch | 1.12.1 |
| Transformers | 4.21.2 |
| Pandas | 1.4.4 |
| NumPy | 1.24.3 |
| Scikit-learn | 1.2.2 |
| IDE | PyCharm / JupyterLab |

---

## 🔗 참고 링크

- [MobileBERT on HuggingFace](https://huggingface.co/google/mobilebert-uncased)  
- [deep_translator 패키지](https://pypi.org/project/deep-translator/)  
- 번역 코드: `anime.py`  
- 시각화 도구: `matplotlib`, `seaborn`

---

📢 리뷰 기반 감정 분석은 사용자 만족도와 직결된다.  
🎯 AI 기반 감정 분석 시스템은 선택이
