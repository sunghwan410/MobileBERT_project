#  MobileBERT를 활용한 애니 리뷰 감정 분석 프로젝트  
<p align="center">
  <img src="https://ifh.cc/g/8hOMNb.jpg" alt="Anime Banner" width="600"/>
</p>

---

##  프로젝트 개요

> 디지털 시대에 접어들며 애니메이션 리뷰는 콘텐츠 흥행의 핵심 지표가 되었습니다.  
> 본 프로젝트는 다양한 애니메이션 리뷰 데이터를 기반으로 감정 분석 모델을 구축하여 다음을 목표로 합니다:

- 리뷰 감정의 경향성 분석  
- 인기 애니메이션 간 평판 비교  
- 추천 시스템 또는 큐레이션 기반 활용 가능성 검토  

> 리뷰 원문은 러시아어이며, 영어로 번역 후 MobileBERT 모델을 파인튜닝하여 감정 분류 (Positive / Neutral / Negative)를 수행합니다.

---

##  데이터셋 구축 및 번역 과정

| 항목       | 내용 |
|------------|------|
| 📄 원본 파일 | Anime_reviews_RU.csv |
| 🔢 총 리뷰 수 | 약 76,000건 |
| 🧾 주요 컬럼 | `anime`, `text`, `rate` |
| 🌍 언어     | 러시아어 |
| 🌐 번역 도구 | Google Translator |
| ⚙️ 병렬 처리 | `anime.py`를 통해 멀티프로세싱 (최대 8코어) |
| 🕒 번역 시간 | 약 2일 소요 |
| 🧹 전처리    | `rate`가 유효하지 않거나 결측치 제거, 약 73,000건 유지 |
<a href='https://ifh.cc/v-wsZgKG' target='_blank'><img src='https://ifh.cc/g/wsZgKG.png' border='0'></a>        <a href='https://ifh.cc/v-dBlDx5' target='_blank'><img src='https://ifh.cc/g/dBlDx5.png' border='0'></a>
anime.py 파일에다 Google Translator로 번역하여 1000건씩 74개의 파일로 나눠 번역을 진행했습니다.

---

## 📊 학습 데이터 구성 방법

| 기준 | 내용 |
|------|------|
| 🔍 추출 비율 | 전체의 약 4% (약 3,000건) 샘플링 |
| ⚖️ 클래스 균형 | `stratify` 샘플링, Positive/Negative 비율 유지 |
| 🧪 분할 | 학습:검증 = 80:20 |

> 전체 분포를 반영하며, 제한된 자원에서도 일반화 성능 확보

---

##  MobileBERT Fine-tuning 결과

###  모델 설정

| 항목 | 설정 |
|------|------|
| Pretrained | `google/mobilebert-uncased` |
| Input length | 256 tokens |
| Class 수 | 3개 또는 2개 (실험 기준) |
| Optimizer | AdamW (`lr=2e-5`) |
| Batch size | 8 |
| Epochs | 4 (3-class) / 10 (2-class) |

###  2진 분류 성능 예시

| Epoch | Train Loss | Val Accuracy |
|-------|------------|---------------|
|   1   | 0.6228     | 0.7733        |
|   2   | 0.3996     | 0.8400        |
|   3   | 0.2624     | **0.8533** ✅ |

---

##  Inference 및 분석

- 전체 리뷰셋에 대해 감정 예측 수행  
- 애니메이션별 긍정/부정 비율 계산  
- `top_bottom5_anime_reviews.csv` 생성  

---

##  감정 분석 시각화

### ✅ 긍정 비율 기준 Top 5 애니

- Your Name  
- Spirited Away  
- Attack on Titan  
- Violet Evergarden  
- Mob Psycho 100

### ⚠️ 부정 비율 기준 Top 5 애니

- Mars of Destruction  
- Pupa  
- School Days  
- Vampire Holmes  
- Boku no Pico

> 결과 저장: `top_bottom5_anime_reviews.csv`

---

## 💻 주요 코드 설명

| 파일명 | 설명 |
|--------|------|
| `anime.py` | 멀티프로세싱 기반 74개 CSV 파일 번역 및 통합 |
| `finetune_mobilebert_anime.py` | 학습 자동화 (입력, 로딩, 학습, 저장 포함) |
| `inference_mobilebert_anime.py` | 전체 데이터셋 감정 예측 및 시각화용 통계 생성 |

---

## 🔚 결론 및 향후 계획

- 번역 품질 및 전처리가 NLP 성능에 직접적 영향  
- 중립 감정은 모호하여 이진 분류 성능이 더 우수  
- 정확도 85% 이상 도달 시 실무 적용 가능성 확인

### 🔮 향후 방향

- RuBERT 기반 러시아어 직접 감정 분석 모델 실험  
- 추천 시스템/감정 기반 큐레이션 적용  
- 장르/연도 기반 감정 트렌드 분석

---

## 🛠 개발 환경 및 라이브러리

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
- `anime.py`: 병렬 번역 처리 코드  
- 시각화 도구: `matplotlib`, `seaborn`

---

📢 **리뷰 기반 감정 분석은 사용자 만족도와 직결됩니다.**  
🎯 **AI 기반 감정 분석 시스템은 이제 선택이 아닌 필수입니다!**
