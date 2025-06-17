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

전체 리뷰셋에 대해 감정 예측을 수행했다.  
애니메이션별 긍정/부정 비율을 계산했고, 결과는 `top_bottom5_anime_reviews.csv`에 저장했다.

---

## 📈 감정 분석 시각화

### ✅ 긍정 비율 기준 Top 5 애니

<div style="display: flex; justify-content: center; flex-wrap: wrap; gap: 10px;">
  <img src="https://ifh.cc/g/TlKfqw.jpg" alt="Your Name" style="width: 150px; height: 225px; object-fit: cover;"/>
  <img src="https://ifh.cc/g/mq0dDq.jpg" alt="Spirited Away" style="width: 150px; height: 225px; object-fit: cover;"/>
  <img src="https://ifh.cc/g/Hr19Bs.jpg" alt="Attack on Titan" style="width: 150px; height: 225px; object-fit: cover;"/>
  <img src="https://ifh.cc/g/RF5cxl.jpg" alt="Violet Evergarden" style="width: 150px; height: 225px; object-fit: cover;"/>
  <img src="https://ifh.cc/g/f4ngWy.jpg" alt="Mob Psycho 100" style="width: 150px; height: 225px; object-fit: cover;"/>
</div>

1. **Your Name (너의 이름은)**  
2. **Spirited Away (센과 치히로의 행방불명)**  
3. **Attack on Titan (진격의 거인)**  
4. **Violet Evergarden (바이올렛 에버가든)**  
5. **Mob Psycho 100 (모브 사이코 100)**  

### ⚠️ 부정 비율 기준 Top 5 애니

<div style="display: flex; justify-content: center; flex-wrap: wrap; gap: 10px;">
  <img src="https://ifh.cc/g/VsDVSk.jpg" alt="Mars of Destruction" style="width: 150px; height: 225px; object-fit: cover;"/>
  <img src="https://ifh.cc/g/qtZgq1.jpg" alt="Pupa" style="width: 150px; height: 225px; object-fit: cover;"/>
  <img src="https://ifh.cc/g/00x0Zb.jpg" alt="School Days" style="width: 150px; height: 225px; object-fit: cover;"/>
  <img src="https://ifh.cc/g/Zdmk8t.jpg" alt="Vampire Holmes" style="width: 150px; height: 225px; object-fit: cover;"/>
  <img src="https://ifh.cc/g/Dp12AN.jpg" alt="Boku no Pico" style="width: 150px; height: 225px; object-fit: cover;"/>
</div>

1. **Mars of Destruction (파괴된 마스)**  
2. **Pupa (퓨파)**  
3. **School Days (스쿨 데이즈)**  
4. **Vampire Holmes (뱀파이어 홈즈)**  
5. **Boku no Pico (보쿠 노 피코)**  

---

## 💻 주요 코드 설명

| 파일명 | 설명 |
|--------|------|
| anime.py | 멀티프로세싱 기반으로 74개 CSV 파일을 번역하고 통합했다. |
| finetune_mobilebert_anime.py | 학습 입력, 전처리, 학습, 저장까지 자동화했다. |
| inference_mobilebert_anime.py | 전체 데이터셋에 감정 예측을 수행하고 시각화용 통계를 생성했다. |

---

## 🔚 결론 및 향후 계획

- 번역 품질과 전처리가 NLP 모델 성능에 큰 영향을 주었다.  
- 중립 감정은 해석이 모호하여 이진 분류의 성능이 더 우수했다.  
- 정확도 85% 이상을 달성하여 실무 적용 가능성을 확인했다.

### 🔮 향후 계획

- **RuBERT** 기반 러시아어 직접 감정 분석 모델을 실험할 예정이다.  
- 추천 시스템 및 감정 기반 큐레이션에 적용할 계획이다.  
- 장르별/연도별 감정 트렌드 분석을 수행할 예정이다.  

---

## 🛠️ 개발 환경 및 라이브러리

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

📢 리뷰 기반 감정 분석은 사용자 만족도와 직결된다.  
🎯 AI 기반 감정 분석 시스템은 이제 선택이 아닌 필수이다.
