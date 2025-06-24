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
📦 러시아어 리뷰 번역 프로세스 상세 보고
애니 리뷰 원문은 전부 러시아어였고, 문장 구조가 비정형이거나 은어·오타가 많아 번역 품질 저하 가능성이 있었다.
이를 해결하기 위해 다음과 같은 구조로 번역 파이프라인을 설계했다.

🔄 사용 도구 및 구현 방식
항목	내용
사용 번역기	GoogleTranslator (via deep_translator)
번역 방향	러시아어 → 영어 (ru → en)
병렬 처리 방식	multiprocessing.Pool로 최대 8코어 활용
적용 함수	translate_parallel() 함수로 일괄 번역
예외 처리	실패 시 원문 유지하여 전체 중단 방지

python
복사
편집
# 병렬 번역 함수 예시
def translate_text(text):
    try:
        return GoogleTranslator(source='ru', target='en').translate(text)
    except Exception:
        return text  # 번역 실패 시 원문 유지
멀티프로세싱으로 병렬 번역을 처리하여 속도를 크게 향상시켰고, 예외 처리로 안정성도 확보했다.

⏱️ 처리 성능 및 속도
전체 번역량: 약 76,000건

번역 완료 수: 약 73,000건

총 소요 시간: 약 48시간

평균 번역 속도: 시간당 약 1,600 ~ 1,800건

병렬 처리 효과: 단일 프로세스 대비 약 5~6배 향상

Google Translator API는 쿼터 제한이 있어 딜레이가 발생할 수 있는데, 자동 재시도 구조를 활용하면 보다 안정적인 처리가 가능하다.

📂 결과 저장 방식
저장 파일: Anime_reviews_EN.csv

추가 컬럼: Translated_Text (원본 text 컬럼도 함께 유지)

인코딩: UTF-8

원문과 번역문을 동시에 보존함으로써, 향후 러시아어 기반 모델 학습에도 활용할 수 있도록 설계했다.

🎯 번역 품질 평가 요약
항목	내용
문장 품질	대부분 자연스럽지만 일부 관용어/은어는 왜곡될 수 있음
감정 보존률	긍정/부정 표현은 잘 전달됨. 중립은 다소 모호하게 번역되기도 함
오류율 추정	약 5~8% 문장은 의미가 왜곡되거나 흐릿함
향후 계획	러시아어 직접 분류 가능한 RuBERT 기반 모델 실험 예정

📝 정리하면,
멀티프로세싱 구조 덕분에 번역 시간을 최소 80% 이상 단축했고, 감정 뉘앙스를 최대한 보존하는 방식으로 파이프라인을 구성했다.
이후 MobileBERT 파인튜닝 성능 향상에 기여한 핵심 요소 중 하나였다.

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
### ✅ 긍정 비율 기준 Top 5 애니 1~5 순서대로 시각화<a href='https://ifh.cc/v-gzBAFk' target='_blank'><img src='https://ifh.cc/g/gzBAFk.png' border='0'></a>   ### ⚠️ 부정 비율 기준 Top 5 애니  <a href='https://ifh.cc/v-aS7mA2' target='_blank'><img src='https://ifh.cc/g/aS7mA2.png' border='0'></a>

---
### ✅ 긍정 비율 기준 Top 5 애니 설명


| 애니메이션                             | 설명                                           |
| --------------------------------- | -------------------------------------------- |
| **Your Name (너의 이름은)**            | 운명처럼 몸이 바뀐 두 소년소녀의 시간과 기억을 넘나드는 로맨스 판타지.     |
| **Spirited Away (센과 치히로의 행방불명)**  | 신비한 온천 마을에 갇힌 소녀의 성장과 모험을 그린 스튜디오 지브리 대표작.   |
| **Attack on Titan (진격의 거인)**      | 거인과 인류의 생존 전쟁 속에서 밝혀지는 진실과 자유의 의미를 다룬 액션 대작. |
| **Violet Evergarden (바이올렛 에버가든)** | 전쟁 후 감정을 배우며 편지를 쓰는 소녀의 치유와 성장 이야기.          |
| **Mob Psycho 100 (모브 사이코 100)**   | 막강한 초능력을 가진 소년이 평범함을 추구하며 성장하는 코미디 액션물.      |


### ⚠️ 부정 비율 기준 Top 5 애니 설명


| 애니메이션                            | 설명                                     |
| -------------------------------- | -------------------------------------- |
| **Mars of Destruction (파괴된 마스)** | 연출과 작화, 스토리 전개 모두에서 혹평을 받은 전설적인 B급 애니. |
| **Pupa (퓨파)**                    | 남매 간의 기괴한 관계와 식인을 소재로 한 고어 단편 애니메이션.   |
| **School Days (스쿨 데이즈)**         | 삼각관계를 극단적으로 묘사한 충격적 결말의 심리 스릴러.        |
| **Vampire Holmes (뱀파이어 홈즈)**     | 추리 요소 없이 혼란스러운 전개로 악평을 받은 패러디 애니.      |
| **Boku no Pico (보쿠 노 피코)**       | 아동과 성인을 둘러싼 논란으로 인터넷 밈이 된 문제작.         |



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
