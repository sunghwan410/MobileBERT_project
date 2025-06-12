                                                  # MobileBERT를 활용한 애니 리뷰 분석 프로젝트

                                        -------------------------------------------------------------------
                      <a href='https://ifh.cc/v-8hOMNb' target='_blank'><img src='https://ifh.cc/g/8hOMNb.jpg' border='0'></a>


                                                           

                                 # 모바일버트를 활용한 애니메이션 리뷰 감정 분석Positive / Neutral / Negative



                                       ## 1. 프로젝트 개요

                                          디지털 시대에 접어들며 영화·애니메이션 리뷰는 콘텐츠의 흥행 여부를 결정하는 핵심 지표로 자리 잡았습니다. 본 프로젝트는 다양한 애니메이션 리뷰 데이터를 활용하여 긍정/중립/부정 감정 분류 모델을 구축함으로써,

                                          리뷰 감정의 경향성 분석

                                         인기 애니메이션의 평판 비교 향후 추천 시스템 또는 큐레이션 기반에 활용 가능성을 확인하는 것이 목적입니다.

            
                                         리뷰 원문은 러시아어로 제공되며, 이를 영어로 번역하고 자연어처리 가능한 형태로 전처리한 후 MobileBERT를 기반으로 fine-tuning 했습니다.

                                      ## 2. 📂 데이터셋 구축 및 번역 과정

                                          📦 원본 정보

                                              원본 파일: Anime_reviews_RU.csv

                                              총 리뷰 수: 약 76,000건

                                              주요 컬럼: anime (제목), text (리뷰), rate (감정 레이블)

                                              언어: 러시아어

                                              🌐 번역 과정: 수작업 이상의 노력

                                                항목

                                                내용

                                                분할

                                                1,000건씩 분할하여 74개 CSV 파일로 나눔

                                                번역 도구

                                                Google Translator + deep_translator 패키지 사용

                                                  병렬 처리

                                              anime.py 코드로 멀티프로세싱 병렬 번역 (최대 8코어) 수행

                                              번역 소요 시간

                                              약 3일 이상, 파일 수 만큼의 반복적 확인 필요

                                                병합 결과

                                            anime_sentiment_1.csv로 통합, 번역 누락은 수동으로 보완

                                            🔍 전처리 및 필터링

                                              Rate가 Positive/Neutral/Negative 외인 경우 제거

                                                text, anime 결측치 제거

                                              최종 데이터 수: 약 73,000건 유지

                                              3. 🎓 학습 데이터 구성 방법

                                                  대규모 리뷰를 그대로 학습에 사용하기에는 리소스가 제한되기 때문에, 다음 기준으로 학습 데이터를 구축했습니다.

기준

내용

추출 비율

전체의 약 4%, 3,000건 샘플링 (Positive/Negative 균형 유지)

분포 유지

클래스 균형 (stratify), 비율 유지한 학습/검증 분할 (80:20)

이러한 방식은 전체 데이터의 감정 분포를 반영하면서도 일반화 성능을 확보하는 데 효과적입니다.

4. 🤖 MobileBERT Fine-tuning 결과

🔧 모델 구조 및 설정

Pretrained: google/mobilebert-uncased

Input length: 256 tokens

Class 수: 3개 또는 2개 (실험에 따라)

Optimizer: AdamW (lr=2e-5)

Batch size: 8

Epochs: 4 (3진 분류) / 10 (이진 분류)

📈 학습 그래프 (3진 분류 vs 2진 분류)

(예시)

3진 분류 결과: 정확도 약 0.55 ~ 0.60 수준 유지
2진 분류 전환 후:

| Epoch | Train Loss | Val Accuracy |
|-------|------------|---------------|
|   1   | 0.6228     | 0.7733        |
|   2   | 0.3996     | 0.8400        |
|   3   | 0.2624     | **0.8533** ✅ |

🧪 Inference 수행

전체 리뷰셋에 대해 예측 수행

결과를 기반으로 애니메이션별 감정 통계 시각화

top_bottom5_anime_reviews.csv 생성 (상하위 5개 애니 선정)

5. 📊 시각화 및 감정 분석

✅ 상하위 5개 애니 감정 통계

긍정 비율 기준 상위 5개/하위 5개 애니 추출

리뷰 수 기준 필터링 포함

rate_dist = df.groupby(['anime', 'rate']).size().unstack(fill_value=0)
rate_dist['positive_ratio'] = rate_dist['Positive'] / rate_dist.sum(axis=1)
rate_dist.sort_values(by='positive_ratio')

상위 5개 애니 (긍정 비율 Top):

Your Name

Spirited Away

Attack on Titan

Violet Evergarden

Mob Psycho 100

하위 5개 애니 (긍정 비율 Bottom):

Mars of Destruction

Pupa

School Days

Vampire Holmes

Boku no Pico

📁 결과 저장: top_bottom5_anime_reviews.csv

6. 🧠 주요 코드 설명

anime.py

74개 파일 분할 번역 수행 (Multiprocessing 기반)

실패한 번역 자동 재시도

finetune_mobilebert_anime.py

입력/출력, 모델 로딩, 학습, 저장 자동화

이진 분류 실험도 동일 코드로 확장 가능

inference_mobilebert_anime.py

전체 데이터셋에 대해 감정 예측 수행

애니별 긍정/부정 비율 계산

7. 💬 결론 및 향후 계획

번역 품질과 전처리가 NLP 정확도에 직결된다는 교훈

중립 감정이 모델에게는 모호하게 작용하여 이진 분류가 더 유리함을 실험적으로 입증함

정확도 85% 이상 달성 시, 실무 시스템에도 충분히 응용 가능

🔮 향후 발전 방향

러시아어 원문 감정 모델 (RuBERT 등)과 비교

감정 분석 결과 기반 추천 시스템 제작

장르/연도별 감정 변화 트렌드 분석

8. 🛠 개발 환경 및 버전 정보

환경

버전

Python

3.9

PyTorch

1.12.1

Transformers

4.21.2

Pandas

1.4.4

NumPy

1.24.3

Scikit-learn

1.2.2

IDE

PyCharm / JupyterLab

🔗 참고 링크

MobileBERT (HuggingFace)

deep_translator

anime.py: 병렬 번역 처리 코드

시각화 도구: matplotlib, seaborn

📢 리뷰 기반 분석이 사용자 만족도와 직결된 시대,
AI 감정 분석은 필수가 되었습니다. 🎯
