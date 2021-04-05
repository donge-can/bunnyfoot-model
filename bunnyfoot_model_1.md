### BunnyFoot_Model 1차 모델 환경



### Task

비절병 True False 분류

- Input : 발바닥 사진
- Ouput : 확률값
  - 0<x<0.5 : 비절병 False
  - 0.5<= x <1 : 비절병 True 

 

### Label

- 0 : 비절병 False
- 1 : 비절병 True

 

### Data

- train_data 
  - True : 46개
  - Fales : 40개
- test_data : train_data 중 27개 랜덤 추출



###  Model 실험 환경

```python
!python --version #3.6.9
keras.__version__ #2.4.3
tf.__version__ #2.4.0
```

