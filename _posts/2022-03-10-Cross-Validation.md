---
title:  "8.9 Cross Validation"
categories: 
  - ML_practice
tags:
  - machine learning
  - ML
  - Cross Validation
  - KFold
  - Hyperparameter tuning
toc: true
toc_sticky: true
toc_label: "Cross Validation"
toc_icon: "blog"
---



# Cross Validation

## 1) 목적
- Hyperparameter 튜닝

## 2) 데이터 불러오기


```python
from sklearn import datasets
raw_wine = datasets.load_wine()

```

# 3) feature, target 데이터 지정



```python
X = raw_wine.data
y = raw_wine.target
```

# 4) train / test 데이터 분할


```python
from sklearn.model_selection import train_test_split
X_tn, X_te, y_tn, y_te = train_test_split(X,y,random_state=0)

```

# 5) 데이터 표준화


```python
from sklearn.preprocessing import StandardScaler
std_scale = StandardScaler()
std_scale.fit(X_tn)
X_tn_std = std_scale.transform(X_tn)
X_te_std = std_scale.transform(X_te)
```

# 6) Grid Search


```python
from sklearn import svm 
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV

param_grid = {'kernel' : ('linear', 'rbf'), 
             'C' : [0.5, 1, 10, 100]}
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
svc = svm.SVC(random_state=0)
grid_cv = GridSearchCV(svc, param_grid, cv=kfold, scoring='accuracy')
grid_cv.fit(X_tn_std, y_tn)
```




    GridSearchCV(cv=StratifiedKFold(n_splits=5, random_state=0, shuffle=True),
                 estimator=SVC(random_state=0),
                 param_grid={'C': [0.5, 1, 10, 100], 'kernel': ('linear', 'rbf')},
                 scoring='accuracy')



- 2: stratified k-fold cross validation은 일반적인 k-fold cross validation과는 달리 라벨링의 비율을 유지하면서 데이터를 추출하는 방법
- 3: GridSearch를 위해 GridSearchCV 함수를 불러옴 
- 4: grid search를 위해 parameter를 정한다. SVM에서 커널은 linear 또는 rbf로 설정. C 값은 0.5, 1, 10, 100 으로 설정
- 5: n_splits=5는 트레이닝 데이터를 5개의 split으로 나눈다라는 것, shuffle은 데이터를 섞는다는 의미
- 6: 학습시킬 모형을 SVM을 기본형으로 다룬다.
- 7: 학습시킬 모형 svc와 파라미터 param_grid, 크로스 벨리데이션 방법 kfold, 모형 평가 방법을 설정
- 8: 표준화된 피처 데이터와 트레이닝 타깃 데이터를 넣고 적합시킨다. 

# 7) Grid Search 결과 확인


```python
grid_cv.cv_results_
```




    {'mean_fit_time': array([0.00101571, 0.00106258, 0.00085115, 0.00109282, 0.0007503 ,
            0.00087047, 0.00070262, 0.00082722]),
     'std_fit_time': array([1.62061749e-04, 2.20231669e-05, 4.01492923e-05, 1.69207370e-04,
            2.22167229e-05, 1.88643192e-05, 1.72882242e-05, 1.32990559e-05]),
     'mean_score_time': array([0.00031829, 0.00031857, 0.00028024, 0.0003047 , 0.00024934,
            0.00026579, 0.00023694, 0.00025845]),
     'std_score_time': array([2.49586790e-05, 3.33922222e-06, 5.75493263e-06, 1.77508997e-05,
            3.91297116e-06, 7.18174384e-06, 5.86218878e-06, 1.09121764e-05]),
     'param_C': masked_array(data=[0.5, 0.5, 1, 1, 10, 10, 100, 100],
                  mask=[False, False, False, False, False, False, False, False],
            fill_value='?',
                 dtype=object),
     'param_kernel': masked_array(data=['linear', 'rbf', 'linear', 'rbf', 'linear', 'rbf',
                        'linear', 'rbf'],
                  mask=[False, False, False, False, False, False, False, False],
            fill_value='?',
                 dtype=object),
     'params': [{'C': 0.5, 'kernel': 'linear'},
      {'C': 0.5, 'kernel': 'rbf'},
      {'C': 1, 'kernel': 'linear'},
      {'C': 1, 'kernel': 'rbf'},
      {'C': 10, 'kernel': 'linear'},
      {'C': 10, 'kernel': 'rbf'},
      {'C': 100, 'kernel': 'linear'},
      {'C': 100, 'kernel': 'rbf'}],
     'split0_test_score': array([0.88888889, 0.96296296, 0.88888889, 0.92592593, 0.88888889,
            0.92592593, 0.88888889, 0.92592593]),
     'split1_test_score': array([0.96296296, 1.        , 0.96296296, 0.96296296, 0.96296296,
            0.96296296, 0.96296296, 0.96296296]),
     'split2_test_score': array([0.92592593, 0.96296296, 0.92592593, 0.96296296, 0.92592593,
            0.96296296, 0.92592593, 0.96296296]),
     'split3_test_score': array([1.        , 0.96153846, 1.        , 0.96153846, 1.        ,
            0.96153846, 1.        , 0.96153846]),
     'split4_test_score': array([0.84615385, 1.        , 0.84615385, 1.        , 0.84615385,
            1.        , 0.84615385, 1.        ]),
     'mean_test_score': array([0.92478632, 0.97749288, 0.92478632, 0.96267806, 0.92478632,
            0.96267806, 0.92478632, 0.96267806]),
     'std_test_score': array([0.05401397, 0.01838435, 0.05401397, 0.02343121, 0.05401397,
            0.02343121, 0.05401397, 0.02343121]),
     'rank_test_score': array([5, 1, 5, 2, 5, 2, 5, 2], dtype=int32)}



- 결과적으로는, 두번째 진행된 fit, 학습 정확도가 제일 높기에 두번째 학습 때의 hyperparameter가 사용될 것

# 8) Grid Search 결과 시각적 확인(데이터 프레임)


```python
import numpy as np
import pandas as pd
np.transpose(pd.DataFrame(grid_cv.cv_results_))
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>mean_fit_time</th>
      <td>0.00101571</td>
      <td>0.00106258</td>
      <td>0.000851154</td>
      <td>0.00109282</td>
      <td>0.000750303</td>
      <td>0.000870466</td>
      <td>0.00070262</td>
      <td>0.000827217</td>
    </tr>
    <tr>
      <th>std_fit_time</th>
      <td>0.000162062</td>
      <td>2.20232e-05</td>
      <td>4.01493e-05</td>
      <td>0.000169207</td>
      <td>2.22167e-05</td>
      <td>1.88643e-05</td>
      <td>1.72882e-05</td>
      <td>1.32991e-05</td>
    </tr>
    <tr>
      <th>mean_score_time</th>
      <td>0.000318289</td>
      <td>0.000318575</td>
      <td>0.000280237</td>
      <td>0.000304699</td>
      <td>0.000249338</td>
      <td>0.000265789</td>
      <td>0.00023694</td>
      <td>0.000258446</td>
    </tr>
    <tr>
      <th>std_score_time</th>
      <td>2.49587e-05</td>
      <td>3.33922e-06</td>
      <td>5.75493e-06</td>
      <td>1.77509e-05</td>
      <td>3.91297e-06</td>
      <td>7.18174e-06</td>
      <td>5.86219e-06</td>
      <td>1.09122e-05</td>
    </tr>
    <tr>
      <th>param_C</th>
      <td>0.5</td>
      <td>0.5</td>
      <td>1</td>
      <td>1</td>
      <td>10</td>
      <td>10</td>
      <td>100</td>
      <td>100</td>
    </tr>
    <tr>
      <th>param_kernel</th>
      <td>linear</td>
      <td>rbf</td>
      <td>linear</td>
      <td>rbf</td>
      <td>linear</td>
      <td>rbf</td>
      <td>linear</td>
      <td>rbf</td>
    </tr>
    <tr>
      <th>params</th>
      <td>{'C': 0.5, 'kernel': 'linear'}</td>
      <td>{'C': 0.5, 'kernel': 'rbf'}</td>
      <td>{'C': 1, 'kernel': 'linear'}</td>
      <td>{'C': 1, 'kernel': 'rbf'}</td>
      <td>{'C': 10, 'kernel': 'linear'}</td>
      <td>{'C': 10, 'kernel': 'rbf'}</td>
      <td>{'C': 100, 'kernel': 'linear'}</td>
      <td>{'C': 100, 'kernel': 'rbf'}</td>
    </tr>
    <tr>
      <th>split0_test_score</th>
      <td>0.888889</td>
      <td>0.962963</td>
      <td>0.888889</td>
      <td>0.925926</td>
      <td>0.888889</td>
      <td>0.925926</td>
      <td>0.888889</td>
      <td>0.925926</td>
    </tr>
    <tr>
      <th>split1_test_score</th>
      <td>0.962963</td>
      <td>1</td>
      <td>0.962963</td>
      <td>0.962963</td>
      <td>0.962963</td>
      <td>0.962963</td>
      <td>0.962963</td>
      <td>0.962963</td>
    </tr>
    <tr>
      <th>split2_test_score</th>
      <td>0.925926</td>
      <td>0.962963</td>
      <td>0.925926</td>
      <td>0.962963</td>
      <td>0.925926</td>
      <td>0.962963</td>
      <td>0.925926</td>
      <td>0.962963</td>
    </tr>
    <tr>
      <th>split3_test_score</th>
      <td>1</td>
      <td>0.961538</td>
      <td>1</td>
      <td>0.961538</td>
      <td>1</td>
      <td>0.961538</td>
      <td>1</td>
      <td>0.961538</td>
    </tr>
    <tr>
      <th>split4_test_score</th>
      <td>0.846154</td>
      <td>1</td>
      <td>0.846154</td>
      <td>1</td>
      <td>0.846154</td>
      <td>1</td>
      <td>0.846154</td>
      <td>1</td>
    </tr>
    <tr>
      <th>mean_test_score</th>
      <td>0.924786</td>
      <td>0.977493</td>
      <td>0.924786</td>
      <td>0.962678</td>
      <td>0.924786</td>
      <td>0.962678</td>
      <td>0.924786</td>
      <td>0.962678</td>
    </tr>
    <tr>
      <th>std_test_score</th>
      <td>0.054014</td>
      <td>0.0183843</td>
      <td>0.054014</td>
      <td>0.0234312</td>
      <td>0.054014</td>
      <td>0.0234312</td>
      <td>0.054014</td>
      <td>0.0234312</td>
    </tr>
    <tr>
      <th>rank_test_score</th>
      <td>5</td>
      <td>1</td>
      <td>5</td>
      <td>2</td>
      <td>5</td>
      <td>2</td>
      <td>5</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>



# 9) Best Score & Hyperparameter


```python
grid_cv.best_score_
```




    0.9774928774928775




```python
grid_cv.best_params_
```




    {'C': 0.5, 'kernel': 'rbf'}



# 10) 최종 모형


```python
# grid search, cross-validation 후 가장 좋은 hyperparameter로 clf 설정
clf = grid_cv.best_estimator_
print(clf)
```

    SVC(C=0.5, random_state=0)


# 11) Cross-validation 스코어 확인(1)



```python
from sklearn.model_selection import cross_validate
metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
cv_scores = cross_validate(clf,X_tn_std,y_tn,cv=kfold,scoring=metrics)
cv_scores
```




    {'fit_time': array([0.00132918, 0.00131297, 0.00119996, 0.00112677, 0.00101924]),
     'score_time': array([0.002707  , 0.00217199, 0.00220799, 0.00206614, 0.00182986]),
     'test_accuracy': array([0.96296296, 1.        , 0.96296296, 0.96153846, 1.        ]),
     'test_precision_macro': array([0.96296296, 1.        , 0.96969697, 0.96969697, 1.        ]),
     'test_recall_macro': array([0.96666667, 1.        , 0.96296296, 0.95833333, 1.        ]),
     'test_f1_macro': array([0.9628483 , 1.        , 0.96451914, 0.96190476, 1.        ])}



# 12) Cross-validation 스코어 확인(2)



```python
from sklearn.model_selection import cross_val_score
cv_score = cross_val_score(clf, X_tn_std, y_tn, cv=kfold, scoring='accuracy')
# split 별 score..!
print(cv_score)
```

    [0.96296296 1.         0.96296296 0.96153846 1.        ]


# 13) 예측


```python
pred_svm = clf.predict(X_te_std)
print(pred_svm)

```

    [0 2 1 0 1 1 0 2 1 1 2 2 0 1 2 1 0 0 1 0 1 0 0 1 1 1 1 1 1 2 0 0 1 0 0 0 2
     1 1 2 0 0 1 1 1]


# 14) Grid Search로 고른 Hyperparameter를 적용한 모델 평가 

## 1. accuracy


```python
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_te, pred_svm)
print(accuracy)
```

    1.0


## 2. confusion matrix


```python
from sklearn.metrics import confusion_matrix
conf = confusion_matrix(y_te, pred_svm)
print(conf)
```

    [[16  0  0]
     [ 0 21  0]
     [ 0  0  8]]


## 3. classification report



```python
from sklearn.metrics import classification_report
cls_rpt = classification_report(y_te, pred_svm)
print(cls_rpt)
```

                  precision    recall  f1-score   support
    
               0       1.00      1.00      1.00        16
               1       1.00      1.00      1.00        21
               2       1.00      1.00      1.00         8
    
        accuracy                           1.00        45
       macro avg       1.00      1.00      1.00        45
    weighted avg       1.00      1.00      1.00        45
    

