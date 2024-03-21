# 평가 지표 저장 모듈
__version__ = 1.0

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score, accuracy_score, ConfusionMatrixDisplay, average_precision_score, roc_auc_score

def plot_confusion_matrix(y, pred, title=None):
    """
    Confusion Matrix 시각화 함수
    [parameter]
        y: ndarray - 정답
        pred: ndarray - 모델 예측값
        title: str - 그래프 제목
    [return]
    [exception]
    """
    
    cm = confusion_matrix(y, pred)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot(cmap='Blues')
    if title:
        plt.title(title)
    plt.show()
    
def print_metrics_classification(y, pred, title = None):
    """
    classification(분류) 결과들을 출력하는 함수
    accuracy, recall, precision, f1_score
    [parameter]
        y: ndarray - 정답
        pred: ndarray - 모델 예측값
        title: str - 그래프 제목
    [return]
    [exception]
    """
    if title:
        print(title)
    print('정확도(accuracy):', accuracy_score(y, pred))
    print('재현율/민감도(recall):', recall_score(y, pred))
    print('정밀도(precision):', precision_score(y, pred))
    print('F1-score:', f1_score(y, pred))

def print_metrics_classification2(y, pred_pos_proba, title=None):
        """
        Average Precision Score와 roc-auc score를 출력
        [parameter]
            y: ndarray - 정답의 label
            pred_pos_proba: ndarray - 모델이 추론한 positive(양성)의 확률
        """
        if title:
            print(title)
        print('Average Precision:', average_precision_score(y, pred_pos_proba))
        print('roc_auc:', roc_auc_score(y, pred_pos_proba))