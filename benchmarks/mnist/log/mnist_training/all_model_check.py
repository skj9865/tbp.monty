from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from load_object_model import load_object_model  # 함수가 있는 모듈
import argparse
import numpy as np


# ------------------------------------------------------------

# 1) 사전 학습 모델 불러오기 (사용자 제공 코드)
DMC_PRETRAIN_DIR = Path("")  # 필요하면 환경 변수 대신 직접 지정

# 0_0 부터 9_9 까지 모든 graph model 을 순회
for i in range(10): # 숫자 0부터 9까지
    object_name = f"{i}_{i}" # 예를 들어, "0_0", "1_1", ..., "9_9"
    
    # 각 모델을 새로운 창에 띄우기 위해 루프 내에서 figure와 axes를 생성
    fig = plt.figure(figsize=(8,8)) # 각 창의 크기 조정
    ax = fig.add_subplot(111, projection='3d')
    
    # Z축 반전 추가
    ax.invert_zaxis() 

    try:
        model = load_object_model(
            model_name="model.pt",
            object_name=object_name,
            features=("rgba","pose_vectors_flat","principal_curvatures_log"),
            checkpoint=None,
            lm_id=0,
        )

        # 2) 위치와 주곡률 방향 벡터
        points = model.pos
        v1 = model.pose_vectors_flat[:, 3:] # (N, 3) 주곡률 방향 1
        # v2 = model.pose_vectors_flat[:, 3:] # (N, 3) 주곡률 방향 2 (선택사항, 필요하면 v1과 함께 그려도 됨)

        # 3-b) 주곡률 방향 벡터 그리기
        scale = 0.005 # 화살표 길이 조절
        
        # 각 숫자 모델에 고유한 색상 부여
        # 10가지 뚜렷한 색상을 제공하는 'tab10' 컬러맵을 사용합니다.
        color = plt.cm.get_cmap('tab10', 10)(i) 
        
        ax.quiver(points[:, 0], points[:, 1], points[:, 2],
                  v1[:, 0], v1[:, 1], v1[:, 2],
                  length=scale, color=color, normalize=True,
                  label=f'Digit {i}') # 범례를 위한 라벨 추가

    except FileNotFoundError:
        print(f"Warning: Model for {object_name} not found. Skipping.")
    except Exception as e:
        print(f"Error loading or processing model for {object_name}: {e}. Skipping.")

    # 3-c) 축 설정 및 레이아웃 조정 (각 창에 대해)
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_zlabel('Z Coordinate')
    ax.set_title(f'Graph Model for Digit {i}') # 각 창의 제목 설정
    
    ax.set_box_aspect([1, 1, 1]) # 평면 객체라면 z 축 축소
    ax.autoscale_view(True,True,True) # 각 모델에 맞게 축 범위 자동 조정

    # 각 창에 범례 추가
    ax.legend(title='Digit') # 범례의 제목을 'Digit'으로 설정

    plt.tight_layout()

# 모든 Figure가 생성된 후 한 번에 표시
plt.show(block=True)