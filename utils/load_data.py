def load_and_prepare_data(scores_path, test_size=0.2, batch_size=16, random_seed=42, visualize=False):
    """
    점수 데이터를 로드하고 전처리한 후 데이터로더를 생성합니다.
    
    Args:
        scores_path (str or Path): 점수 데이터가 저장된 경로
        test_size (float): 테스트 데이터 비율
        batch_size (int): 배치 크기
        random_seed (int): 랜덤 시드
        
    Returns:
        dict: 데이터셋, 데이터로더, 메타데이터를 포함하는 딕셔너리
    """
    from pathlib import Path
    import numpy as np
    import torch
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt
    
    # 경로 변환
    save_path = Path(scores_path)
    
    # 결과 및 점수 로드
    results = np.load(save_path / 'results.npy')
    brightness_scores = np.load(save_path / 'brightness_scores.npy')
    noisiness_scores = np.load(save_path / 'noisiness_scores.npy')
    quality_scores = np.load(save_path / 'quality_scores.npy')
    
    # 데이터 형태 확인 및 재구성
    unique_img_indices = np.unique(brightness_scores[:, 0]).astype(int)
    num_images = len(unique_img_indices)
    unique_T_values = np.unique(brightness_scores[:, 1])
    num_T = len(unique_T_values)
    
    print(f"총 이미지 수: {num_images}")
    print(f"T 값의 개수: {num_T}")
    print(f"총 데이터 포인트: {len(brightness_scores)}")
    print(f"T 값 범위: {unique_T_values.min()} ~ {unique_T_values.max()}")
    
    # 데이터 재구성 (N, T, 3) 형태로
    X = np.zeros((num_images, num_T, 3))
    
    # 각 이미지, 각 T값에 대한 점수 할당
    for i in range(len(brightness_scores)):
        img_idx = int(brightness_scores[i, 0])
        img_pos = np.where(unique_img_indices == img_idx)[0][0]
        t_val = brightness_scores[i, 1]
        t_idx = np.where(unique_T_values == t_val)[0][0]
        
        X[img_pos, t_idx, 0] = brightness_scores[i, 2]  # 밝기 점수
        X[img_pos, t_idx, 1] = noisiness_scores[i, 2]   # 잡음 점수
        X[img_pos, t_idx, 2] = quality_scores[i, 2]     # 품질 점수
    
    # 라벨 데이터
    y = results[:, 0]
    
    # T값 분포 분석 및 시각화
    unique_values, counts = np.unique(y, return_counts=True)
    
    # 결과 출력
    print("\n유니크 T 값 분포:")
    # for value, count in zip(unique_values, counts):
        # print(f"T 값 {value:.2f}: {count}개")
    if visualize:
        # 시각화
        plt.figure(figsize=(12, 6))
        plt.bar(unique_values, counts)
        plt.xlabel('T 값')
        plt.ylabel('개수')
        plt.title('T 값 분포')
        plt.grid(axis='y')
        plt.xticks(unique_values)
        plt.show()
        
    # 데이터를 torch tensor로 변환
    X = torch.FloatTensor(X)
    y = torch.FloatTensor(y)
    
    # 학습 및 테스트 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_seed
    )
    
    print(f'\n학습 데이터: {X_train.shape}, {y_train.shape}')
    print(f'테스트 데이터: {X_test.shape}, {y_test.shape}')
    
    # 데이터 로더 생성
    train_loader = torch.utils.data.DataLoader(
        list(zip(X_train, y_train)), batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        list(zip(X_test, y_test)), batch_size=batch_size, shuffle=False
    )
    
    # 메타데이터 및 데이터셋/데이터로더 반환
    return {
        # 데이터셋
        'X': X,
        'y': y,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        
        # 데이터로더
        'train_loader': train_loader,
        'test_loader': test_loader,
        
        # 메타데이터
        'unique_T_values': unique_T_values,
        'unique_img_indices': unique_img_indices,
        'save_path': save_path,
        'num_images': num_images,
        'num_T': num_T,
        'input_dim': X.shape[2],  # 특성 차원
        'T_distribution': dict(zip(unique_values.tolist(), counts.tolist()))
    }