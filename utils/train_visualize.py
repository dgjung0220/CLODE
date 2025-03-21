import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm
from network.clip_regressor import TtoTRegressor


def initialize_model(input_dim, hidden_dim, device, learning_rate=1e-4, loss_type='huber'):
    """
    모델, 손실 함수, 옵티마이저를 초기화합니다.
    
    Args:
        input_dim (int): 입력 차원
        hidden_dim (int): 은닉층 차원
        device (torch.device): 학습에 사용할 디바이스
        learning_rate (float): 학습률
        loss_type (str): 손실 함수 유형 ('mse', 'mae', 'huber')
        
    Returns:
        tuple: 모델, 손실 함수, 옵티마이저, 스케줄러
    """
    regressor = TtoTRegressor(input_dim, hidden_dim).to(device)
    
    # 손실 함수 설정
    if loss_type == 'mse':
        criterion_reg = nn.MSELoss()
    elif loss_type == 'mae':
        criterion_reg = nn.L1Loss()
    else:  # huber
        criterion_reg = nn.HuberLoss(delta=1.0)
    
    # 옵티마이저 및 스케줄러 설정
    optimizer_reg = optim.Adam(regressor.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_reg, mode='min', factor=0.5, patience=10, verbose=True
    )
    
    return regressor, criterion_reg, optimizer_reg, scheduler

def train_epoch(model, train_loader, criterion, optimizer, device):
    """
    한 에포크 동안 모델을 학습합니다.
    
    Args:
        model (nn.Module): 학습할 모델
        train_loader (DataLoader): 학습 데이터 로더
        criterion (nn.Module): 손실 함수
        optimizer (optim.Optimizer): 옵티마이저
        device (torch.device): 학습에 사용할 디바이스
        
    Returns:
        float: 에포크 평균 손실
    """
    model.train()
    epoch_loss = 0.0
    
    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        
        # 예측 및 손실 계산
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)
        
        # 역전파 및 최적화
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    return epoch_loss / len(train_loader)

def evaluate(model, test_loader, criterion, device):
    """
    모델을 평가합니다.
    
    Args:
        model (nn.Module): 평가할 모델
        test_loader (DataLoader): 테스트 데이터 로더
        criterion (nn.Module): 손실 함수
        device (torch.device): 평가에 사용할 디바이스
        
    Returns:
        float: 평균 손실
    """
    model.eval()
    test_loss = 0.0
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            y_pred = model(X_batch)
            test_loss += criterion(y_pred, y_batch).item()
    
    return test_loss / len(test_loader)

def train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, 
               device, num_epochs=3000):
    """
    모델 학습의 전체 과정을 실행합니다.
    
    Args:
        model (nn.Module): 학습할 모델
        train_loader (DataLoader): 학습 데이터 로더
        test_loader (DataLoader): 테스트 데이터 로더
        criterion (nn.Module): 손실 함수
        optimizer (optim.Optimizer): 옵티마이저
        scheduler: 학습률 스케줄러
        device (torch.device): 학습에 사용할 디바이스
        num_epochs (int): 학습할 에포크 수
        
    Returns:
        tuple: 학습 손실 리스트, 테스트 손실 리스트, 최적 모델 상태, 최적 에포크
    """
    train_losses = []
    test_losses = []
    best_test_loss = float('inf')
    best_model = None
    best_epoch = 0
    
    # tqdm을 전체 에포크에 대해 생성
    progress_bar = tqdm(total=num_epochs, desc="Training Progress", position=0, leave=True)
    
    for epoch in range(num_epochs):
        # 학습
        avg_train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(avg_train_loss)
        
        # 평가
        avg_test_loss = evaluate(model, test_loader, criterion, device)
        test_losses.append(avg_test_loss)
        
        # 최적 모델 저장
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            best_model = model.state_dict().copy()
            best_epoch = epoch
        
        # 스케줄러 업데이트
        scheduler.step(avg_test_loss)
        
        # 진행 표시줄 업데이트
        progress_bar.set_postfix({
            'Train Loss': f'{avg_train_loss:.6f}',
            'Test Loss': f'{avg_test_loss:.6f}',
            'Best Epoch': best_epoch
        })
        progress_bar.update(1)
        
    # tqdm 종료
    progress_bar.close()
    
    print(f"\n학습 완료! 최적 에포크: {best_epoch}, 최적 테스트 손실: {best_test_loss:.6f}")
    
    return train_losses, test_losses, best_model, best_epoch

def save_model(model_state, hidden_dim, learning_rate, best_epoch, model_path="/home/lbw/CLODE_model"):
    """
    모델을 저장합니다.
    
    Args:
        model_state: 저장할 모델의 상태 사전
        hidden_dim (int): 은닉층 차원
        learning_rate (float): 학습률
        best_epoch (int): 최적 에포크
        model_path (str): 저장할 경로
        
    Returns:
        Path: 저장된 모델의 경로
    """
    model_path = Path(model_path)
    model_path.mkdir(exist_ok=True, parents=True)

    model_name = f'att_regression_{hidden_dim}_{learning_rate}_{best_epoch}.pth'
    model_path_name = model_path / model_name

    torch.save(model_state, model_path_name)
    print(f"모델 저장 완료: {model_path_name}")
    
    return model_path_name

def plot_losses(train_losses, test_losses, save_path=None):
    """
    학습 및 테스트 손실을 시각화합니다.
    
    Args:
        train_losses (list): 학습 손실 리스트
        test_losses (list): 테스트 손실 리스트
        save_path (Path, optional): 그래프 저장 경로
    """
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Testing Loss')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()

def evaluate_and_visualize(model, X_data, y_data, model_path_name, device, title='Test Results', save_path=None):
    """
    모델을 평가하고 결과를 시각화합니다.
    
    Args:
        model (nn.Module): 평가할 모델
        X_data (torch.Tensor): 입력 데이터
        y_data (torch.Tensor): 정답 데이터
        model_path_name (Path): 모델 파일 경로
        device (torch.device): 평가에 사용할 디바이스
        title (str): 그래프 제목
        save_path (Path, optional): 그래프 저장 경로
        
    Returns:
        tuple: MAE, RMSE, 예측값, 실제값
    """
    model.eval()
    model.load_state_dict(torch.load(model_path_name))
    
    with torch.no_grad():
        X_data_device = X_data.to(device)
        y_pred = model(X_data_device).cpu().numpy()
        y_true = y_data.numpy()
        
        # 평균 절대 오차 (MAE) 계산
        mae = np.mean(np.abs(y_pred - y_true))
        # 평균 제곱근 오차 (RMSE) 계산
        rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
        
        print(f'MAE: {mae:.4f}, RMSE: {rmse:.4f}')
        
        # 예측 vs 실제 시각화
        plt.figure(figsize=(8, 8))
        plt.scatter(y_true, y_pred, alpha=0.6)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
        plt.xlabel('True T')
        plt.ylabel('Predicted T')
        plt.title(f'{title}: True vs Predicted T (MAE: {mae:.4f})')
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
        
        plt.show()
        
    return mae, rmse, y_pred, y_true
