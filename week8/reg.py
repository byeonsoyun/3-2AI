import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# --- 0. 환경 및 시드 설정 ---
torch.manual_seed(42)
np.random.seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# --- 1. 데이터 로드 및 생성 (오류 해결 및 단순화) ---
# 사용자가 제공한 Age와 Premium 데이터를 DataFrame으로 직접 생성합니다.
# 실제 과제에서는 pd.read_csv('insurance.csv')를 사용해야 합니다.
data = {
    'Age': np.arange(18, 68, 1),  # 18세부터 67세까지 50개 데이터 생성
    'Premium': np.arange(10000, 36000, 520) + np.random.normal(0, 1500, 50) # 단순 선형 관계 + 노이즈
}
df = pd.DataFrame(data)

print("\n원본 데이터 처음 5줄:")
print(df.head())
print("\n데이터 타입 확인:")
print(df.dtypes)

# --- 2. 데이터 전처리 ---
# 1. 특성(X)과 타겟(y) 정의
X = df['Age'].values.reshape(-1, 1) # 단일 특성이므로 reshape(-1, 1) 필수
y = df['Premium'].values

# 2. 데이터 분할 (80:20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

print(f"\n훈련 데이터 크기: {X_train.shape[0]}개")
print(f"테스트 데이터 크기: {X_test.shape[0]}개")

# 3. 특성(X) 정규화
scaler_X = StandardScaler() # X 스케일러
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
print("✓ Age 특성 정규화 완료.")

# 4. 타겟(y) 정규화 추가 (새로운 스케일러 사용)
scaler_y = StandardScaler() # y 스케일러
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1))
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1))
print("✓ Premium 타겟 정규화 완료.")

# --- 3. PyTorch 텐서 및 DataLoader 생성 ---
# 정규화된 데이터만 사용합니다.
X_train_tensor = torch.FloatTensor(X_train_scaled)
y_train_tensor = torch.FloatTensor(y_train_scaled) # ⚠️ y_train_scaled 사용
X_test_tensor = torch.FloatTensor(X_test_scaled)
y_test_tensor = torch.FloatTensor(y_test_scaled) # ⚠️ y_test_scaled 사용

batch_size = 8
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# --- 4. 모델 정의 (RegressionModel) ---
class RegressionModel(nn.Module):
    def __init__(self, input_size, hidden_size1=32, hidden_size2=16): # 히든 사이즈를 줄였습니다.
        super(RegressionModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size2, 1)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

# 모델 초기화: input_size = 1
input_size = X_train_scaled.shape[1] 
model = RegressionModel(input_size).to(device)

# --- 5. 손실 함수 및 옵티마이저 설정 ---
criterion = nn.MSELoss()
learning_rate = 0.001
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# --- 6. 훈련 루프 ---
num_epochs = 200 # 에폭을 200으로 늘려 학습 안정성을 높입니다.
train_losses = []
test_losses = []

print("\n--- Training Start ---")
for epoch in range(num_epochs):
    # Training phase
    model.train()
    train_loss = 0.0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        predictions = model(X_batch)
        loss = criterion(predictions, y_batch)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    avg_train_loss = train_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    
    # Evaluation phase
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            test_loss += loss.item()
    
    avg_test_loss = test_loss / len(test_loader)
    test_losses.append(avg_test_loss)
    
    if (epoch + 1) % 20 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}] | Train Loss: {avg_train_loss:.2f} | Test Loss: {avg_test_loss:.2f}")

print("--- Training Complete! ---")

# --- 7. 모델 평가 및 결과 ---
model.eval()
with torch.no_grad():
    X_test_device = X_test_tensor.to(device)
    y_pred = model(X_test_device).cpu().numpy()
    y_true = y_test_tensor.numpy()

mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_true, y_pred)

print("\n==== Model Performance ====")
print(f"Mean Squared Error (MSE):  {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R² Score: {r2:.4f}")
print("===========================")

# --- 8. 시각화 (선택적) ---
plt.figure(figsize=(10, 6))
plt.scatter(y_true, y_pred, alpha=0.7, s=30)
plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 
         'r--', lw=2, label='Perfect Prediction')
plt.xlabel('Actual Premium')
plt.ylabel('Predicted Premium')
plt.title('Actual vs Predicted Premium')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()