# License: BSD
# Author: Sasank Chilamkurthy (Modified for Cat/Dog Transfer Learning, Updated for EfficientNet)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
from PIL import Image
from tempfile import TemporaryDirectory

cudnn.benchmark = True
plt.ion()  # interactive mode

# -----------------------------------------------------------------------------
# 1. 데이터 전처리 및 경로 설정
# -----------------------------------------------------------------------------
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# 데이터 경로 설정 (주어진 데이터 구조에 맞춰 조정)
data_dir = '..\\week6\\data\\hymenoptera_data'

# Dataset / DataLoader 설정
try:
    image_datasets = {
        x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
        for x in ['train', 'val']
    }
except FileNotFoundError:
    print(f"\n[오류] 데이터 폴더 '{data_dir}'를 찾을 수 없습니다. 경로를 확인하세요.")
    exit()

dataloaders = {
    x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4, shuffle=True, num_workers=4)
    for x in ['train', 'val']
}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes  # 예: ['ants', 'bees']

# -----------------------------------------------------------------------------
# 2. Device 설정
# -----------------------------------------------------------------------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")


# -----------------------------------------------------------------------------
# 3. 유틸리티 함수 정의
# -----------------------------------------------------------------------------
def imshow(inp, title=None):
    """Tensor 이미지를 시각화"""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title:
        plt.title(title)
    plt.pause(0.001)


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    """모델 학습 및 검증 루프"""
    since = time.time()
    with TemporaryDirectory() as tempdir:
        best_model_path = os.path.join(tempdir, 'best_model.pt')
        torch.save(model.state_dict(), best_model_path)
        best_acc = 0.0

        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)

            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0
                running_corrects = 0

                for inputs, labels in dataloaders[phase]:
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                if phase == 'train':
                    scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]
                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), best_model_path)

            print()

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:4f}')

        model.load_state_dict(torch.load(best_model_path))
    return model


def visualize_model(model, num_images=6):
    """여러 이미지의 예측 결과 시각화"""
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for inputs, labels in dataloaders['val']:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images // 2, 2, images_so_far)
                ax.axis('off')
                ax.set_title(f'predicted: {class_names[preds[j]]}')
                imshow(inputs.cpu().data[j])
                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
    model.train(mode=was_training)


def visualize_model_predictions(model, img_path):
    """단일 이미지 예측 결과 시각화"""
    model.eval()
    img = Image.open(img_path)
    img = data_transforms['val'](img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img)
        _, preds = torch.max(outputs, 1)
        plt.figure()
        plt.axis('off')
        plt.title(f'Predicted: {class_names[preds[0]]}')
        imshow(img.cpu().data[0])


# -----------------------------------------------------------------------------
# 4. 메인 실행
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    print("\n--- 메인 실행 시작 ---")

    # 데이터 샘플 시각화
    inputs, classes = next(iter(dataloaders['train']))
    out = torchvision.utils.make_grid(inputs)
    imshow(out, title=[class_names[x] for x in classes])

    # --------------------------------------------------------
    # (1) Finetuning - EfficientNet 전체 파라미터 학습
    # --------------------------------------------------------
    model_ft = models.efficientnet_b3(weights='IMAGENET1K_V1')
    num_ftrs = model_ft.classifier[1].in_features
    model_ft.classifier[1] = nn.Linear(num_ftrs, len(class_names))  # 2개 클래스

    model_ft = model_ft.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    print("\n[Start] Finetuning")
    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=25)
    print("[End] Finetuning")

    # --------------------------------------------------------
    # (2) Feature Extraction - EfficientNet 백본 고정
    # --------------------------------------------------------
    model_conv = models.efficientnet_b3(weights='IMAGENET1K_V1')
    for param in model_conv.parameters():
        param.requires_grad = False

    num_ftrs = model_conv.classifier[1].in_features
    model_conv.classifier[1] = nn.Linear(num_ftrs, len(class_names))
    model_conv = model_conv.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer_conv = optim.SGD(model_conv.classifier.parameters(), lr=0.001, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

    print("\n[Start] Feature Extraction")
    model_conv = train_model(model_conv, criterion, optimizer_conv, exp_lr_scheduler, num_epochs=25)
    print("[End] Feature Extraction")

    # --------------------------------------------------------
    # (3) 시각화
    # --------------------------------------------------------
    print("\n[Start] 모델 예측 시각화 (6장)")
    visualize_model(model_conv, num_images=6)
    print("[End] 모델 예측 시각화")

    print("\n[Start] 단일 이미지 예측 시각화")
    visualize_model_predictions(
        model_conv,
        img_path='..\\week6\\data\\hymenoptera_data\\val\\bees\\72100438_73de9f17af.jpg'
    )
    print("[End] 단일 이미지 예측 시각화")

    plt.ioff()
    plt.show()
