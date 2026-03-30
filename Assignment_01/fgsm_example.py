#%%
import torch 
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import mnist_cnn
print("libraries loaded")

#%%
model = mnist_cnn.MNIST_CNN()
#%%
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 데이터 로드
    transform = transforms.Compose([
    transforms.ToTensor(), # PyTorch 텐서로 변환
    transforms.Normalize((0.1307,), (0.3081,)) # MNIST 평균 0.1307, 표준편차 0.3081로 정규화
    ])

    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    # 모델 로드
    model = MNIST_CNN()
    





cnn = MNIST_CNN()

files = glob.glob("./training_result/mnist*.pt")
latest_file = max(files, key = lambda x: os.path.basename(x).replace("mnist_", ""))
print(f"Latest file: {latest_file}")

cnn.load_state_dict(torch.load(latest_file, map_location = device))
test_acc = evaluate_model(cnn, test_loader, device = device)

print("Test accuracy: ", test_acc)