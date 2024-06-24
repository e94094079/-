#!/usr/bin/env python
# coding: utf-8

# In[2]:


import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torch import nn, optim
import os
from PIL import Image
import numpy as np
from tqdm import tqdm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")#檢查所使用的是GPU還是CPU
category_to_id = {"glass": 0, "paper": 1, "cardboard": 2, "plastic": 3, "metal": 4, "trash": 5}#將不同類別的垃圾進行分類
id_to_category = {v: k for k, v in category_to_id.items()}  
num_classes = len(category_to_id)# 獲取類別數量
print(f"Number of classes: {num_classes}")

#自定義的class用來載入垃圾分類的數據
class GarbageDataset(Dataset):
    def __init__(self, root_dir, transform=None, prop=1.0):
        self.root_dir = root_dir
        self.transform = transform
         # 獲取所有類別的資料夾名稱
        self.classes = os.listdir(root_dir)
         # 計算每個類別中的圖片數量，並且乘上prop來決定使用多少
        self.num_images_per_class = {category: int(len(os.listdir(os.path.join(root_dir, category))) * prop) for category in self.classes}
        #計算圖片總量
        self.total_num_images = sum(self.num_images_per_class.values())
        for category, num_images in self.num_images_per_class.items():
            print(f"Class {category}: {num_images} images")

    def __len__(self):
        # 返回資料集的總長度
        return self.total_num_images

    def __getitem__(self, idx):#獲取圖像和標籤
        cumulative_count = 0
        for category in self.classes:
            num_files = self.num_images_per_class[category]
            # 檢查當前索引是否在該類別的範圍內
            if idx < cumulative_count + num_files:
                file_idx = idx - cumulative_count
                # 獲取圖片路徑
                img_name = os.path.join(self.root_dir, category, os.listdir(os.path.join(self.root_dir, category))[file_idx])
                image = Image.open(img_name)
                if self.transform:
                    image = self.transform(image)
                label = category_to_id[category]
                return image, label
            cumulative_count += num_files
        raise IndexError("Index out of range")

# 定義訓練資料的圖片轉換
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),  # 調整圖片大小到224x224
    transforms.RandomHorizontalFlip(),  # 隨機翻轉圖片
    transforms.ToTensor(),  
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 標準化圖片
])

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.ToTensor(),  
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 標準化圖片
])

# 設定資料比例，1.0表示使用全部資料
prop = 1.0  
# 創建訓練的資料集
train_dataset = GarbageDataset(root_dir="D:/archive/Garbage classification/Garbage classification", transform=transform_train, prop=prop)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# 創建驗證資料集
val_dataset = GarbageDataset(root_dir="D:/archive/Garbage classification/Garbage classification", transform=transform_test)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# 載入訓練的模型
model = torchvision.models.mobilenet_v2(pretrained=True)
num_features = model.classifier[1].in_features
# 修改模型的分類器層以適應新的分類數量
model.classifier[1] = nn.Linear(num_features, num_classes)  
print(f"Model output features: {num_classes}")

# 將模型移動到GPU或CPU
model = model.to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


num_epochs = 50
# 使用tqdm顯示訓練進度
for epoch in tqdm(range(num_epochs)):
    total_train_loss = 0.0
    total_train_correct = 0
    total_train_samples = 0
    model.train()  # 設置模型為訓練模式
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        print(f"Output shape: {output.shape}, Target shape: {target.shape}")
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()
        # 計算預測正確的樣本數量
        _, predicted = torch.max(output, 1)
        total_train_correct += (predicted == target).sum().item()
        total_train_samples += target.size(0)
        
        # DEBUG，每10個print一次
        if batch_idx % 10 == 0:  
            print(f'Train Batch {batch_idx}: data shape: {data.shape}, target shape: {target.shape}, output shape: {output.shape}')
            print(f'Predicted: {predicted}, Target: {target}')
    # 計算準確率        
    avg_train_loss = total_train_loss / len(train_loader)
    train_accuracy = total_train_correct / total_train_samples
    print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {avg_train_loss}, Training Accuracy: {train_accuracy}")
    
    # 驗證模型
    model.eval()  
    with torch.no_grad():
        total_val_loss = 0.0
        total_val_correct = 0
        total_val_samples = 0
        for batch_idx, (data, target) in enumerate(val_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            total_val_loss += loss.item()
            # 計算預測正確的樣本數量
            _, predicted = torch.max(output, 1)
            total_val_correct += (predicted == target).sum().item()
            total_val_samples += target.size(0)

            # DEBUG，每10個print一次
            if batch_idx % 10 == 0:  
                print(f'Validation Batch {batch_idx}: data shape: {data.shape}, target shape: {target.shape}, output shape: {output.shape}')
                print(f'Predicted: {predicted}, Target: {target}')

    avg_val_loss = total_val_loss / len(val_loader)
    val_accuracy = total_val_correct / total_val_samples
    print(f"Epoch {epoch + 1}/{num_epochs}, Validation Loss: {avg_val_loss}, Validation Accuracy: {val_accuracy}")

    # 每5輪保存一次模型
    if (epoch + 1) % 5 == 0:
        torch.save(model.state_dict(), f'model_{epoch}.pt')


# In[ ]:


import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
# 定義一個函數，用於處理圖像分類
def classify_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        image = Image.open(file_path)
        # 調整圖像大小，讓最大不超過400像素
        image.thumbnail((400, 400))  
        img_tk = ImageTk.PhotoImage(image)
        # 在圖像顯示區域顯示選擇的圖像
        panel.config(image=img_tk)
        panel.image = img_tk


        image_tensor = transform_test(image).unsqueeze(0).to(device)
        

        model.eval()
        with torch.no_grad():
            output = model(image_tensor)
        
        _, predicted = torch.max(output, 1)
        result_id = predicted.item()
        # 根據類別ID查找對應的分類名稱
        result = id_to_category[result_id]
        
        result_label.config(text=f"Result: {result}")

# 創建主應用程式窗口
root = tk.Tk()
root.title("Garbage Classifier")
# 創建一個標籤，用於顯示圖像
panel = tk.Label(root)
panel.pack()
# 創建一個按鈕，點擊後會打開文件選擇對話框
btn = tk.Button(root, text="Select Image", command=classify_image)
btn.pack()
# 創建一個標籤，用於顯示分類結果
result_label = tk.Label(root, text="Result:")
result_label.pack()

root.mainloop()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




