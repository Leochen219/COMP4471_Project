import os

import json
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
from transformers import CLIPTokenizer
from torchvision import transforms
#pip install transformers Pillow torchvision first
class CleanCOCODataset(Dataset):
    def __init__(self, json_path, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        
        # 1. 初始化现成的 CLIP Tokenizer
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        
        # 2. 读取 JSON 文件
        with open(json_path, 'r') as f:
            data = json.load(f)
            
        # 3. 重新组织数据：将 caption 按 image_id 归类
        self.img_to_captions = {}
        for ann in data['annotations']:
            img_id = ann['image_id']
            caption = self._clean_text(ann['caption']) # 基础清洗
            
            if img_id not in self.img_to_captions:
                self.img_to_captions[img_id] = []
            self.img_to_captions[img_id].append(caption)
            
        # 建立文件名的映射，并过滤掉没有 caption 的图片
        self.img_data = []
        for img in data['images']:
            if img['id'] in self.img_to_captions:
                self.img_data.append({
                    'file_name': img['file_name'],
                    'id': img['id']
                })

    def _clean_text(self, text):
        """简单的文本清洗逻辑"""
        if not isinstance(text, str):
            text = str(text)
        # 去除首尾空格，转小写
        return text.strip().lower()

    def __len__(self):
        return len(self.img_data)

    def __getitem__(self, idx):
        img_info = self.img_data[idx]
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        
        # 4. 图像清洗与读取
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            # 如果图片损坏，返回一张全黑的图作为 fallback (防止程序崩溃)
            image = Image.new('RGB', (224, 224), (0, 0, 0))
            
        if self.transform:
            image = self.transform(image)
            
        # 5. 随机采样策略：从 5 句话中随机选一句
        captions = self.img_to_captions[img_info['id']]
        selected_caption = random.choice(captions)
        
        # 6. 使用现成 Tokenizer 转换文本 (自动截断/填充到 77 长度)
        text_inputs = self.tokenizer(
            selected_caption, 
            padding='max_length', 
            truncation=True, 
            max_length=77, 
            return_tensors="pt"
        )
        
        # squeeze 去掉多余的 batch 维度
        token_ids = text_inputs['input_ids'].squeeze(0)
        attention_mask = text_inputs['attention_mask'].squeeze(0)

        return image, token_ids, attention_mask
