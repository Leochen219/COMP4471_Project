import gradio as gr
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPTokenizer

from configs import load_config
from inference import load_model, encode_texts
from data.transforms import get_val_transform

# ==========================================
# 1. 全局初始化：加载配置和模型 (只在启动时加载一次)
# ==========================================
print("正在初始化 Web 界面和加载模型...")
cfg = load_config("configs/demo.yaml")
device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
checkpoint_path = "checkpoints/demo_cliptext/best.pt"  # 确保你已经运行过训练并生成了这个文件

# 加载模型、转换器和分词器
model = load_model(cfg, checkpoint_path, device)
transform = get_val_transform(cfg.image_size)
tokenizer = CLIPTokenizer.from_pretrained(
    getattr(cfg, "text_encoder_name", "openai/clip-vit-base-patch32")
)

# ==========================================
# 2. 核心推理函数 (适配 Gradio 的输入输出)
# ==========================================
def match_image_texts(image, text_input):
    """
    接收用户上传的图片 (PIL 对象) 和 多行文本，返回每个文本的契合度得分。
    """
    if image is None:
        return {"请先上传一张图片!": 0.0}
    if not text_input.strip():
        return {"请输入至少一句候选文案!": 0.0}

    # 将输入的字符串按行分割成列表，并去除空行
    candidate_texts = [t.strip() for t in text_input.split('\n') if t.strip()]
    
    # 编码图像 (针对 PIL Image 直接处理，而不是读取路径)
    image_tensor = transform(image).convert("RGB") if image.mode != "RGB" else transform(image)
    image_tensor = image_tensor.unsqueeze(0).to(device)
    
    with torch.no_grad():
        image_embed = model.encode_image(image_tensor)  # [1, D]
        # 复用 inference.py 中的文本编码函数
        text_embeds = encode_texts(model, candidate_texts, tokenizer, cfg.text_max_length, device)  # [N, D]
        
        # 计算余弦相似度并放缩 (类似 CLIP 的 logit scale)，这里简单用 softmax 转化为直观的概率分布
        similarities = (image_embed @ text_embeds.t()).squeeze(0)  # [N]
        # 使用 Softmax 将相似度转化为 0~1 之间的"契合度概率"，总和为 1，更适合页面展示
        probs = F.softmax(similarities, dim=0) 
        
    # 组合成 Gradio Label 组件需要的字典格式: {"文本": 分数}
    results = {text: float(prob) for text, prob in zip(candidate_texts, probs.tolist())}
    return results

# ==========================================
# 3. 构建网页 UI (Gradio Blocks)
# ==========================================
with gr.Blocks(title="图文契合度测试引擎") as demo:
    gr.Markdown("# 🖼️ 朋友圈文案契合度测试 (Zero-Shot Vision-Language Alignment)")
    gr.Markdown("上传一张你的照片，然后输入几句你想发的文案（每行一句）。模型会自动帮你测算哪句文案和照片的意境最匹配！")
    
    with gr.Row():
        with gr.Column():
            # 左侧：输入区
            img_input = gr.Image(type="pil", label="📸 上传照片")
            # 预设一些有趣的默认文案供测试
            default_texts = "A magnificent view of the Colosseum in Rome\nEnjoying a cozy hot chocolate community event\nA sleek computer science workstation with dual monitors\nA delicious plate of Sichuan-style cuisine\nA photo of a cute dog resting"
            txt_input = gr.Textbox(
                lines=6, 
                label="✍️ 候选文案 (每行一句)", 
                placeholder="在此输入多行文案...",
                value=default_texts
            )
            submit_btn = gr.Button("🚀 计算契合度", variant="primary")
            
        with gr.Column():
            # 右侧：输出区
            output_label = gr.Label(label="✨ 最佳匹配文案榜单")

    # 绑定按钮点击事件到推理函数
    submit_btn.click(
        fn=match_image_texts,
        inputs=[img_input, txt_input],
        outputs=output_label
    )

if __name__ == "__main__":
    # 启动网页服务
    demo.launch(share=False, server_port=7860)