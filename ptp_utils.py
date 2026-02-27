# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np # 用于矩阵和数组计算
import torch # PyTorch深度学习框架核心库
from PIL import Image, ImageDraw, ImageFont # 图像处理库，用于生成和编辑图像
import cv2 # OpenCV，主要用于在这里往图像上写字
from typing import Optional, Union, Tuple, List, Callable, Dict # 类型提示
from IPython.display import display # 用于在Jupyter Notebook中直接显示图片
from tqdm.notebook import tqdm # 进度条工具，方便查看去噪步骤

def text_under_image(image: np.ndarray, text: str, text_color: Tuple[int, int, int] = (0, 0, 0)):
    """在图像底部添加一段文本（主要用于可视化注意力热力图时标注对应的单词）"""
    h, w, c = image.shape # 获取图像的高度、宽度和通道数
    offset = int(h * .2) # 设定底部留白的高度，约为原图高度的20%
    # 创建一张新的白色背景图，高度包含了原图加留白
    img = np.ones((h + offset, w, c), dtype=np.uint8) * 255
    font = cv2.FONT_HERSHEY_SIMPLEX # 设置OpenCV字体
    # font = ImageFont.truetype("/usr/share/fonts/truetype/noto/NotoMono-Regular.ttf", font_size) # 备用字体选项
    img[:h] = image # 将原图贴到新图的上半部分
    # 计算文本占用的像素大小，以便居中
    textsize = cv2.getTextSize(text, font, 1, 2)[0]
    # 计算文本的左上角坐标 (居中对齐)
    text_x, text_y = (w - textsize[0]) // 2, h + offset - textsize[1] // 2
    # 使用OpenCV将文本绘制到留白区域
    cv2.putText(img, text, (text_x, text_y ), font, 1, text_color, 2)
    return img # 返回带文字的图像


def view_images(images, num_rows=1, offset_ratio=0.02):
    """将多张图像拼接成一个网格(Grid)以便于同时查看对比"""
    if type(images) is list:
        num_empty = len(images) % num_rows # 如果是列表，计算最后一行缺几张图才能补齐
    elif images.ndim == 4:
        num_empty = images.shape[0] % num_rows # 如果是4维张量(Batch, H, W, C)，同样计算空缺
    else:
        images = [images] # 单张图转为列表
        num_empty = 0

    empty_images = np.ones(images[0].shape, dtype=np.uint8) * 255 # 创建全白的占位图
    # 统一转换数据类型为uint8，并在末尾补齐白图
    images = [image.astype(np.uint8) for image in images] + [empty_images] * num_empty
    num_items = len(images)

    h, w, c = images[0].shape # 获取单张图的尺寸
    offset = int(h * offset_ratio) # 计算图像之间的间距
    num_cols = num_items // num_rows # 计算列数
    # 创建一张巨大的白色背景图，尺寸刚好能放下所有带间距的子图
    image_ = np.ones((h * num_rows + offset * (num_rows - 1),
                      w * num_cols + offset * (num_cols - 1), 3), dtype=np.uint8) * 255
    # 遍历行和列，将每一张图填入对应的大图位置
    for i in range(num_rows):
        for j in range(num_cols):
            image_[i * (h + offset): i * (h + offset) + h:, j * (w + offset): j * (w + offset) + w] = images[
                i * num_cols + j]

    pil_img = Image.fromarray(image_) # 转回PIL格式
    display(pil_img) # 在Notebook中显示


def diffusion_step(model, controller, latents, context, t, guidance_scale, low_resource=False):
    """执行SD模型单步的去噪计算 (包含 Classifier-Free Guidance)"""
    if low_resource:
        # 显存优化模式：将 无条件(uncond) 和 有条件(text) 分两次过UNet，虽然慢但省显存
        noise_pred_uncond = model.unet(latents, t, encoder_hidden_states=context[0])["sample"]
        noise_prediction_text = model.unet(latents, t, encoder_hidden_states=context[1])["sample"]
    else:
        # 正常模式：把latents复制两份(对应无条件和有条件)，拼成Batch一起过UNet
        latents_input = torch.cat([latents] * 2)
        noise_pred = model.unet(latents_input, t, encoder_hidden_states=context)["sample"]
        # 将输出切分成 无条件 和 有条件 两部分
        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
        
    # 执行 Classifier-Free Guidance (CFG) 公式： 最终噪声 = 无条件噪声 + scale * (条件噪声 - 无条件噪声)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
    # 调用调度器(Scheduler)根据预测的噪声计算上一步(更清晰)的隐变量
    latents = model.scheduler.step(noise_pred, t, latents)["prev_sample"]
    # 【核心！】调用控制器的 step_callback。如果开启了LocalBlend局部融合，这步会将背景特征替换回原图
    latents = controller.step_callback(latents)
    return latents


def latent2image(vae, latents):
    """将潜空间变量(Latents)通过VAE解码回像素空间的高清图像"""
    latents = 1 / 0.18215 * latents # 缩放因子。SD模型在编码时除以了这个常数以保证方差，解码时需乘回来
    image = vae.decode(latents)['sample'] # 调用VAE的解码器
    image = (image / 2 + 0.5).clamp(0, 1) # 从 [-1, 1] 的数值范围线性映射并截断到 [0, 1]
    image = image.cpu().permute(0, 2, 3, 1).numpy() # 把张量挪到CPU，从 (Batch, Channel, H, W) 转为 (Batch, H, W, Channel)，并转为NumPy格式
    image = (image * 255).astype(np.uint8) # 缩放到 [0, 255] 以符合RGB图像标准
    return image


def init_latent(latent, model, height, width, generator, batch_size):
    """初始化纯噪声隐变量。如果是编辑任务，通常传入和原图相同种子的纯噪声。"""
    if latent is None: # 如果没传初始噪声
        latent = torch.randn(
            # SD 1.5/2.0的隐变量空间比像素空间缩小了8倍，所以宽长需除以8
            (1, model.unet.config.in_channels, height // 8, width // 8),
            generator=generator, # 指定随机数生成器以保证可复现性
        )
    # 将同一份噪声扩展(复制)到Batch size的大小（所有编辑后的图与原图共用同一份起始噪声，这是P2P生效的前提）
    latents = latent.expand(batch_size,  model.unet.config.in_channels, height // 8, width // 8).to(model.device)
    return latent, latents


@torch.no_grad() # 禁用梯度计算，加速推理
def text2image_ldm(
    model,
    prompt:  List[str],
    controller,
    num_inference_steps: int = 50,
    guidance_scale: Optional[float] = 7.,
    generator: Optional[torch.Generator] = None,
    latent: Optional[torch.FloatTensor] = None,
):
    register_attention_control(model, controller)
    height = width = 256
    batch_size = len(prompt)
    
    uncond_input = model.tokenizer([""] * batch_size, padding="max_length", max_length=77, return_tensors="pt")
    uncond_embeddings = model.bert(uncond_input.input_ids.to(model.device))[0]
    
    text_input = model.tokenizer(prompt, padding="max_length", max_length=77, return_tensors="pt")
    text_embeddings = model.bert(text_input.input_ids.to(model.device))[0]
    latent, latents = init_latent(latent, model, height, width, generator, batch_size)
    context = torch.cat([uncond_embeddings, text_embeddings])
    
    model.scheduler.set_timesteps(num_inference_steps)
    for t in tqdm(model.scheduler.timesteps):
        latents = diffusion_step(model, controller, latents, context, t, guidance_scale)
    
    image = latent2image(model.vqvae, latents)
   
    return image, latent


@torch.no_grad()
def text2image_ldm_stable(
    model,
    prompt: List[str],
    controller,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    generator: Optional[torch.Generator] = None,
    latent: Optional[torch.FloatTensor] = None,
    low_resource: bool = False,
):
    """用于 Stable Diffusion (1.x/2.x) 的主生成循环。"""
    # 【关键！】在生成前，将我们的控制器注入到UNet的注意力层中
    register_attention_control(model, controller)
    height = width = 512 # 默认生成分辨率为512x512
    batch_size = len(prompt) # Batch大小取决于提示词列表长度

    # 1. 编码条件提示词(Conditional text)
    text_input = model.tokenizer(
        prompt, padding="max_length", max_length=model.tokenizer.model_max_length,
        truncation=True, return_tensors="pt",
    )
    text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]
    
    # 2. 编码无条件提示词(Unconditional text/Negative prompt，这里全为空字符串)
    max_length = text_input.input_ids.shape[-1]
    uncond_input = model.tokenizer(
        [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
    )
    uncond_embeddings = model.text_encoder(uncond_input.input_ids.to(model.device))[0]
    
    # 组织Context
    context = [uncond_embeddings, text_embeddings]
    if not low_resource:
        context = torch.cat(context) # 非低资源模式下将其拼接在一起
        
    # 初始化隐变量噪声
    latent, latents = init_latent(latent, model, height, width, generator, batch_size)
    
    # 设置调度器的去噪时间步
    model.scheduler.set_timesteps(num_inference_steps)
    # 开始迭代去噪 (主循环)
    for t in tqdm(model.scheduler.timesteps):
        # 每一步调用上面定义的 diffusion_step 
        latents = diffusion_step(model, controller, latents, context, t, guidance_scale, low_resource)
    
    # 降噪完成后，用VAE把潜变量解码为最终图像
    image = latent2image(model.vae, latents)
  
    return image, latent # 返回生成的图像和最后的潜状态


def register_attention_control(model, controller):
    """
    【最核心的函数 - Monkey Patching 机制】
    遍历SD模型的UNet，将其内部所有的 Attention(注意力) 层进行“移花接木”，
    用我们自定义的转发(forward)逻辑替换原有的计算流程。
    """
    
    def ca_forward(self, place_in_unet):
        """定义一个全新的 forward 函数闭包，用于替换模型内置的 forward。"""
        # 兼容处理：获取注意力层的输出投射模块
        to_out = self.to_out
        if type(to_out) is torch.nn.modules.container.ModuleList:
            to_out = self.to_out[0]
        else:
            to_out = self.to_out

        def forward(x, encoder_hidden_states=None, attention_mask=None):
            # 这段逻辑几乎是对原生Attention内部计算 Q,K,V 的复刻
            batch_size, sequence_length, dim = x.shape
            h = self.heads # 注意力头数
            q = self.to_q(x) # 算 Query
            # 判断是否为交叉注意力：如果传了encoder_hidden_states(文本特征)，就是交叉注意力；否则是自注意力
            is_cross = encoder_hidden_states is not None
            encoder_hidden_states = encoder_hidden_states if is_cross else x
            k = self.to_k(encoder_hidden_states) # 算 Key
            v = self.to_v(encoder_hidden_states) # 算 Value
            
            # 张量形状重塑以适应多头注意力计算
            q = self.head_to_batch_dim(q)
            k = self.head_to_batch_dim(k)
            v = self.head_to_batch_dim(v)

            # 计算 Q 和 K 的点积相似度矩阵
            sim = torch.einsum("b i d, b j d -> b i j", q, k) * self.scale

            # 处理Mask逻辑（原版逻辑自带的）
            if attention_mask is not None:
                attention_mask = attention_mask.reshape(batch_size, -1)
                max_neg_value = -torch.finfo(sim.dtype).max
                attention_mask = attention_mask[:, None, :].repeat(h, 1, 1)
                sim.masked_fill_(~attention_mask, max_neg_value)

            # attention = softmax(Q*K)，这是真正控制网络看向哪里的“注意力图”！
            attn = sim.softmax(dim=-1) 
            
            # 【魔法在这里发生！】 
            # 把算出的原始注意力图丢进我们自建的 P2P controller 中！
            # controller 可能会原样返回(存储模式)，也可能把它替换成原图的注意力(编辑模式)
            attn = controller(attn, is_cross, place_in_unet)
            
            # 拿到(可能被篡改过)的注意力图后，再和 Value 相乘，得到注意力层的输出
            out = torch.einsum("b i j, b j d -> b i d", attn, v)
            out = self.batch_to_head_dim(out)
            return to_out(out) # 最终输出

        return forward # 返回这套新的前向传播闭包

    # 若未提供controller，用一个空的代替（即不改变原行为）
    class DummyController:
        def __call__(self, *args): return args[0]
        def __init__(self): self.num_att_layers = 0

    if controller is None:
        controller = DummyController()

    def register_recr(net_, count, place_in_unet):
        """递归函数：深入遍历UNet网络模型树"""
        # 如果当前模块类名是 'Attention'，意味着找到了一个注意力层
        if net_.__class__.__name__ == 'Attention':
            # 将该层的 forward 强行赋值为我们刚才包装的 ca_forward（这就是Monkey Patching！）
            net_.forward = ca_forward(net_, place_in_unet)
            return count + 1 # 找到的层数+1
        elif hasattr(net_, 'children'):
            # 如果不是，则递归遍历它的子模块
            for net__ in net_.children():
                count = register_recr(net__, count, place_in_unet)
        return count

    # 统计UNet网络中 Down(下采样)、Mid(中间层)、Up(上采样)三个模块里分别有多少个注意力层
    cross_att_count = 0
    sub_nets = model.unet.named_children()
    for net in sub_nets:
        if "down" in net[0]:
            cross_att_count += register_recr(net[1], 0, "down")
        elif "up" in net[0]:
            cross_att_count += register_recr(net[1], 0, "up")
        elif "mid" in net[0]:
            cross_att_count += register_recr(net[1], 0, "mid")

    # 告知控制器整个网络中一共有多少个注意力层需要拦截
    controller.num_att_layers = cross_att_count

    
def get_word_inds(text: str, word_place: int, tokenizer):
    """
    因为分词器(Tokenizer)通常使用BPE算法，一个长单词可能会被拆成多个Token (如'burger' -> 'bur', 'ger')。
    此函数的作用是：你输入想要找的“单词”(字符串)或“单词位置”(第几个)，
    它能帮你精准返回该单词在模型Token列表里对应的所有位置索引(Index)。
    """
    split_text = text.split(" ")
    if type(word_place) is str:
        # 找到目标单词在以空格分割的列表中的位置
        word_place = [i for i, word in enumerate(split_text) if word_place == word]
    elif type(word_place) is int:
        word_place = [word_place]
    out = []
    if len(word_place) > 0:
        # 将文本编码并立刻解码，去掉首尾的特殊符号(BOS/EOS)
        words_encode = [tokenizer.decode([item]).strip("#") for item in tokenizer.encode(text)][1:-1]
        cur_len, ptr = 0, 0

        # 遍历分词结果，将子Token拼凑起来对比，直到长度匹配，从而推断出正确的Token索引
        for i in range(len(words_encode)):
            cur_len += len(words_encode[i])
            if ptr in word_place:
                out.append(i + 1) # 保存索引 (由于前面有BOS符，索引需要+1)
            if cur_len >= len(split_text[ptr]):
                ptr += 1
                cur_len = 0
    return np.array(out)


def update_alpha_time_word(alpha, bounds: Union[float, Tuple[float, float]], prompt_ind: int,
                           word_inds: Optional[torch.Tensor]=None):
    """
    更新编辑权重 Alpha 值。它控制在“多少步到多少步”的区间内，特定单词的替换比例生效（值为1），其余步数为0。
    """
    if type(bounds) is float:
        bounds = 0, bounds
    # 根据步长比例计算真实的起止时间步索引
    start, end = int(bounds[0] * alpha.shape[0]), int(bounds[1] * alpha.shape[0])
    if word_inds is None:
        word_inds = torch.arange(alpha.shape[2]) # 如果没有指定具体单词，就应用于所有Token
    alpha[: start, prompt_ind, word_inds] = 0 # 起始步前不介入
    alpha[start: end, prompt_ind, word_inds] = 1 # 介入区间设为1
    alpha[end:, prompt_ind, word_inds] = 0 # 结束步后不介入
    return alpha


def get_time_words_attention_alpha(prompts, num_steps,
                                   cross_replace_steps: Union[float, Dict[str, Tuple[float, float]]],
                                   tokenizer, max_num_words=77):
    """
    生成一个张量（Tensor），精确控制：在扩散模型的哪一步(Step)，针对哪个词(Word)，我们要开启交叉注意力替换。
    有些词我们需要从头替换到尾，有些词（比如后期修饰色彩的词）我们可能只在前面几步注入源结构。
    """
    if type(cross_replace_steps) is not dict:
        cross_replace_steps = {"default_": cross_replace_steps} # 如果传了单值，设为默认全词策略
    if "default_" not in cross_replace_steps:
        cross_replace_steps["default_"] = (0., 1.) # 默认0到100%覆盖
        
    # 初始化一个5维大张量：(去噪步数, prompt数量, 1, 1, Token数)
    alpha_time_words = torch.zeros(num_steps + 1, len(prompts) - 1, max_num_words)
    
    # 1. 首先给所有Token赋予默认的替换起止步数
    for i in range(len(prompts) - 1):
        alpha_time_words = update_alpha_time_word(alpha_time_words, cross_replace_steps["default_"], i)
        
    # 2. 如果用户针对特定的词汇设定了特别的替换比例（比如 {"dog": 0.8} 表示替换前80%的步数）
    for key, item in cross_replace_steps.items():
        if key != "default_":
             # 找出这个特定单词在所有提示词中的Token索引
             inds = [get_word_inds(prompts[i], key, tokenizer) for i in range(1, len(prompts))]
             for i, ind in enumerate(inds):
                 if len(ind) > 0:
                    # 覆盖默认设置，应用针对该单词的特制起止步数
                    alpha_time_words = update_alpha_time_word(alpha_time_words, item, i, ind)
                    
    # 重塑为与注意力矩阵匹配的维度以备相乘融合
    alpha_time_words = alpha_time_words.reshape(num_steps + 1, len(prompts) - 1, 1, 1, max_num_words)
    return alpha_time_words
