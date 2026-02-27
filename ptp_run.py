# 导入类型提示，用于规范函数输入输出类型
from typing import Optional, Union, Tuple, List, Callable, Dict
import torch # 导入PyTorch深度学习框架
# 从diffusers库导入Stable Diffusion流水线
from diffusers import StableDiffusionPipeline
import torch.nn.functional as nnf # 导入PyTorch的神经网络函数库（如插值、池化等）
import numpy as np # 导入NumPy用于数组计算
import abc # 导入抽象基类模块，用于定义接口
import ptp_utils # 导入P2P作者提供的工具库（如图像显示、获取token索引等）
import seq_aligner # 导入P2P作者提供的序列对齐工具（用于匹配原prompt和编辑后的prompt）

LOW_RESOURCE = False  # 资源受限标志。如果为True，则减少无条件生成（unconditional）分支的注意力计算以节省显存
NUM_DIFFUSION_STEPS = 50 # 扩散模型的去噪步数（Inference steps）设为50
GUIDANCE_SCALE = 7.5 # 无分类器引导（Classifier-free guidance）的权重尺度，标准设为7.5
MAX_NUM_WORDS = 77 # Stable Diffusion的最大Token数量限制为77
# 判断是否有可用的GPU，有则使用cuda:0，否则退回使用CPU
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
# 从Hugging Face加载Stable Diffusion v1.5预训练模型，并将其放置到指定的设备（GPU/CPU）上
sd_pipeline = StableDiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5").to(device)
# 从加载的pipeline中提取分词器（tokenizer），用于将文本转化为Token ID
tokenizer = sd_pipeline.tokenizer


class LocalBlend:
    """局部融合类：用于在局部编辑时，根据特定单词的注意力图生成一个空间Mask，只在Mask区域内应用编辑。"""
    
    def __call__(self, x_t, attention_store):
        # 魔法方法，使得类的实例可以像函数一样被调用
        k = 1 # 定义池化操作的核大小参数
        # 提取指定层（下采样层的第2、3层，上采样层的前3层）的交叉注意力图
        maps = attention_store["down_cross"][2:4] + attention_store["up_cross"][:3]
        # 将这些注意力图重塑为形状: (batch, heads, 1, 16, 16, 77)，适配空间维度
        maps = [item.reshape(self.alpha_layers.shape[0], -1, 1, 16, 16, MAX_NUM_WORDS) for item in maps]
        # 将重塑后的特征图在通道/头维度进行拼接
        maps = torch.cat(maps, dim=1)
        # 将注意力图与设定的目标词汇层（alpha_layers）相乘，在Token维度(-1)求和，并在注意力头维度(1)求平均
        maps = (maps * self.alpha_layers).sum(-1).mean(1)
        # 使用最大池化对注意力图进行膨胀/平滑处理，生成Mask的基础形状
        mask = nnf.max_pool2d(maps, (k * 2 + 1, k * 2 +1), (1, 1), padding=(k, k))
        # 将Mask双线性插值放大到与当前隐变量 x_t 相同的空间分辨率
        mask = nnf.interpolate(mask, size=(x_t.shape[2:]))
        # 归一化：将Mask除以空间上的最大值，使其取值范围在 0 到 1 之间
        mask = mask / mask.max(2, keepdims=True)[0].max(3, keepdims=True)[0]
        # 二值化：将Mask中大于阈值（threshold）的部分设为1（True），其余为0（False）
        mask = mask.gt(self.threshold)
        # 将原图的Mask和编辑图的Mask相加（防止某一方完全没激活），并转为浮点数
        mask = (mask[:1] + mask[1:]).float()
        # 核心融合步骤：原隐变量 x_t[:1] 加上 Mask区域内的差异 (x_t - x_t[:1])
        x_t = x_t[:1] + mask * (x_t - x_t[:1])
        return x_t # 返回融合后的隐变量
       
    def __init__(self, prompts: List[str], words: [List[List[str]]], threshold=.3):
        # 初始化函数，传入所有提示词，需要进行局部编辑的目标词汇，以及二值化阈值
        alpha_layers = torch.zeros(len(prompts),  1, 1, 1, 1, MAX_NUM_WORDS) # 初始化一个全0张量来标记目标词汇的位置
        for i, (prompt, words_) in enumerate(zip(prompts, words)): # 遍历每一个prompt及其对应的目标词汇
            if type(words_) is str: # 如果传入的词汇是字符串而不是列表
                words_ = [words_] # 将其包装成列表形式
            for word in words_: # 遍历每一个目标词汇
                # 使用工具函数获取目标词汇在当前prompt中的Token索引（位置）
                ind = ptp_utils.get_word_inds(prompt, word, tokenizer)
                # 将该索引对应位置的值设为1，表示这里是我们关注的区域
                alpha_layers[i, :, :, :, :, ind] = 1
        self.alpha_layers = alpha_layers.to(device) # 将标记好的层放到GPU上
        self.threshold = threshold # 保存阈值


class AttentionControl(abc.ABC):
    """注意力控制抽象基类：所有注意力操作（存储、替换、重加权）的父类"""
    
    def step_callback(self, x_t):
        # 每一步去噪结束时的回调函数，默认原样返回隐变量
        return x_t
    
    def between_steps(self):
        # 在两个去噪步之间调用的钩子函数，默认无操作
        return
    
    @property
    def num_uncond_att_layers(self):
        # 属性：计算无条件（负向提示词）的注意力层数量。开启LOW_RESOURCE时为num_att_layers，否则为0
        return self.num_att_layers if LOW_RESOURCE else 0
    
    @abc.abstractmethod
    def forward (self, attn, is_cross: bool, place_in_unet: str):
        # 抽象方法：必须由子类实现，用于定义如何修改或记录注意力矩阵
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        # 魔法方法：拦截Stable Diffusion内部计算出的注意力矩阵
        # 判断当前正在处理的层是否属于有条件（文本引导）分支
        if self.cur_att_layer >= self.num_uncond_att_layers:
            if LOW_RESOURCE: # 如果是低资源模式
                attn = self.forward(attn, is_cross, place_in_unet) # 直接处理传入的注意力
            else: # 正常模式（batch包含 unconditional 和 conditional 两部分）
                h = attn.shape[0] # 获取 batch_size * head 的总数
                # 仅对后半部分（有条件生成部分，即文本引导的部分）进行forward拦截处理
                attn[h // 2:] = self.forward(attn[h // 2:], is_cross, place_in_unet)
        self.cur_att_layer += 1 # 当前处理的层数计数器加1
        # 如果当前层数等于总层数（条件层 + 无条件层），说明当前去噪步的所有UNet层已处理完
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0 # 层计数器归零
            self.cur_step += 1 # 去噪步数加1
            self.between_steps() # 调用步间处理函数（用于累加或整理当前步的数据）
        return attn # 返回（可能被修改过的）注意力矩阵
    
    def reset(self):
        # 重置计数器的状态，用于开始新的一张图片的生成
        self.cur_step = 0 # 步数清零
        self.cur_att_layer = 0 # 层数清零

    def __init__(self):
        # 初始化基本计数变量
        self.cur_step = 0 # 当前所处的去噪时间步
        self.num_att_layers = -1 # 总注意力层数（通常在运行中被推断出）
        self.cur_att_layer = 0 # 当前正在处理的注意力层索引

class EmptyControl(AttentionControl):
    """空控制器：什么都不做，用于生成基线（Baseline）对比图"""
    
    def forward (self, attn, is_cross: bool, place_in_unet: str):
        # 原封不动地返回注意力矩阵，不加干涉
        return attn
    
    
class AttentionStore(AttentionControl):
    """注意力存储器：用于捕获并保存生成过程中的注意力图，是P2P分析和可视化的基础"""

    @staticmethod
    def get_empty_store():
        # 静态方法，返回一个初始化的空字典，按UNet的结构（下、中、上）和注意力类型（交叉、自注意）分类
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [],  "mid_self": [],  "up_self": []}

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        # 拦截注意力图
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}" # 根据位置和类型生成字典的Key
        if attn.shape[1] <= 32 ** 2:  # 只有当空间分辨率较小（<=32x32）时才保存，避免OOM（内存溢出）
            self.step_store[key].append(attn) # 将注意力图加入当前步的临时存储器中
        return attn # 返回原注意力图

    def between_steps(self):
        # 在每一步结束后调用，用于将当前步的注意力图汇总到全局存储中
        if len(self.attention_store) == 0: # 如果是第一步
            self.attention_store = self.step_store # 直接赋值
        else: # 如果不是第一步
            for key in self.attention_store: # 遍历每一个位置（如 down_cross）
                for i in range(len(self.attention_store[key])): # 累加对应层的注意力图
                    self.attention_store[key][i] += self.step_store[key][i]
        self.step_store = self.get_empty_store() # 清空当前步的临时存储器，为下一步做准备

    def get_average_attention(self):
        # 计算整个扩散过程中的平均注意力图（将累加的总和除以当前经过的步数）
        average_attention = {key: [item / self.cur_step for item in self.attention_store[key]] for key in self.attention_store}
        return average_attention # 返回均值字典

    def reset(self):
        # 重置控制器状态
        super(AttentionStore, self).reset() # 调用父类的复位（清空计数）
        self.step_store = self.get_empty_store() # 清空临时存储
        self.attention_store = {} # 清空全局存储

    def __init__(self):
        # 初始化存储器
        super(AttentionStore, self).__init__() # 调用父类初始化
        self.step_store = self.get_empty_store() # 初始化临时存储
        self.attention_store = {} # 初始化全局存储

        
class AttentionControlEdit(AttentionStore, abc.ABC):
    """注意力控制编辑基类：P2P编辑操作的核心父类，负责替换/注入注意力机制"""
    
    def step_callback(self, x_t):
        # 每步回调：如果启用了局部融合（LocalBlend），则应用局部Mask融合操作
        if self.local_blend is not None:
            x_t = self.local_blend(x_t, self.attention_store)
        return x_t # 返回融合/未修改的隐变量
        
    def replace_self_attention(self, attn_base, att_replace):
        # 替换自注意力：在保持高分辨率结构（布局）不变时极为关键
        if att_replace.shape[2] <= 16 ** 2: # 仅针对空间分辨率低于或等于16x16的特征图进行自注意力注入
            # 将基图（原图）的自注意力复制/扩展到与编辑图相同的Batch大小并返回
            return attn_base.unsqueeze(0).expand(att_replace.shape[0], *attn_base.shape)
        else:
            # 对于高分辨率层，保留自身的自注意力以生成细节
            return att_replace
    
    @abc.abstractmethod
    def replace_cross_attention(self, attn_base, att_replace):
        # 抽象方法：如何替换交叉注意力。这取决于具体的编辑类型（Replace/Refine/Reweight）
        raise NotImplementedError
    
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        # 拦截机制：这是P2P魔法发生的地方
        super(AttentionControlEdit, self).forward(attn, is_cross, place_in_unet) # 先调用父类存储当前注意力
        # 判断当前是否需要执行替换操作：如果是交叉注意力，或者是处于设定时间步窗口内的自注意力
        if is_cross or (self.num_self_replace[0] <= self.cur_step < self.num_self_replace[1]):
            h = attn.shape[0] // (self.batch_size) # 计算当前UNet特征层的注意力头数 (heads)
            # 重塑张量，把 batch (不同prompt) 和 heads 分开
            attn = attn.reshape(self.batch_size, h, *attn.shape[1:])
            # 提取原图（基础）的注意力，和其余编辑图的注意力
            attn_base, attn_repalce = attn[0], attn[1:]
            if is_cross: # 如果是文本-图像交叉注意力
                alpha_words = self.cross_replace_alpha[self.cur_step] # 获取当前去噪步对应的融合权重（通常随时间衰减）
                # 调用子类实现的交叉注意力替换方法，再按 alpha_words 权重进行软融合
                attn_repalce_new = self.replace_cross_attention(attn_base, attn_repalce) * alpha_words + (1 - alpha_words) * attn_repalce
                attn[1:] = attn_repalce_new # 把修改后的交叉注意力写回编辑图中
            else: # 如果是图像-图像自注意力
                attn[1:] = self.replace_self_attention(attn_base, attn_repalce) # 注入基图的自注意力以锁定布局
            # 将维度重新展平回模型需要的格式 (batch*heads, seq_len, ...)
            attn = attn.reshape(self.batch_size * h, *attn.shape[2:])
        return attn # 返回被P2P修改后的注意力矩阵
    
    def __init__(self, prompts, num_steps: int,
                 cross_replace_steps: Union[float, Tuple[float, float], Dict[str, Tuple[float, float]]],
                 self_replace_steps: Union[float, Tuple[float, float]],
                 local_blend: Optional[LocalBlend]):
        # 初始化编辑控制器
        super(AttentionControlEdit, self).__init__() # 继承存储器的初始化
        self.batch_size = len(prompts) # batch大小等于提示词列表的长度（1个原词 + n个编辑词）
        # 获取交叉注意力替换的Alpha权重调度器（定义在什么时候停止注入源交叉注意力）
        self.cross_replace_alpha = ptp_utils.get_time_words_attention_alpha(prompts, num_steps, cross_replace_steps, tokenizer).to(device)
        if type(self_replace_steps) is float: # 如果自注意力替换步数只给了一个浮点数
            self_replace_steps = 0, self_replace_steps # 将其转换为元组，表示从第0步开始，到该比例步数结束
        # 将比例转换为具体的去噪步数区间 (例如：从第0步到第25步)
        self.num_self_replace = int(num_steps * self_replace_steps[0]), int(num_steps * self_replace_steps[1])
        self.local_blend = local_blend # 保存局部融合器对象

class AttentionReplace(AttentionControlEdit):
    """替换词汇编辑类：适用于 "a dog" 变为 "a cat" 等单词替换场景"""

    def replace_cross_attention(self, attn_base, att_replace):
        # 爱因斯坦求和约定：将原图注意力(attn_base) 乘以序列映射矩阵(self.mapper)，实现特征对齐
        # 'hpw'是head, pixel, word_base， 'bwn' 是batch, word_base, word_new
        # 结果变为 'bhpn'，即新prompt中各Token对应原图结构分布的注意力
        return torch.einsum('hpw,bwn->bhpn', attn_base, self.mapper)
      
    def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float,
                 local_blend: Optional[LocalBlend] = None):
        super(AttentionReplace, self).__init__(prompts, num_steps, cross_replace_steps, self_replace_steps, local_blend)
        # 获取替换映射矩阵（标记新旧文本序列中相同、替换或删除的单词索引关系）
        self.mapper = seq_aligner.get_replacement_mapper(prompts, tokenizer).to(device)
        

class AttentionRefine(AttentionControlEdit):
    """提纯/修饰编辑类：适用于添加修饰词（如 "a dog" -> "a fluffy red dog"）"""

    def replace_cross_attention(self, attn_base, att_replace):
        # 根据对齐映射mapper重新排列基图交叉注意力，以对齐新Prompt
        attn_base_replace = attn_base[:, :, self.mapper].permute(2, 0, 1, 3)
        # 按alpha掩码将新生成的注意力(对于新增的修饰词)和基准注意力(对于原有的词)进行拼接融合
        attn_replace = attn_base_replace * self.alphas + att_replace * (1 - self.alphas)
        return attn_replace

    def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float,
                 local_blend: Optional[LocalBlend] = None):
        super(AttentionRefine, self).__init__(prompts, num_steps, cross_replace_steps, self_replace_steps, local_blend)
        # 获得序列精炼/扩充映射矩阵及对应的Alpha蒙版（标记哪些是旧词、哪些是新添加的词）
        self.mapper, alphas = seq_aligner.get_refinement_mapper(prompts, tokenizer)
        self.mapper, alphas = self.mapper.to(device), alphas.to(device) # 转置到设备
        self.alphas = alphas.reshape(alphas.shape[0], 1, 1, alphas.shape[1]) # 塑形以匹配注意力矩阵维度


class AttentionReweight(AttentionControlEdit):
    """重加权编辑类：用于增强或削弱特定单词的影响力（如“增强burger的权重”）"""

    def replace_cross_attention(self, attn_base, att_replace):
        # 如果存在前置控制器（比如先替换再重加权），则先调用前置控制器的替换逻辑
        if self.prev_controller is not None:
            attn_base = self.prev_controller.replace_cross_attention(attn_base, att_replace)
        # 将基础注意力图乘以均衡器（equalizer）权重系数，以放大或缩小特定单词的响应
        attn_replace = attn_base[None, :, :, :] * self.equalizer[:, None, None, :]
        return attn_replace

    def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float, equalizer,
                local_blend: Optional[LocalBlend] = None, controller: Optional[AttentionControlEdit] = None):
        super(AttentionReweight, self).__init__(prompts, num_steps, cross_replace_steps, self_replace_steps, local_blend)
        self.equalizer = equalizer.to(device) # 加载调整特定Token权重的均衡器向量
        self.prev_controller = controller # 保存可选的组合控制器


def get_equalizer(text: str, word_select: Union[int, Tuple[int, ...]], values: Union[List[float],
                  Tuple[float, ...]]):
    # 构建权重均衡器矩阵：针对指定的文本(text)中的选中词(word_select)赋予不同的注意力乘数(values)
    if type(word_select) is int or type(word_select) is str:
        word_select = (word_select,) # 转为元组
    equalizer = torch.ones(len(values), 77) # 初始化全为1的均衡矩阵 (批次 x 77长度)
    values = torch.tensor(values, dtype=torch.float32) # 将目标值转为张量
    for word in word_select: # 遍历要修改权重的单词
        inds = ptp_utils.get_word_inds(text, word, tokenizer) # 获取单词对应的Token索引
        equalizer[:, inds] = values # 将对应的索引位上的权重改写为用户指定的值
    return equalizer

from PIL import Image # 导入Python图像库

def aggregate_attention(attention_store: AttentionStore, res: int, from_where: List[str], is_cross: bool, select: int):
    # 聚合工具：将存储的多个层的注意力图进行平均和尺寸统一
    out = []
    attention_maps = attention_store.get_average_attention() # 获取时序上平均过的注意力图字典
    num_pixels = res ** 2 # 计算目标分辨率的像素总数 (例如 16x16 = 256)
    for location in from_where: # 遍历来自UNet的不同位置 (up, mid, down)
        for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]: # 遍历对应的注意力层列表
            if item.shape[1] == num_pixels: # 如果当前注意力图的分辨率符合目标分辨率(如16x16)
                # 提取第select个批次的数据（通常0是原图，1是编辑图），重塑为图像尺寸
                cross_maps = item.reshape(len(prompts), -1, res, res, item.shape[-1])[select]
                out.append(cross_maps) # 加入输出列表
    out = torch.cat(out, dim=0) # 在注意力头（或层）维度进行拼接
    out = out.sum(0) / out.shape[0] # 求所有层的平均值，得到一张综合注意力图
    return out.cpu() # 将聚合后的张量拉回CPU并返回


def show_cross_attention(attention_store: AttentionStore, res: int, from_where: List[str], select: int = 0):
    # 可视化工具：渲染文本-图像交叉注意力图，展示每个单词关注图像的哪个部分
    tokens = tokenizer.encode(prompts[select]) # 对指定的提示词进行分词
    decoder = tokenizer.decode # 获取解码器（Token ID -> 字符串）
    attention_maps = aggregate_attention(attention_store, res, from_where, True, select) # 调用上面的聚合函数获取综合交叉注意力
    images = []
    for i in range(len(tokens)): # 遍历句子中的每一个Token
        image = attention_maps[:, :, i] # 提取当前Token对应的二维空间注意力图
        image = 255 * image / image.max() # 将0-1浮点值域缩放到 0-255，便于作为图像显示
        image = image.unsqueeze(-1).expand(*image.shape, 3) # 扩展为一个三通道的灰度图
        image = image.numpy().astype(np.uint8) # 转为NumPy 8位无符号整型数组
        image = np.array(Image.fromarray(image).resize((256, 256))) # 转成PIL Image，插值放大到256x256后再转回NumPy
        image = ptp_utils.text_under_image(image, decoder(int(tokens[i]))) # 使用工具在图像下方打印对应的单词
        images.append(image) # 收集所有带标签的注意力热力图
    ptp_utils.view_images(np.stack(images, axis=0)) # 调用P2P的显示工具画成网格
    

def show_self_attention_comp(attention_store: AttentionStore, res: int, from_where: List[str],
                        max_com=10, select: int = 0):
    # 可视化工具：展示自注意力的主成分分析（PCA/SVD）可视化图
    # 获取聚合的自注意力矩阵，并展平成二维矩阵 (像素数 x 像素数)
    attention_maps = aggregate_attention(attention_store, res, from_where, False, select).numpy().reshape((res ** 2, res ** 2))
    # 通过SVD分解（奇异值分解），提取特征矩阵的主成分（提取图像的主体语义块）
    u, s, vh = np.linalg.svd(attention_maps - np.mean(attention_maps, axis=1, keepdims=True))
    images = []
    for i in range(max_com): # 取前 max_com（通常是10）个主成分
        image = vh[i].reshape(res, res) # 还原回空间形状
        image = image - image.min() # Min-Max 归一化到 0-1
        image = 255 * image / image.max() # 放缩到 0-255
        image = np.repeat(np.expand_dims(image, axis=2), 3, axis=2).astype(np.uint8) # 扩展为三通道RGB图
        image = Image.fromarray(image).resize((256, 256)) # 放大分辨率
        image = np.array(image)
        images.append(image) # 加入显示列表
    ptp_utils.view_images(np.concatenate(images, axis=1)) # 水平拼接并显示所有主成分

def run_and_display(prompts, controller, latent=None, run_baseline=False, generator=None):
    # 核心执行包装函数：运行Stable Diffusion推理过程并展示结果图
    if run_baseline: # 如果开启了Baseline运行（无P2P介入）
        print("w.o. prompt-to-prompt")
        # 递归调用自身，传入EmptyControl（相当于原始Stable Diffusion）生成基线图
        images, latent = run_and_display(prompts, EmptyControl(), latent=latent, run_baseline=False, generator=generator)
        print("with prompt-to-prompt")
    # 调用 P2P 提供的定制版 SD 去噪循环 (text2image_ldm_stable)，它在内部会调用我们传进去的 controller
    images, x_t = ptp_utils.text2image_ldm_stable(sd_pipeline, prompts, controller, latent=latent, num_inference_steps=NUM_DIFFUSION_STEPS, guidance_scale=GUIDANCE_SCALE, generator=generator, low_resource=LOW_RESOURCE)
    ptp_utils.view_images(images) # 渲染最终的生成图像
    return images, x_t # 返回生成的图像数组以及最后一步的隐状态

# 以下是执行代码（Demo 运行）
g_cpu = torch.Generator().manual_seed(8888) # 设定随机种子以保证结果可以复现
prompts = ["A painting of a squirrel eating a burger"] # 测试提示词：一幅松鼠吃汉堡的画
controller = AttentionStore() # 实例化一个注意力存储控制器（这里只是为了提取并可视化注意力，没做编辑）
# 执行生成并显示结果图像，传入前面初始化的参数
image, x_t = run_and_display(prompts, controller, latent=None, run_baseline=False, generator=g_cpu)
# 显示包含 up(上采样) 和 down(下采样) 阶段 16x16 分辨率的交叉注意力图
show_cross_attention(controller, res=16, from_where=("up", "down"))