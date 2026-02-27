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
import torch # 导入 PyTorch
import numpy as np # 导入 NumPy


class ScoreParams:
    """得分参数类：用于定义序列对齐时的奖惩分数规则"""
    def __init__(self, gap, match, mismatch):
        self.gap = gap # 引入空格（Gap）的惩罚分
        self.match = match # 字符匹配的奖励分
        self.mismatch = mismatch # 字符不匹配的惩罚分

    def mis_match_char(self, x, y):
        """判断两个字符（在这里是 Token ID）是否匹配，并返回对应的分数"""
        if x != y:
            return self.mismatch # 不匹配，扣分
        else:
            return self.match # 匹配，加分
        

# ⚠️ 注意：这段代码里定义了两个 get_matrix 函数。
# 在 Python 中，后面定义的同名函数会覆盖前面的。所以这个纯列表实现的版本实际上是不生效的（被废弃的冗余代码）。
def get_matrix(size_x, size_y, gap):
    matrix = []
    for i in range(len(size_x) + 1):
        sub_matrix = []
        for j in range(len(size_y) + 1):
            sub_matrix.append(0)
        matrix.append(sub_matrix)
    for j in range(1, len(size_y) + 1):
        matrix[0][j] = j*gap
    for i in range(1, len(size_x) + 1):
        matrix[i][0] = i*gap
    return matrix


def get_matrix(size_x, size_y, gap):
    """真正生效的矩阵初始化函数 (基于 NumPy，速度更快)"""
    # 初始化一个大小为 (size_x+1) x (size_y+1) 的全零动态规划矩阵
    matrix = np.zeros((size_x + 1, size_y + 1), dtype=np.int32)
    # 初始化第一行：不断插入 gap 的累积惩罚分
    matrix[0, 1:] = (np.arange(size_y) + 1) * gap
    # 初始化第一列：不断插入 gap 的累积惩罚分
    matrix[1:, 0] = (np.arange(size_x) + 1) * gap
    return matrix


def get_traceback_matrix(size_x, size_y):
    """初始化回溯矩阵：用于记录动态规划时每一步是最优解是从哪个方向来的，以便最后反推对齐路径"""
    matrix = np.zeros((size_x + 1, size_y +1), dtype=np.int32)
    matrix[0, 1:] = 1 # 1 表示只能从左边过来 (插入 Gap)
    matrix[1:, 0] = 2 # 2 表示只能从上面过来 (插入 Gap)
    matrix[0, 0] = 4  # 4 表示起点/终点标记
    return matrix


def global_align(x, y, score):
    """
    核心算法：Needleman-Wunsch 全局对齐算法。
    用于计算原序列 x 和新序列 y 的最优对齐方式。
    """
    matrix = get_matrix(len(x), len(y), score.gap) # 获取初始化的得分矩阵
    trace_back = get_traceback_matrix(len(x), len(y)) # 获取初始化的回溯矩阵
    
    # 动态规划填表过程
    for i in range(1, len(x) + 1):
        for j in range(1, len(y) + 1):
            # 计算从三个方向走到当前格子的得分
            left = matrix[i, j - 1] + score.gap # 从左边来（序列 y 插入空格）
            up = matrix[i - 1, j] + score.gap # 从上面来（序列 x 插入空格）
            diag = matrix[i - 1, j - 1] + score.mis_match_char(x[i - 1], y[j - 1]) # 从左上角对角线来（匹配或替换）
            
            # 当前格子的最高得分
            matrix[i, j] = max(left, up, diag)
            
            # 记录是从哪个方向得到最高分的（用于回溯）
            if matrix[i, j] == left:
                trace_back[i, j] = 1 # 1: 左
            elif matrix[i, j] == up:
                trace_back[i, j] = 2 # 2: 上
            else:
                trace_back[i, j] = 3 # 3: 对角线 (匹配/替换)
    return matrix, trace_back # 返回得分矩阵和回溯路径矩阵


def get_aligned_sequences(x, y, trace_back):
    """根据回溯矩阵，反向推导出 x 和 y 的对齐序列，以及它们的映射关系"""
    x_seq = []
    y_seq = []
    i = len(x) # 从矩阵右下角（终点）开始回溯
    j = len(y)
    mapper_y_to_x = [] # 记录新序列 y 到原序列 x 的位置映射
    
    while i > 0 or j > 0:
        if trace_back[i, j] == 3: # 如果是对角线来的（匹配或替换）
            x_seq.append(x[i-1])
            y_seq.append(y[j-1])
            i = i-1
            j = j-1
            mapper_y_to_x.append((j, i)) # 记录 y 的第 j 个 token 对应 x 的第 i 个 token
        elif trace_back[i][j] == 1: # 如果是从左边来的（x缺省，相当于y新增了词）
            x_seq.append('-') # x 序列补占位符
            y_seq.append(y[j-1])
            j = j-1
            mapper_y_to_x.append((j, -1)) # 新增的词在原序列中找不到对应，记为 -1
        elif trace_back[i][j] == 2: # 如果是从上面来的（y缺省，相当于删除了词）
            x_seq.append(x[i-1])
            y_seq.append('-') # y 序列补占位符
            i = i-1
        elif trace_back[i][j] == 4: # 到达起点
            break
            
    mapper_y_to_x.reverse() # 因为是从后往前推的，最后需要把映射表反转回来
    return x_seq, y_seq, torch.tensor(mapper_y_to_x, dtype=torch.int64)


def get_mapper(x: str, y: str, tokenizer, max_len=77):
    """
    为 Refine 模式（比如添加修饰词 "a dog" -> "a red dog"）生成 Token 映射表和 Alpha 遮罩
    """
    x_seq = tokenizer.encode(x) # 将原文本转为 Token ID 序列
    y_seq = tokenizer.encode(y) # 将新文本转为 Token ID 序列
    score = ScoreParams(0, 1, -1) # 设定对齐分数：空格0分，匹配得1分，不匹配扣1分
    
    # 跑动态规划对齐
    matrix, trace_back = global_align(x_seq, y_seq, score)
    # 获取新旧 Token 的位置映射表
    mapper_base = get_aligned_sequences(x_seq, y_seq, trace_back)[-1]
    
    alphas = torch.ones(max_len) # 初始化全 1 的 Alpha 遮罩
    # 找到哪些 token 是新加的（原序列中对应 -1）。新加的 token 的 alpha 设为 0，原有 token 的 alpha 设为 1。
    # 这决定了生成时，原有的词继承旧图特征，新加的词生成新特征。
    alphas[: mapper_base.shape[0]] = mapper_base[:, 1].ne(-1).float()
    
    mapper = torch.zeros(max_len, dtype=torch.int64)
    # 将 y 到 x 的索引映射填入 mapper
    mapper[:mapper_base.shape[0]] = mapper_base[:, 1]
    # 对超出的部分进行补齐处理
    mapper[mapper_base.shape[0]:] = len(y_seq) + torch.arange(max_len - len(y_seq))
    return mapper, alphas


def get_refinement_mapper(prompts, tokenizer, max_len=77):
    """对多个 prompt 进行 Refine 映射的包装函数"""
    x_seq = prompts[0] # 以第一句话为基准原图
    mappers, alphas = [], []
    for i in range(1, len(prompts)): # 遍历所有编辑后的 prompt
        mapper, alpha = get_mapper(x_seq, prompts[i], tokenizer, max_len)
        mappers.append(mapper)
        alphas.append(alpha)
    return torch.stack(mappers), torch.stack(alphas) # 堆叠成张量返回


def get_word_inds(text: str, word_place: int, tokenizer):
    """获取指定单词在分词器 Token 序列中的索引位置（复用函数，此前已解析过）"""
    split_text = text.split(" ")
    if type(word_place) is str:
        word_place = [i for i, word in enumerate(split_text) if word_place == word]
    elif type(word_place) is int:
        word_place = [word_place]
    out = []
    if len(word_place) > 0:
        words_encode = [tokenizer.decode([item]).strip("#") for item in tokenizer.encode(text)][1:-1]
        cur_len, ptr = 0, 0

        for i in range(len(words_encode)):
            cur_len += len(words_encode[i])
            if ptr in word_place:
                out.append(i + 1)
            if cur_len >= len(split_text[ptr]):
                ptr += 1
                cur_len = 0
    return np.array(out)


def get_replacement_mapper_(x: str, y: str, tokenizer, max_len=77):
    """
    为 Replace 模式（比如替换单词 "dog" -> "cat"）生成交叉注意力映射矩阵。
    处理复杂情况：1 个词可能被替换成了包含多个 Token 的长词，或者反之。
    """
    words_x = x.split(' ') # 按空格拆分原句
    words_y = y.split(' ') # 按空格拆分新句
    if len(words_x) != len(words_y):
        # 注意：这里的 Replace 编辑强制要求新旧句子单词数量相等。如果不等会报错。
        raise ValueError(f"attention replacement edit can only be applied on prompts with the same length"
                         f" but prompt A has {len(words_x)} words and prompt B has {len(words_y)} words.")
        
    # 找到发生替换操作的单词索引位置
    inds_replace = [i for i in range(len(words_y)) if words_y[i] != words_x[i]]
    # 获取这些被替换的原单词对应的 Token 索引列表
    inds_source = [get_word_inds(x, i, tokenizer) for i in inds_replace]
    # 获取这些新的替换单词对应的 Token 索引列表
    inds_target = [get_word_inds(y, i, tokenizer) for i in inds_replace]
    
    # 初始化一个 77x77 的二维映射矩阵
    mapper = np.zeros((max_len, max_len))
    i = j = 0
    cur_inds = 0
    # 双指针遍历序列，构建映射矩阵
    while i < max_len and j < max_len:
        # 如果当前位置是需要替换的单词位置
        if cur_inds < len(inds_source) and inds_source[cur_inds][0] == i:
            inds_source_, inds_target_ = inds_source[cur_inds], inds_target[cur_inds]
            
            # 情况 1：新旧单词分解出的 token 数量一样 (例如 1 对 1)
            if len(inds_source_) == len(inds_target_):
                mapper[inds_source_, inds_target_] = 1 # 直接一对一映射为 1
                
            # 情况 2：新旧单词分解出的 token 数量不一样 (例如 1 个 token 替换成 3 个 token)
            else:
                ratio = 1 / len(inds_target_) # 将原 token 的注意力权重平分给新的多个 token
                for i_t in inds_target_:
                    mapper[inds_source_, i_t] = ratio # 赋值比例
                    
            cur_inds += 1 # 移动到下一个替换词对
            i += len(inds_source_)
            j += len(inds_target_)
            
        # 如果当前位置还没到需要替换的单词，则它们是相同的单词
        elif cur_inds < len(inds_source):
            mapper[i, j] = 1 # 原封不动一对一继承注意力
            i += 1
            j += 1
            
        # 如果所有的替换词都已经处理完了，剩下相同的单词
        else:
            mapper[j, j] = 1 # 原封不动一对一继承
            i += 1
            j += 1

    return torch.from_numpy(mapper).float() # 转为 PyTorch 张量返回


def get_replacement_mapper(prompts, tokenizer, max_len=77):
    """对多个 prompt 进行 Replace 映射矩阵生成的包装函数"""
    x_seq = prompts[0]
    mappers = []
    for i in range(1, len(prompts)):
        # 逐个调用底层的 _ 函数
        mapper = get_replacement_mapper_(x_seq, prompts[i], tokenizer, max_len)
        mappers.append(mapper)
    return torch.stack(mappers) # 堆叠打包返回