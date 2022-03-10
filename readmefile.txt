一.训练
1.在项目根目录下构造charactors.txt文档，文档中只包含一行,表示全部字符集（包括blank字符），包括所有的字符，每个字符之间用逗号分隔
2.准备数据集
训练集图片存放路径应当存放有所有的训练图片，路径形式如下所示:
train_img_dir
    ----1.jpg
    ----2.jpg
        ......
训练集标签是一个json文件，内容如下所示:
{
    "1.jpg": text1,
    "2.jpg": text2,
    ......
}
验证集图片和标签格式与训练集相同。
你可以使用generate_data.py来根据charactors.txt来生成训练数据集，需要首先配置charactors.txt中的参数，参数如下所示：
img_save_dir： 生成的图片存放路径
labels_pth： 生成的label文件存放路径
img_height_min： 生成图片最小高度
img_height_max： 生成图片最大高度
img_width_min： 生成图片最小宽度
img_width_max： 生成图片最大宽度
blank_index： 空白符在charactors.txt中的索引位置（charactors.txt中的逗号不占位，或表示为去除所有逗号后空白符的索引位置）
charactor_count_min： 一张图片中最少包含多少个字符
charactor_count_max： 一张图片中最多包含多少个字符
sample_count： 生成多少张图片
3.配置train.py中的参数
CUDA_VISIBLE_DEVICES: 表示使用的gpu序号，例如"0,1"
epoch: 训练多少个epoch
batch_size： batch size
init_lr: 初始学习率
min_lr： 最小学习率
cosine_lr_sch_cycle_times： 余弦学习率变化策略总共将学习率变化几个周期，一个周期表示学习率从初始学习率降至最小学习率后再升至初始学习率
input_h： 模型输入图片的高度，为16的倍数
train_img_dir： 训练集图片存放目录
valid_img_dir： 验证集图片存放目录
train_label_pth： 训练集标签路径
valid_label_pth： 验证集标签路径
print_step： 训练多少个step输出一次训练信息
blank_index： 空白符在charactors.txt中的索引位置（charactors.txt中的逗号不占位，或表示为去除所有逗号后空白符的索引位置）
num_workers： 使用几个线程加载数据
4.运行train.py
```
python train.py
```
二.预测
1.配置predict.py中的参数：
blank_index： 空白符在charactors.txt中的索引位置（charactors.txt中的逗号不占位，或表示为去除所有逗号后空白符的索引位置），和训练时保持一致
input_h： 模型输入图片的高度，为16的倍数，保证和训练时一致
use_best_model： True运用验证集上表现最佳的模型预测，False运用最后一个epoch存储的模型预测
img_pth：要预测的图片路径
