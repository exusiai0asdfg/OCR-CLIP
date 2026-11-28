# OCL


Implement an OCL-based object category classification inference experiment.

Set up your own Python environment by downloading and installing **Miniforge / uv / Python**, and then begin the experiment.

## References

1. OCL paper: **Beyond Object Recognition: A New Benchmark towards Object Concept Learning**

2. **OCL dataset preparation**
   * The official data preparation scripts are provided in the OCL GitHub repository:
     [https://github.com/silicx/ObjectConceptLearning/tree/main/data](https://github.com/silicx/ObjectConceptLearning/tree/main/data)
   * Please read and understand the provided scripts and dataset structure before implementing the DataLoader.
   * In particular, the repository provides `prepare_data.sh`, `download_web_images.py`, and the `resources` directory for dataset preparation.
   * After downloading the dataset, inspect the image data and annotation files carefully and understand how **category**, **attribute**, and **affordance** annotations are represented.

3. Download and prepare the required test data using the scripts provided in the official OCL GitHub repository.

4. CLIP paper/repository: **Contrastive Language-Image Pre-Training**
   [https://github.com/openai/CLIP](https://github.com/openai/CLIP)

## Step 1: Prepare the DataLoader

1. Implement a custom dataset class by inheriting from PyTorch's native `Dataset`:

   * Load the dataset and corresponding annotations;
   * Filter the data by sorting category labels according to the number of samples, and keep only the **10 categories with the largest number of samples**;
   * Perform image preprocessing and text tokenization.

2. Create a test data loader using PyTorch's native `DataLoader`:

   * Set `batch_size = 4`.

## Step 2: Prepare the CLIP Model

1. Load a pretrained CLIP model using the available API.

## Step 3: Inference and Evaluation

1. Instantiate the dataset and model, and move the model to GPU, including:

   * `load_dataset`
   * `load_model`
   * `model.to(device)`

2. Set the model to inference mode and freeze all model parameters:

   * Iterate over all model parameters and set gradient computation to `False`.

3. Iterate over the test set and perform inference:

   * Each image may contain multiple objects.
   * Compute the cosine similarity between each object and the candidate **category** labels.
   * Use the cosine similarity scores for ranking.
   * Compute the model's **Top-k accuracy**.
   * Perform forward inference batch by batch and report the average accuracy over the entire test set.

4. Evaluate the prediction performance for **attribute** and **affordance** labels using the **mAP (mean Average Precision)** metric.






# OCL

实现一个基于 OCL 的物体类别分类推理实验。

通过下载并安装 **Miniforge / uv / Python** 来配置你自己的 Python 环境，然后开始实验。

## 参考资料

1. OCL 论文：**Beyond Object Recognition: A New Benchmark towards Object Concept Learning**(超越物体识别：面向物体概念学习的新基准)

2. **OCL 数据集准备**
   * 官方数据准备脚本位于 OCL 的 GitHub 仓库中：
     [https://github.com/silicx/ObjectConceptLearning/tree/main/data](https://github.com/silicx/ObjectConceptLearning/tree/main/data)
   * 在实现 DataLoader 之前，请先阅读并理解仓库中提供的脚本和数据集结构。
   * 具体来说，该仓库提供了 `prepare_data.sh`、`download_web_images.py` 以及 `resources` 目录用于数据集准备。
   * 下载数据集后，请仔细检查图像数据和标注文件，并理解**类别(category)**、**属性(attribute)**和**可供性(affordance)**标注是如何表示的。

3. 使用官方 OCL GitHub 仓库中提供的脚本下载并准备所需的测试数据。

4. CLIP 论文/代码仓库：**Contrastive Language-Image Pre-Training**(对比语言-图像预训练)
   [https://github.com/openai/CLIP](https://github.com/openai/CLIP)

## 步骤 1：准备 DataLoader

1. 继承 PyTorch 原生的 `Dataset` 类，实现一个自定义数据集类：

   * 加载数据集及对应的标注；
   * 按样本数量对类别标签进行排序以筛选数据，仅保留**样本数量最多的前 10 个类别**；
   * 执行图像预处理和文本分词(tokenization)。

2. 使用 PyTorch 原生的 `DataLoader` 创建测试数据加载器：

   * 设置 `batch_size = 4`。

## 步骤 2：准备 CLIP 模型

1. 使用可用的 API 加载预训练的 CLIP 模型。

## 步骤 3：推理与评估

1. 实例化数据集和模型，并将模型迁移至 GPU，包括：

   * `load_dataset`
   * `load_model`
   * `model.to(device)`

2. 将模型设置为推理模式并冻结所有模型参数：

   * 遍历所有模型参数，并将梯度计算设为 `False`。

3. 遍历测试集并执行推理：

   * 每张图像可能包含多个物体。
   * 计算每个物体与候选**类别(category)**标签之间的余弦相似度。
   * 使用余弦相似度分数进行排序。
   * 计算模型的 **Top-k 准确率**。
   * 按批次(batch)逐批执行前向推理，并报告在整个测试集上的平均准确率。

4. 使用 **mAP(mean Average Precision，平均精度均值)** 指标评估模型对**属性(attribute)**和**可供性(affordance)**标签的预测性能。