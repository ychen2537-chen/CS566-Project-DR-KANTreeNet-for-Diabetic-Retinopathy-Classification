# KANTreeNet 视频演示演讲稿 / KANTreeNet Video Demo Presentation Script

## 开场 / Introduction

**中文：**
大家好！今天我将为大家展示如何使用 KANTreeNet 模型生成糖尿病视网膜病变（DR）分类的分析视频。这个视频展示了模型从输入眼底图像到最终分类结果的完整处理流程。

**English:**
Hello everyone! Today I will demonstrate how to generate an analysis video using the KANTreeNet model for Diabetic Retinopathy (DR) classification. This video showcases the complete processing pipeline from input fundus images to final classification results.

---

## 第一部分：视频生成过程 / Part 1: Video Generation Process

### 1.1 技术架构 / Technical Architecture

**中文：**
视频生成脚本基于 Python 和深度学习框架 PyTorch 开发。主要使用了以下技术栈：
- OpenCV 用于图像处理和视频编码
- Matplotlib 用于可视化帧的生成
- PyTorch 用于模型推理和特征提取
- NumPy 用于数组操作和数据处理

**English:**
The video generation script is developed using Python and the PyTorch deep learning framework. The main technologies used include:
- OpenCV for image processing and video encoding
- Matplotlib for generating visualization frames
- PyTorch for model inference and feature extraction
- NumPy for array operations and data processing

### 1.2 生成流程 / Generation Workflow

**中文：**
视频生成过程分为五个主要步骤：

第一步：图像预处理
- 从磁盘加载原始眼底图像
- 将图像调整为 448×448 像素，这是模型的输入尺寸
- 应用 ImageNet 标准化（均值和方差归一化）

第二步：模型推理
- 加载预训练的 EnhancedDRKANTreeNet 模型
- 执行前向传播，同时提取中间层特征
- 获取血管树、病变注意力、DAM 增强特征和 ViT 全局上下文

第三步：特征可视化
- 将提取的特征图转换为热力图
- 使用颜色映射（JET colormap）将热力图叠加到原始图像上
- 生成每个处理步骤的可视化帧

第四步：帧序列构建
- 按照处理顺序组织所有可视化帧
- 为每个步骤设置显示时长（默认 2 秒）
- 确保所有帧尺寸一致

第五步：视频编码
- 使用 OpenCV 的 VideoWriter 将帧序列编码为 MP4 格式
- 设置帧率（默认 2 FPS）和视频参数
- 保存最终的分析视频

**English:**
The video generation process consists of five main steps:

**Step 1: Image Preprocessing**
- Load the original fundus image from disk
- Resize the image to 448×448 pixels, which is the model's input size
- Apply ImageNet normalization (mean and variance normalization)

**Step 2: Model Inference**
- Load the pre-trained EnhancedDRKANTreeNet model
- Perform forward propagation while extracting intermediate layer features
- Obtain vessel tree, lesion attention, DAM-enhanced features, and ViT global context

**Step 3: Feature Visualization**
- Convert extracted feature maps into heatmaps
- Overlay heatmaps onto the original image using color mapping (JET colormap)
- Generate visualization frames for each processing step

**Step 4: Frame Sequence Construction**
- Organize all visualization frames in processing order
- Set display duration for each step (default 2 seconds)
- Ensure all frames have consistent dimensions

**Step 5: Video Encoding**
- Use OpenCV's VideoWriter to encode the frame sequence into MP4 format
- Set frame rate (default 2 FPS) and video parameters
- Save the final analysis video

---

## 第二部分：视频内容详解 / Part 2: Detailed Video Content

### 步骤 1：原始眼底图像 / Step 1: Original Fundus Image

**中文：**
视频的第一帧展示原始输入的眼底图像。这是从数据集或临床环境中获取的原始图像，可能具有不同的分辨率和尺寸。这一步骤让观众了解模型的输入来源。

**English:**
The first frame of the video displays the original input fundus image. This is the raw image obtained from the dataset or clinical environment, which may have different resolutions and sizes. This step allows the audience to understand the source of the model's input.

### 步骤 2：图像预处理 / Step 2: Image Preprocessing

**中文：**
第二帧显示经过预处理后的图像。图像被调整为 448×448 像素，并应用了标准化处理。这一步确保了图像符合模型的输入要求，同时保持了重要的视觉特征。

**English:**
The second frame shows the preprocessed image. The image is resized to 448×448 pixels and normalized. This step ensures the image meets the model's input requirements while preserving important visual features.

### 步骤 3：血管树分支 / Step 3: Vessel Tree Branch

**中文：**
第三帧展示血管树分支的提取结果。KANTreeNet 使用 EnhancedVesselTreeNet 模块从图像的绿色通道中提取血管状结构。这些血管信息对于 DR 诊断非常重要，因为血管异常是糖尿病视网膜病变的重要指标。可视化以灰度图形式展示，白色区域表示检测到的血管结构。

**English:**
The third frame displays the extracted vessel tree branch. KANTreeNet uses the EnhancedVesselTreeNet module to extract vessel-like structures from the green channel of the image. This vessel information is crucial for DR diagnosis, as vascular abnormalities are important indicators of diabetic retinopathy. The visualization is shown as a grayscale image, with white areas representing detected vessel structures.

### 步骤 4：病变注意力 / Step 4: Lesion Attention

**中文：**
第四帧展示病变注意力热力图。这是模型自动学习到的注意力机制，能够识别和聚焦于疑似病变区域。该注意力机制结合了：
- ResNet 提取的高级特征
- 原始图像的病变特征
- 通道注意力机制
- 严重 DR 检测器

热力图以彩色叠加的方式显示在原始图像上，红色和黄色区域表示模型认为可能存在病变的高关注区域。

**English:**
The fourth frame displays the lesion attention heatmap. This is an automatically learned attention mechanism that identifies and focuses on suspected lesion regions. The attention mechanism combines:
- High-level features extracted by ResNet
- Lesion features from the original image
- Channel attention mechanism
- Severe DR detector

The heatmap is overlaid on the original image in color, with red and yellow areas indicating high-attention regions where the model believes lesions may exist.

### 步骤 5：DAM 增强局部结构 / Step 5: DAM-Enhanced Local Structures

**中文：**
第五帧展示经过 KANDAM（KAN Dual Attention Module）增强后的局部结构特征。DAM 模块进一步强化了具有判别性的纹理和局部结构信息，这些信息对于区分不同严重程度的 DR 至关重要。热力图显示了模型在局部细节上的关注重点。

**English:**
The fifth frame displays the local structure features enhanced by KANDAM (KAN Dual Attention Module). The DAM module further strengthens discriminative textures and local structure information, which are crucial for distinguishing different severity levels of DR. The heatmap shows the model's focus on local details.

### 步骤 6：ViT-S 全局上下文 / Step 6: ViT-S Global Context

**中文：**
第六帧展示 Vision Transformer (ViT-S) 提供的全局上下文信息。ViT-S 将图像分割成 14×14 个补丁（patches），每个补丁被编码为一个 token。通过计算这些 patch token 的范数，我们得到了全局注意力热力图。这个热力图展示了模型对整个图像全局结构的理解，有助于捕捉病变之间的空间关系和整体模式。

**English:**
The sixth frame displays the global context information provided by Vision Transformer (ViT-S). ViT-S divides the image into 14×14 patches, with each patch encoded as a token. By computing the norms of these patch tokens, we obtain a global attention heatmap. This heatmap shows the model's understanding of the global structure of the entire image, helping to capture spatial relationships between lesions and overall patterns.

### 步骤 7：最终分类结果 / Step 7: Final Classification Result

**中文：**
最后一帧展示模型的最终分类结果。以柱状图的形式显示五个 DR 严重程度类别的概率分布：
- No DR（无 DR）
- Mild（轻度）
- Moderate（中度）
- Severe（重度）
- Proliferative（增殖性）

柱状图清晰地展示了模型对每个类别的置信度，最高概率的类别即为模型的预测结果。这个结果综合了前面所有步骤提取的特征信息，包括血管树、病变注意力、DAM 增强特征和 ViT 全局上下文。

**English:**
The final frame displays the model's final classification result. It shows the probability distribution across five DR severity classes in a bar chart format:
- No DR
- Mild
- Moderate
- Severe
- Proliferative

The bar chart clearly shows the model's confidence for each category, with the highest probability category being the model's prediction. This result integrates all feature information extracted in the previous steps, including vessel tree, lesion attention, DAM-enhanced features, and ViT global context.

---

## 第三部分：技术亮点 / Part 3: Technical Highlights

### 3.1 多模态特征融合 / Multi-modal Feature Fusion

**中文：**
KANTreeNet 的创新之处在于它融合了多种不同类型的特征：
1. **局部特征**：通过 ResNet 提取的卷积特征
2. **血管特征**：专门的血管树网络提取的血管结构
3. **注意力特征**：自动学习的病变注意力机制
4. **全局特征**：ViT-S 提供的全局上下文信息

这种多模态融合使得模型能够从多个角度理解眼底图像，提高了分类的准确性和可解释性。

**English:**
The innovation of KANTreeNet lies in its fusion of multiple different types of features:
1. **Local Features**: Convolutional features extracted by ResNet
2. **Vessel Features**: Vessel structures extracted by a specialized vessel tree network
3. **Attention Features**: Automatically learned lesion attention mechanism
4. **Global Features**: Global context information provided by ViT-S

This multi-modal fusion allows the model to understand fundus images from multiple perspectives, improving classification accuracy and interpretability.

### 3.2 可解释性可视化 / Interpretability Visualization

**中文：**
视频演示的一个重要价值是提供了模型决策过程的可视化。通过展示每个处理步骤的中间结果，观众可以：
- 理解模型关注的重点区域
- 验证模型是否关注了正确的病变区域
- 诊断模型的潜在问题
- 增强对模型决策的信任

这种可解释性对于医疗 AI 应用至关重要，因为它帮助医生理解模型的推理过程。

**English:**
An important value of the video demonstration is providing visualization of the model's decision-making process. By showing intermediate results at each processing step, the audience can:
- Understand the key regions the model focuses on
- Verify whether the model focuses on the correct lesion regions
- Diagnose potential issues with the model
- Enhance trust in the model's decisions

This interpretability is crucial for medical AI applications, as it helps doctors understand the model's reasoning process.

### 3.3 端到端自动化 / End-to-End Automation

**中文：**
整个视频生成过程是完全自动化的。只需提供一张眼底图像，脚本就会：
- 自动加载模型权重
- 执行完整的推理流程
- 提取所有中间特征
- 生成可视化帧
- 编码输出视频

这使得研究人员和临床医生可以轻松地为任何眼底图像生成分析视频，无需手动干预。

**English:**
The entire video generation process is fully automated. Simply providing a fundus image, the script will:
- Automatically load model weights
- Execute the complete inference pipeline
- Extract all intermediate features
- Generate visualization frames
- Encode the output video

This allows researchers and clinicians to easily generate analysis videos for any fundus image without manual intervention.

---

## 第四部分：应用场景 / Part 4: Application Scenarios

### 4.1 教学演示 / Educational Demonstration

**中文：**
这个视频可以用于：
- 向医学生和研究人员解释深度学习模型的工作原理
- 展示 KANTreeNet 架构的优势
- 演示多模态特征融合的效果

**English:**
This video can be used for:
- Explaining how deep learning models work to medical students and researchers
- Demonstrating the advantages of the KANTreeNet architecture
- Showing the effects of multi-modal feature fusion

### 4.2 模型验证 / Model Validation

**中文：**
临床医生可以使用这些视频来：
- 验证模型是否关注了正确的解剖结构
- 检查模型对病变区域的识别是否准确
- 评估模型的可信度和可靠性

**English:**
Clinicians can use these videos to:
- Verify whether the model focuses on the correct anatomical structures
- Check if the model's identification of lesion regions is accurate
- Evaluate the model's credibility and reliability

### 4.3 研究分析 / Research Analysis

**中文：**
研究人员可以利用视频来：
- 分析不同病例的特征模式
- 比较模型在不同类型病变上的表现
- 发现模型的潜在改进方向

**English:**
Researchers can use videos to:
- Analyze feature patterns in different cases
- Compare model performance on different types of lesions
- Identify potential directions for model improvement

---

## 结语 / Conclusion

**中文：**
通过这个视频演示，我们展示了 KANTreeNet 模型在糖尿病视网膜病变分类任务中的完整工作流程。从原始图像到最终分类结果，每一步都经过精心设计，既保证了分类准确性，又提供了良好的可解释性。这种可视化方法不仅有助于理解模型的工作原理，也为医疗 AI 的临床应用提供了重要的透明度。

感谢大家的聆听！

**English:**
Through this video demonstration, we have showcased the complete workflow of the KANTreeNet model in the task of diabetic retinopathy classification. From the original image to the final classification result, each step is carefully designed to ensure both classification accuracy and good interpretability. This visualization method not only helps understand how the model works but also provides important transparency for the clinical application of medical AI.

Thank you for your attention!

---

## 附录：使用方法 / Appendix: Usage Instructions

**中文：**
生成视频的命令示例：

```bash
python kantree_video_demo.py \
    --image_path train_images/000c1434d8d7.png \
    --output_video kantree_analysis_demo.mp4 \
    --weights model_checkpoint.pth \
    --fps 2 \
    --step_sec 2.0
```

**English:**
Example command to generate the video:

```bash
python kantree_video_demo.py \
    --image_path train_images/000c1434d8d7.png \
    --output_video kantree_analysis_demo.mp4 \
    --weights model_checkpoint.pth \
    --fps 2 \
    --step_sec 2.0
```


