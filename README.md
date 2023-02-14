# 3DGQA算法使用说明

This repository is for the **T-CSVT 2022** paper "[Toward Explainable 3D Grounded Visual Question Answering: A New Benchmark and Strong Baseline](https://ieeexplore.ieee.org/abstract/document/9984686/)" [arxiv version](https://arxiv.org/abs/2209.12028)

## 算法描述

In this work, we formally define and address a 3D grounded VQA task by collecting a new 3D question answering (GQA) dataset, referred to as flexible and explainable 3D GQA (FE-3DGQA), with diverse and relatively free-form question-answer pairs, as well as dense and completely grounded bounding box annotations. To achieve more explainable answers, we label the objects appeared in the complex QA pairs with different semantic types, including answer-grounded objects (both appeared and not appeared in the questions), and contextual objects for answer-grounded objects. We also propose a new 3D VQA framework to effectively predict the completely visually grounded and explainable answer. Extensive experiments verify that our newly collected benchmark datasets can be effectively used to evaluate various 3D VQA methods from different aspects and our newly proposed framework also achieves the state-of-the-art performance on the new benchmark dataset.

## 数据准备

1. 在你想要放置数据的地方创建一个名为data的文件夹。
2. Fill out [this form](https://forms.gle/aLtzXN12DsYDMSXX6). Once your request is accepted, you will receive an email with the download link. Download the ScanRefer dataset and unzip it under `data/`.
3. Downloadand the preprocessed [GLoVE embeddings (~990MB)](http://kaldir.vc.in.tum.de/glove.p) and put them under `data/`.
4. Download the ScanNetV2 dataset and put (or link) `scans/` under (or to) `data/scannet/scans/` (Please follow the [ScanNet Instructions](data/scannet/README.md) for downloading the ScanNet dataset).
5. 将[ScanRefer仓库](https://github.com/daveredrum/ScanRefer/tree/master/data/scannet)中的`meta_data`文件夹以及其他5个`.py`文件放在`data/scannet`目录下

> After this step, there should be folders containing the ScanNet scene data under the `data/scannet/scans/` with names like `scene0000_00`

4. Pre-process ScanNet data. A folder named `scannet_data/` will be generated under `data/scannet/` after running the following command. Roughly 3.8GB free space is needed for this step:

```shell
cd data/scannet/
python batch_load_scannet_data.py
```

> After this step, you can check if the processed scene data is valid by running:
>
> ```shell
> python visualize.py --scene_id scene0000_00
> ```

## 安装

CUDA版本: 11.6   python版本: 3.9.16
其他依赖库的安装命令如下：

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
```

可使用如下命令下载安装算法包：
```bash
pip install FE_3DGQA==0.1.17
```


## 3. 使用示例及运行参数说明

```python
from FE_3DGQA import GroudingQuestionAnswering

m = GroudingQuestionAnswering()

# 训练  m.run_train(你的data文件夹所在的绝对路径) 示例如下
m.run_train('/data2/user1/pip_test/data')

# 训练结果会保存在 data 文件夹的同级目录下的 outputs 文件夹中，如示例路径，即保存在 /data2/user1/pip_test/outputs/{训练开始时间_3DGQA} 文件夹下

# 推理  m.inference(你的模型所在的绝对路径，你的data文件夹所在的绝对路径) 示例如下
m.inference('/data2/user1/pip_test/outputs/2023-02-14_07-56-16_3DGQA', '/data2/wangzhen/pip_test/data')

# 推理结果会保存在第一个参数中的路径下，如示例路径，即保存在 /data2/user1/pip_test/outputs/2023-02-14_07-56-16_3DGQA 文件夹下
```

## 4. 论文
```
@article{zhao2022towards,
  author={Zhao, Lichen and Cai, Daigang and Zhang, Jing and Sheng, Lu and Xu, Dong and Zheng, Rui and Zhao, Yinjie and Wang, Lipeng and Fan, Xibo},
  journal={IEEE Transactions on Circuits and Systems for Video Technology}, 
  title={Towards Explainable 3D Grounded Visual Question Answering: A New Benchmark and Strong Baseline}, 
  year={2022},
  doi={10.1109/TCSVT.2022.3229081}
}
```