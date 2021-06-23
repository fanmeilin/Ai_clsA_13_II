## 准备

运行 demo.py 需要从dvc下载所需要的库和文件，运行：

```shell
$ ./setup.sh
```

## 部署

``external_lib`` 是跑 demo.py 用的，部署时可以删掉以减重。或者在干净的 repo 下运行：

```shell
$ ./setup.sh release
```



## 模型训练

该回归模型训练自 ai-lab，对应的 repo 地址，版本号和 tab 如下：

* repo address: ``git@192.168.10.30:Algorithm4/Programs/ADC/zhaimenghua/cnn-factory/multilabel-classifier.git``
* commit hash: ``070153c2bec275bd4079b0f4263f14333ecce57f``

可以运行：

```python
python train-test.py
```

复现或这更新该模型（原模型在 val dataset 上的 mse 是 0.78，比这小可以放心更新）。把训练好的模型 ``ai-lab/tmp/*/latest.pth`` 拷贝到 ``model/``，配置文件 ``ai-lab/tmp/*/[CNN NAME].py`` 拷贝到 ``model/`` 。 

