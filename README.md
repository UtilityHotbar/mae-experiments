This is a modified version of the repo [Masked Autoencoders: A PyTorch Implementation](https://github.com/facebookresearch/mae). It has been modified to perform experiments on whether MAE classifiers possess strong occlusion sensitivity and whether they evaluate well on [Imagenet-X](https://facebookresearch.github.io/imagenetx/site/home) compared to traditional Vision transformer and Resnet models.

Code in `occlusion_test.py`, `imagenet_x_eval.py`, `imagenet_x_eval_timm.py`, `heatmap.py`, `heatmap_data.py`, and `data_collect.py` contain my extensions to run these experiments. Other files may have been modified (`models_vit.py`).