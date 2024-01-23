参考
[MONAI_tutorial](https://github.com/Project-MONAI/tutorials/blob/main/3d_segmentation/unetr_btcv_segmentation_3d.ipynb)
# 讲解视频
[https://www.bilibili.com/video/BV12N411J7s7/](https://www.bilibili.com/video/BV12N411J7s7/)
# demo演示
demo.ipynb

![image.png](https://cdn.nlark.com/yuque/0/2024/png/28587781/1705987642909-6b2295c2-0a99-44e8-a111-d85b94b078d3.png#averageHue=%23bfd8db&clientId=ub75aaa61-1668-4&from=paste&height=651&id=ubdb04450&originHeight=976&originWidth=2000&originalType=binary&ratio=1.5&rotation=0&showTitle=false&size=340779&status=done&style=none&taskId=u5d6da9a3-db56-4e21-b35e-50cceffcfaf&title=&width=1333.3333333333333)

# 训练自己的分割模型pipline

1. 配置环境
```
pip install -r requirements.txt
```

2. 整理数据集，参见视频教程
3. 若为dicom格式，转换为 NIFTI 

 	      dicom2nii.ipynb

4. 生成数据集json文件

        generate_json.ipynb

5. 训练

        train_brain.ipynb

6. 推理

        inference.ipynb

7. 计算体积

 	      compute.ipynb
