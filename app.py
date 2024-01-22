import os

import matplotlib.pyplot as plt

from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.transforms import (
    EnsureChannelFirstd,
    Compose,
    LoadImaged,
    Orientationd,
    ScaleIntensityRanged,
    Rotate90d,
)
from monai.networks.nets import UNETR

from monai.data import (
    SmartCacheDataset,
)

import numpy as np
import torch

import gradio as gr
import matplotlib.pyplot as plt
import torch
import nibabel as nib
import numpy as np
import SimpleITK as sitk

def dcm2nii(dcms_path, nii_path):
	# 1.构建dicom序列文件阅读器，并执行（即将dicom序列文件“打包整合”）
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(dcms_path)
    reader.SetFileNames(dicom_names)
    image2 = reader.Execute()
	# 2.将整合后的数据转为array，并获取dicom文件基本信息
    image_array = sitk.GetArrayFromImage(image2)  # z, y, x
    origin = image2.GetOrigin()  # x, y, z
    print(origin)
    spacing = image2.GetSpacing()  # x, y, z
    print(spacing)
    direction = image2.GetDirection()  # x, y, z
    print(direction)

    # 3.将array转为img，并保存为.nii.gz
    image3 = sitk.GetImageFromArray(image_array)
    image3.SetSpacing(spacing)
    image3.SetDirection(direction)
    image3.SetOrigin(origin)
    sitk.WriteImage(image3, nii_path)

def calculate_volume(mask_image_path):
    # 读取分割结果的图像文件
    mask_image = sitk.ReadImage(mask_image_path)

    # 获取图像的大小、原点和间距
    size = mask_image.GetSize()
    origin = mask_image.GetOrigin()
    spacing = mask_image.GetSpacing()

    # 将 SimpleITK 图像转换为 NumPy 数组
    mask_array = sitk.GetArrayFromImage(mask_image)

    # if len(np.unique(mask_array)) != 5:
    #     print(mask_image_path[-15:-12])
    #     print(np.unique(mask_array))
    
    # 计算非零像素的数量
    one_voxels = (mask_array == 1).sum()
    two_voxels = (mask_array == 2).sum()
    three_voxels = (mask_array == 3).sum()
    four_voxels = (mask_array == 4).sum()
    # print(one_voxels,two_voxels,three_voxels,four_voxels)
    # 计算像素的体积（以立方毫米为单位）
    voxel_volume_mm3 = spacing[0] * spacing[1] * spacing[2]

    # 计算体积（以 mm³ 为单位）
    V_Right_ventricular_cistern = one_voxels * voxel_volume_mm3 / 1000.0
    V_Right_cerebral_sulcus = two_voxels * voxel_volume_mm3 / 1000.0
    V_Left_ventricular_cistern = three_voxels * voxel_volume_mm3 / 1000.0
    V_Left_cerebral_sulcus = four_voxels * voxel_volume_mm3 / 1000.0
    # 如果需要以其他单位（例如 cm³）显示，请进行适当的单位转换
    # volume_cm3 = volume_mm3 / 1000.0

    return size,spacing,V_Right_ventricular_cistern, V_Right_cerebral_sulcus, V_Left_ventricular_cistern, V_Left_cerebral_sulcus
    
def process_nii_file(input_nii_file, slice, mode):
            
    if mode == "Step1:Segment":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        root_dir = "./code/run/"
        model = UNETR(
            in_channels=1,
            out_channels=5,
            img_size=(96, 96, 16),
            feature_size=16,
            hidden_size=768,
            mlp_dim=3072,
            num_heads=12,
            pos_embed="perceptron",
            norm_name="instance",
            res_block=True,
            dropout_rate=0.0,
        ).to(device)
        model.load_state_dict(torch.load(os.path.join(root_dir, "best_metric_model67v2.pth")))
        
        test_transforms = Compose(
            [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            Orientationd(keys=["image"], axcodes="RAS"),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-50,
                a_max=100,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            Rotate90d(keys=["image"], k=1)
            # ResizeWithPadOrCropd(keys=["image"], spatial_size=(512, 512, 16)),
            ]
        )
        test_file = [{'image':input_nii_file.name}]
        # test_file = [{'image':r'F:\sth\23Fall\fcpro\brain_image_copy\image\60020599.nii.gz'}]
        test_image = SmartCacheDataset(data=test_file, transform=test_transforms)[0]['image']

        with torch.no_grad():

            inputs = torch.unsqueeze(test_image, 1).cuda()

            val_outputs = sliding_window_inference(inputs, (96, 96, 16), 8, model, overlap=0.8)
        
        # Display the images
        fig1 = plt.figure()
        plt.title("image")
        plt.axis('off') # Remove axis
        plt.imshow(inputs.cpu().numpy()[0, 0, :, :, slice], cmap="gray")

        fig2 = plt.figure()
        plt.title("output")
        plt.axis('off') # Remove axis
        plt.imshow(torch.argmax(val_outputs, dim=1).detach().cpu()[0, :, :, slice])

        val_outputs = torch.argmax(val_outputs, dim=1).detach().cpu()[0, :, :, :]
        val_outputs = val_outputs.numpy().astype('int16')
        # val_outputs = np.transpose(val_outputs, (2, 1, 0))
        val_outputs = np.rot90(val_outputs, k=3)
        val_outputs = nib.Nifti1Image(val_outputs, np.eye(4))
        nib.save(val_outputs, f'D:/{input_nii_file.name[-15:-7]}_mask.nii.gz')

        return ["指定切片分割结果如下, mask文件已保存至D:/", fig1, fig2]
    
    if mode == "Step2:Volumn":
        maskFilePath = input_nii_file.name
        size,spacing,V_Right_ventricular_cistern, V_Right_cerebral_sulcus, V_Left_ventricular_cistern, V_Left_cerebral_sulcus = calculate_volume(maskFilePath)

        vol = f"""右侧脑室脑池的体积为{V_Right_ventricular_cistern}cm³\n 右侧脑沟的体积为{V_Right_cerebral_sulcus}cm³\n 左侧脑室脑池的体积为{V_Left_ventricular_cistern}cm³\n 左侧脑沟的体积为{V_Left_cerebral_sulcus}cm³"""
        fig1 = plt.figure()
        fig2 = plt.figure()
        return [vol, fig1, fig2]

# Define the Gradio interface
iface = gr.Interface(
    fn=process_nii_file,
    inputs=
    [gr.File(file_count='single', file_types=['.nii.gz']), 
    gr.Slider(0, 24, value=8, label="Select Slice", step=1),
    gr.Radio(
            ["Step1:Segment", "Step2:Volumn"], label="mode"
        ),
    ],
    
    outputs=[gr.Text(label="Output"), gr.Plot(label="image"), gr.Plot(label="mask")],  # Display both "image" and "output"
)

iface.launch(share=True)
