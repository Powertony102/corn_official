import h5py
import math
import nibabel as nib
import numpy as np
from medpy import metric
from myutils.metrics import calculate_metric_percase
import torch
import torch.nn.functional as F
from tqdm import tqdm
from skimage.measure import label
from scipy.spatial.distance import cdist

def getLargestCC(segmentation):
    labels = label(segmentation)
    assert( labels.max() != 0 ) # assume at least 1 CC
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
    return largestCC

## Edited By Xinze LI at 2024/1/8 1:14
def dice_coefficient(prediction, label):
    # 计算交集
    intersection = np.sum(prediction & label)

    # 计算每个集合中真值的数量
    prediction_sum = np.sum(prediction)
    label_sum = np.sum(label)

    # 计算 Dice 系数
    if (prediction_sum + label_sum) == 0:
        # 避免除以零的情况
        return 1.0
    else:
        dice = 2.0 * intersection / (prediction_sum + label_sum)

    return dice

def jc_coefficient(image1, image2):
    """
    Calculate the Jaccard Coefficient (Intersection over Union) for binary images.

    Args:
    image1 (numpy.ndarray): The first binary image.
    image2 (numpy.ndarray): The second binary image.

    Returns:
    float: Jaccard Coefficient between the two images.
    """
    # Ensure that the images are of the same size
    if image1.shape != image2.shape:
        raise ValueError("Images must have the same dimensions.")

    # Calculate the intersection and union
    intersection = (image1 & image2).sum()
    union = (image1 | image2).sum()

    # Avoid division by zero
    if union == 0:
        return 1.0 if intersection == 0 else 0.0

    # Calculate Jaccard Coefficient
    jc = intersection / union
    return jc

def hd95_coefficient(binary_image1, binary_image2):
    """
    Calculate the 95% Hausdorff Distance (HD95) between two binary images.

    Args:
    binary_image1 (numpy.ndarray): First binary image.
    binary_image2 (numpy.ndarray): Second binary image.

    Returns:
    float: 95% Hausdorff Distance.
    """
    # Extract the indices of the non-zero elements in each image
    pts_image1 = np.argwhere(binary_image1)
    pts_image2 = np.argwhere(binary_image2)

    print(f"pts pass")

    if pts_image1.size == 0 or pts_image2.size == 0:
        raise ValueError("One of the images is empty.")

    # Compute the pairwise distances between non-zero elements
    distances1 = cdist(pts_image1, pts_image2)
    distances2 = cdist(pts_image2, pts_image1)
    print(f"dist pass")

    # Compute the 95th percentile of the minimum distances for each point set
    hd1 = np.percentile(np.min(distances1, axis=1), 95)
    hd2 = np.percentile(np.min(distances2, axis=1), 95)

    print(f"hd pass")

    # The HD95 is the maximum of these two values
    hd95_value = max(hd1, hd2)
    return hd95_value

def asd_coefficient(binary_image1, binary_image2):
    """
    Calculate the Average Surface Distance (ASD) between two binary images.

    Args:
    binary_image1 (numpy.ndarray): First binary image.
    binary_image2 (numpy.ndarray): Second binary image.

    Returns:
    float: Average Surface Distance.
    """
    # Extract the indices of the non-zero elements in each image
    pts_image1 = np.argwhere(binary_image1)
    pts_image2 = np.argwhere(binary_image2)

    if pts_image1.size == 0 or pts_image2.size == 0:
        raise ValueError("One of the images is empty.")

    # Compute the pairwise distances between non-zero elements
    distances1 = cdist(pts_image1, pts_image2)
    distances2 = cdist(pts_image2, pts_image1)

    # Compute the average distances
    avg_dist1 = np.mean(np.min(distances1, axis=1))
    avg_dist2 = np.mean(np.min(distances2, axis=1))

    # The ASD is the average of these two values
    asd_value = (avg_dist1 + avg_dist2) / 2
    return asd_value


def var_all_case(model, num_classes, patch_size=(112, 112, 80), stride_xy=18, stride_z=4, dataset_name="LA"):
    if dataset_name == "LA":
        with open('./data/LA/train.list', 'r') as f:
            image_list = f.readlines()
        image_list = ["./data/LA/2018LA_Seg_Training Set/" + item.replace('\n', '') + "/mri_norm2.h5" for item in image_list]
    elif dataset_name == "Pancreas_CT":
        with open('./data/Pancreas/test.list', 'r') as f:
            image_list = f.readlines()
        image_list = ["./data/Pancreas/Pancreas_h5/" + item.replace('\n', '') + "_norm.h5" for item in image_list]
    loader = tqdm(image_list)
    total_dice = 0.0
    for image_path in loader:
        h5f = h5py.File(image_path, 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        prediction, score_map = test_single_case_first_output(model, image, stride_xy, stride_z, patch_size, num_classes=num_classes)
        # 将 prediction 和 label 转换为布尔类型
        prediction_bool = prediction.astype(np.bool_)
        label_bool = label.astype(np.bool_)
    
        if np.sum(prediction_bool) == 0:
            dice = 0
        else:
            # dice = metric.binary.dc(prediction_bool, label_bool)
            dice = dice_coefficient(prediction_bool, label_bool)
        total_dice += dice
        
    
    
    avg_dice = total_dice / len(image_list)
    print('average metric is {}'.format(avg_dice))
    return avg_dice

def test_all_case(model_name, num_outputs, model, image_list, num_classes, patch_size=(112, 112, 80), stride_xy=18, stride_z=4, save_result=True, test_save_path=None, preproc_fn=None, metric_detail=1, nms=0):

    loader = tqdm(image_list) if not metric_detail else image_list
    ith = 0
    total_metric = 0.0
    total_metric_average = 0.0
    for image_path in loader:
        print(f"Current testing: {image_path}")
        h5f = h5py.File(image_path, 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        
        if preproc_fn is not None:
            image = preproc_fn(image)
        prediction, score_map = test_single_case_first_output(model, image, stride_xy, stride_z, patch_size, num_classes=num_classes)
        if num_outputs > 1:
            prediction_average, score_map_average = test_single_case_average_output(model, image, stride_xy, stride_z, patch_size, num_classes=num_classes)
        if nms:
            prediction = getLargestCC(prediction)
            if num_outputs > 1:
                prediction_average = getLargestCC(prediction_average)
            
        if np.sum(prediction)==0:
            # print(f"Current Testing But all 0: {ith}")
            single_metric = (0,0,0,0)
            if num_outputs > 1:
                single_metric_average = (0,0,0,0)
        else:
            single_metric = calculate_metric_percase(prediction, label[:])
            if num_outputs > 1:
                single_metric_average  = calculate_metric_percase(prediction_average, label[:])
            
        if metric_detail:
            print('%02d,\t%.5f, %.5f, %.5f, %.5f' % (ith, single_metric[0], single_metric[1], single_metric[2], single_metric[3]))
            if num_outputs > 1:
                print('%02d,\t%.5f, %.5f, %.5f, %.5f' % (ith, single_metric_average[0], single_metric_average[1], single_metric_average[2], single_metric_average[3]))
        
        total_metric += np.asarray(single_metric)
        if num_outputs > 1:
            total_metric_average += np.asarray(single_metric_average) 
        
        if save_result:
            nib.save(nib.Nifti1Image(prediction.astype(np.float32), np.eye(4)), test_save_path +  "%02d_pred.nii.gz" % ith)
            nib.save(nib.Nifti1Image(score_map[0].astype(np.float32), np.eye(4)), test_save_path +  "%02d_scores.nii.gz" % ith)
            if num_outputs > 1:
                nib.save(nib.Nifti1Image(prediction_average.astype(np.float32), np.eye(4)), test_save_path +  "%02d_pred_average.nii.gz" % ith)
                nib.save(nib.Nifti1Image(score_map_average[0].astype(np.float32), np.eye(4)), test_save_path +  "%02d_scores_average.nii.gz" % ith)
            nib.save(nib.Nifti1Image(image[:].astype(np.float32), np.eye(4)), test_save_path +  "%02d_img.nii.gz" % ith)
            nib.save(nib.Nifti1Image(label[:].astype(np.float32), np.eye(4)), test_save_path +  "%02d_gt.nii.gz" % ith)
        
        ith += 1

    avg_metric = total_metric / len(image_list)
    print('average metric is decoder 1 {}'.format(avg_metric))
    if num_outputs > 1:
        avg_metric_average = total_metric_average / len(image_list)
        print('average metric of all decoders is {}'.format(avg_metric_average))
    
    with open(test_save_path+'../{}_performance.txt'.format(model_name), 'w') as f:
        f.writelines('average metric of decoder 1 is {} \n'.format(avg_metric))
        if num_outputs > 1:
            f.writelines('average metric of all decoders is {} \n'.format(avg_metric_average))
    return avg_metric


def test_single_case_first_output(model, image, stride_xy, stride_z, patch_size, num_classes=1):
    w, h, d = image.shape

    # if the size of image is less than patch_size, then padding it
    add_pad = False
    if w < patch_size[0]:
        w_pad = patch_size[0]-w
        add_pad = True
    else:
        w_pad = 0
    if h < patch_size[1]:
        h_pad = patch_size[1]-h
        add_pad = True
    else:
        h_pad = 0
    if d < patch_size[2]:
        d_pad = patch_size[2]-d
        add_pad = True
    else:
        d_pad = 0
    wl_pad, wr_pad = w_pad//2,w_pad-w_pad//2
    hl_pad, hr_pad = h_pad//2,h_pad-h_pad//2
    dl_pad, dr_pad = d_pad//2,d_pad-d_pad//2
    if add_pad:
        image = np.pad(image, [(wl_pad,wr_pad),(hl_pad,hr_pad), (dl_pad, dr_pad)], mode='constant', constant_values=0)
    ww,hh,dd = image.shape

    sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
    sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
    sz = math.ceil((dd - patch_size[2]) / stride_z) + 1
    # print("{}, {}, {}".format(sx, sy, sz))
    score_map = np.zeros((num_classes, ) + image.shape).astype(np.float32)
    cnt = np.zeros(image.shape).astype(np.float32)

    for x in range(0, sx):
        xs = min(stride_xy*x, ww-patch_size[0])
        for y in range(0, sy):
            ys = min(stride_xy * y,hh-patch_size[1])
            for z in range(0, sz):
                zs = min(stride_z * z, dd-patch_size[2])
                test_patch = image[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]]
                test_patch = np.expand_dims(np.expand_dims(test_patch,axis=0),axis=0).astype(np.float32)
                test_patch = torch.from_numpy(test_patch).cuda()

                with torch.no_grad():
                    y = model(test_patch)
                    if len(y) > 1:
                        y = y[0]
                    y = F.softmax(y, dim=1)
                y = y.cpu().data.numpy()
                y = y[0,1,:,:,:]
                score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                  = score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + y
                cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                  = cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + 1

    score_map = score_map/np.expand_dims(cnt,axis=0)
    label_map = (score_map[0]>0.5).astype(int)
    if add_pad:
        label_map = label_map[wl_pad:wl_pad+w,hl_pad:hl_pad+h,dl_pad:dl_pad+d]
        score_map = score_map[:,wl_pad:wl_pad+w,hl_pad:hl_pad+h,dl_pad:dl_pad+d]
    return label_map, score_map

def test_single_case_average_output(net, image, stride_xy, stride_z, patch_size, num_classes=1):
    w, h, d = image.shape

    # if the size of image is less than patch_size, then padding it
    add_pad = False
    if w < patch_size[0]:
        w_pad = patch_size[0]-w
        add_pad = True
    else:
        w_pad = 0
    if h < patch_size[1]:
        h_pad = patch_size[1]-h
        add_pad = True
    else:
        h_pad = 0
    if d < patch_size[2]:
        d_pad = patch_size[2]-d
        add_pad = True
    else:
        d_pad = 0
    wl_pad, wr_pad = w_pad//2,w_pad-w_pad//2
    hl_pad, hr_pad = h_pad//2,h_pad-h_pad//2
    dl_pad, dr_pad = d_pad//2,d_pad-d_pad//2
    if add_pad:
        image = np.pad(image, [(wl_pad,wr_pad),(hl_pad,hr_pad), (dl_pad, dr_pad)], mode='constant', constant_values=0)
    ww,hh,dd = image.shape

    sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
    sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
    sz = math.ceil((dd - patch_size[2]) / stride_z) + 1
    # print("{}, {}, {}".format(sx, sy, sz))
    score_map = np.zeros((num_classes, ) + image.shape).astype(np.float32)
    cnt = np.zeros(image.shape).astype(np.float32)

    for x in range(0, sx):
        xs = min(stride_xy*x, ww-patch_size[0])
        for y in range(0, sy):
            ys = min(stride_xy * y,hh-patch_size[1])
            for z in range(0, sz):
                zs = min(stride_z * z, dd-patch_size[2])
                test_patch = image[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]]
                test_patch = np.expand_dims(np.expand_dims(test_patch,axis=0),axis=0).astype(np.float32)
                test_patch = torch.from_numpy(test_patch).cuda()

                with torch.no_grad():
                    y_logit = net(test_patch)
                    num_outputs = len(y_logit)
                    y=torch.zeros(y_logit[0].shape).cuda()
                    for idx in range(num_outputs):
                        y += y_logit[idx]
                    y/=num_outputs
                    
                y = y.cpu().data.numpy()
                y = y[0,1,:,:,:]
                score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                  = score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + y
                cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                  = cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + 1

    score_map = score_map/np.expand_dims(cnt,axis=0)
    label_map = (score_map[0]>0.5).astype(np.int)
    if add_pad:
        label_map = label_map[wl_pad:wl_pad+w,hl_pad:hl_pad+h,dl_pad:dl_pad+d]
        score_map = score_map[:,wl_pad:wl_pad+w,hl_pad:hl_pad+h,dl_pad:dl_pad+d]
    return label_map, score_map
