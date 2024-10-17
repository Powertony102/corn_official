import torch
import numpy as np

def coral_loss(source, target):
    """
    计算 CORAL 损失：源和目标特征协方差矩阵的 Frobenius 范数差异
    """
    # 计算Frobenius范数的平方
    d = source.size(1)
    loss = torch.norm(source - target, p='fro') ** 2 / (4 * d * d)
    return loss

def cal_coral_correlation(features, prob, label, memory, num_classes, temperature=0.1, num_filtered=64):
    correlation_list = []

    for i in range(num_classes):
        class_mask = label == i
        class_feature = features[class_mask, :]
        class_prob = prob[class_mask]
        
        
        # 选择置信度最高的部分特征
        _, high_conf_indices = torch.sort(class_prob, descending=True)
        high_conf_indices = high_conf_indices[:num_filtered // 4]
        high_conf_features = class_feature[high_conf_indices]
        
        # 随机选择一些置信度较低的特征
        low_conf_indices = torch.sort(class_prob, descending=False)[1][:num_filtered // 4]
        low_conf_features = class_feature[low_conf_indices]

        # 合并高置信度和低置信度特征
        selected_features = torch.cat((high_conf_features, low_conf_features), dim=0)

        logits_list = []
        for memory_c in memory:
            if memory_c is not None and selected_features.shape[0] > 1 and memory_c.shape[0] > 1:
                memory_c_tensor = torch.from_numpy(memory_c).cuda() if isinstance(memory_c, np.ndarray) else memory_c

                """
                I = torch.ones(selected_features.shape[0])
                n_s = selected_features.shape[0] - 1
                source_cov = (torch.matmul(selected_features.T, selected_features) - 
                            torch.matmul(torch.matmul(I, selected_features).T, torch.matmul(I.T, selected_features)) / n_s) / (n_s - 1)
                # source_cov = torch.matmul(selected_features.T, selected_features) / (selected_features.shape[0] - 1)
                # target_cov = torch.matmul(memory_c_tensor.T, memory_c_tensor) / (memory_c_tensor.shape[0] - 1)
                n_t = memory_c_tensor.shape[0]
                target_cov = (torch.matmul(memory_c_tensor.T, memory_c_tensor) - 
                            torch.matmul(torch.matmul(I, memory_c_tensor).T, torch.matmul(I.T, memory_c_tensor)) / n_t) / (n_t - 1)

                covariance_diff = coral_loss(source_cov, target_cov)
                """
                # 使用 torch.cov 计算协方差矩阵
                source_cov = torch.cov(selected_features.T)
                target_cov = torch.cov(memory_c_tensor.T)

                # 计算CORAL损失
                covariance_diff = coral_loss(source_cov, target_cov)
                # correlation_score = 1 / (1 + covariance_diff)
                # logits_list.append(correlation_score)
                correlation_list.append(covariance_diff)
        """
        if logits_list:
            logits = torch.stack(logits_list)
            correlation = torch.softmax(logits / temperature, dim=0)
            correlation_list.append(correlation)
        """

    if not correlation_list:
        return [], False
    else:
        correlation_list = torch.stack(correlation_list)  # 确保返回的结果是一个二维张量
        return correlation_list, True
