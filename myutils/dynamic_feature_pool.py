import torch
import random
import numpy as np

class DynamicFeaturePool:
    def __init__(self, num_labeled_samples, num_cls):
        self.memory_feature = [None] * num_labeled_samples
        self.memory_label = [None] * num_labeled_samples
        self.num_cls = num_cls
        self.alpha = 0.9  # Weight Factors

    def update_labeled_features(self, features, labels, idxs):
        for i in range(len(idxs)):
            idx = idxs[i].cpu().numpy()
            new_feature = features[i].detach().cpu().numpy()

            if self.memory_feature[idx] is not None and self.memory_feature[idx].shape == new_feature.shape:
                updated_feature = self.alpha * self.memory_feature[idx] + (1 - self.alpha) * new_feature
                self.memory_feature[idx] = updated_feature
            else:
                self.memory_feature[idx] = new_feature

            self.memory_label[idx] = labels[i].detach().cpu().numpy()

    def sample_labeled_features(self, num_sampled_per_cls):
        tmp_feature_list = []
        tmp_label_list = []
        for i in range(len(self.memory_label)):
            if self.memory_label[i] is not None:
                tmp_feature_list.append(self.memory_feature[i])
                tmp_label_list.append(self.memory_label[i])
        tmp_feature_list = np.concatenate(tmp_feature_list, axis=0)
        tmp_label_list = np.concatenate(tmp_label_list, axis=0)
        selected_feature_list = []
        for c in range(self.num_cls):
            mask_c = tmp_label_list == c
            features_c = tmp_feature_list[mask_c]
            if features_c.shape[0] >= num_sampled_per_cls:
                num_features = features_c.shape[0]
                # 选择置信度高的部分样本
                high_conf_indices = random.sample(range(num_features), num_sampled_per_cls // 2)
                high_conf_features = features_c[high_conf_indices]

                # 随机选择一些置信度较低的样本
                low_conf_indices = random.sample(range(num_features), num_sampled_per_cls // 2)
                low_conf_features = features_c[low_conf_indices]

                # 合并高置信度和低置信度样本
                selected_features = np.concatenate((high_conf_features, low_conf_features), axis=0)
                selected_feature_list.append(selected_features)
            else:
                selected_feature_list.append(None)
        return selected_feature_list

def sample_labeled_features_from_both_memory_bank(memory_a, memory_b, num_sampled_per_cls):
    tmp_feature_list_a, tmp_feature_list_b = [], []
    tmp_label_list = []
    for i in range(len(memory_a.memory_label)):
        if memory_a.memory_label[i] is not None:
            tmp_feature_list_a.append(memory_a.memory_feature[i])
            tmp_feature_list_b.append(memory_b.memory_feature[i])
            tmp_label_list.append(memory_a.memory_label[i])
    tmp_feature_list_a = np.concatenate(tmp_feature_list_a, axis=0)
    tmp_feature_list_b = np.concatenate(tmp_feature_list_b, axis=0)
    tmp_label_list = np.concatenate(tmp_label_list, axis=0)

    selected_feature_list_a, selected_feature_list_b = [], []
    for c in range(memory_a.num_cls):
        mask_c = tmp_label_list == c
        features_a = tmp_feature_list_a[mask_c]
        features_b = tmp_feature_list_b[mask_c]
        if features_a.shape[0] >= num_sampled_per_cls:
            num_features = features_a.shape[0]
            # 选择置信度高的部分样本
            high_conf_indices = random.sample(range(num_features), num_sampled_per_cls // 2)
            high_conf_features_a = features_a[high_conf_indices]
            high_conf_features_b = features_b[high_conf_indices]

            # 随机选择一些置信度较低的样本
            low_conf_indices = random.sample(range(num_features), num_sampled_per_cls // 2)
            low_conf_features_a = features_a[low_conf_indices]
            low_conf_features_b = features_b[low_conf_indices]

            # 合并高置信度和低置信度样本
            selected_features_a = np.concatenate((high_conf_features_a, low_conf_features_a), axis=0)
            selected_features_b = np.concatenate((high_conf_features_b, low_conf_features_b), axis=0)

            selected_feature_list_a.append(selected_features_a)
            selected_feature_list_b.append(selected_features_b)
        else:
            selected_feature_list_a.append(None)
            selected_feature_list_b.append(None)
    return selected_feature_list_a, selected_feature_list_b
