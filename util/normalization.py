# vim: set tabstop=8 softtabstop=0 expandtab shiftwidth=4 smarttab

# Statistics calculation for normalization of samples during prediction.
#
# Copyright 2021 KappaZeta Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np


def set_normalization(generator, split, sub_batch):
    samples = len(split) // sub_batch
    sum_std = []
    sum_mean = []
    all_min = []
    all_max = []
    for i in range(sub_batch):
        curr_std, curr_mean_list, curr_unique_list, curr_min, curr_max = generator.get_normal_par(
            split[i * samples:(i + 1) * samples])
        sum_std.append(curr_std)
        sum_mean.append(curr_mean_list)
        all_min.append(curr_min)
        all_max.append(curr_max)

    sum_std = np.asarray(sum_std)
    sum_mean = np.asarray(sum_mean)
    all_min = np.asarray(all_min)
    all_max = np.asarray(all_max)

    final_std = np.sum(sum_std, axis=0)
    final_std = final_std / sub_batch
    final_mean = np.sum(sum_mean, axis=0)
    final_mean = final_mean / sub_batch
    final_min = np.min(all_min, axis=0)
    final_max = np.min(all_max, axis=0)

    generator.set_std(final_std.tolist())
    generator.set_means(final_mean.tolist())
    generator.set_min(final_min.tolist())
    generator.set_max(final_max.tolist())
    return final_std.tolist(), final_mean.tolist(), final_min.tolist(), final_max.tolist()
