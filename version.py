# vim: set tabstop=8 softtabstop=0 expandtab shiftwidth=4 smarttab

# KappaMask predictor version and changelog.
#
# Copyright 2021 - 2022 KappaZeta Ltd.
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

__version__ = '1.1.0'
min_cm_vsm_version = '0.2.6'

# 1.0.5 - Calls to cm_vsm now less dependent on platform. Switch from miniconda to micromamba.
# 1.0.4 - Support processing withing polygon-limited area of interest. Sub-tiles no longer flipped.
# 1.0.3 - Mosaic performance optimization.
# 1.0.2 - Mosaic function is unified.
# 1.0.1 - L1C support, new weights files.
# 1.0.0 - km_predict version implementation, logger implementation, image rotating  on re-creation fix.
