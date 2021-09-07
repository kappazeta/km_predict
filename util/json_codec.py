# vim: set tabstop=8 softtabstop=0 expandtab shiftwidth=4 smarttab

# JSON encoder with support for Numpy int64.
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

import json
import numpy as np


class CMFJSONEncoder(json.JSONEncoder):
    """
    JSON encoder with support for numpy data types.
    """

    def default(self, obj):
        if isinstance(obj, np.int64):
            return int(obj)
        return super(CMFJSONEncoder, self).default(obj)
