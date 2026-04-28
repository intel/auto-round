# # Copyright (C) 2026 Intel Corporation
# # SPDX-License-Identifier: Apache-2.0

cmake -B build -DCMAKE_BUILD_TYPE=Release -DARK_UT=ON
cmake --build build -j 32
cmake -B xbuild -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=icpx -DARK_XPU=ON -DARK_UT=ON
cmake --build xbuild -j 32