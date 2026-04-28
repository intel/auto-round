# rm -rf build
rm -f ./auto_round_kernel/*.so
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 32
cp build/auto_round_kernel*.so ./auto_round_kernel
# rm -rf xbuild
# -DBTLA_UT_DEBUG=ON -DBTLA_UT_BENCHMARK=ON
cmake -B xbuild -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=icx -DARK_XPU=ON  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build xbuild -j 32
cp xbuild/auto_round_kernel*.so ./auto_round_kernel