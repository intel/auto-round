cmake -B build -DCMAKE_BUILD_TYPE=Release -DARK_UT=ON -GNinja
cmake --build build -j 32
cmake -B xbuild -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=icx -DARK_XPU=ON -DARK_UT=ON -GNinja
cmake --build xbuild -j 32