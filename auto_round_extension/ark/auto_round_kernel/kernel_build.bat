rd /s /q build
cmake -B build -DCMAKE_BUILD_TYPE=Release -GNinja
cmake --build build -j 32
copy build\auto_round_kernel*.pyd auto_round_kernel\
rd /s /q xbuild
cmake -B xbuild -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=icx -DARK_XPU=ON -GNinja
cmake --build xbuild -j 32
copy xbuild\auto_round_kernel*.pyd auto_round_kernel\