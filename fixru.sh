F1=auto_round_extension/ark/test/bench_compare_mxfp4_moe.py
F2=auto_round_extension/ark/test/bench_mxfp8_mxfp4_moe.py

chmod +x "$F1" "$F2"
git update-index --chmod=+x "$F1" "$F2"   # 让 git 也记下可执行权限

pre-commit run --all-files                 # 应全部 Passed
git add -A
git commit -m "Fix ruff EXE001: make bench scripts executable; apply ruff auto-fixes"
git push origin feat/mxfp8-mxfp4-moe-bdpas
