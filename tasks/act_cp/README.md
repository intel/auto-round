Task:
    - analysis the peak memory in theory for auto-round tuning
    - Reduce the peak cuda memory usage in auto-round tuning by leverage the pytorch activation checkpoint
    - The correctness MUST be ENSURE

Note:
- quant model code: 
- activation checkpoint: https://pytorch.org/blog/activation-checkpointing-techniques/
- python env /home/yiliu7/workspace/ar-local/bin/python
- using gpu2 for all test
- 