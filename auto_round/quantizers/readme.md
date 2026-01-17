# AutoRound Quantizer
主要的功能组件，包含不同的算法，量化的具体执行逻辑。

## 结构与调用流程
AutoRundQuantizer根据粒度从大到小分为三层（可扩展）： algs、model_type、data_type，从每层中继承方法动态的构造一个Quantizers, 同层间互斥，不同层间可以自由组合。

AutoRoundQuantizer
- algs
    - RTN
    - Tuning(auto_round)
- model_type
    - llm
    - mllm
    - diffusion
- data_type
    - gguf
    - nvfp/mxfp
### 1. AutoRoundQuantizer
主入口，根据配置，使用__new__方法动态构造一个Quantizer, 从AlgsQuantizer, ModelTypeQuantizer, DataTypeQuantizer中继承方法，小粒度层可覆写大粒度层方法

### 2. AlgsQuantizer