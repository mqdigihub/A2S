# A2S
Adaptive synchronous strategy (A2S) is a distributed training paradigm based on parameter server framework which improves the BSP and ASP paradigms by adaptively adopting different parallel training schemes for workers with different training speeds.
# Installation
Anaconda3  
Python3.6  
Tensorflow1.14.0
# Usage
Worker side:  
  python vggnet_worker.py --partition 0  
PS side:  
  python A2S.py --lr 0.9
# Publication
Miao-quan Tan, Wai-xi Liu*, Luo J, et al. Adaptive synchronous strategy for distributed machine learning[J]. International Journal of Intelligent Systems, 2022, 37(12): 11713-11741. https://onlinelibrary.wiley.com/doi/10.1002/int.23060
