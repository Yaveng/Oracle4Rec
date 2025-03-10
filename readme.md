Hi there👋

This is the official implementation of **Oracle4Rec**, which is accepted by WSDM 2025.

We hope this code helps you well. If you use this code in your work, please cite our paper.

```
Oracle-guided Dynamic User Preference Modeling for Sequential Recommendation
Jiafeng Xia, Dongsheng Li, Hansu Gu, Tun Lu, Peng Zhang, Li Shang and Ning Gu.
The 18th ACM International Conference on Web Search and Data Mining (WSDM). 2025
```



### How to run this code

##### Step 1: Check the compatibility of your running environment. Generally, different running environments will still have a chance to cause different experimental results though all random processes are fixed in the code. Our running environments are

```
- CPU: Intel(R) Xeon(R) Silver 4110
- GPU: Nvidia TITAN V (12GB)
- Memory: 376.6 GB
- Operating System: Ubuntu 16.04
- CUDA Version: 10.1
- Python Packages:
  - numpy: 1.19.2
  - pandas: 1.1.3
  - python: 3.6
  - pytorch: 1.7.0
  - scikit-learn: 0.23.2
  - scipy: 1.5.2
  
OR

- CPU: Intel(R) Xeon(R) Gold 5218
- GPU: Nvidia GeForce RTX 2080 Ti (11GB)
- Memory: 471.7 GB
- Operating System: Ubuntu 16.04
- CUDA Version: 10.0
- Python Packages:
  - numpy: 1.19.2
  - pandas: 1.1.3
  - python: 3.6
  - pytorch: 1.9.0
  - scikit-learn: 0.23.2
  - scipy: 1.5.2
```

Both environments have passed our test.



##### Step 2: prepare the dataset.

Please put the datasets under the directory `data/`. If you use your own datasets, check their format so as to make sure that it matches the input format of `Oracle4Rec`.



##### Step 3: Run the code.

* For `ML1M` dataset, please use the following code:

  ```sh
  python main.py --data_name ml1m
  ```

* For `Beauty` dataset, please use the following code:

  ```sh
  python main.py --data_name Beauty --lr 0.001 --num_filter_layers 3 --alpha 0.005 --hidden_size 128 --num_hidden_layers 3 --decay_factor 0.05 --ratio 0.75
  ```


* For ```Sports``` dataset, please use the following code:

  ```sh
  python main.py --data_name Sports --lr 0.001 --num_filter_layers 3 --alpha 0.01 --hidden_size 128 --num_hidden_layers 3 --decay_factor 0.05 --ratio 0.75
  ```

  

### Acknowledgement

Our code is built upon [FMLP-Rec](https://github.com/Woeee/FMLP-Rec), we thank authors for their efforts.
