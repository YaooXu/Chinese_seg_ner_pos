# 基于BILSTM-CRF的中文分词系统

## 预测文件的方式
```bash
python3 test.py --test_file ./icwb2-data/testing/pku_test.utf8  --target_file ./result
```
--test_file: 要预测的原文件

--target_file: 生成的结果文件

> 模型已经默认选定，由于默认使用的是CPU，预测时间稍微有点长

## 运行demo
```bash
python3 demo.py
```

