------

This code develop a multi-task learning neural combinatorial optimization (NCO) model for cross-problem generalization for routing problems. 

![](.\results.jpg)

A comparison of gaps on eleven VRPs (Left: box plot, Right: radar plot). **ST** represents the unified model trained with single-task learning on CVRP, **ST\_all** represents the unified model with single-task learning on OVRPBLTW, and **MT** represents our approach, i.e., the unified model with multi-task learning on five VRPs. **ST\_FT** and **MT\_FT** are the fine-tuning models

## Files in MTNCO

+ MTPOMO: the implementation of multi-task learning with attribute composition based on POMO.
+ Trained_models: the pre-trained unified models with problem size 50 and 100 
+ Test_instances: test instances of 11 VRPs
+ utils: utils

## Train & Test

cd MTPOMO/POMO/

**Train:**  python train_n50.py

**Test:**  python test_n50.py



## Acknowledgements

Our implementation is based on the code of [POMO](https://github.com/yd-kwon/POMO/tree/master/NEW_py_ver). Thanks to them.