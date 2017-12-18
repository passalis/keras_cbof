
# Keras implementation of the CBoF method

This is an easy to use **keras** (re)-implementation of the Convolutional Bag-of-Features (CBoF) pooling method (as presented in [Bag-of-Features Pooling for Deep Convolutional Neural Networks](http://openaccess.thecvf.com/content_iccv_2017/html/Passalis_Learning_Bag-Of-Features_Pooling_ICCV_2017_paper.html)). CBoF is a useful tool that allows for **decoupling the size of the representation extracted** from a deep CNN from both the **size** and the **number** of the extracted feature maps. Therefore, it allows for **reducing the size** of the used CNNs as well as  for **improving the scale-invariance** of the models.

To use the CBoF pooling method, simply insert the *BoF_Pooling* layer between the last convolution and a fully connected layer:
```python

from cbof import BoF_Pooling, initialize_bof_layers
...
model.add(BoF_Pooling(n_codewords, spatial_level=0))
...
initialize_bof_layers(model, x_train)
```
Remember to initialize the BoF layer (the *initialize_bof_layers()* function automatically initializes all the BoF layers in a keras model). The number of codewords (that defines the dimensionality of the extracted representation) as well as the spatial level must be defined. Two spatial levels are current supported: 0 (no spatial segmentation) and 1 (spatial segmentation in 4 regions).


In [example.py](example.py) the ability of CBoF to reduce the size of a CNN used for classifying the digits of the MNIST dataset is demonstrated. Some results are reported bellow (for the SPP method, the implementation available [here](https://github.com/yhenon/keras-spp) was used).


**Baseline CNN**:

|Pooling  | FC Input | # Params | Error | Error ( Scale = 0.8) |
| ---             | ---   | ---       | ---   | ---   |
Max (4 f.)	  |	100   |	58,318    |	1.21  | -     |
Μax (32 f.)   | 800	  | 424,810	  |	0.65	| -     |
Max (64 f.)	  | 1600	| 843,658	  |	0.66  | -     |

(The notation *f* is used to refer to the number of *filters* in the last convolutional layer)

**Global Pooling**:


| Pooling  | FC Input | # Params | Error | Error ( Scale = 0.8) |
| ---             | ---   | ---       | ---   | ---   |
GMP (24 f.)	  | 24	  | 25,186		| 3.24  |	5.60  |
BoF (32 f., 16 c.)       |	16	  | 23,930		| **2.43**	| **3.90**	|
GMP (64 f.)	  | 64	  | 57,226		| 1.81  |	3.24  |
BoF (32 f., 64 c.)      |	64	  | 50,090		| **0.96**	| **2.24**	|
GMP (128 f.)	  | 128	  | 108,490 	| 1.35  |	3.48  |
BoF (32 f., 128 c.)      |	128	  | 84,970		| **0.78**	| **2.23**	| 

(The notation *f* is used to refer to the number of *filters* in the last convolutional layer, while the notation *c* is used to refer to the number of codewords)

**Spatial Pooling**:


| Pooling  | FC Input | # Params | Error | Error ( Scale = 0.8) |
| ---             | ---   | ---       | ---   | ---   |
SPP (8 f.)	    |	40	  | 28,754	  |	1.85  |	3.96  |
Spatial BoF (32, 8 c.)      |	32	  | 31,858		| **1.26**	| **2.18**	| 
SPP (16 f.)    |	80	  | 51,546	  |	1.25  |	2.61  |
Spatial BoF (32, 16 c.)     |	64	  | 48,506	  |	**0.81**	| **1.57**  |	
SPP (32 f. )   |	160	  | 97,130	  |	0.83  |	1.94  |
SPP (64 f. )   |	320	  | 188,298		  |	0.76  |	1.74  |
Spatial SBoF (64, 32 c.)	    |128	  | 92,074	  |	**0.61**	| **1.40**	| 

					
(The notation *f* is used to refer to the number of *filters*, while the notation *c* is used to refer to the number of codewords)         

If you use this implementation, please cite our paper:

<pre>
@InProceedings{Passalis_2017_ICCV,
author = {Passalis, Nikolaos and Tefas, Anastasios},
title = {Learning Bag-Of-Features Pooling for Deep Convolutional Neural Networks},
booktitle = {The IEEE International Conference on Computer Vision (ICCV)},
month = {Oct},
year = {2017}
}
</pre>

Note that there are some minor differences from the [original implementation](https://github.com/passalis/cbof) to improve the pooling performance.
