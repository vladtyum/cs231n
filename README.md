CS231n.md

http://cs231n.github.io/assignments2017/assignment1/
Following assignment1

Too 'out of the box' that im not even cathing it

CIFAR let's firstly get clear picture of it

I want to be able to load pictures, request, display them


This source is useful
[http://jetyoung.github.io/2015/12/18/Python ....](http://jetyoung.github.io/2015/12/18/Python%E8%AF%BB%E5%85%A5CIFAR-10%E6%95%B0%E6%8D%AE%E5%BA%93/)

problem with using cPickle
because it's not well ported for Py3

should be \_pickle [ link >>  -how-to-install-cpickle-on-python-3-4)](https://askubuntu.com/questions/742782/how-to-install-cpickle-on-python-3-4)

Meanwhile CS231n use
`from six.moves import cPickle as pickle`

six.moves used because
[link >> six](https://media.readthedocs.org/pdf/six/latest/six.pdf)
> Python 3 reorganized the standard library and moved several functions to different modules.
Six provides a consistent
interface to them through the fake
six.moves
module

So i'm going both ways
trying out \_pickle and six.moves import cPickle

here guy trying to do the same
[loading-an-image-from-cifar-10-dataset](https://stackoverflow.com/questions/45104970/loading-an-image-from-cifar-10-dataset)

tried import `from six.moves import cPickle as pickle`

and dir(pickle) gives much more stuff, maybe that's the reason why it's used by them


> here guy trying to do the same
[loading-an-image-from-cifar-10-dataset](https://stackoverflow.com/questions/45104970/loading-an-image-from-cifar-10-dataset)
there is Udacity folder, so most likely Udacity has cource for it
f = open('/home/jayanth/udacity/cifar-10-batches-py/data_batch_1', 'rb')


```python
from six.moves import cPickle as pickle
from  PIL import Image
import numpy as np
import matplotlib.pyplot as plt

f = open('cifar-10-batches-py/data_batch_1', 'rb')
tupled_data= pickle.load(f, encoding='bytes')
f.close()
img = tupled_data[b'data']
single_img = np.array(img[5])
#that reshaping didn't work
# single_img_reshaped = single_img.reshape(32,32,3)

# this works well
single_img_reshaped = np.transpose(np.reshape(single_img,(3, 32,32)), (1,2,0))

plt.imshow(single_img_reshaped)
plt.show()
```
Reshaping thing is a bit unclear, can take it as black box for now


Duck.Really forgot nparray stuff
getting back to [https://www.python-course.eu/numpy_create_arrays.php](https://www.python-course.eu/numpy_create_arrays.php)


```Python
x = np.array([ [67, 63, 87],
               [77, 69, 59],
               [85, 87, 99],
               [79, 72, 71],
               [63, 89, 93],
               [68, 92, 78]])
print(np.shape(x))
```
> The shape of an array tells us also something about the order in which the indices are processed, i.e.
> first rows, then columns and after that the further dimensions.


> shape" can also be used to change the shape of an array.

```python
x.shape = (3, 6)
print(x)
```

```python
[[67 63 87 77 69 59]
 [85 87 99 79 72 71]
 [63 89 93 68 92 78]]
 ```

if so why converstion image above has to be so complicated
`single_img_reshaped = np.transpose(np.reshape(single_img,(3, 32,32)), (1,2,0))`
[ ] let's try to play with resizing 
