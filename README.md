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
> Python 3 reorganised the standard library and moved several functions to different modules.
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


Refresh memory about numpy array stuff
[https://www.python-course.eu/numpy_create_arrays.php](https://www.python-course.eu/numpy_create_arrays.php)


```python
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


25 Dec 2017  23:55

Converting start making sense if take small picture 5*5 RGBA
and play with it

This was [quick-tour-of-numpy](http://www.pythoninformer.com/python-libraries/numpy/quick-tour-of-numpy/) useful

❓ Question about that weird CIFAR transformation still stands

📝 It would be cool to write converter which takes arrays, and put numbers for each pixel together by small font and on top of it bigger pixel with generated colour. That'll be very informative and handy.


26 Dec 2017 16:55
### KNN classifier

cs231n put too many cool-compicated things in one bowl so it becomes black box.

let's do simple KNN things

[medium/k-nearest-neighbors](https://medium.com/@adi.bronshtein/a-quick-introduction-to-k-nearest-neighbors-algorithm-62214cea29c7)  

good to go through later [SkiKit MachineLearning](http://scikit-learn.org/stable/tutorial/statistical_inference/supervised_learning.html)

##### KNN is a non-parametric, lazy learning algorithm.
`non-parametric`**does not make any assumptions on the underlying data distribution.**

> Therefore, KNN could and probably should be one of the first choices for a classification study when there is little or no prior knowledge about the distribution data.

`lazy`(as opposed to an eager algorithm). **does not use the training data points to do any generalization**

Then deeper and deeper
Distance metric, similarity function
all-together brings me to
[ImageSearch GettingStarted](https://www.pyimagesearch.com/2014/01/27/hobbits-and-histograms-a-how-to-guide-to-building-your-first-image-search-engine-in-python/)
Even that not deep enough

Depper will be histogram calculation


[opencv.org/3.1.0/.../histogram_begins.html](https://docs.opencv.org/3.1.0/d1/db7/tutorial_py_histogram_begins.html)

> could be useful reading  
> [Photo Termins](https://www.cambridgeincolour.com/learn-photography-concepts.htm)

I want to take histogram from camera and show it on the screen
