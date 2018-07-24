# Tensorflow.js Shape Backend

This is a backend for [Tensorflow.js](https://github.com/tensorflow/tfjs) that just computes the shapes and dtypes of tensors. 

This could be used to build an editor extension that lets you get the shape of a tensor on hover, or to build asynchronous computation backends (Tensorflow.js needs to know the shape of tensors synchronously, but is fine with the actual computation being asynchronous).


## Usage


```js
import * as tf from '@tensorflow/tfjs'
import 'tfjs-shapes'

tf.setBackend('tfjs-shapes')

const ones = tf.ones([10, 10, 10, 10])
const pooled = tf.maxPool(ones, 2, 2, 'same')

console.log(pooled)
/** should print something like
 * Tensor {
  isDisposedInternal: false,
  size: 2500,
  shape: [ 10, 5, 5, 10 ],
  dtype: 'float32',
  strides: [ 250, 50, 10 ],
  dataId: {},
  id: 3,
  rankType: '4' }
  */

```

Or

```js
import { shapesOnly } from 'tfjs-shapes'

const { shape, dtype } = shapesOnly(() => {
    const ones = tf.ones([10, 10, 10, 10])
    const pooled = tf.maxPool(ones, 2, 2, 'same')
    return pooled
})
// shape === [10, 5, 5, 10]
// dtype === 'float32'

```



### TODO

* Implement 
    * BatchToSpaceND & SpaceToBatchND
* More tests
* Better name for `shapesOnly`

