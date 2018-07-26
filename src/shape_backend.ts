/**
 * @license
 * Copyright 2018 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import {DataType, KernelBackend, Rank, Tensor, Tensor1D, Tensor2D, Tensor3D, Tensor4D} from '@tensorflow/tfjs-core';
import * as axis_util from './axis_util';
import * as broadcast_util from './broadcast_util';
import * as concat_util from './concat_util';
import {Conv2DInfo} from './conv_util';
import {getStridedSlicedInfo} from './slice_util';
import * as types from './types';

type TypedArray = Float32Array|Int32Array|Uint8Array;

function unimplemented(name: string): never {
  throw new Error(`${name} not implemented`);
}

export class MetadataTensor<T extends Rank> extends Tensor<T> {
  public static of(shape: Tensor['shape'], dtype: Tensor['dtype']) {
    return new Tensor(shape.slice(), dtype);
  }
}

export type DataId = object;

export interface BackendTimingInfo {
  kernelMs: number;
}

export class ShapeBackend implements KernelBackend {
  private data_ = new WeakSet<DataId>();
  private tensorCount_ = 0;

  // Unary op that doesn't change tensor shape
  private mappingOp_<T extends Tensor>(x: T, ...args: any[]): T {
    return MetadataTensor.of(x.shape, x.dtype) as T;
  }

  public reverse = this.mappingOp_;

  public neg = this.mappingOp_;

  public ceil = this.mappingOp_;

  public floor = this.mappingOp_;

  public sign = this.mappingOp_;

  public round = this.mappingOp_;

  public exp = this.mappingOp_;

  public expm1 = this.mappingOp_;

  public log = this.mappingOp_;

  public log1p = this.mappingOp_;

  public sqrt = this.mappingOp_;

  public rsqrt = this.mappingOp_;

  public square = this.mappingOp_;

  public reciprocal = this.mappingOp_;

  public relu = this.mappingOp_;

  public elu = this.mappingOp_;

  public eluDer = this.mappingOp_;

  public selu = this.mappingOp_;

  public clip = this.mappingOp_;

  public abs = this.mappingOp_;

  public sigmoid = this.mappingOp_;

  public softplus = this.mappingOp_;

  public sin = this.mappingOp_;

  public cos = this.mappingOp_;

  public tan = this.mappingOp_;

  public asin = this.mappingOp_;

  public acos = this.mappingOp_;

  public atan = this.mappingOp_;

  public atan2 = this.mappingOp_;

  public sinh = this.mappingOp_;

  public cosh = this.mappingOp_;

  public tanh = this.mappingOp_;

  public asinh = this.mappingOp_;

  public acosh = this.mappingOp_;

  public atanh = this.mappingOp_;

  public erf = this.mappingOp_;

  public step = this.mappingOp_;

  public read(dataId: DataId): Promise<TypedArray> {
    if (!this.data_.has(dataId)) {
      throw new Error('DataId not registered');
    }
    return Promise.resolve(new Uint8Array(0));
  }

  public readSync(dataId: DataId): TypedArray {
    if (!this.data_.has(dataId)) {
      throw new Error('DataId not registered');
    }
    return new Uint8Array(0);
  }

  public disposeData(dataId): void {
    if (this.data_.has(dataId)) {
      this.data_.delete(dataId);
      this.tensorCount_--;
    }
  }
  public write(dataId: DataId, values: TypedArray): void {}

  public fromPixels(
      pixels: ImageData|HTMLImageElement|HTMLCanvasElement|HTMLVideoElement,
      numChannels: number): Tensor3D {
    return MetadataTensor.of(
               [pixels.width, pixels.height, numChannels], 'int32') as
        MetadataTensor<Rank.R3>;
  }

  public register(dataId: DataId, shape: number[], dtype: DataType): void {
    if (this.data_.has(dataId)) {
      throw new Error('DataId is already registered');
    }
    this.data_.add(dataId);
    this.tensorCount_++;
  }

  public memory(): {unreliable: boolean;} {
    return {unreliable: true, numTensors: this.tensorCount_} as {
      unreliable: boolean;
    };
  }  // Backend-specific information.

  public time(f: () => void): Promise<BackendTimingInfo> {
    const st = performance.now();
    f();
    const et = st - performance.now();
    return Promise.resolve({kernelMs: et});
  }

  public matMul(
      a: Tensor2D, b: Tensor2D, transposeA: boolean,
      transposeB: boolean): MetadataTensor<Rank.R2> {
    const leftDim = transposeA ? a.shape[1] : a.shape[0];
    const rightDim = transposeB ? b.shape[0] : b.shape[1];
    return MetadataTensor.of([leftDim, rightDim], a.dtype) as
        MetadataTensor<Rank.R2>;
  }

  public slice<T extends Tensor>(x: T, begin: number[], size: number[]): T {
    return MetadataTensor.of(size, x.dtype) as T;
  }

  public stridedSlice<T extends Tensor>(
      x: T, begin: number[], end: number[], strides: number[],
      beginMask: number, endMask: number): T {
    const [, size] =
        getStridedSlicedInfo(x.shape, begin, end, strides, beginMask, endMask);

    if (size.some((axis) => axis === 0)) {
      return MetadataTensor.of([], x.dtype) as T;
    }

    return MetadataTensor.of(size, x.dtype) as T;
  }

  // Concats 2d tensors along axis=1. See comments in MathBackend.concat().
  public concat(a: Tensor2D, b: Tensor2D): Tensor2D {
    const outShape = concat_util.computeOutShape(
                         a.shape, b.shape, 1 /* axis */) as [number, number];

    return MetadataTensor.of(outShape, a.dtype) as MetadataTensor<Rank.R2>;
  }

  public add(a: Tensor, b: Tensor): Tensor {
    return this.broadcastedBinaryOp_(
               a, b, types.upcastType(a.dtype, b.dtype),
               (aValue, bValue) => aValue + bValue) as Tensor;
  }

  public subtract(a: Tensor, b: Tensor): Tensor {
    return this.broadcastedBinaryOp_(
               a, b, types.upcastType(a.dtype, b.dtype),
               (aValue, bValue) => aValue - bValue) as Tensor;
  }

  public pow<T extends Tensor>(a: T, b: Tensor): T {
    return this.broadcastedBinaryOp_(
               a, b, a.dtype, (aValue, bValue) => Math.pow(aValue, bValue)) as
        T;
  }

  public multiply(a: Tensor, b: Tensor): Tensor {
    return this.broadcastedBinaryOp_(
               a, b, types.upcastType(a.dtype, b.dtype),
               (aValue, bValue) => aValue * bValue) as Tensor;
  }

  public realDivide(a: Tensor, b: Tensor): Tensor {
    const op = (a: number, b: number) => a / b;
    const outputDtype = 'float32';
    return this.broadcastedBinaryOp_(a, b, outputDtype, op) as Tensor;
  }

  public floorDiv(a: Tensor, b: Tensor): Tensor {
    const op = (a: number, b: number) => Math.floor(a / b);
    const outputDtype = 'int32';
    return this.broadcastedBinaryOp_(a, b, outputDtype, op) as Tensor;
  }

  public sum(x: Tensor, axes: number[]): Tensor {
    axis_util.assertAxesAreInnerMostDims('sum', axes, x.rank);
    const [outShape] = axis_util.computeOutAndReduceShapes(x.shape, axes);
    const resultDtype = types.upcastType(x.dtype, 'int32');
    return MetadataTensor.of(outShape, resultDtype);
  }

  public unsortedSegmentSum<T extends Tensor>(
      x: T, segmentIds: Tensor1D, numSegments: number): Tensor {
    const inShape = x.shape.slice();
    inShape[0] = numSegments;
    return MetadataTensor.of(inShape, x.dtype);
  }

  public argMin(x: Tensor, axis: number): Tensor {
    const axes = [axis];
    axis_util.assertAxesAreInnerMostDims('argMin', axes, x.rank);
    const [outShape] = axis_util.computeOutAndReduceShapes(x.shape, axes);
    return MetadataTensor.of(outShape, 'int32');
  }

  public argMax(x: Tensor, axis: number): Tensor {
    const axes = [axis];
    axis_util.assertAxesAreInnerMostDims('argMax', axes, x.rank);
    const [outShape] = axis_util.computeOutAndReduceShapes(x.shape, axes);
    return MetadataTensor.of(outShape, 'int32');
  }

  public cumsum(x: Tensor, axis: number, exclusive: boolean, reverse: boolean):
      Tensor {
    if (axis !== x.rank - 1) {
      throw new Error(
          `backend.cumsum in CPU expects an inner-most axis=${x.rank - 1} ` +
          `but got axis=${axis}`);
    }
    const resultDtype = types.upcastType(x.dtype, 'int32');
    return MetadataTensor.of(x.shape, resultDtype);
  }

  public equal(a: Tensor, b: Tensor): Tensor {
    return this.broadcastedBinaryOp_(a, b, 'bool', (aVal, bVal) => {
      return (aVal === bVal) ? 1 : 0;
    });
  }

  public notEqual(a: Tensor, b: Tensor): Tensor {
    return this.broadcastedBinaryOp_(a, b, 'bool', (aVal, bVal) => {
      return (aVal !== bVal) ? 1 : 0;
    });
  }

  public less(a: Tensor, b: Tensor): Tensor {
    return this.broadcastedBinaryOp_(a, b, 'bool', (aVal, bVal) => {
      return (aVal < bVal) ? 1 : 0;
    });
  }

  public lessEqual(a: Tensor, b: Tensor): Tensor {
    return this.broadcastedBinaryOp_(a, b, 'bool', (aVal, bVal) => {
      return (aVal <= bVal) ? 1 : 0;
    });
  }

  public greater(a: Tensor, b: Tensor): Tensor {
    return this.broadcastedBinaryOp_(a, b, 'bool', (aVal, bVal) => {
      return (aVal > bVal) ? 1 : 0;
    });
  }

  public greaterEqual(a: Tensor, b: Tensor): Tensor {
    return this.broadcastedBinaryOp_(a, b, 'bool', (aVal, bVal) => {
      return (aVal >= bVal) ? 1 : 0;
    });
  }

  public logicalNot<T extends Tensor>(x: T): T {
    return MetadataTensor.of(x.shape, 'bool') as T;
  }

  public logicalAnd(a: Tensor, b: Tensor): Tensor {
    return this.broadcastedBinaryOp_(a, b, 'bool', (aVal, bVal) => {
      return aVal && bVal;
    });
  }

  public logicalOr(a: Tensor, b: Tensor): Tensor {
    return this.broadcastedBinaryOp_(a, b, 'bool', (aVal, bVal) => {
      return aVal || bVal;
    });
  }

  public where<T extends Tensor>(condition: T): Tensor2D {
    throw new Error(
        'Impossible to determine output shape for tf.where');
  }

  public select(condition: Tensor, a: Tensor, b: Tensor): Tensor {
    return MetadataTensor.of(a.shape, a.dtype);
  }

  public nonMaxSuppression(
      boxes: Tensor2D, scores: Tensor1D, maxOutputSize: number,
      iouThreshold: number, scoreThreshold: number): Tensor1D {
    throw new Error(
        'Impossible to determine output shape for tf.nonMaxSuppresion');
  }

  public topKIndices<T extends Tensor>(x: T, k: number): Tensor1D {
    return MetadataTensor.of([k], 'int32') as MetadataTensor<Rank.R1>;
  }

  public topKValues<T extends Tensor>(x: T, k: number): Tensor1D {
    return MetadataTensor.of([k], x.dtype) as MetadataTensor<Rank.R1>;
  }

  public topk<T extends Tensor>(x: T, k: number, sorted: boolean): [T, T] {
    return [
      MetadataTensor.of([x.shape[0], k], 'int32'),
      MetadataTensor.of([x.shape[0], k], x.dtype),
    ] as [T, T];
  }

  public min(x: Tensor, axes: number[]): Tensor {
    axis_util.assertAxesAreInnerMostDims('min', axes, x.rank);
    const [outShape] = axis_util.computeOutAndReduceShapes(x.shape, axes);
    return MetadataTensor.of(outShape, x.dtype);
  }

  public minimum(a: Tensor, b: Tensor): Tensor {
    return this.broadcastedBinaryOp_(
        a, b, a.dtype, (aVal, bVal) => Math.min(aVal, bVal));
  }

  public mod(a: Tensor, b: Tensor): Tensor {
    return this.broadcastedBinaryOp_(a, b, a.dtype, (aVal, bVal) => {
      const rem = aVal % bVal;
      if ((aVal < 0 && bVal < 0) || (aVal >= 0 && bVal >= 0)) {
        return rem;
      } else {
        return (rem + bVal) % bVal;
      }
    });
  }

  public max(x: Tensor, axes: number[]): Tensor {
    axis_util.assertAxesAreInnerMostDims('max', axes, x.rank);
    const [outShape] = axis_util.computeOutAndReduceShapes(x.shape, axes);
    return MetadataTensor.of(outShape, x.dtype);
  }

  public maximum(a: Tensor, b: Tensor): Tensor {
    return this.broadcastedBinaryOp_(
        a, b, a.dtype, (aVal, bVal) => Math.max(aVal, bVal));
  }

  public all(x: Tensor, axes: number[]): Tensor {
    axis_util.assertAxesAreInnerMostDims('all', axes, x.rank);
    const [outShape] = axis_util.computeOutAndReduceShapes(x.shape, axes);
    return MetadataTensor.of(outShape, x.dtype);
  }

  public any(x: Tensor, axes: number[]): Tensor {
    axis_util.assertAxesAreInnerMostDims('any', axes, x.rank);
    const [outShape] = axis_util.computeOutAndReduceShapes(x.shape, axes);
    return MetadataTensor.of(outShape, x.dtype);
  }

  public squaredDifference(a: Tensor, b: Tensor): Tensor {
    return this.broadcastedBinaryOp_(a, b, a.dtype, (aVal, bVal) => {
      const diff = aVal - bVal;
      return diff * diff;
    });
  }

  public int<T extends Tensor>(x: T): T {
    return MetadataTensor.of(x.shape, 'int32') as T;
  }

  public conv2d(x: Tensor4D, filter: Tensor4D, convInfo: Conv2DInfo): Tensor4D {
    return MetadataTensor.of(convInfo.outShape, x.dtype) as
        MetadataTensor<Rank.R4>;
  }

  public conv2dDerInput(dy: Tensor4D, filter: Tensor4D, convInfo: Conv2DInfo):
      Tensor4D {
    return MetadataTensor.of(convInfo.inShape, 'float32') as
        MetadataTensor<Rank.R4>;
  }

  public conv2dDerFilter(x: Tensor4D, dy: Tensor4D, convInfo: Conv2DInfo):
      Tensor4D {
    return MetadataTensor.of(convInfo.filterShape, 'float32') as
        MetadataTensor<Rank.R4>;
  }

  public depthwiseConv2D(x: Tensor4D, filter: Tensor4D, convInfo: Conv2DInfo):
      Tensor4D {
    return MetadataTensor.of(convInfo.outShape, x.dtype) as
        MetadataTensor<Rank.R4>;
  }

  public depthwiseConv2DDerInput(
      dy: Tensor4D, filter: Tensor4D, convInfo: Conv2DInfo): Tensor4D {
    return MetadataTensor.of(convInfo.inShape, 'float32') as
        MetadataTensor<Rank.R4>;
  }

  public depthwiseConv2DDerFilter(
      x: Tensor4D, dy: Tensor4D, convInfo: Conv2DInfo): Tensor4D {
    return MetadataTensor.of(convInfo.filterShape, 'float32') as
        MetadataTensor<Rank.R4>;
  }

  public tile<T extends Tensor>(x: T, reps: number[]): T {
    const newShape: number[] = new Array(x.rank);
    for (let i = 0; i < newShape.length; i++) {
      newShape[i] = x.shape[i] * reps[i];
    }
    return MetadataTensor.of(newShape, x.dtype) as T;
  }

  public pad<T extends Tensor>(
      x: T, paddings: Array<[number, number]>, constantValue: number): T {
    const outShape = paddings.map(
        (p, i) => p[0] /* beforePad */ + x.shape[i] + p[1] /* afterPad */);
    return MetadataTensor.of(outShape, x.dtype) as T;
  }

  public transpose<T extends Tensor>(x: T, perm: number[]): T {
    const newShape: number[] = new Array(x.rank);
    for (let i = 0; i < newShape.length; i++) {
      newShape[i] = x.shape[perm[i]];
    }
    return MetadataTensor.of(newShape, x.dtype) as T;
  }

  public gather<T extends Tensor>(x: T, indices: Tensor1D, axis: number): T {
    const newShape: number[] = x.shape.slice();
    newShape[axis] = indices.shape[0];
    return MetadataTensor.of(newShape, x.dtype) as T;
  }

  public batchToSpaceND<T extends Tensor>(
      x: T, blockShape: number[], crops: number[][]): T {
    return unimplemented('batchToShapeND');
  }

  public spaceToBatchND<T extends Tensor>(
      x: T, blockShape: number[], paddings: Array<[number, number]>): T {
    return unimplemented('batchToShapeND');
  }

  public maxPool(x: Tensor4D, convInfo: Conv2DInfo): Tensor4D {
    return this.pool_(x, convInfo, 'max');
  }

  public maxPoolBackprop(
      dy: Tensor4D, x: Tensor4D, y: Tensor4D, convInfo: Conv2DInfo): Tensor4D {
    return MetadataTensor.of(x.shape, 'float32') as MetadataTensor<Rank.R4>;
  }

  public avgPoolBackprop(dy: Tensor4D, x: Tensor4D, convInfo: Conv2DInfo):
      Tensor4D {
    return MetadataTensor.of(x.shape, 'float32') as MetadataTensor<Rank.R4>;
  }

  public cast<T extends Tensor<Rank>>(x: T, dtype: DataType): T {
    return MetadataTensor.of(x.shape, dtype) as T;
  }

  public reshape<T extends Tensor<Rank>, R extends Rank>(
      x: T, shape: Tensor['shape']): Tensor<R> {
    return MetadataTensor.of(shape, x.dtype) as Tensor<R>;
  }

  public avgPool(x: Tensor4D, convInfo: Conv2DInfo): Tensor4D {
    return this.pool_(x, convInfo, 'avg').toFloat();
  }

  public resizeBilinear(
      x: Tensor4D, newHeight: number, newWidth: number,
      alignCorners: boolean): Tensor4D {
    const [batch, , , numChannels] = x.shape;
    return MetadataTensor.of(
               [batch, newHeight, newWidth, numChannels], x.dtype) as
        MetadataTensor<Rank.R4>;
  }

  public resizeBilinearBackprop(
      dy: Tensor4D, x: Tensor4D, alignCorners: boolean) {
    return MetadataTensor.of(x.shape, x.dtype) as MetadataTensor<Rank.R4>;
  }

  public resizeNearestNeighbor(
      x: Tensor4D, newHeight: number, newWidth: number,
      alignCorners: boolean): Tensor4D {
    const [batch, , , numChannels] = x.shape;
    return MetadataTensor.of(
               [batch, newHeight, newWidth, numChannels], x.dtype) as
        MetadataTensor<Rank.R4>;
  }

  public resizeNearestNeighborBackprop(
      dy: Tensor4D, x: Tensor4D, alignCorners: boolean) {
    return MetadataTensor.of(x.shape, x.dtype) as MetadataTensor<Rank.R4>;
  }

  public batchNormalization(
      x: Tensor4D, mean: Tensor4D|Tensor1D, variance: Tensor4D|Tensor1D,
      varianceEpsilon: number, scale?: Tensor4D|Tensor1D,
      offset?: Tensor4D|Tensor1D): Tensor4D {
    return MetadataTensor.of(x.shape, x.dtype) as MetadataTensor<Rank.R4>;
  }

  public localResponseNormalization4D(
      x: Tensor4D, radius: number, bias: number, alpha: number,
      beta: number): Tensor4D {
    return MetadataTensor.of(x.shape, 'float32') as MetadataTensor<Rank.R4>;
  }

  public LRNGrad(
      dy: Tensor4D, inputImage: Tensor4D, outputImage: Tensor4D,
      depthRadius: number, bias: number, alpha: number,
      beta: number): Tensor4D {
    const batch = dy.shape[0];
    const rows = dy.shape[1];
    const cols = dy.shape[2];
    const depth = dy.shape[3];
    return MetadataTensor.of([batch, rows, cols, depth], 'float32') as
        MetadataTensor<Rank.R4>;
  }

  public multinomial(
      logits: Tensor2D, normalized: boolean, numSamples: number,
      seed: number): Tensor2D {
    return MetadataTensor.of([logits.shape[0], numSamples], 'int32') as
        MetadataTensor<Rank.R2>;
  }

  public oneHot(
      indices: Tensor1D, depth: number, onValue: number,
      offValue: number): Tensor2D {
    return MetadataTensor.of([indices.size, depth], 'int32') as
        MetadataTensor<Rank.R2>;
  }

  public dispose() {}

  private pool_(x: Tensor4D, convInfo: Conv2DInfo, poolType: 'max'|'avg'):
      Tensor4D {
    return MetadataTensor.of(convInfo.outShape, 'float32') as
        MetadataTensor<Rank.R4>;
  }

  private broadcastedBinaryOp_(
      a: Tensor, b: Tensor, dtype: DataType,
      op: (a: number, b: number) => number): Tensor {
    const newShape =
        broadcast_util.assertAndGetBroadcastShape(a.shape, b.shape);

    return MetadataTensor.of(newShape, dtype);
  }
}
