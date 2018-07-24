import * as tfc from '@tensorflow/tfjs-core';
import {ShapeBackend} from './shape_backend';

export * from './shape_backend';

export const SHAPE_BACKEND_NAME = 'tfjs-shapes';

tfc.ENV.registerBackend(SHAPE_BACKEND_NAME, () => new ShapeBackend());

/** this would be cool but needs TS 3.0
 * type ArgTypes<Fn> =
  Fn extends (...args: infer ArgT) => any ? ArgT : any[]; 
 */

export function shapesOnly(fn: () => any) {
  const currBackend = tfc.getBackend();
  tfc.setBackend(SHAPE_BACKEND_NAME);
  const ret = tfc.tidy(fn);
  tfc.setBackend(currBackend);
  if(ret instanceof Promise) {
    throw new Error('shapesOnly doesn\'t support async functions');
  }
  return ret;
};
