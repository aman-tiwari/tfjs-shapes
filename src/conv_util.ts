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

export interface PadInfo {
  top: number,
  left: number,
  right: number,
  bottom: number,
  type: string
}

export interface Conv2DInfo {
  batchSize: number,
  inHeight: number,
  inWidth: number,
  inChannels: number,
  outHeight: number,
  outWidth: number,
  outChannels: number,
  dataFormat: "channelsFirst"|"channelsLast",
  strideHeight: number,
  strideWidth: number,
  dilationHeight: number,
  dilationWidth: number,
  filterHeight: number,
  filterWidth: number,
  padInfo: PadInfo,
  inShape: [number, number, number, number],
  outShape: [number, number, number, number],
  filterShape: [number, number, number, number]
}
