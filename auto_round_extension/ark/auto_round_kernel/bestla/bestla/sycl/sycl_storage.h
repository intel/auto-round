//  Copyright (c) 2023 Intel Corporation
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
#pragma once
#include "sycl_utils.h"
#include "bestla/bestla_storage.h"

namespace bestla {
namespace sycl_storage {
class StorageWeightKBlockNInteger {
 public:
  BTLA_PROLOGUEB_IDS mPrologueID = BTLA_PROLOGUEB_IDS::Undef;
  uint64_t mCoreId = 0;
  BTLA_DTYPE mDType = BTLA_DTYPE::F32;
  BTLA_DTYPE mScaT = BTLA_DTYPE::F32, mZpT = BTLA_DTYPE::F32, mRedT = BTLA_DTYPE::S32;
  int mNPad = 0, mKPad = 0;
  int mN = 0, mK = 0;
  int mBlockSize = 1;
  int mDqBlockSize = 0;
  int mCStep = 0;
  int8_t* mQBuf = nullptr;
  size_t mWSize = 0;
  int8_t* mSBuf = nullptr;
  size_t mCSize = 0;
  int8_t *mZpBuf = nullptr, *mRedBuf = nullptr;
  size_t mZpSize = 0, mRedSize = 0;
  int8_t* mDQCorrectionBuf = nullptr;
  int8_t* mShuffleIndices = nullptr;
  size_t mDQCorSize = 0, mShufSize = 0;

  StorageWeightKBlockNInteger(bestla::storage::gemm::StorageWeightKBlockNInteger& _hoststor) {
    mPrologueID = BTLA_PROLOGUEB_IDS::WeightKBlockNInteger;
    mCoreId = 0;
    mDType = _hoststor.mDType;
    mScaT = _hoststor.SDtype();
    mZpT = _hoststor.ZDtype();
    mNPad = _hoststor.mNPad;
    mKPad = _hoststor.mKPad;
    mN = _hoststor.mN;
    mK = _hoststor.mK;
    mBlockSize = _hoststor.mBlockSize;
    mDqBlockSize = _hoststor.mDqBlockSize;
    mWSize = _hoststor.template WSize<int8_t>();
    mCSize = _hoststor.CSize() * utils::bestla_dtype_bytes(mScaT);
    mCStep = _hoststor.CStep();

    if (_hoststor.template ZPtr<void>()) {
      mZpSize = _hoststor.CSize() * utils::bestla_dtype_bytes(mZpT);
    }
    if (_hoststor.template RPtr<void>()) {
      mRedSize = _hoststor.CSize() * utils::bestla_dtype_bytes(mRedT);
    }
    // TODO DQ,shuffle support
  }

  size_t getDeviceSize() { return mWSize + mCSize + mZpSize + mRedSize + mDQCorSize + mShufSize; }

  void assign(int8_t* dptr) {
    mQBuf = dptr;
    dptr += mWSize;
    mSBuf = dptr;
    dptr += mCSize;
    if (mZpSize) {
      mZpBuf = dptr;
      dptr += mZpSize;
    }
    if (mRedSize) {
      mRedBuf = dptr;
      dptr += mRedSize;
    }
    if (mDQCorSize) {
      mDQCorrectionBuf = dptr;
      dptr += mDQCorSize;
    }
    if (mShuffleIndices) {
      mDQCorrectionBuf = dptr;
      dptr += mShufSize;
    }
  }

  void fromHost(bestla::storage::gemm::StorageWeightKBlockNInteger& _hoststor, sycl::queue* queue) {
    if (_hoststor.template WPtr<void>() && mQBuf) {
      queue->memcpy(mQBuf, _hoststor.template WPtr<void>(), mWSize);
    }
    if (_hoststor.template SPtr<void>() && mSBuf) {
      queue->memcpy(mSBuf, _hoststor.template SPtr<void>(), mCSize);
    }
    if (_hoststor.template ZPtr<void>() && mZpBuf) {
      queue->memcpy(mZpBuf, _hoststor.template ZPtr<void>(), mZpSize);
    }
    if (_hoststor.template RPtr<void>() && mRedBuf) {
      queue->memcpy(mRedBuf, _hoststor.template RPtr<void>(), mRedSize);
    }
    queue->wait();
  }

  void toHost(bestla::storage::gemm::StorageWeightKBlockNInteger& _hoststor, sycl::queue* queue) {
    if (mQBuf) {
      queue->memcpy(_hoststor.template WPtr<void>(), mQBuf, mWSize);
    }
    if (mSBuf) {
      queue->memcpy(_hoststor.template SPtr<void>(), mSBuf, mCSize);
    }
    if (mZpBuf) {
      queue->memcpy(_hoststor.template ZPtr<void>(), mZpBuf, mZpSize);
    }
    if (mRedBuf) {
      queue->memcpy(_hoststor.template RPtr<void>(), mRedBuf, mRedSize);
    }
    queue->wait();
  }
};
}  // namespace sycl_storage
}  // namespace bestla
