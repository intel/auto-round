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
#include "bestla.h"
#include "bestla_gemm.h"
#include "bestla_utils.h"

namespace bestla {
namespace storage {

constexpr size_t Alignment = 64;
class ISerialObject {
 protected:
  virtual size_t getSerializedSize() = 0;

  virtual void serializeToBuffer(int8_t*& wptr) = 0;

  virtual void deserializeBuffer(int8_t*& rptr, bool map_buf) = 0;
};

class ISerializable : public ISerialObject {
 public:
  virtual ~ISerializable() = default;

  virtual void assign(int8_t* buf) = 0;

  virtual void serialize(int8_t* wptr) = 0;

  virtual void deserialize(int8_t* rptr) = 0;
  size_t mSize = 0;

 protected:
  virtual size_t getSerializedSize() override {
    size_t totalsize = 0;
    totalsize += sizeof(mSize);
    return totalsize;
  }
  virtual void serializeToBuffer(int8_t*& wptr) override { utils::serialize(wptr, mSize); }
  virtual void deserializeBuffer(int8_t*& rptr, bool map_buf) override {
    if (!map_buf) {
      mSize = utils::deserialize<size_t>(rptr);
    } else {
      utils::serialize<size_t>(rptr, mSize);
    }
  }
};

class ISerialObjectV2 {
 public:
  virtual size_t get_serialized_size() { return get_misc_size() + get_buffer_size(); }
  virtual size_t get_misc_size() = 0;
  virtual size_t get_buffer_size() { return 0; };

  virtual void serialize_misc(int8_t*& wptr) = 0;
  virtual void deserialize_misc(int8_t*& wptr) = 0;
  virtual void set_buffer(int8_t*& bufptr){};
};

class ICollection {
 public:
  virtual ~ICollection() = default;
  BTLA_PROLOGUEB_IDS prologue_id_ = BTLA_PROLOGUEB_IDS::Undef;
  size_t misc_size_, buf_size_;
  static constexpr size_t get_misc_size() { return sizeof(prologue_id_) + sizeof(misc_size_) + sizeof(buf_size_); }

  virtual void set_buffers(int8_t* _buf) = 0;

  virtual void copy_buffers(int8_t* _dst_buf) = 0;

  virtual int8_t* serialize(int8_t* _ptr) {
    utils::serialize(_ptr, prologue_id_);
    utils::serialize(_ptr, misc_size_);
    utils::serialize(_ptr, buf_size_);
    return _ptr;
  }

  virtual int8_t* deserialize(int8_t* _ptr) {
    prologue_id_ = utils::deserialize<BTLA_PROLOGUEB_IDS>(_ptr);
    misc_size_ = utils::deserialize<size_t>(_ptr);
    buf_size_ = utils::deserialize<size_t>(_ptr);
    return _ptr;
  }
};

template <int ALIGN>
class ObjectAlignedBuffer : public ISerialObject {
 public:
  template <typename T>
  inline constexpr T* get() const {
    return reinterpret_cast<T*>(mBufPtr);
  }

  template <typename T>
  inline size_t size() {
    return mBufSize / sizeof(T);
  }

  void resize(size_t bytes) { mBufSize = bytes; }

  // ser
  int8_t* mBufPtr = nullptr;
  size_t mBufSize = 0;
  size_t mBufOffset = 0;

  size_t getSerializedSize() override {
    size_t totalsize = 0;
    totalsize += sizeof(mBufSize);
    totalsize += sizeof(mBufOffset);
    totalsize += mBufSize + ALIGN;
    return totalsize;
  }

  virtual void serializeToBuffer(int8_t*& wptr) override {
    utils::serialize(wptr, mBufSize);
    auto tmpptr = wptr + sizeof(mBufOffset);
    mBufOffset = utils::pointer_align<ALIGN>(tmpptr) - tmpptr;
    utils::serialize(wptr, mBufOffset);
    wptr += mBufOffset;
    if (wptr != mBufPtr) {
      std::memcpy(wptr, mBufPtr, mBufSize);
    }
    wptr += mBufSize;
  }
  virtual void deserializeBuffer(int8_t*& rptr, bool map_buf) override {
    if (!map_buf) {
      mBufSize = utils::deserialize<size_t>(rptr);
      mBufOffset = utils::deserialize<size_t>(rptr);
    } else {
      utils::serialize<size_t>(rptr, mBufSize);
      auto tmpptr = rptr + sizeof(mBufOffset);
      mBufOffset = utils::pointer_align<ALIGN>(tmpptr) - tmpptr;
      utils::serialize(rptr, mBufOffset);
    }
    rptr += mBufOffset;
    mBufPtr = rptr;
    rptr += mBufSize;
  }
};

class ObjectBuffer : public ISerialObjectV2 {
 public:
  // ser
  int8_t* buf_ptr_ = nullptr;
  size_t buf_size_ = 0;

  size_t get_misc_size() override {
    size_t totalsize = 0;
    totalsize += sizeof(buf_size_);
    return totalsize;
  }

  void serialize_misc(int8_t*& wptr) override { utils::serialize(wptr, buf_size_); }
  void deserialize_misc(int8_t*& rptr) override { buf_size_ = utils::deserialize<size_t>(rptr); }

  size_t get_buffer_size() override { return buf_size_; }

  void set_buffer(int8_t*& bufptr) override {
    buf_ptr_ = bufptr;
    bufptr += buf_size_;
  }
};

template <int ALIGN>
class ObjectOptionalBuffer : public ObjectAlignedBuffer<ALIGN> {
 public:
  void resize(size_t bytes) {
    ObjectAlignedBuffer<ALIGN>::resize(bytes);
    mNotEmpty = bytes > 0;
  }

  // ser
  bool mNotEmpty{false};

  virtual size_t getSerializedSize() override {
    size_t totalsize = 0;
    totalsize += sizeof(mNotEmpty);
    if (mNotEmpty) {
      totalsize += ObjectAlignedBuffer<ALIGN>::getSerializedSize();
    }
    return totalsize;
  }
  virtual void serializeToBuffer(int8_t*& wptr) override {
    utils::serialize(wptr, mNotEmpty);
    if (mNotEmpty) {
      ObjectAlignedBuffer<ALIGN>::serializeToBuffer(wptr);
    }
  }
  virtual void deserializeBuffer(int8_t*& rptr, bool map_buf) override {
    if (!map_buf) {
      mNotEmpty = utils::deserialize<bool>(rptr);
    } else {
      utils::serialize<bool>(rptr, mNotEmpty);
    }
    if (mNotEmpty) {
      ObjectAlignedBuffer<ALIGN>::deserializeBuffer(rptr, map_buf);
    }
  }
};

namespace gemm {
enum class Layout : uint32_t {
  PLAIN = 0,  // torch default N*K or M*K
  T = PLAIN,
  NT,
  NPack,
};

class ObjectQuantCorrection : public ISerialObject {
  // ser
 public:
  size_t mCSize = 0;
  int mCStep = 0;
  BTLA_DTYPE mScaT = BTLA_DTYPE::F32, mZpT = BTLA_DTYPE::F32;
  ObjectAlignedBuffer<Alignment> mScaleBuf;
  ObjectOptionalBuffer<Alignment> mZpBuf, mRedBuf;
  ObjectOptionalBuffer<Alignment> mDQCorrectionBuf;

  // non-ser
 public:
  int mScaEleSize = 0, mZpEleSize = 0;

  size_t resize(int Rows, int Step, BTLA_DTYPE scalet, BTLA_DTYPE zpt, bool _is_asym, bool _has_reduce) {
    mScaT = scalet;
    mZpT = zpt;
    updateSize();
    mCStep = Step;
    mCSize = static_cast<size_t>(Rows) * Step;
    mScaleBuf.resize(mCSize * mScaEleSize);
    if (_is_asym) {
      mZpBuf.resize(mCSize * mZpEleSize);
    }
    if (_has_reduce) {
      mRedBuf.resize(mCSize * sizeof(int));
    }
    return getSerializedSize();
  }

  virtual size_t getSerializedSize() override {
    size_t totalsize = getMiscSize();
    totalsize += mScaleBuf.getSerializedSize();
    totalsize += mZpBuf.getSerializedSize();
    totalsize += mRedBuf.getSerializedSize();
    totalsize += mDQCorrectionBuf.getSerializedSize();
    return totalsize;
  }
  virtual void serializeToBuffer(int8_t*& wptr) override {
    utils::serialize(wptr, mScaT);
    utils::serialize(wptr, mZpT);
    utils::serialize(wptr, mCStep);
    utils::serialize(wptr, mCSize);
    mScaleBuf.serializeToBuffer(wptr);
    mZpBuf.serializeToBuffer(wptr);
    mRedBuf.serializeToBuffer(wptr);
    mDQCorrectionBuf.serializeToBuffer(wptr);
  }
  virtual void deserializeBuffer(int8_t*& rptr, bool locate_buf) override {
    if (!locate_buf) {
      mScaT = utils::deserialize<BTLA_DTYPE>(rptr);
      mZpT = utils::deserialize<BTLA_DTYPE>(rptr);
      updateSize();
      mCStep = utils::deserialize<int>(rptr);
      mCSize = utils::deserialize<size_t>(rptr);
    } else {
      utils::serialize<BTLA_DTYPE>(rptr, mScaT);
      utils::serialize<BTLA_DTYPE>(rptr, mZpT);
      utils::serialize<int>(rptr, mCStep);
      utils::serialize<size_t>(rptr, mCSize);
    }
    mScaleBuf.deserializeBuffer(rptr, locate_buf);
    mZpBuf.deserializeBuffer(rptr, locate_buf);
    mRedBuf.deserializeBuffer(rptr, locate_buf);
    mDQCorrectionBuf.deserializeBuffer(rptr, locate_buf);
  }
  void enable_double_quant(size_t scale_size, BTLA_DTYPE stype) {
    if (stype == BTLA_DTYPE::DQ8_BNB) {
      auto super_scale_size = scale_size * sizeof(float);
      auto super_zp_size = sizeof(float);
      mDQCorrectionBuf.resize(super_scale_size + super_zp_size);
    } else {
      assert(0);
    }
  };

 protected:
  inline void updateSize() {
    mScaEleSize = int(utils::bestla_dtype_bytes(mScaT));
    mZpEleSize = int(utils::bestla_dtype_bytes(mZpT));
  }

  inline constexpr size_t getMiscSize() {
    size_t totalsize = 0;
    totalsize += sizeof(mScaT);
    totalsize += sizeof(mZpT);
    totalsize += sizeof(mCStep);
    totalsize += sizeof(mCSize);
    return totalsize;
  }
};

class IWeightBase : public storage::ISerializable {
 public:
  BTLA_PROLOGUEB_IDS mPrologueID = BTLA_PROLOGUEB_IDS::Undef;
  uint64_t mCoreId = 0;
  BTLA_DTYPE mDType = BTLA_DTYPE::F32;
  Layout mLayout = Layout::PLAIN;
  int mNPad = 0, mKPad = 0;
  int mN = 0, mK = 0;

  IWeightBase(uint64_t _id) { mCoreId = _id; }

  // bytes offset to mPrologueID
  static constexpr inline size_t offset() { return sizeof(mSize); }

 protected:
  void resize(int NPad, int KPad, int N, int K, BTLA_DTYPE dtype) {
    mNPad = NPad;
    mKPad = KPad;
    mN = N;
    mK = K;
    mDType = dtype;
  }

  virtual size_t getSerializedSize() { return ISerializable::getSerializedSize() + getMiscSize(); }

  virtual void serializeToBuffer(int8_t*& wptr) {
    ISerializable::serializeToBuffer(wptr);
    utils::serialize(wptr, mPrologueID);
    utils::serialize(wptr, mCoreId);
    utils::serialize(wptr, mNPad);
    utils::serialize(wptr, mKPad);
    utils::serialize(wptr, mN);
    utils::serialize(wptr, mK);
    utils::serialize(wptr, mDType);
    utils::serialize(wptr, mLayout);
  }

  virtual void deserializeBuffer(int8_t*& rptr, bool map_buf) {
    ISerializable::deserializeBuffer(rptr, map_buf);
    if (!map_buf) {
      mPrologueID = utils::deserialize<BTLA_PROLOGUEB_IDS>(rptr);
      mCoreId = utils::deserialize<uint64_t>(rptr);
      mNPad = utils::deserialize<int>(rptr);
      mKPad = utils::deserialize<int>(rptr);
      mN = utils::deserialize<int>(rptr);
      mK = utils::deserialize<int>(rptr);
      mDType = utils::deserialize<BTLA_DTYPE>(rptr);
      mLayout = utils::deserialize<Layout>(rptr);
    } else {
      utils::serialize<BTLA_PROLOGUEB_IDS>(rptr, mPrologueID);
      utils::serialize<uint64_t>(rptr, mCoreId);
      utils::serialize<int>(rptr, mNPad);
      utils::serialize<int>(rptr, mKPad);
      utils::serialize<int>(rptr, mN);
      utils::serialize<int>(rptr, mK);
      utils::serialize<BTLA_DTYPE>(rptr, mDType);
      utils::serialize<Layout>(rptr, mLayout);
    }
  }

  inline constexpr size_t getMiscSize() {
    size_t totalsize = 0;
    totalsize += sizeof(mPrologueID);
    totalsize += sizeof(mCoreId);
    totalsize += sizeof(mNPad);
    totalsize += sizeof(mKPad);
    totalsize += sizeof(mN);
    totalsize += sizeof(mK);
    totalsize += sizeof(mDType);
    totalsize += sizeof(mLayout);
    return totalsize;
  }
};

class ObjectWeightInfo : public storage::ISerialObjectV2 {
 public:
  uint64_t core_id_ = 0;
  BTLA_DTYPE dtype_ = BTLA_DTYPE::F32;
  Layout layout_ = Layout::PLAIN;
  int npad_ = 0, kpad_ = 0;
  int n_ = 0, k_ = 0;

  ObjectWeightInfo(uint64_t _id) { core_id_ = _id; }

  void resize(int NPad, int KPad, int N, int K, BTLA_DTYPE dtype, Layout _layout) {
    npad_ = NPad;
    kpad_ = KPad;
    n_ = N;
    k_ = K;
    dtype_ = dtype;
    layout_ = _layout;
  }

  size_t get_misc_size() override {
    size_t totalsize = 0;
    totalsize += sizeof(core_id_);
    totalsize += sizeof(npad_);
    totalsize += sizeof(kpad_);
    totalsize += sizeof(n_);
    totalsize += sizeof(k_);
    totalsize += sizeof(dtype_);
    totalsize += sizeof(layout_);
    return totalsize;
  }

  void serialize_misc(int8_t*& wptr) override {
    utils::serialize(wptr, core_id_);
    utils::serialize(wptr, npad_);
    utils::serialize(wptr, kpad_);
    utils::serialize(wptr, n_);
    utils::serialize(wptr, k_);
    utils::serialize(wptr, dtype_);
    utils::serialize(wptr, layout_);
  }

  void deserialize_misc(int8_t*& rptr) override {
    core_id_ = utils::deserialize<uint64_t>(rptr);
    npad_ = utils::deserialize<int>(rptr);
    kpad_ = utils::deserialize<int>(rptr);
    n_ = utils::deserialize<int>(rptr);
    k_ = utils::deserialize<int>(rptr);
    dtype_ = utils::deserialize<BTLA_DTYPE>(rptr);
    layout_ = utils::deserialize<Layout>(rptr);
  }
};

class IWeightKBlockBase : public IWeightBase {
 public:
  int mBlockSize = 1;
  int mDqBlockSize = 0;
  IWeightKBlockBase(uint64_t _id) : IWeightBase(_id) {}
  void resize(int NPad, int KPad, int Block, int N, int K, BTLA_DTYPE dtype) {
    IWeightBase::resize(NPad, KPad, N, K, dtype);
    mBlockSize = Block;
  }

 protected:
  virtual size_t getSerializedSize() {
    size_t totalsize = IWeightBase::getSerializedSize() + getMiscSize();
    return totalsize;
  }

  virtual void serializeToBuffer(int8_t*& wptr) {
    IWeightBase::serializeToBuffer(wptr);
    utils::serialize(wptr, mBlockSize);
    utils::serialize(wptr, mDqBlockSize);
  }

  virtual void deserializeBuffer(int8_t*& rptr, bool map_buf) {
    IWeightBase::deserializeBuffer(rptr, map_buf);
    if (!map_buf) {
      mBlockSize = utils::deserialize<int>(rptr);
      mDqBlockSize = utils::deserialize<int>(rptr);
    } else {
      utils::serialize(rptr, mBlockSize);
      utils::serialize(rptr, mDqBlockSize);
    }
  }

  inline constexpr size_t getMiscSize() {
    size_t totalsize = sizeof(mBlockSize);
    totalsize += sizeof(mDqBlockSize);
    return totalsize;
  }
};

class IActivationBase : public storage::ISerializable {
 public:
  BTLA_PROLOGUEB_IDS mPrologueID = BTLA_PROLOGUEB_IDS::Undef;
  uint64_t mCoreId = 0;
  BTLA_DTYPE mDType = BTLA_DTYPE::F32;
  Layout mLayout = Layout::PLAIN;
  int mMPad = 0, mKPad = 0;
  int mM = 0, mK = 0;

  IActivationBase(uint64_t _id) { mCoreId = _id; }

  // bytes offset to mPrologueID
  static constexpr inline size_t offset() { return sizeof(mSize); }

 protected:
  void resize(int NPad, int KPad, int N, int K, BTLA_DTYPE dtype) {
    mMPad = NPad;
    mKPad = KPad;
    mM = N;
    mK = K;
    mDType = dtype;
  }

  virtual size_t getSerializedSize() { return ISerializable::getSerializedSize() + getMiscSize(); }

  virtual void serializeToBuffer(int8_t*& wptr) {
    ISerializable::serializeToBuffer(wptr);
    utils::serialize(wptr, mPrologueID);
    utils::serialize(wptr, mCoreId);
    utils::serialize(wptr, mMPad);
    utils::serialize(wptr, mKPad);
    utils::serialize(wptr, mM);
    utils::serialize(wptr, mK);
    utils::serialize(wptr, mDType);
    utils::serialize(wptr, mLayout);
  }

  virtual void deserializeBuffer(int8_t*& rptr, bool map_buf) {
    ISerializable::deserializeBuffer(rptr, map_buf);
    if (!map_buf) {
      mPrologueID = utils::deserialize<BTLA_PROLOGUEB_IDS>(rptr);
      mCoreId = utils::deserialize<uint64_t>(rptr);
      mMPad = utils::deserialize<int>(rptr);
      mKPad = utils::deserialize<int>(rptr);
      mM = utils::deserialize<int>(rptr);
      mK = utils::deserialize<int>(rptr);
      mDType = utils::deserialize<BTLA_DTYPE>(rptr);
      mLayout = utils::deserialize<Layout>(rptr);
    } else {
      utils::serialize<BTLA_PROLOGUEB_IDS>(rptr, mPrologueID);
      utils::serialize<uint64_t>(rptr, mCoreId);
      utils::serialize<int>(rptr, mMPad);
      utils::serialize<int>(rptr, mKPad);
      utils::serialize<int>(rptr, mM);
      utils::serialize<int>(rptr, mK);
      utils::serialize<BTLA_DTYPE>(rptr, mDType);
      utils::serialize<Layout>(rptr, mLayout);
    }
  }

  inline constexpr size_t getMiscSize() {
    size_t totalsize = 0;
    totalsize += sizeof(mPrologueID);
    totalsize += sizeof(mCoreId);
    totalsize += sizeof(mMPad);
    totalsize += sizeof(mKPad);
    totalsize += sizeof(mM);
    totalsize += sizeof(mK);
    totalsize += sizeof(mDType);
    totalsize += sizeof(mLayout);
    return totalsize;
  }
};

class IActivationKBlockBase : public IActivationBase {
 public:
  int mBlockSize = 1;
  IActivationKBlockBase(uint64_t _id) : IActivationBase(_id) {}
  void resize(int MPad, int KPad, int Block, int N, int K, BTLA_DTYPE dtype) {
    IActivationBase::resize(MPad, KPad, N, K, dtype);
    mBlockSize = Block;
  }

 protected:
  virtual size_t getSerializedSize() {
    size_t totalsize = IActivationBase::getSerializedSize() + getMiscSize();
    return totalsize;
  }

  virtual void serializeToBuffer(int8_t*& wptr) {
    IActivationBase::serializeToBuffer(wptr);
    utils::serialize(wptr, mBlockSize);
  }

  virtual void deserializeBuffer(int8_t*& rptr, bool map_buf) {
    IActivationBase::deserializeBuffer(rptr, map_buf);
    if (!map_buf) {
      mBlockSize = utils::deserialize<int>(rptr);
    } else {
      utils::serialize(rptr, mBlockSize);
    }
  }

  inline constexpr size_t getMiscSize() {
    size_t totalsize = sizeof(mBlockSize);
    return totalsize;
  }
};

class StoragePackedWeight : public IWeightBase {
 public:
  ObjectAlignedBuffer<Alignment> mWBuf;
  StoragePackedWeight(uint64_t _id) : IWeightBase(_id) { mPrologueID = BTLA_PROLOGUEB_IDS::WeightPack; }

  size_t resize(int NPad, int KPad, int N, int K, BTLA_DTYPE dtype, Layout _layout) {
    IWeightBase::resize(NPad, KPad, N, K, dtype);
    auto bsize = static_cast<size_t>(NPad) * KPad * utils::bestla_dtype_bytes(dtype);
    mWBuf.resize(bsize);
    mSize = IWeightBase::getSerializedSize() + mWBuf.getSerializedSize();
    mSize = utils::padto(mSize, Alignment);
    mLayout = _layout;
    return mSize;
  }

  template <typename T>
  inline constexpr T* WPtr() const {
    return mWBuf.get<T>();
  }

  virtual void assign(int8_t* buf) override {
    IWeightBase::deserializeBuffer(buf, true);
    mWBuf.deserializeBuffer(buf, true);
  }

  virtual void serialize(int8_t* wptr) {
    IWeightBase::serializeToBuffer(wptr);
    mWBuf.serializeToBuffer(wptr);
  }

  virtual void deserialize(int8_t* rptr) override {
    IWeightBase::deserializeBuffer(rptr, false);
    mWBuf.deserializeBuffer(rptr, false);
  }
};

class StorageWeight : public ICollection {
 public:
  ObjectWeightInfo info_;
  ObjectBuffer buf_;

  StorageWeight(uint64_t _id = 0) : info_(_id) {
    prologue_id_ = BTLA_PROLOGUEB_IDS::WeightPack;
    update_size();
  }

  void set_plain_data(int _n, int _k, int _ld, BTLA_DTYPE dtype, void* _ptr) {
    info_.resize(_n, _ld, _n, _k, dtype, gemm::Layout::PLAIN);
    buf_.buf_size_ = (size_t)_n * _ld * ele_size();
    update_size();
    set_buffers((int8_t*)_ptr);
  }

  void set_nt_data(int _n, int _k, int _ld, BTLA_DTYPE dtype, void* _ptr) {
    info_.resize(_ld, _k, _n, _k, dtype, gemm::Layout::NT);
    buf_.buf_size_ = (size_t)_k * _ld * ele_size();
    update_size();
    set_buffers((int8_t*)_ptr);
  }

  void set_buffers(int8_t* _buf) override { buf_.set_buffer(_buf); }

  void copy_buffers(int8_t* _dst_buf) override { std::memcpy(_dst_buf, buf_.buf_ptr_, buf_size_); }

  int8_t* serialize(int8_t* _ptr) {
    auto raw_ptr = _ptr;
    ICollection::serialize(_ptr);
    _ptr += get_misc_size();
    info_.serialize_misc(_ptr);
    buf_.serialize_misc(_ptr);
    return raw_ptr + misc_size_;
  }

  int8_t* deserialize(int8_t* _ptr) {
    auto raw_ptr = _ptr;
    ICollection::deserialize(_ptr);
    _ptr += get_misc_size();
    info_.deserialize_misc(_ptr);
    buf_.deserialize_misc(_ptr);
    return raw_ptr + misc_size_;
  }

  inline constexpr size_t ele_size() const { return bestla::utils::bestla_dtype_bytes(info_.dtype_); }

  size_t resize(int NPad, int KPad, int N, int K, BTLA_DTYPE dtype, Layout _layout) {
    info_.resize(NPad, KPad, N, K, dtype, _layout);
    auto bsize = static_cast<size_t>(NPad) * KPad * utils::bestla_dtype_bytes(dtype);
    buf_.buf_size_ = bsize;
    update_size();
    return buf_size_;
  }

  void update_size() {
    buf_size_ = info_.get_buffer_size() + buf_.get_buffer_size();
    misc_size_ = utils::padto(get_misc_size() + info_.get_misc_size() + buf_.get_misc_size(), Alignment);
  }

  size_t total_size() { return buf_size_ + misc_size_; }

  inline constexpr int8_t* ptr() const { return buf_.buf_ptr_; }
};

class ObjectCorrection : public ISerialObjectV2 {
  // ser
 public:
  size_t size_ = 0;
  int step_ = 0;
  BTLA_DTYPE sdtype_ = BTLA_DTYPE::F32, zdtype_ = BTLA_DTYPE::S8;
  ObjectBuffer scale_buf_, zp_buf_, red_buf_, dq_buf_;

  // non-ser
 public:
  inline int8_t* sptr() const { return scale_buf_.buf_size_ ? scale_buf_.buf_ptr_ : nullptr; }
  inline int8_t* zptr() const { return zp_buf_.buf_size_ ? zp_buf_.buf_ptr_ : nullptr; }
  inline int8_t* rptr() const { return red_buf_.buf_size_ ? red_buf_.buf_ptr_ : nullptr; }

  void resize(int Rows, int Step, BTLA_DTYPE scalet, BTLA_DTYPE zpt, bool _is_asym, bool _has_reduce) {
    sdtype_ = scalet;
    zdtype_ = zpt;
    step_ = Step;
    size_ = static_cast<size_t>(Rows) * Step;
    scale_buf_.buf_size_ = size_ * utils::bestla_dtype_bytes(sdtype_);
    if (_is_asym) {
      zp_buf_.buf_size_ = size_ * utils::bestla_dtype_bytes(zdtype_);
    }
    if (_has_reduce) {
      red_buf_.buf_size_ = size_ * sizeof(int);
    }
  }

  size_t get_misc_size() override {
    size_t totalsize = 0;
    totalsize += sizeof(size_);
    totalsize += sizeof(step_);
    totalsize += sizeof(sdtype_);
    totalsize += sizeof(zdtype_);
    totalsize += scale_buf_.get_misc_size();
    totalsize += zp_buf_.get_misc_size();
    totalsize += red_buf_.get_misc_size();
    totalsize += dq_buf_.get_misc_size();
    return totalsize;
  }

  void serialize_misc(int8_t*& wptr) override {
    utils::serialize(wptr, size_);
    utils::serialize(wptr, step_);
    utils::serialize(wptr, sdtype_);
    utils::serialize(wptr, zdtype_);
    scale_buf_.serialize_misc(wptr);
    zp_buf_.serialize_misc(wptr);
    red_buf_.serialize_misc(wptr);
    dq_buf_.serialize_misc(wptr);
  }

  void deserialize_misc(int8_t*& rptr) override {
    size_ = utils::deserialize<size_t>(rptr);
    step_ = utils::deserialize<int>(rptr);
    sdtype_ = utils::deserialize<BTLA_DTYPE>(rptr);
    zdtype_ = utils::deserialize<BTLA_DTYPE>(rptr);
    scale_buf_.deserialize_misc(rptr);
    zp_buf_.deserialize_misc(rptr);
    red_buf_.deserialize_misc(rptr);
    dq_buf_.deserialize_misc(rptr);
  }

  size_t get_buffer_size() override {
    return scale_buf_.get_buffer_size() + zp_buf_.get_buffer_size() + red_buf_.get_buffer_size() +
           dq_buf_.get_buffer_size();
  }

  void set_buffer(int8_t*& bufptr) override {
    scale_buf_.set_buffer(bufptr);
    zp_buf_.set_buffer(bufptr);
    red_buf_.set_buffer(bufptr);
    dq_buf_.set_buffer(bufptr);
  }

  void copy_buffers(int8_t*& _dst_buf) {
    if (scale_buf_.buf_size_) {
      std::memcpy(_dst_buf, scale_buf_.buf_ptr_, scale_buf_.buf_size_);
      _dst_buf += scale_buf_.buf_size_;
    }
    if (zp_buf_.buf_size_) {
      std::memcpy(_dst_buf, zp_buf_.buf_ptr_, zp_buf_.buf_size_);
      _dst_buf += zp_buf_.buf_size_;
    }
    if (red_buf_.buf_size_) {
      std::memcpy(_dst_buf, red_buf_.buf_ptr_, red_buf_.buf_size_);
      _dst_buf += red_buf_.buf_size_;
    }
    if (dq_buf_.buf_size_) {
      std::memcpy(_dst_buf, dq_buf_.buf_ptr_, dq_buf_.buf_size_);
      _dst_buf += dq_buf_.buf_size_;
    }
  }

  void enable_double_quant(size_t scale_size, BTLA_DTYPE stype) {
    if (stype == BTLA_DTYPE::DQ8_BNB) {
      auto super_scale_size = scale_size * sizeof(float);
      auto super_zp_size = sizeof(float);
      dq_buf_.buf_size_ = super_scale_size + super_zp_size;
    } else {
      assert(0);
    }
  };
};
class StorageWeightNInt : public ICollection {
 public:
  ObjectWeightInfo info_;
  int block_ = -1, n_block_ = 0;
  ObjectBuffer buf_;
  ObjectCorrection corr_;

  StorageWeightNInt(uint64_t _id = 0) : info_(_id) {
    prologue_id_ = BTLA_PROLOGUEB_IDS::WeightKBlockNInteger;
    update_size();
  }

  void t_from(const StorageWeightNInt& src) {
    block_ = src.block_;
    n_block_ = src.n_block_;
    info_ = src.info_;
    info_.kpad_ = info_.k_;
    info_.npad_ = info_.n_;
    info_.layout_ = gemm::Layout::T;
    auto bytes = nbit_size(static_cast<size_t>(info_.n_) * info_.k_, info_.dtype_);
    buf_.buf_size_ = bytes;
    corr_.resize(info_.n_, n_block_, src.corr_.sdtype_, src.corr_.zdtype_, src.corr_.zptr() != nullptr,
                 src.corr_.rptr() != nullptr);
    update_size();
  }

  void set_plain_data(int _n, int _k, int _ld, BTLA_DTYPE dtype, void* _ptr) {
    info_.resize(_n, _ld, _n, _k, dtype, gemm::Layout::PLAIN);
    buf_.buf_size_ = (size_t)_n * _ld * ele_size();
    update_size();
    set_buffers((int8_t*)_ptr);
  }

  void set_nt_data(int _n, int _k, int _ld, BTLA_DTYPE dtype, void* _ptr) {
    info_.resize(_ld, _k, _n, _k, dtype, gemm::Layout::NT);
    buf_.buf_size_ = (size_t)_k * _ld * ele_size();
    update_size();
    set_buffers((int8_t*)_ptr);
  }

  void set_buffers(int8_t* _buf) override {
    buf_.set_buffer(_buf);
    corr_.set_buffer(_buf);
  }

  void copy_buffers(int8_t* _dst_buf) override {
    std::memcpy(_dst_buf, buf_.buf_ptr_, buf_.buf_size_);
    _dst_buf += buf_.buf_size_;
    corr_.copy_buffers(_dst_buf);
  }

  int8_t* serialize(int8_t* _ptr) {
    auto raw_ptr = _ptr;
    ICollection::serialize(_ptr);
    _ptr += get_misc_size();
    info_.serialize_misc(_ptr);
    utils::serialize(_ptr, block_);
    utils::serialize(_ptr, n_block_);
    buf_.serialize_misc(_ptr);
    corr_.serialize_misc(_ptr);
    return raw_ptr + misc_size_;
  }

  int8_t* deserialize(int8_t* _ptr) {
    auto raw_ptr = _ptr;
    ICollection::deserialize(_ptr);
    _ptr += get_misc_size();
    info_.deserialize_misc(_ptr);
    block_ = utils::deserialize<int>(_ptr);
    n_block_ = utils::deserialize<int>(_ptr);
    buf_.deserialize_misc(_ptr);
    corr_.deserialize_misc(_ptr);
    return raw_ptr + misc_size_;
  }

  inline constexpr size_t ele_size() const { return bestla::utils::bestla_dtype_bytes(info_.dtype_); }

  static size_t nbit_size(size_t elecount, BTLA_DTYPE dtype) {
    auto bitsize = elecount * utils::bestla_dtype_bits(dtype);
    auto bytes = utils::updiv(bitsize, 8);  // add 3bits, 5btis, 7bits size calculation here
    if (dtype == BTLA_DTYPE::S3_CLIP)
      bytes = utils::updiv(elecount * 2, 8) + utils::updiv(elecount * 1, 8);
    else if (dtype == BTLA_DTYPE::S5_CLIP)
      bytes = utils::updiv(elecount * 4, 8) + utils::updiv(elecount * 1, 8);
    else if (dtype == BTLA_DTYPE::S6_CLIP)
      bytes = utils::updiv(elecount * 4, 8) + utils::updiv(elecount * 2, 8);
    else if (dtype == BTLA_DTYPE::S7_CLIP)
      bytes = utils::updiv(elecount * 4, 8) + utils::updiv(elecount * 2, 8) + utils::updiv(elecount * 1, 8);
    return bytes;
  }

  size_t resize(int NPad, int KPad, int Block, int N, int K, BTLA_DTYPE dtype, Layout _layout, BTLA_DTYPE scalet,
                bool _is_asym) {
    info_.resize(NPad, KPad, N, K, dtype, _layout);
    block_ = Block;
    auto bytes = nbit_size(static_cast<size_t>(NPad) * KPad, dtype);
    buf_.buf_size_ = bytes;
    n_block_ = utils::updiv(KPad, Block);
    auto gemm_comp = bestla::gemm::CoreAttr::get_comp(info_.core_id_);
    auto is_cint = bestla::gemm::CompTypeHelper::is_integer(gemm_comp);
    BTLA_DTYPE zpt = BTLA_DTYPE::S8;
    corr_.resize(n_block_, NPad, scalet, zpt, _is_asym, is_cint);
    if (scalet == BTLA_DTYPE::DQ8_BNB) {
      corr_.enable_double_quant(utils::updiv(n_block_ * N, Block), scalet);
    }
    update_size();
    return buf_size_;
  }

  void update_size() {
    buf_size_ = info_.get_buffer_size() + buf_.get_buffer_size() + corr_.get_buffer_size();
    misc_size_ =
        utils::padto(get_misc_size() + info_.get_misc_size() + buf_.get_misc_size() + corr_.get_misc_size(), Alignment);
  }

  size_t total_size() { return buf_size_ + misc_size_; }

  inline constexpr int8_t* ptr() const { return buf_.buf_ptr_; }
};

class StorageReduce : public ISerializable {
 public:
  using CorrectionType = ObjectQuantCorrection;
  int m = 0, k = 0, lda = 0, kblock = 1;
  ObjectAlignedBuffer<Alignment> mRedBuf;
  size_t resize(int _m, int _k, int _kblock, BTLA_DTYPE redt) {
    kblock = _kblock;
    m = _m;
    k = _k;
    lda = utils::updiv(_k, _kblock);
    size_t bufsize = static_cast<size_t>(m) * lda * utils::bestla_dtype_bytes(redt);
    mRedBuf.resize(bufsize);
    mSize = getSerializedSize();
    mSize = utils::padto(mSize, Alignment);
    return mSize;
  }
  template <typename QT_T>
  inline QT_T* RPtr() {
    return mRedBuf.get<QT_T>();
  }

  virtual void assign(int8_t* buf) override {
    ISerializable::deserializeBuffer(buf, true);
    deserializeBuffer(buf, true);
    mRedBuf.deserializeBuffer(buf, true);
  }

  virtual void serialize(int8_t* wptr) {
    ISerializable::serializeToBuffer(wptr);
    serializeToBuffer(wptr);
    mRedBuf.serializeToBuffer(wptr);
  }

  virtual void deserialize(int8_t* rptr) override {
    ISerializable::deserializeBuffer(rptr, false);
    deserializeBuffer(rptr, false);
    mRedBuf.deserializeBuffer(rptr, false);
  }

 protected:
  virtual size_t getSerializedSize() {
    return ISerializable::getSerializedSize() + getMiscSize() + mRedBuf.getSerializedSize();
  }

  virtual void serializeToBuffer(int8_t*& wptr) {
    utils::serialize(wptr, m);
    utils::serialize(wptr, k);
    utils::serialize(wptr, lda);
    utils::serialize(wptr, kblock);
  }

  virtual void deserializeBuffer(int8_t*& rptr, bool map_buf) {
    if (!map_buf) {
      m = utils::deserialize<int>(rptr);
      lda = utils::deserialize<int>(rptr);
      kblock = utils::deserialize<int>(rptr);
    } else {
      utils::serialize(rptr, m);
      utils::serialize(rptr, k);
      utils::serialize(rptr, lda);
      utils::serialize(rptr, kblock);
    }
  }

  inline constexpr size_t getMiscSize() {
    size_t totalsize = 0;
    totalsize += sizeof(m);
    totalsize += sizeof(k);
    totalsize += sizeof(lda);
    totalsize += sizeof(kblock);
    return totalsize;
  }
};

class StorageReorderActivation : public IActivationKBlockBase {
 public:
  ObjectAlignedBuffer<Alignment> mABuf;
  StorageReorderActivation(uint64_t _id) : IActivationKBlockBase(_id) { mPrologueID = BTLA_PROLOGUEB_IDS::WeightPack; }

  size_t resize(int MPad, int KPad, int M, int K, int KBlock, BTLA_DTYPE dtype) {
    IActivationKBlockBase::resize(MPad, KPad, KBlock, M, K, dtype);
    auto bsize = static_cast<size_t>(MPad) * KPad * utils::bestla_dtype_bytes(dtype);
    mABuf.resize(bsize);
    mSize = IActivationKBlockBase::getSerializedSize() + mABuf.getSerializedSize();
    mSize = utils::padto(mSize, Alignment);
    return mSize;
  }

  template <typename T>
  inline constexpr T* APtr() const {
    return mABuf.get<T>();
  }

  virtual void assign(int8_t* buf) override {
    IActivationKBlockBase::deserializeBuffer(buf, true);
    mABuf.deserializeBuffer(buf, true);
  }

  virtual void serialize(int8_t* wptr) {
    IActivationKBlockBase::serializeToBuffer(wptr);
    mABuf.serializeToBuffer(wptr);
  }

  virtual void deserialize(int8_t* rptr) override {
    IActivationKBlockBase::deserializeBuffer(rptr, false);
    mABuf.deserializeBuffer(rptr, false);
  }
};

class StorageQuantActivation : public IActivationKBlockBase {
 public:
  using CorrectionType = ObjectQuantCorrection;
  CorrectionType mCorrection;
  ObjectAlignedBuffer<Alignment> mQBuf;
  StorageQuantActivation(uint64_t _id = 0) : IActivationKBlockBase(_id) {
    mPrologueID = BTLA_PROLOGUEB_IDS::WeightPack;
  }

  size_t resize(int _mpad, int _kpad, int _m, int _k, int _kblock, BTLA_DTYPE buft, BTLA_DTYPE scalet, BTLA_DTYPE zpt,
                bool is_asym, bool has_reduce) {
    IActivationKBlockBase::resize(_mpad, _kpad, _kblock, _m, _k, buft);
    mCorrection.resize(_mpad, utils::updiv(_kpad, _kblock), scalet, zpt, is_asym, has_reduce);
    size_t bufsize = static_cast<size_t>(_mpad) * _kpad * utils::bestla_dtype_bytes(buft);
    mQBuf.resize(bufsize);
    mSize = getSerializedSize();
    mSize = utils::padto(mSize, Alignment);
    return mSize;
  }

  template <typename QT_T>
  inline constexpr QT_T* APtr() {
    return mQBuf.get<QT_T>();
  }

  template <typename T>
  inline constexpr size_t ASize() {
    return mQBuf.size<T>();
  }

  template <typename QT_T>
  inline constexpr QT_T* ZPtr() {
    return mCorrection.mZpBuf.get<QT_T>();
  }

  template <typename QT_T>
  inline constexpr QT_T* SPtr() {
    return mCorrection.mScaleBuf.get<QT_T>();
  }

  template <typename QT_T>
  inline constexpr QT_T* RPtr() {
    return mCorrection.mRedBuf.get<QT_T>();
  }

  inline constexpr BTLA_DTYPE ZDtype() { return mCorrection.mZpT; }
  inline constexpr BTLA_DTYPE SDtype() { return mCorrection.mScaT; }
  inline constexpr bool IsAsym() { return mCorrection.mZpBuf.mNotEmpty; }
  inline constexpr bool HasReduce() { return mCorrection.mRedBuf.mNotEmpty; }
  inline constexpr size_t CSize() { return mCorrection.mCSize; }
  inline constexpr int CStep() { return mCorrection.mCStep; }

  virtual void assign(int8_t* buf) override {
    IActivationKBlockBase::deserializeBuffer(buf, true);
    deserializeBuffer(buf, true);
  }

  virtual void serialize(int8_t* wptr) {
    IActivationKBlockBase::serializeToBuffer(wptr);
    serializeToBuffer(wptr);
  }

  virtual void deserialize(int8_t* rptr) override {
    IActivationKBlockBase::deserializeBuffer(rptr, false);
    deserializeBuffer(rptr, false);
  }

 protected:
  virtual size_t getSerializedSize() {
    return ISerializable::getSerializedSize() + getMiscSize() + mQBuf.getSerializedSize() +
           mCorrection.getSerializedSize();
  }

  virtual void serializeToBuffer(int8_t*& wptr) {
    mQBuf.serializeToBuffer(wptr);
    mCorrection.serializeToBuffer(wptr);
  }

  virtual void deserializeBuffer(int8_t*& rptr, bool map_buf) {
    mQBuf.deserializeBuffer(rptr, map_buf);
    mCorrection.deserializeBuffer(rptr, map_buf);
  }

  inline constexpr size_t getMiscSize() {
    size_t totalsize = 0;
    return totalsize;
  }
};

class StorageWeightKBlockNInteger : public IWeightKBlockBase {
 public:
  using InfoType = IWeightKBlockBase;
  using QWeightType = ObjectAlignedBuffer<Alignment>;
  using CorrectionType = ObjectQuantCorrection;
  QWeightType mQBuf;
  CorrectionType mCorrection;
  ObjectOptionalBuffer<Alignment> mShuffleIndices;
  StorageWeightKBlockNInteger(uint64_t _type) : IWeightKBlockBase(_type) {
    mPrologueID = BTLA_PROLOGUEB_IDS::WeightKBlockNInteger;
  }

  StorageWeightKBlockNInteger toTrans() {
    StorageWeightKBlockNInteger trans(-1);
    trans.mK = mK;
    trans.mN = mN;
    trans.mNPad = mNPad;
    trans.mKPad = mKPad;
    trans.mBlockSize = mBlockSize;
    trans.mDType = mDType;
    trans.mQBuf.resize(mQBuf.size<int8_t>());
    int nk_scale = utils::updiv(mKPad, mBlockSize);
    trans.mCorrection.resize(mNPad, nk_scale, mCorrection.mScaT, mCorrection.mZpT, mCorrection.mZpBuf.size<int>() > 0,
                             mCorrection.mRedBuf.size<int>() > 0);
    trans.update_size();
    return trans;
  }

  size_t resize(int NPad, int KPad, int Block, int N, int K, BTLA_DTYPE qtype, BTLA_DTYPE scalet, bool IsAsym) {
    BTLA_DTYPE zpt = BTLA_DTYPE::S8;
    InfoType::resize(NPad, KPad, Block, N, K, qtype);
    auto bits = utils::bestla_dtype_bits(qtype);
    auto elesize = static_cast<size_t>(NPad) * KPad;
    auto bytes = utils::updiv(elesize * bits, 8);  // add 3bits, 5btis, 7bits size calculation here
    if (qtype == BTLA_DTYPE::S3_CLIP)
      bytes =
          utils::updiv(static_cast<size_t>(KPad) * NPad * 2, 8) + utils::updiv(static_cast<size_t>(KPad) * NPad * 1, 8);
    else if (qtype == BTLA_DTYPE::S5_CLIP)
      bytes =
          utils::updiv(static_cast<size_t>(KPad) * NPad * 4, 8) + utils::updiv(static_cast<size_t>(KPad) * NPad * 1, 8);
    else if (qtype == BTLA_DTYPE::S6_CLIP)
      bytes =
          utils::updiv(static_cast<size_t>(KPad) * NPad * 4, 8) + utils::updiv(static_cast<size_t>(KPad) * NPad * 2, 8);
    else if (qtype == BTLA_DTYPE::S7_CLIP)
      bytes = utils::updiv(static_cast<size_t>(KPad) * NPad * 4, 8) +
              utils::updiv(static_cast<size_t>(KPad) * NPad * 2, 8) +
              utils::updiv(static_cast<size_t>(KPad) * NPad * 1, 8);
    mQBuf.resize(bytes);
    int nk_scale = utils::updiv(KPad, Block);
    auto gemm_comp = bestla::gemm::CoreAttr::get_comp(mCoreId);
    auto is_cint = bestla::gemm::CompTypeHelper::is_integer(gemm_comp);
    mCorrection.resize(nk_scale, NPad, scalet, zpt, IsAsym, is_cint);
    if (scalet == BTLA_DTYPE::DQ8_BNB) initDoubleQuantBlkSize(Block, nk_scale, IsAsym, N, scalet);
    update_size();
    return mSize;
  }

  void initDoubleQuantBlkSize(int dq_blksize, int nk_scale, bool asym, int N, BTLA_DTYPE stype) {
    mDqBlockSize = dq_blksize;
    if (asym || dq_blksize % 8 != 0) assert(0);
    mCorrection.enable_double_quant(utils::updiv(nk_scale * N, dq_blksize), stype);
  }

  void enable_shuffle() {
    auto indicessize = mK * sizeof(int);
    mShuffleIndices.resize(indicessize);
    update_size();
  }

  inline constexpr BTLA_DTYPE ZDtype() { return mCorrection.mZpT; }
  inline constexpr BTLA_DTYPE SDtype() { return mCorrection.mScaT; }
  inline constexpr bool IsAsym() { return mCorrection.mZpBuf.mNotEmpty; }
  inline constexpr bool HasReduce() { return mCorrection.mRedBuf.mNotEmpty; }
  inline constexpr bool IsDoubleQuant() { return mCorrection.mDQCorrectionBuf.mNotEmpty; }
  inline constexpr size_t CSize() { return mCorrection.mCSize; }
  inline constexpr int CStep() { return mCorrection.mCStep; }

  template <typename T>
  inline constexpr size_t WSize() {
    return mQBuf.size<T>();
  }

  template <typename T>
  inline constexpr T* WPtr() const {
    return mQBuf.get<T>();
  }

  template <typename T>
  inline constexpr T* SPtr() {
    return mCorrection.mScaleBuf.get<T>();
  }

  template <typename T>
  inline constexpr T* ZPtr() {
    return mCorrection.mZpBuf.get<T>();
  }

  template <typename T>
  inline constexpr T* RPtr() {
    return mCorrection.mRedBuf.get<T>();
  }

  template <typename T>
  inline constexpr T* DQPtr() {
    return mCorrection.mDQCorrectionBuf.get<T>();
  }

  inline constexpr int* ShfIndice() { return mShuffleIndices.get<int>(); }

  void update_size() {
    mSize = InfoType::getSerializedSize() + mQBuf.getSerializedSize() + mCorrection.getSerializedSize() +
            mShuffleIndices.getSerializedSize();
    mSize = utils::padto(mSize, Alignment);
  }

  virtual void assign(int8_t* buf) override {
    InfoType::deserializeBuffer(buf, true);
    mQBuf.deserializeBuffer(buf, true);
    mCorrection.deserializeBuffer(buf, true);
    mShuffleIndices.deserializeBuffer(buf, true);
  }

  virtual void serialize(int8_t* wptr) {
    InfoType::serializeToBuffer(wptr);
    mQBuf.serializeToBuffer(wptr);
    mCorrection.serializeToBuffer(wptr);
    mShuffleIndices.serializeToBuffer(wptr);
  }

  virtual void deserialize(int8_t* rptr) override {
    InfoType::deserializeBuffer(rptr, false);
    mQBuf.deserializeBuffer(rptr, false);
    mCorrection.deserializeBuffer(rptr, false);
    mShuffleIndices.deserializeBuffer(rptr, false);
  }
};

class StorageWeightKBlockNFloat : public StorageWeightKBlockNInteger {
 public:
  StorageWeightKBlockNFloat(uint64_t _type) : StorageWeightKBlockNInteger(_type) {
    mPrologueID = BTLA_PROLOGUEB_IDS::WeightKBlockNFloat;
  }

  size_t resize(int NPad, int KPad, int Block, int N, int K, BTLA_DTYPE ftype, BTLA_DTYPE scalet) {
    StorageWeightKBlockNInteger::InfoType::resize(NPad, KPad, Block, N, K, ftype);
    auto bits = utils::bestla_dtype_bits(ftype);
    auto elesize = static_cast<size_t>(NPad) * KPad;
    auto bytes = utils::updiv(elesize * bits, 8);  // add fp6 size calculation here
    StorageWeightKBlockNInteger::mQBuf.resize(bytes);
    int nk_scale = utils::updiv(KPad, Block);
    StorageWeightKBlockNInteger::mCorrection.resize(nk_scale, NPad, scalet, BTLA_DTYPE::EleBitsUndef, false, false);
    if (scalet == BTLA_DTYPE::DQ8_BNB) initDoubleQuantBlkSize(Block, nk_scale, false, N, scalet);
    update_size();
    mSize = StorageWeightKBlockNInteger::InfoType::getSerializedSize() +
            StorageWeightKBlockNInteger::mQBuf.getSerializedSize() +
            StorageWeightKBlockNInteger::mCorrection.getSerializedSize();
    mSize = utils::padto(mSize, Alignment);
    return mSize;
  }
};

class PackedWeightParser {
 public:
  static gemm::IWeightBase* deserialBuffer(const void* serialized_buf) {
    if (serialized_buf == nullptr) {
      return nullptr;
    }
    auto tmpptr = const_cast<void*>(serialized_buf);
    auto rptr = reinterpret_cast<int8_t*>(tmpptr);
    rptr += IWeightBase::offset();
    int mProID = utils::deserialize<int>(rptr);
    IWeightBase* ptr = nullptr;
    if (mProID >= int(BTLA_PROLOGUEB_IDS::Begin) && mProID < int(BTLA_PROLOGUEB_IDS::End)) {
      rptr = reinterpret_cast<int8_t*>(tmpptr);
      auto type = static_cast<BTLA_PROLOGUEB_IDS>(mProID);
      switch (type) {
        case BTLA_PROLOGUEB_IDS::WeightPack:
          ptr = new gemm::StoragePackedWeight(0);
          break;
        case BTLA_PROLOGUEB_IDS::WeightKBlockNInteger:
          ptr = new gemm::StorageWeightKBlockNInteger(0);
          break;
        case BTLA_PROLOGUEB_IDS::WeightKBlockNFloat:
          ptr = new gemm::StorageWeightKBlockNFloat(0);
          break;
        default:
          break;
      }
      if (ptr) {
        ptr->deserialize(rptr);
      }
    }
    return ptr;
  }
};

}  // namespace gemm
class CollectionParser {
 public:
  static inline bool valid_prologue(BTLA_PROLOGUEB_IDS pro_id) {
    return pro_id >= BTLA_PROLOGUEB_IDS::Begin && pro_id < BTLA_PROLOGUEB_IDS::End;
  }
  static BTLA_PROLOGUEB_IDS parse_prologue_id(const void* serialized_buf) {
    auto rptr = (int8_t*)(serialized_buf);
    return utils::deserialize<BTLA_PROLOGUEB_IDS>(rptr);
  }

  static ICollection* parse_buffer(const void* serialized_buf) {
    if (serialized_buf == nullptr) {
      return nullptr;
    }
    auto tmpptr = const_cast<void*>(serialized_buf);
    auto rptr = reinterpret_cast<int8_t*>(tmpptr);
    auto pro_id = utils::deserialize<BTLA_PROLOGUEB_IDS>(rptr);
    ICollection* ptr = nullptr;
    if (pro_id >= BTLA_PROLOGUEB_IDS::Begin && pro_id < BTLA_PROLOGUEB_IDS::End) {
      rptr = reinterpret_cast<int8_t*>(tmpptr);
      switch (pro_id) {
        case BTLA_PROLOGUEB_IDS::WeightPack:
          ptr = new gemm::StorageWeight;
          break;
        case BTLA_PROLOGUEB_IDS::WeightKBlockNInteger:
          ptr = new gemm::StorageWeightNInt;
          break;
        default:
          break;
      }
      if (ptr) {
        auto data_ptr = ptr->deserialize(rptr);
        ptr->set_buffers(data_ptr);
      }
    }
    return ptr;
  }
};

}  // namespace storage
}  // namespace bestla
