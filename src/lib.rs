#![feature(clone_from_slice)]
#![feature(zero_one)]

extern crate byteorder;

use byteorder::{ReadBytesExt, WriteBytesExt, LittleEndian};

use std::io::{Read, Write};
use std::mem::{size_of};
use std::num::{Zero};
use std::slice::{from_raw_parts, from_raw_parts_mut};

pub trait Shape: Copy {
  type Stride: Copy;

  fn to_least_stride(&self) -> Self::Stride;
  fn len(&self) -> usize;
  fn offset(&self, stride: Self::Stride) -> usize;

  fn major_iter(self) -> MajorIter<Self> where Self: Default {
    MajorIter{
      idx:          Default::default(),
      upper_bound:  self,
    }
  }
}

pub struct MajorIter<S> where S: Shape {
  idx:          S,
  upper_bound:  S,
}

impl Iterator for MajorIter<(usize, usize, usize)> {
  type Item = (usize, usize, usize);

  fn next(&mut self) -> Option<(usize, usize, usize)> {
    // FIXME(20160203): this only terminates "once".
    self.idx.0 += 1;
    if self.idx.0 < self.upper_bound.0 {
      return Some(self.idx);
    }
    self.idx.0 = 0;
    self.idx.1 += 1;
    if self.idx.1 < self.upper_bound.1 {
      return Some(self.idx);
    }
    self.idx.1 = 0;
    self.idx.2 += 1;
    if self.idx.2 < self.upper_bound.2 {
      return Some(self.idx);
    }
    None
  }
}

impl Shape for usize {
  type Stride = ();

  fn to_least_stride(&self) -> () {
    ()
  }

  fn len(&self) -> usize {
    *self
  }

  fn offset(&self, _: ()) -> usize {
    *self
  }
}

impl Shape for (usize, usize) {
  type Stride = usize;

  fn to_least_stride(&self) -> usize {
    self.0
  }

  fn len(&self) -> usize {
    self.0 * self.1
  }

  fn offset(&self, stride: usize) -> usize {
    self.0 + self.1 * stride
  }
}

impl Shape for (usize, usize, usize) {
  type Stride = (usize, usize);

  fn to_least_stride(&self) -> (usize, usize) {
    (self.0, self.1)
  }

  fn len(&self) -> usize {
    self.0 * self.1 * self.2
  }

  fn offset(&self, stride: (usize, usize)) -> usize {
    self.0 + self.1 * stride.0 + self.2 * stride.1 * stride.0
  }
}

pub trait SerialDataType: Copy {
  fn serial_id() -> u8;
}

impl SerialDataType for u8 {
  fn serial_id() -> u8 { 0 }
}

impl SerialDataType for f32 {
  fn serial_id() -> u8 { 1 }
}

pub trait Array<'a, T, S> where T: 'a + Copy, S: Shape {
  type View: ArrayView<'a, T, S>;
  type ViewMut: ArrayViewMut<'a, T, S>;

  fn as_view(&'a self) -> Self::View;
  fn as_view_mut(&'a mut self) -> Self::ViewMut;
}

pub trait AsyncArray<'ctx, 'a, T, S> where 'ctx: 'a, T: 'a + Copy, S: Shape {
  type Ctx;
  type View: ArrayView<'a, T, S>;
  type ViewMut: ArrayViewMut<'a, T, S>;

  fn as_view(&'a mut self, ctx: &'a Self::Ctx) -> Self::View;
  fn as_view_mut(&'a mut self, ctx: &'a Self::Ctx) -> Self::ViewMut;
}

pub trait ArrayView<'a, T, S> where T: 'a + Copy, S: Shape {
  fn bound(&self) -> S;
  fn stride(&self) -> S::Stride;
  fn len(&self) -> usize;
  unsafe fn as_ptr(&self) -> *const T;
  fn view(self, lo: S, hi: S) -> Self;
}

pub trait ArrayViewMut<'a, T, S>/*: ArrayView<'a, T, S>*/ where T: 'a + Copy, S: Shape {
  fn bound(&self) -> S;
  fn stride(&self) -> S::Stride;
  fn len(&self) -> usize;
  unsafe fn as_ptr(&self) -> *const T;
  unsafe fn as_mut_ptr(&mut self) -> *mut T;
  fn view_mut(self, lo: S, hi: S) -> Self;
}

pub trait ArrayZeroExt<T, S> where T: Copy, S: Shape {
  fn zeros(bound: S) -> Self;
}

pub trait NdArraySerialize<T, S> where T: SerialDataType + Copy, S: Shape {
  fn serial_size(bound: S) -> usize;
  fn deserialize(reader: &mut Read) -> Result<Self, ()> where Self: Sized;
  fn serialize(&self, writer: &mut Write) -> Result<(), ()>;
}

pub struct Array2d<T> where T: Copy {
  data:     Vec<T>,
  bound:    (usize, usize),
  stride:   usize,
}

impl<T> Array2d<T> where T: Copy {
  pub unsafe fn new(bound: (usize, usize)) -> Array2d<T> {
    let len = bound.len();
    let mut data = Vec::with_capacity(len);
    data.set_len(len);
    Array2d{
      data:     data,
      bound:    bound,
      stride:   bound.to_least_stride(),
    }
  }

  pub fn as_slice(&self) -> &[T] {
    &self.data
  }

  pub fn as_mut_slice(&mut self) -> &mut [T] {
    &mut self.data
  }
}

impl<T> ArrayZeroExt<T, (usize, usize)> for Array2d<T> where T: Zero + Copy {
  fn zeros(bound: (usize, usize)) -> Array2d<T> {
    let len = bound.len();
    let mut data = Vec::with_capacity(len);
    unsafe { data.set_len(len) };
    for i in 0 .. len {
      data[i] = T::zero();
    }
    Array2d{
      data:     data,
      bound:    bound,
      stride:   bound.to_least_stride(),
    }
  }
}

impl<T> NdArraySerialize<T, (usize, usize)> for Array2d<T> where T: SerialDataType + Copy {
  fn serial_size(bound: (usize, usize)) -> usize {
    24 + bound.len()
  }

  fn deserialize(reader: &mut Read) -> Result<Array2d<T>, ()> {
    let magic0 = reader.read_u8()
      .ok().expect("failed to deserialize!");
    let magic1 = reader.read_u8()
      .ok().expect("failed to deserialize!");
    assert_eq!(magic0, b'N');
    assert_eq!(magic1, b'D');
    let version = reader.read_u8()
      .ok().expect("failed to deserialize!");
    assert_eq!(version, 0);
    let data_ty = reader.read_u8()
      .ok().expect("failed to deserialize!");
    let ndim = reader.read_u32::<LittleEndian>()
      .ok().expect("failed to deserialize!");
    assert_eq!(data_ty, T::serial_id());
    assert_eq!(ndim, 2);
    let bound0 = reader.read_u64::<LittleEndian>()
      .ok().expect("failed to deserialize!") as usize;
    let bound1 = reader.read_u64::<LittleEndian>()
      .ok().expect("failed to deserialize!") as usize;
    let dims = (bound0, bound1);
    let mut arr = unsafe { Array2d::new(dims) };
    {
      let mut data_bytes = unsafe { from_raw_parts_mut(arr.data.as_mut_ptr() as *mut u8, size_of::<f32>() * arr.data.len()) };
      let mut read_idx: usize = 0;
      loop {
        match reader.read(&mut data_bytes[read_idx ..]) {
          Ok(n) => {
            read_idx += n;
            if n == 0 {
              break;
            }
          }
          Err(e) => panic!("failed to deserialize: {:?}", e),
        }
      }
      assert_eq!(read_idx, data_bytes.len());
    }
    Ok(arr)
  }

  fn serialize(&self, writer: &mut Write) -> Result<(), ()> {
    let ty_id = T::serial_id();
    writer.write_u32::<LittleEndian>(0x0000444e | ((ty_id as u32) << 24))
      .ok().expect("failed to serialize!");
    writer.write_u32::<LittleEndian>(2)
      .ok().expect("failed to serialize!");
    let (bound0, bound1) = self.bound;
    writer.write_u64::<LittleEndian>(bound0 as u64)
      .ok().expect("failed to serialize!");
    writer.write_u64::<LittleEndian>(bound1 as u64)
      .ok().expect("failed to serialize!");
    if self.bound.to_least_stride() == self.stride {
      let bytes = unsafe { from_raw_parts(self.data.as_ptr() as *const u8, size_of::<f32>() * self.data.len()) };
      writer.write_all(bytes)
        .ok().expect("failed to serialize!");
    } else {
      unimplemented!();
    }
    Ok(())
  }
}

impl<'a, T> Array<'a, T, (usize, usize)> for Array2d<T> where T: 'a + Copy {
  type View     = Array2dView<'a, T>;
  type ViewMut  = Array2dViewMut<'a, T>;

  fn as_view(&'a self) -> Array2dView<'a, T> {
    Array2dView{
      data:     &self.data,
      bound:    self.bound,
      stride:   self.stride,
    }
  }

  fn as_view_mut(&'a mut self) -> Array2dViewMut<'a, T> {
    Array2dViewMut{
      data:     &mut self.data,
      bound:    self.bound,
      stride:   self.stride,
    }
  }
}

pub struct Array2dView<'a, T> where T: 'a + Copy {
  data:     &'a [T],
  bound:    (usize, usize),
  stride:   usize,
}

impl<'a, T> ArrayView<'a, T, (usize, usize)> for Array2dView<'a, T> where T: 'a + Copy {
  fn bound(&self) -> (usize, usize) {
    self.bound
  }

  fn stride(&self) -> usize {
    self.stride
  }

  fn len(&self) -> usize {
    self.bound.len()
  }

  unsafe fn as_ptr(&self) -> *const T {
    self.data.as_ptr()
  }

  fn view(self, lo: (usize, usize), hi: (usize, usize)) -> Array2dView<'a, T> {
    // TODO(20151215)
    unimplemented!();
  }
}

impl<'a, T> Array2dView<'a, T> where T: 'a + Copy {
  pub fn as_slice(&self) -> &[T] {
    self.data
  }
}

pub struct Array2dViewMut<'a, T> where T: 'a + Copy {
  data:     &'a mut [T],
  bound:    (usize, usize),
  stride:   usize,
}

impl<'a, T> ArrayViewMut<'a, T, (usize, usize)> for Array2dViewMut<'a, T> where T: 'a + Copy {
  fn bound(&self) -> (usize, usize) {
    self.bound
  }

  fn stride(&self) -> usize {
    self.stride
  }

  fn len(&self) -> usize {
    self.bound.len()
  }

  unsafe fn as_ptr(&self) -> *const T {
    self.data.as_ptr()
  }

  unsafe fn as_mut_ptr(&mut self) -> *mut T {
    self.data.as_mut_ptr()
  }

  fn view_mut(self, lo: (usize, usize), hi: (usize, usize)) -> Array2dViewMut<'a, T> {
    // TODO(20151215)
    unimplemented!();
  }
}

impl<'a, T> Array2dViewMut<'a, T> where T: 'a + Copy {
  pub fn as_mut_slice(&mut self) -> &mut [T] {
    self.data
  }
}

pub struct BitArray3d {
  data:     Vec<u64>,
  bound:    (usize, usize, usize),
  raw_len:  usize,
}

impl BitArray3d {
  pub unsafe fn new(bound: (usize, usize, usize)) -> BitArray3d {
    let len = bound.len();
    let raw_len = (len + 64 - 1) / 64;
    let mut data = Vec::with_capacity(raw_len);
    unsafe { data.set_len(raw_len) };
    BitArray3d{
      data:     data,
      bound:    bound,
      raw_len:  raw_len,
    }
  }

  pub fn from_byte_array(arr: &Array3d<u8>) -> BitArray3d {
    assert!(arr.stride == arr.bound.to_least_stride());
    let mut raw_arr = unsafe { BitArray3d::new(arr.bound) };
    let len = raw_arr.bound.len();
    let raw_len = (len + 64 - 1) / 64;
    {
      let mut idx = 0;
      for p in 0 .. raw_len {
        raw_arr.data[p] = 0;
        for s in 0 .. 64 {
          if arr.data[idx] != 0 {
            raw_arr.data[p] |= 1u64 << s;
          }
          idx += 1;
          if idx >= len {
            break;
          }
        }
      }
    }
    raw_arr
  }

  pub fn into_bytes(&self, nonzero_value: u8) -> Array3d<u8> {
    let mut array = unsafe { Array3d::new(self.bound) };
    let len = self.bound.len();
    let raw_len = self.raw_len;
    {
      let mut idx = 0;
      for p in 0 .. raw_len {
        let mask = self.data[p];
        for s in 0 .. 64 {
          if 64 * p + s < len {
            array.data[idx] = match (mask >> s) & 1 {
              0 => 0u8,
              1 => nonzero_value,
              _ => unreachable!(),
            };
            idx += 1;
          } else {
            break;
          }
        }
      }
    }
    array
  }

  pub fn write_bytes(&self, nonzero_value: u8, output: &mut Array3dViewMut<u8>) {
    assert_eq!(self.bound(), output.bound());
    assert_eq!(self.stride(), output.stride());
    let len = self.bound.len();
    let raw_len = self.raw_len;
    {
      let mut data = output.as_mut_slice();
      let mut idx = 0;
      for p in 0 .. raw_len {
        let mask = self.data[p];
        for s in 0 .. 64 {
          if 64 * p + s < len {
            data[idx] = match (mask >> s) & 1 {
              0 => 0u8,
              1 => nonzero_value,
              _ => unreachable!(),
            };
            idx += 1;
          } else {
            break;
          }
        }
      }
    }
  }

  pub fn bound(&self) -> (usize, usize, usize) {
    self.bound
  }

  pub fn stride(&self) -> (usize, usize) {
    self.bound.to_least_stride()
  }
}

impl BitArray3d {
  pub fn serial_size(bound: (usize, usize, usize)) -> usize {
    32 + (bound.len() + 64 - 1) / 64 * 8
  }

  pub fn deserialize(reader: &mut Read) -> Result<BitArray3d, ()> {
    let magic0 = reader.read_u8()
      .ok().expect("failed to deserialize!");
    let magic1 = reader.read_u8()
      .ok().expect("failed to deserialize!");
    assert_eq!(magic0, b'N');
    assert_eq!(magic1, b'D');
    let version = reader.read_u8()
      .ok().expect("failed to deserialize!");
    assert_eq!(version, 0);
    let data_ty = reader.read_u8()
      .ok().expect("failed to deserialize!");
    let ndim = reader.read_u32::<LittleEndian>()
      .ok().expect("failed to deserialize!");
    let expected_data_ty = 255u8;
    assert_eq!(data_ty, expected_data_ty);
    assert_eq!(ndim, 3);
    let bound0 = reader.read_u64::<LittleEndian>()
      .ok().expect("failed to deserialize!") as usize;
    let bound1 = reader.read_u64::<LittleEndian>()
      .ok().expect("failed to deserialize!") as usize;
    let bound2 = reader.read_u64::<LittleEndian>()
      .ok().expect("failed to deserialize!") as usize;
    let dims = (bound0, bound1, bound2);
    let mut arr = unsafe { BitArray3d::new(dims) };
    {
      let mut data_bytes = unsafe { from_raw_parts_mut(arr.data.as_mut_ptr() as *mut u8, 8 * arr.raw_len) };
      let mut read_idx: usize = 0;
      loop {
        match reader.read(&mut data_bytes[read_idx ..]) {
          Ok(n) => {
            read_idx += n;
            if n == 0 {
              break;
            }
          }
          Err(e) => panic!("failed to deserialize: {:?}", e),
        }
      }
      assert_eq!(read_idx, data_bytes.len());
    }
    Ok(arr)
  }

  pub fn serialize(&self, writer: &mut Write) -> Result<(), ()> {
    let ty_id = 255u8;
    writer.write_u32::<LittleEndian>(0x0000444e | ((ty_id as u32) << 24))
      .ok().expect("failed to serialize!");
    writer.write_u32::<LittleEndian>(3)
      .ok().expect("failed to serialize!");
    let (bound0, bound1, bound2) = self.bound;
    writer.write_u64::<LittleEndian>(bound0 as u64)
      .ok().expect("failed to serialize!");
    writer.write_u64::<LittleEndian>(bound1 as u64)
      .ok().expect("failed to serialize!");
    writer.write_u64::<LittleEndian>(bound2 as u64)
      .ok().expect("failed to serialize!");
    let bytes = unsafe { from_raw_parts(self.data.as_ptr() as *const u8, 8 * self.raw_len) };
    writer.write_all(bytes)
      .ok().expect("failed to serialize!");
    Ok(())
  }
}

#[derive(Clone)]
pub struct Array3d<T> where T: Copy {
  data:     Vec<T>,
  bound:    (usize, usize, usize),
  stride:   (usize, usize),
}

impl<'a, T> Array<'a, T, (usize, usize, usize)> for Array3d<T> where T: 'a + Copy {
  type View     = Array3dView<'a, T>;
  type ViewMut  = Array3dViewMut<'a, T>;

  fn as_view(&'a self) -> Array3dView<'a, T> {
    Array3dView{
      data:     &self.data,
      bound:    self.bound,
      stride:   self.stride,
    }
  }

  fn as_view_mut(&'a mut self) -> Array3dViewMut<'a, T> {
    Array3dViewMut{
      data:     &mut self.data,
      bound:    self.bound,
      stride:   self.stride,
    }
  }
}

impl<T> Array3d<T> where T: Copy {
  pub unsafe fn new(bound: (usize, usize, usize)) -> Array3d<T> {
    let len = bound.len();
    let mut data = Vec::with_capacity(len);
    data.set_len(len);
    Array3d{
      data:     data,
      bound:    bound,
      stride:   bound.to_least_stride(),
    }
  }

  pub fn with_data(data: Vec<T>, bound: (usize, usize, usize)) -> Array3d<T> {
    let len = bound.len();
    assert_eq!(len, data.len());
    Array3d{
      data:     data,
      bound:    bound,
      stride:   bound.to_least_stride(),
    }
  }

  pub fn as_slice(&self) -> &[T] {
    &self.data
  }

  pub fn as_mut_slice(&mut self) -> &mut [T] {
    &mut self.data
  }

  pub fn bound(&self) -> (usize, usize, usize) {
    self.bound
  }

  pub fn stride(&self) -> (usize, usize) {
    self.stride
  }
}

impl<T> NdArraySerialize<T, (usize, usize, usize)> for Array3d<T> where T: SerialDataType + Copy {
  fn serial_size(bound: (usize, usize, usize)) -> usize {
    32 + bound.len()
  }

  fn deserialize(reader: &mut Read) -> Result<Array3d<T>, ()> {
    let magic0 = reader.read_u8()
      .ok().expect("failed to deserialize!");
    let magic1 = reader.read_u8()
      .ok().expect("failed to deserialize!");
    assert_eq!(magic0, b'N');
    assert_eq!(magic1, b'D');
    let version = reader.read_u8()
      .ok().expect("failed to deserialize!");
    assert_eq!(version, 0);
    let data_ty = reader.read_u8()
      .ok().expect("failed to deserialize!");
    let ndim = reader.read_u32::<LittleEndian>()
      .ok().expect("failed to deserialize!");
    let expected_data_ty = T::serial_id();
    assert_eq!(data_ty, expected_data_ty);
    assert_eq!(ndim, 3);
    let bound0 = reader.read_u64::<LittleEndian>()
      .ok().expect("failed to deserialize!") as usize;
    let bound1 = reader.read_u64::<LittleEndian>()
      .ok().expect("failed to deserialize!") as usize;
    let bound2 = reader.read_u64::<LittleEndian>()
      .ok().expect("failed to deserialize!") as usize;
    let dims = (bound0, bound1, bound2);
    let mut arr = unsafe { Array3d::new(dims) };
    {
      let mut data_bytes = unsafe { from_raw_parts_mut(arr.data.as_mut_ptr() as *mut u8, size_of::<T>() * arr.data.len()) };
      let mut read_idx: usize = 0;
      loop {
        match reader.read(&mut data_bytes[read_idx ..]) {
          Ok(n) => {
            read_idx += n;
            if n == 0 {
              break;
            }
          }
          Err(e) => panic!("failed to deserialize: {:?}", e),
        }
      }
      assert_eq!(read_idx, data_bytes.len());
    }
    Ok(arr)
  }

  fn serialize(&self, writer: &mut Write) -> Result<(), ()> {
    let ty_id = T::serial_id();
    writer.write_u32::<LittleEndian>(0x0000444e | ((ty_id as u32) << 24))
      .ok().expect("failed to serialize!");
    writer.write_u32::<LittleEndian>(3)
      .ok().expect("failed to serialize!");
    let (bound0, bound1, bound2) = self.bound;
    writer.write_u64::<LittleEndian>(bound0 as u64)
      .ok().expect("failed to serialize!");
    writer.write_u64::<LittleEndian>(bound1 as u64)
      .ok().expect("failed to serialize!");
    writer.write_u64::<LittleEndian>(bound2 as u64)
      .ok().expect("failed to serialize!");
    if self.bound.to_least_stride() == self.stride {
      let bytes = unsafe { from_raw_parts(self.data.as_ptr() as *const u8, size_of::<T>() * self.data.len()) };
      writer.write_all(bytes)
        .ok().expect("failed to serialize!");
    } else {
      unimplemented!();
    }
    Ok(())
  }
}

pub struct Array3dView<'a, T> where T: 'a + Copy {
  data:     &'a [T],
  bound:    (usize, usize, usize),
  stride:   (usize, usize),
}

impl<'a, T> ArrayView<'a, T, (usize, usize, usize)> for Array3dView<'a, T> where T: 'a + Copy {
  fn bound(&self) -> (usize, usize, usize) {
    self.bound
  }

  fn stride(&self) -> (usize, usize) {
    self.stride
  }

  fn len(&self) -> usize {
    self.bound.len()
  }

  unsafe fn as_ptr(&self) -> *const T {
    self.data.as_ptr()
  }

  fn view(self, lo: (usize, usize, usize), hi: (usize, usize, usize)) -> Array3dView<'a, T> {
    // TODO(20151215)
    unimplemented!();
  }
}

pub struct Array3dViewMut<'a, T> where T: 'a + Copy {
  data:     &'a mut [T],
  bound:    (usize, usize, usize),
  stride:   (usize, usize),
}

impl<'a, T> Array3dViewMut<'a, T> where T: 'a + Copy {
  pub fn as_mut_slice(&mut self) -> &mut [T] {
    self.data
  }

  pub fn copy_from(&mut self, src: &Array3dView<'a, T>) {
    assert_eq!(self.bound(), src.bound());
    if self.stride() == self.bound().to_least_stride() && self.stride() == src.stride() {
      self.data.clone_from_slice(src.data);
    } else {
      // FIXME(20160202)
      panic!("unimplemented: strided 3d array copy");
    }
  }
}

impl<'a, T> ArrayViewMut<'a, T, (usize, usize, usize)> for Array3dViewMut<'a, T> where T: 'a + Copy {
  fn bound(&self) -> (usize, usize, usize) {
    self.bound
  }

  fn stride(&self) -> (usize, usize) {
    self.stride
  }

  fn len(&self) -> usize {
    self.bound.len()
  }

  unsafe fn as_ptr(&self) -> *const T {
    self.data.as_ptr()
  }

  unsafe fn as_mut_ptr(&mut self) -> *mut T {
    self.data.as_mut_ptr()
  }

  fn view_mut(self, lo: (usize, usize, usize), hi: (usize, usize, usize)) -> Array3dViewMut<'a, T> {
    let new_bound = (hi.0 - lo.0, hi.1 - lo.1, hi.2 - lo.2);
    let new_offset = lo.offset(self.stride);
    // FIXME(20160203): array index arithmetic.
    //let new_offset_end = hi.offset(self.stride);
    assert_eq!(self.stride, self.bound.to_least_stride());
    let new_offset_end = new_offset + new_bound.len();
    assert!(new_offset <= self.data.len());
    assert!(new_offset_end <= self.data.len());
    assert!(new_offset <= new_offset_end);
    Array3dViewMut{
      data:     &mut self.data[new_offset .. new_offset_end],
      bound:    new_bound,
      stride:   self.stride,
    }
  }
}
