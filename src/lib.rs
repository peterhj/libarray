use std::io::{Read, Write};

pub trait Shape: Copy {
  type Stride;

  fn to_least_stride(&self) -> Self::Stride;
  fn len(&self) -> usize;
  fn offset(&self, stride: Self::Stride) -> usize;
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
  fn view(&self, lo: S, hi: S) -> Self;
}

pub trait ArrayViewMut<'a, T, S>/*: ArrayView<'a, T, S>*/ where T: 'a + Copy, S: Shape {
  fn bound(&self) -> S;
  fn stride(&self) -> S::Stride;
  fn len(&self) -> usize;
  unsafe fn as_ptr(&self) -> *const T;
  unsafe fn as_mut_ptr(&mut self) -> *mut T;
  fn view_mut(&mut self, lo: S, hi: S) -> Self;
}

pub trait ArrayZeroExt<T, S> where T: Copy, S: Shape {
  fn zeros(bound: S) -> Self;
}

pub struct Array2d<T> where T: Copy {
  data:     Vec<T>,
  bound:    (usize, usize),
  stride:   usize,
}

impl ArrayZeroExt<i32, (usize, usize)> for Array2d<i32> {
  fn zeros(bound: (usize, usize)) -> Array2d<i32> {
    let len = bound.len();
    let mut data = Vec::with_capacity(len);
    unsafe { data.set_len(len) };
    for i in (0 .. len) {
      data[i] = 0;
    }
    Array2d{
      data:     data,
      bound:    bound,
      stride:   bound.to_least_stride(),
    }
  }
}

impl ArrayZeroExt<f32, (usize, usize)> for Array2d<f32> {
  fn zeros(bound: (usize, usize)) -> Array2d<f32> {
    let len = bound.len();
    let mut data = Vec::with_capacity(len);
    unsafe { data.set_len(len) };
    for i in (0 .. len) {
      data[i] = 0.0;
    }
    Array2d{
      data:     data,
      bound:    bound,
      stride:   bound.to_least_stride(),
    }
  }
}

impl<T> Array2d<T> where T: Copy {
  pub fn deserialize(reader: &mut Read) -> Result<Array2d<T>, ()> {
    // TODO(20151218)
    unimplemented!();
  }

  pub fn serialize(&self, writer: &mut Write) {
    // TODO(20151218)
    unimplemented!();
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

  fn view(&self, lo: (usize, usize), hi: (usize, usize)) -> Array2dView<'a, T> {
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

  fn view_mut(&mut self, lo: (usize, usize), hi: (usize, usize)) -> Array2dViewMut<'a, T> {
    // TODO(20151215)
    unimplemented!();
  }
}

impl<'a, T> Array2dViewMut<'a, T> where T: 'a + Copy {
  pub fn as_mut_slice(&mut self) -> &mut [T] {
    self.data
  }
}

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
  pub fn deserialize(reader: &mut Read) -> Result<Array3d<T>, ()> {
    // TODO(20151218)
    unimplemented!();
  }

  pub fn serialize(&self, writer: &mut Write) {
    // TODO(20151218)
    unimplemented!();
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

  fn view(&self, lo: (usize, usize, usize), hi: (usize, usize, usize)) -> Array3dView<'a, T> {
    // TODO(20151215)
    unimplemented!();
  }
}

pub struct Array3dViewMut<'a, T> where T: 'a + Copy {
  data:     &'a mut [T],
  bound:    (usize, usize, usize),
  stride:   (usize, usize),
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

  fn view_mut(&mut self, lo: (usize, usize, usize), hi: (usize, usize, usize)) -> Array3dViewMut<'a, T> {
    // TODO(20151215)
    unimplemented!();
  }
}
