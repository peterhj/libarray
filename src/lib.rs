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
  type ViewMut: ArrayView<'a, T, S>;

  fn as_view(&'a self) -> Self::View;
  fn as_view_mut(&'a mut self) -> Self::ViewMut;
}

pub trait AsyncArray<'ctx, 'a, T, S> where 'ctx: 'a, T: 'a + Copy, S: Shape {
  type Ctx;
  type View: ArrayView<'a, T, S>;
  type ViewMut: ArrayViewMut<'a, T, S>;

  fn as_view_async(&'a mut self, ctx: &'a Self::Ctx) -> Self::View;
  fn as_view_mut_async(&'a mut self, ctx: &'a Self::Ctx) -> Self::ViewMut;
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
  unsafe fn as_mut_ptr(&mut self) -> *mut T;
  fn view_mut(&mut self, lo: S, hi: S) -> Self;
}
