// Copyright ©2015 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package netlib provides an interface to bindings for a C LAPACK library.
package netlib

import (
	"math"

	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/lapack"
	"gonum.org/v1/netlib/lapack/lapacke"
)

// Implementation is the cgo-based C implementation of LAPACK routines.
type Implementation struct{}

var _ lapack.Float64 = Implementation{}

// Dgeqp3 computes a QR factorization with column pivoting of the
// m×n matrix A: A*P = Q*R using Level 3 BLAS.
//
// The matrix Q is represented as a product of elementary reflectors
//
//	Q = H_0 H_1 . . . H_{k-1}, where k = min(m,n).
//
// Each H_i has the form
//
//	H_i = I - tau * v * v^T
//
// where tau and v are real vectors with v[0:i-1] = 0 and v[i] = 1;
// v[i:m] is stored on exit in A[i:m, i], and tau in tau[i].
//
// jpvt specifies a column pivot to be applied to A. If
// jpvt[j] is at least zero, the jth column of A is permuted
// to the front of A*P (a leading column), if jpvt[j] is -1
// the jth column of A is a free column. If jpvt[j] < -1, Dgeqp3
// will panic. On return, jpvt holds the permutation that was
// applied; the jth column of A*P was the jpvt[j] column of A.
// jpvt must have length n or Dgeqp3 will panic.
//
// tau holds the scalar factors of the elementary reflectors.
// It must have length min(m, n), otherwise Dgeqp3 will panic.
//
// work must have length at least max(1,lwork), and lwork must be at least
// 3*n+1, otherwise Dgeqp3 will panic. For optimal performance lwork must
// be at least 2*n+(n+1)*nb, where nb is the optimal blocksize. On return,
// work[0] will contain the optimal value of lwork.
//
// If lwork == -1, instead of performing Dgeqp3, only the optimal value of lwork
// will be stored in work[0].
//
// Dgeqp3 is an internal routine. It is exported for testing purposes.
func (impl Implementation) Dgeqp3(m, n int, a []float64, lda int, jpvt []int, tau, work []float64, lwork int) {
	minmn := min(m, n)
	iws := 3*n + 1
	if minmn == 0 {
		iws = 1
	}
	switch {
	case m < 0:
		panic(mLT0)
	case n < 0:
		panic(nLT0)
	case lda < max(1, n):
		panic(badLdA)
	case lwork < iws && lwork != -1:
		panic(badLWork)
	case len(work) < max(1, lwork):
		panic(shortWork)
	}

	// Quick return if possible.
	if minmn == 0 {
		work[0] = 1
		return
	}

	// Don't update jpvt if querying lwkopt.
	if lwork == -1 {
		lapacke.Dgeqp3(m, n, a, lda, nil, nil, work, -1)
		return
	}

	switch {
	case len(a) < (m-1)*lda+n:
		panic(shortA)
	case len(jpvt) != n:
		panic(badLenJpvt)
	case len(tau) < minmn:
		panic(shortTau)
	}

	jpvt32 := make([]int32, n)
	for i, v := range jpvt {
		v++
		if v != int(int32(v)) || v < 0 || n < v {
			panic(badJpvt)
		}
		jpvt32[i] = int32(v)
	}
	lapacke.Dgeqp3(m, n, a, lda, jpvt32, tau, work, lwork)
	for i, v := range jpvt32 {
		jpvt[i] = int(v - 1)
	}
}

// Dgerqf computes an RQ factorization of the m×n matrix A,
//
//	A = R * Q.
//
// On exit, if m <= n, the upper triangle of the subarray
// A[0:m, n-m:n] contains the m×m upper triangular matrix R.
// If m >= n, the elements on and above the (m-n)-th subdiagonal
// contain the m×n upper trapezoidal matrix R.
// The remaining elements, with tau, represent the
// orthogonal matrix Q as a product of min(m,n) elementary
// reflectors.
//
// The matrix Q is represented as a product of elementary reflectors
//
//	Q = H_0 H_1 . . . H_{min(m,n)-1}.
//
// Each H(i) has the form
//
//	H_i = I - tau_i * v * v^T
//
// where v is a vector with v[0:n-k+i-1] stored in A[m-k+i, 0:n-k+i-1],
// v[n-k+i:n] = 0 and v[n-k+i] = 1.
//
// tau must have length min(m,n), work must have length max(1, lwork),
// and lwork must be -1 or at least max(1, m), otherwise Dgerqf will panic.
// On exit, work[0] will contain the optimal length for work.
//
// Dgerqf is an internal routine. It is exported for testing purposes.
func (impl Implementation) Dgerqf(m, n int, a []float64, lda int, tau, work []float64, lwork int) {
	switch {
	case m < 0:
		panic(mLT0)
	case n < 0:
		panic(nLT0)
	case lda < max(1, n):
		panic(badLdA)
	case lwork < max(1, m) && lwork != -1:
		panic(badLWork)
	case len(work) < max(1, lwork):
		panic(shortWork)
	}

	// Quick return if possible.
	k := min(m, n)
	if k == 0 {
		work[0] = 1
		return
	}

	if lwork == -1 {
		lapacke.Dgerqf(m, n, a, lda, tau, work, -1)
		return
	}

	switch {
	case len(a) < (m-1)*lda+n:
		panic(shortA)
	case len(tau) != k:
		panic(badLenTau)
	}

	lapacke.Dgerqf(m, n, a, lda, tau, work, lwork)
}

// Dlacn2 estimates the 1-norm of an n×n matrix A using sequential updates with
// matrix-vector products provided externally.
//
// Dlacn2 is called sequentially and it returns the value of est and kase to be
// used on the next call.
// On the initial call, kase must be 0.
// In between calls, x must be overwritten by
//
//	A * X    if kase was returned as 1,
//	A^T * X  if kase was returned as 2,
//
// and all other parameters must not be changed.
// On the final return, kase is returned as 0, v contains A*W where W is a
// vector, and est = norm(V)/norm(W) is a lower bound for 1-norm of A.
//
// v, x, and isgn must all have length n and n must be at least 1, otherwise
// Dlacn2 will panic. isave is used for temporary storage.
//
// Dlacn2 is an internal routine. It is exported for testing purposes.
func (impl Implementation) Dlacn2(n int, v, x []float64, isgn []int, est float64, kase int, isave *[3]int) (float64, int) {
	switch {
	case n < 1:
		panic(nLT1)
	case len(v) < n:
		panic(shortV)
	case len(x) < n:
		panic(shortX)
	case len(isgn) < n:
		panic(shortIsgn)
	case isave[0] < 0 || 5 < isave[0]:
		panic(badIsave)
	case isave[0] == 0 && kase != 0:
		panic(badIsave)
	}

	isgn32 := make([]int32, n)
	pest := []float64{est}
	// Save one allocation by putting isave and kase into the same slice.
	isavekase := []int32{int32(isave[0]), int32(isave[1]), int32(isave[2]), int32(kase)}
	lapacke.Dlacn2(n, v, x, isgn32, pest, isavekase[3:], isavekase[:3])
	for i, v := range isgn32 {
		isgn[i] = int(v)
	}
	isave[0] = int(isavekase[0])
	isave[1] = int(isavekase[1])
	isave[2] = int(isavekase[2])

	return pest[0], int(isavekase[3])
}

// Dlacpy copies the elements of A specified by uplo into B. Uplo can specify
// a triangular portion with blas.Upper or blas.Lower, or can specify all of the
// elemest with blas.All.
func (impl Implementation) Dlacpy(uplo blas.Uplo, m, n int, a []float64, lda int, b []float64, ldb int) {
	switch {
	case uplo != blas.Upper && uplo != blas.Lower && uplo != blas.All:
		panic(badUplo)
	case m < 0:
		panic(mLT0)
	case n < 0:
		panic(nLT0)
	case lda < max(1, n):
		panic(badLdA)
	case ldb < max(1, n):
		panic(badLdB)
	}

	if m == 0 || n == 0 {
		return
	}

	switch {
	case len(a) < (m-1)*lda+n:
		panic(shortA)
	case len(b) < (m-1)*ldb+n:
		panic(shortB)
	}

	lapacke.Dlacpy(byte(uplo), m, n, a, lda, b, ldb)
}

// Dlapmr rearranges the rows of the m×n matrix X as specified by the permutation
// k[0],k[1],...,k[m-1] of the integers 0,...,m-1.
//
// If forward is true, a forward permutation is applied:
//
//	X[k[i],0:n] is moved to X[i,0:n] for i=0,1,...,m-1.
//
// If forward is false, a backward permutation is applied:
//
//	X[i,0:n] is moved to X[k[i],0:n] for i=0,1,...,m-1.
//
// k must have length m, otherwise Dlapmr will panic.
func (impl Implementation) Dlapmr(forward bool, m, n int, x []float64, ldx int, k []int) {
	switch {
	case m < 0:
		panic(mLT0)
	case n < 0:
		panic(nLT0)
	case ldx < max(1, n):
		panic(badLdX)
	}

	// Quick return if possible.
	if m == 0 || n == 0 {
		return
	}

	switch {
	case len(x) < (m-1)*ldx+n:
		panic(shortX)
	case len(k) != m:
		panic(badLenK)
	}

	// Quick return if possible.
	if m == 1 {
		return
	}

	var forwrd int32
	if forward {
		forwrd = 1
	}
	k32 := make([]int32, m)
	for i, v := range k {
		v++ // Convert to 1-based indexing.
		if v != int(int32(v)) {
			panic("lapack: k element out of range")
		}
		k32[i] = int32(v)
	}
	lapacke.Dlapmr(forwrd, m, n, x, ldx, k32)
}

// Dlapmt rearranges the columns of the m×n matrix X as specified by the
// permutation k_0, k_1, ..., k_n-1 of the integers 0, ..., n-1.
//
// If forward is true a forward permutation is performed:
//
//	X[0:m, k[j]] is moved to X[0:m, j] for j = 0, 1, ..., n-1.
//
// otherwise a backward permutation is performed:
//
//	X[0:m, j] is moved to X[0:m, k[j]] for j = 0, 1, ..., n-1.
//
// k must have length n, otherwise Dlapmt will panic. k is zero-indexed.
func (impl Implementation) Dlapmt(forward bool, m, n int, x []float64, ldx int, k []int) {
	switch {
	case m < 0:
		panic(mLT0)
	case n < 0:
		panic(nLT0)
	case ldx < max(1, n):
		panic(badLdX)
	}

	// Quick return if possible.
	if m == 0 || n == 0 {
		return
	}

	switch {
	case len(x) < (m-1)*ldx+n:
		panic(shortX)
	case len(k) != n:
		panic(badLenK)
	}

	// Quick return if possible.
	if n == 1 {
		return
	}

	var forwrd int32
	if forward {
		forwrd = 1
	}
	k32 := make([]int32, n)
	for i, v := range k {
		v++ // Convert to 1-based indexing.
		if v != int(int32(v)) {
			panic("lapack: k element out of range")
		}
		k32[i] = int32(v)
	}
	lapacke.Dlapmt(forwrd, m, n, x, ldx, k32)
}

// Dlapy2 is the LAPACK version of math.Hypot.
//
// Dlapy2 is an internal routine. It is exported for testing purposes.
func (Implementation) Dlapy2(x, y float64) float64 {
	return lapacke.Dlapy2(x, y)
}

// Dlarfb applies a block reflector to a matrix.
//
// In the call to Dlarfb, the mxn c is multiplied by the implicitly defined matrix h as follows:
//
//	c = h * c if side == Left and trans == NoTrans
//	c = c * h if side == Right and trans == NoTrans
//	c = h^T * c if side == Left and trans == Trans
//	c = c * h^T if side == Right and trans == Trans
//
// h is a product of elementary reflectors. direct sets the direction of multiplication
//
//	h = h_1 * h_2 * ... * h_k if direct == Forward
//	h = h_k * h_k-1 * ... * h_1 if direct == Backward
//
// The combination of direct and store defines the orientation of the elementary
// reflectors. In all cases the ones on the diagonal are implicitly represented.
//
// If direct == lapack.Forward and store == lapack.ColumnWise
//
//	V = [ 1        ]
//	    [v1   1    ]
//	    [v1  v2   1]
//	    [v1  v2  v3]
//	    [v1  v2  v3]
//
// If direct == lapack.Forward and store == lapack.RowWise
//
//	V = [ 1  v1  v1  v1  v1]
//	    [     1  v2  v2  v2]
//	    [         1  v3  v3]
//
// If direct == lapack.Backward and store == lapack.ColumnWise
//
//	V = [v1  v2  v3]
//	    [v1  v2  v3]
//	    [ 1  v2  v3]
//	    [     1  v3]
//	    [         1]
//
// If direct == lapack.Backward and store == lapack.RowWise
//
//	V = [v1  v1   1        ]
//	    [v2  v2  v2   1    ]
//	    [v3  v3  v3  v3   1]
//
// An elementary reflector can be explicitly constructed by extracting the
// corresponding elements of v, placing a 1 where the diagonal would be, and
// placing zeros in the remaining elements.
//
// t is a k×k matrix containing the block reflector, and this function will panic
// if t is not of sufficient size. See Dlarft for more information.
//
// work is a temporary storage matrix with stride ldwork.
// work must be of size at least n×k side == Left and m×k if side == Right, and
// this function will panic if this size is not met.
//
// Dlarfb is an internal routine. It is exported for testing purposes.
func (Implementation) Dlarfb(side blas.Side, trans blas.Transpose, direct lapack.Direct, store lapack.StoreV, m, n, k int, v []float64, ldv int, t []float64, ldt int, c []float64, ldc int, work []float64, ldwork int) {
	nv := m
	if side == blas.Right {
		nv = n
	}
	switch {
	case side != blas.Left && side != blas.Right:
		panic(badSide)
	case trans != blas.Trans && trans != blas.NoTrans:
		panic(badTrans)
	case direct != lapack.Forward && direct != lapack.Backward:
		panic(badDirect)
	case store != lapack.ColumnWise && store != lapack.RowWise:
		panic(badStoreV)
	case m < 0:
		panic(mLT0)
	case n < 0:
		panic(nLT0)
	case k < 0:
		panic(kLT0)
	case store == lapack.ColumnWise && ldv < max(1, k):
		panic(badLdV)
	case store == lapack.RowWise && ldv < max(1, nv):
		panic(badLdV)
	case ldt < max(1, k):
		panic(badLdT)
	case ldc < max(1, n):
		panic(badLdC)
	case ldwork < max(1, k):
		panic(badLdWork)
	}

	if m == 0 || n == 0 {
		return
	}

	nw := n
	if side == blas.Right {
		nw = m
	}
	switch {
	case store == lapack.ColumnWise && len(v) < (nv-1)*ldv+k:
		panic(shortV)
	case store == lapack.RowWise && len(v) < (k-1)*ldv+nv:
		panic(shortV)
	case len(t) < (k-1)*ldt+k:
		panic(shortT)
	case len(c) < (m-1)*ldc+n:
		panic(shortC)
	case len(work) < (nw-1)*ldwork+k:
		panic(shortWork)
	}

	// TODO(vladimir-ch): Replace the following two lines with
	//  checkMatrix(nw, k, work, ldwork)
	// if and when the issue
	//  https://github.com/Reference-LAPACK/lapack/issues/37
	// has been resolved.
	ldwork = nw
	work = make([]float64, ldwork*k)

	lapacke.Dlarfb(byte(side), byte(trans), byte(direct), byte(store), m, n, k, v, ldv, t, ldt, c, ldc, work, ldwork)
}

// Dlarfg generates an elementary reflector for a Householder matrix. It creates
// a real elementary reflector of order n such that
//
//	H * (alpha) = (beta)
//	    (    x)   (   0)
//	H^T * H = I
//
// H is represented in the form
//
//	H = 1 - tau * (1; v) * (1 v^T)
//
// where tau is a real scalar.
//
// On entry, x contains the vector x, on exit it contains v.
//
// Dlarfg is an internal routine. It is exported for testing purposes.
func (impl Implementation) Dlarfg(n int, alpha float64, x []float64, incX int) (beta, tau float64) {
	switch {
	case n < 0:
		panic(nLT0)
	case incX <= 0:
		panic(badIncX)
	}

	if n <= 1 {
		return alpha, 0
	}

	aincX := incX
	if aincX < 0 {
		aincX = -aincX
	}
	if len(x) < 1+(n-2)*aincX {
		panic(shortX)
	}

	_alpha := []float64{alpha}
	_tau := []float64{0}
	lapacke.Dlarfg(n, _alpha, x, incX, _tau)
	return _alpha[0], _tau[0]
}

// Dlarft forms the triangular factor T of a block reflector H, storing the answer
// in t.
//
//	H = I - V * T * V^T  if store == lapack.ColumnWise
//	H = I - V^T * T * V  if store == lapack.RowWise
//
// H is defined by a product of the elementary reflectors where
//
//	H = H_0 * H_1 * ... * H_{k-1}  if direct == lapack.Forward
//	H = H_{k-1} * ... * H_1 * H_0  if direct == lapack.Backward
//
// t is a k×k triangular matrix. t is upper triangular if direct = lapack.Forward
// and lower triangular otherwise. This function will panic if t is not of
// sufficient size.
//
// store describes the storage of the elementary reflectors in v. Please see
// Dlarfb for a description of layout.
//
// tau contains the scalar factors of the elementary reflectors H_i.
//
// Dlarft is an internal routine. It is exported for testing purposes.
func (Implementation) Dlarft(direct lapack.Direct, store lapack.StoreV, n, k int, v []float64, ldv int, tau []float64, t []float64, ldt int) {
	mv, nv := n, k
	if store == lapack.RowWise {
		mv, nv = k, n
	}
	switch {
	case direct != lapack.Forward && direct != lapack.Backward:
		panic(badDirect)
	case store != lapack.RowWise && store != lapack.ColumnWise:
		panic(badStoreV)
	case n < 0:
		panic(nLT0)
	case k < 1:
		panic(kLT1)
	case ldv < max(1, nv):
		panic(badLdV)
	case len(tau) < k:
		panic(shortTau)
	case ldt < max(1, k):
		panic(shortT)
	}

	if n == 0 {
		return
	}

	switch {
	case len(v) < (mv-1)*ldv+nv:
		panic(shortV)
	case len(t) < (k-1)*ldt+k:
		panic(shortT)
	}

	lapacke.Dlarft(byte(direct), byte(store), n, k, v, ldv, tau, t, ldt)
}

// Dlange computes the matrix norm of the general m×n matrix a. The input norm
// specifies the norm computed.
//
//	lapack.MaxAbs: the maximum absolute value of an element.
//	lapack.MaxColumnSum: the maximum column sum of the absolute values of the entries.
//	lapack.MaxRowSum: the maximum row sum of the absolute values of the entries.
//	lapack.Frobenius: the square root of the sum of the squares of the entries.
//
// If norm == lapack.MaxColumnSum, work must be of length n, and this function will panic otherwise.
// There are no restrictions on work for the other matrix norms.
func (impl Implementation) Dlange(norm lapack.MatrixNorm, m, n int, a []float64, lda int, work []float64) float64 {
	switch {
	case norm != lapack.MaxRowSum && norm != lapack.MaxColumnSum && norm != lapack.Frobenius && norm != lapack.MaxAbs:
		panic(badNorm)
	case lda < max(1, n):
		panic(badLdA)
	}

	// Quick return if possible.
	if m == 0 || n == 0 {
		return 0
	}

	switch {
	case len(a) < (m-1)*lda+n:
		panic(badLdA)
	case norm == lapack.MaxColumnSum && len(work) < n:
		panic(shortWork)
	}

	return lapacke.Dlange(byte(norm), m, n, a, lda, work)
}

// Dlansy computes the specified norm of an n×n symmetric matrix. If
// norm == lapack.MaxColumnSum or norm == lapackMaxRowSum work must have length
// at least n, otherwise work is unused.
func (impl Implementation) Dlansy(norm lapack.MatrixNorm, uplo blas.Uplo, n int, a []float64, lda int, work []float64) float64 {
	switch {
	case norm != lapack.MaxRowSum && norm != lapack.MaxColumnSum && norm != lapack.Frobenius && norm != lapack.MaxAbs:
		panic(badNorm)
	case uplo != blas.Upper && uplo != blas.Lower:
		panic(badUplo)
	case n < 0:
		panic(nLT0)
	case lda < max(1, n):
		panic(badLdA)
	}

	// Quick return if possible.
	if n == 0 {
		return 0
	}

	switch {
	case len(a) < (n-1)*lda+n:
		panic(shortA)
	case (norm == lapack.MaxColumnSum || norm == lapack.MaxRowSum) && len(work) < n:
		panic(shortWork)
	}

	return lapacke.Dlansy(byte(norm), byte(uplo), n, a, lda, work)
}

// Dlantr computes the specified norm of an m×n trapezoidal matrix A. If
// norm == lapack.MaxColumnSum work must have length at least n, otherwise work
// is unused.
func (impl Implementation) Dlantr(norm lapack.MatrixNorm, uplo blas.Uplo, diag blas.Diag, m, n int, a []float64, lda int, work []float64) float64 {
	switch {
	case norm != lapack.MaxRowSum && norm != lapack.MaxColumnSum && norm != lapack.Frobenius && norm != lapack.MaxAbs:
		panic(badNorm)
	case uplo != blas.Upper && uplo != blas.Lower:
		panic(badUplo)
	case diag != blas.Unit && diag != blas.NonUnit:
		panic(badDiag)
	case m < 0:
		panic(mLT0)
	case uplo == blas.Upper && m > n:
		panic(mGTN)
	case n < 0:
		panic(nLT0)
	case uplo == blas.Lower && n > m:
		panic(nGTM)
	case lda < max(1, n):
		panic(badLdA)
	}

	// Quick return if possible.
	minmn := min(m, n)
	if minmn == 0 {
		return 0
	}

	switch {
	case len(a) < (m-1)*lda+n:
		panic(shortA)
	case norm == lapack.MaxColumnSum && len(work) < n:
		panic(shortWork)
	}

	if norm == lapack.MaxRowSum && len(work) < m {
		// Allocate new work to be on the safe side because the expectation of LAPACKE on
		// row-major input is unclear.
		work = make([]float64, m)
	}
	return lapacke.Dlantr(byte(norm), byte(uplo), byte(diag), m, n, a, lda, work)
}

// Dlarfx applies an elementary reflector H to a real m×n matrix C, from either
// the left or the right, with loop unrolling when the reflector has order less
// than 11.
//
// H is represented in the form
//
//	H = I - tau * v * v^T,
//
// where tau is a real scalar and v is a real vector. If tau = 0, then H is
// taken to be the identity matrix.
//
// v must have length equal to m if side == blas.Left, and equal to n if side ==
// blas.Right, otherwise Dlarfx will panic.
//
// c and ldc represent the m×n matrix C. On return, C is overwritten by the
// matrix H * C if side == blas.Left, or C * H if side == blas.Right.
//
// work must have length at least n if side == blas.Left, and at least m if side
// == blas.Right, otherwise Dlarfx will panic. work is not referenced if H has
// order < 11.
func (impl Implementation) Dlarfx(side blas.Side, m, n int, v []float64, tau float64, c []float64, ldc int, work []float64) {
	switch {
	case side != blas.Left && side != blas.Right:
		panic(badSide)
	case m < 0:
		panic(mLT0)
	case n < 0:
		panic(nLT0)
	case ldc < max(1, n):
		panic(badLdC)
	}

	// Quick return if possible.
	if m == 0 || n == 0 {
		return
	}

	nh := m
	lwork := n
	if side == blas.Right {
		nh = n
		lwork = m
	}
	switch {
	case len(v) < nh:
		panic(shortV)
	case len(c) < (m-1)*ldc+n:
		panic(shortC)
	case nh > 10 && len(work) < lwork:
		panic(shortWork)
	}

	lapacke.Dlarfx(byte(side), m, n, v, tau, c, ldc, work)
}

// Dlascl multiplies an m×n matrix by the scalar cto/cfrom.
//
// cfrom must not be zero, and cto and cfrom must not be NaN, otherwise Dlascl
// will panic.
//
// Dlascl is an internal routine. It is exported for testing purposes.
func (impl Implementation) Dlascl(kind lapack.MatrixType, kl, ku int, cfrom, cto float64, m, n int, a []float64, lda int) {
	switch kind {
	default:
		panic(badMatrixType)
	case 'H', 'B', 'Q', 'Z': // See dlascl.f.
	case lapack.General, lapack.UpperTri, lapack.LowerTri:
		if lda < max(1, n) {
			panic(badLdA)
		}
	}
	switch {
	case cfrom == 0:
		panic(zeroCFrom)
	case math.IsNaN(cfrom):
		panic(nanCFrom)
	case math.IsNaN(cto):
		panic(nanCTo)
	case m < 0:
		panic(mLT0)
	case n < 0:
		panic(nLT0)
	}

	if n == 0 || m == 0 {
		return
	}

	switch kind {
	case lapack.General, lapack.UpperTri, lapack.LowerTri:
		if len(a) < (m-1)*lda+n {
			panic(shortA)
		}
	}

	lapacke.Dlascl(byte(kind), kl, ku, cfrom, cto, m, n, a, lda)
}

// Dlaset sets the off-diagonal elements of A to alpha, and the diagonal
// elements to beta. If uplo == blas.Upper, only the elements in the upper
// triangular part are set. If uplo == blas.Lower, only the elements in the
// lower triangular part are set. If uplo is otherwise, all of the elements of A
// are set.
//
// Dlaset is an internal routine. It is exported for testing purposes.
func (impl Implementation) Dlaset(uplo blas.Uplo, m, n int, alpha, beta float64, a []float64, lda int) {
	switch {
	case m < 0:
		panic(mLT0)
	case n < 0:
		panic(nLT0)
	case lda < max(1, n):
		panic(badLdA)
	}

	minmn := min(m, n)
	if minmn == 0 {
		return
	}

	if len(a) < (m-1)*lda+n {
		panic(shortA)
	}

	lapacke.Dlaset(byte(uplo), m, n, alpha, beta, a, lda)
}

// Dlasrt sorts the numbers in the input slice d. If s == lapack.SortIncreasing,
// the elements are sorted in increasing order. If s == lapack.SortDecreasing,
// the elements are sorted in decreasing order. For other values of s Dlasrt
// will panic.
//
// Dlasrt is an internal routine. It is exported for testing purposes.
func (impl Implementation) Dlasrt(s lapack.Sort, n int, d []float64) {
	switch {
	case s != lapack.SortIncreasing && s != lapack.SortDecreasing:
		panic(badSort)
	case n < 0:
		panic(nLT0)
	case len(d) < n:
		panic(shortD)
	}

	lapacke.Dlasrt(byte(s), n, d[:n])
}

// Dlaswp swaps the rows k1 to k2 of a rectangular matrix A according to the
// indices in ipiv so that row k is swapped with ipiv[k].
//
// n is the number of columns of A and incX is the increment for ipiv. If incX
// is 1, the swaps are applied from k1 to k2. If incX is -1, the swaps are
// applied in reverse order from k2 to k1. For other values of incX Dlaswp will
// panic. ipiv must have length k2+1, otherwise Dlaswp will panic.
//
// The indices k1, k2, and the elements of ipiv are zero-based.
//
// Dlaswp is an internal routine. It is exported for testing purposes.
func (impl Implementation) Dlaswp(n int, a []float64, lda, k1, k2 int, ipiv []int, incX int) {
	switch {
	case n < 0:
		panic(nLT0)
	case k2 < 0:
		panic(badK2)
	case k1 < 0 || k2 < k1:
		panic(badK1)
	case lda < max(1, n):
		panic(badLdA)
	case len(a) < (k2-1)*lda+n:
		panic(shortA)
	case len(ipiv) != k2+1:
		panic(badLenIpiv)
	case incX != 1 && incX != -1:
		panic(absIncNotOne)
	}

	if n == 0 {
		return
	}

	ipiv32 := make([]int32, k2+1)
	for i, v := range ipiv {
		v++
		if v != int(int32(v)) {
			panic("lapack: ipiv element out of range")
		}
		ipiv32[i] = int32(v)
	}
	lapacke.Dlaswp(n, a, lda, k1+1, k2+1, ipiv32, incX)
}

// Dpbcon returns an estimate of the reciprocal of the condition number (in the
// 1-norm) of an n×n symmetric positive definite band matrix using the Cholesky
// factorization
//
//	A = Uᵀ*U  if uplo == blas.Upper
//	A = L*Lᵀ  if uplo == blas.Lower
//
// computed by Dpbtrf. The estimate is obtained for norm(inv(A)), and the
// reciprocal of the condition number is computed as
//
//	rcond = 1 / (anorm * norm(inv(A))).
//
// The length of work must be at least 3*n and the length of iwork must be at
// least n.
func (impl Implementation) Dpbcon(uplo blas.Uplo, n, kd int, ab []float64, ldab int, anorm float64, work []float64, iwork []int) (rcond float64) {
	switch {
	case uplo != blas.Upper && uplo != blas.Lower:
		panic(badUplo)
	case n < 0:
		panic(nLT0)
	case kd < 0:
		panic(kdLT0)
	case ldab < kd+1:
		panic(badLdA)
	case anorm < 0:
		panic(badNorm)
	}

	// Quick return if possible.
	if n == 0 {
		return 1
	}

	switch {
	case len(ab) < (n-1)*ldab+kd+1:
		panic(shortAB)
	case len(work) < 3*n:
		panic(shortWork)
	case len(iwork) < n:
		panic(shortIWork)
	}

	_ldab := n
	_ab := make([]float64, (kd+1)*_ldab)
	bandTriToLapacke(uplo, n, kd, ab, ldab, _ab, _ldab)
	_rcond := []float64{0}
	_iwork := make([]int32, n)
	lapacke.Dpbcon(byte(uplo), n, kd, _ab, _ldab, anorm, _rcond, work, _iwork)
	return _rcond[0]
}

// Dpbtrf computes the Cholesky factorization of an n×n symmetric positive
// definite band matrix
//
//	A = U^T * U  if uplo == blas.Upper
//	A = L * L^T  if uplo == blas.Lower
//
// where U is an upper triangular band matrix and L is lower triangular. kd is
// the number of super- or sub-diagonals of A.
//
// The band storage scheme is illustrated below when n = 6 and kd = 2. Elements
// marked * are not used by the function.
//
//	uplo == blas.Upper
//	On entry:         On return:
//	 a00  a01  a02     u00  u01  u02
//	 a11  a12  a13     u11  u12  u13
//	 a22  a23  a24     u22  u23  u24
//	 a33  a34  a35     u33  u34  u35
//	 a44  a45   *      u44  u45   *
//	 a55   *    *      u55   *    *
//
//	uplo == blas.Lower
//	On entry:         On return:
//	  *    *   a00       *    *   l00
//	  *   a10  a11       *   l10  l11
//	 a20  a21  a22      l20  l21  l22
//	 a31  a32  a33      l31  l32  l33
//	 a42  a43  a44      l42  l43  l44
//	 a53  a54  a55      l53  l54  l55
func (impl Implementation) Dpbtrf(uplo blas.Uplo, n, kd int, ab []float64, ldab int) (ok bool) {
	switch {
	case uplo != blas.Upper && uplo != blas.Lower:
		panic(badUplo)
	case n < 0:
		panic(nLT0)
	case kd < 0:
		panic(kdLT0)
	case ldab < kd+1:
		panic(badLdA)
	}

	// Quick return if possible.
	if n == 0 {
		return true
	}

	if len(ab) < (n-1)*ldab+kd+1 {
		panic(shortAB)
	}

	ldabConv := n
	abConv := make([]float64, (kd+1)*ldabConv)
	bandTriToLapacke(uplo, n, kd, ab, ldab, abConv, ldabConv)
	info := lapacke.Dpbtrf(byte(uplo), n, kd, abConv, ldabConv)
	bandTriToGonum(uplo, n, kd, abConv, ldabConv, ab, ldab)
	return info
}

// Dpbtrs solves a system of linear equations A*X = B with an n×n symmetric
// positive definite band matrix A using the Cholesky factorization
//
//	A = U^T * U  if uplo == blas.Upper
//	A = L * L^T  if uplo == blas.Lower
//
// computed by Dpbtrf. kd is the number of super- or sub-diagonals of A. See the
// documentation for Dpbtrf for a description of the band storage format of A.
//
// On entry, b contains the n×nrhs right hand side matrix B. On return, it is
// overwritten with the solution matrix X.
func (Implementation) Dpbtrs(uplo blas.Uplo, n, kd, nrhs int, ab []float64, ldab int, b []float64, ldb int) {
	switch {
	case uplo != blas.Upper && uplo != blas.Lower:
		panic(badUplo)
	case n < 0:
		panic(nLT0)
	case kd < 0:
		panic(kdLT0)
	case nrhs < 0:
		panic(nrhsLT0)
	case ldab < kd+1:
		panic(badLdA)
	case ldb < max(1, nrhs):
		panic(badLdB)
	}

	// Quick return if possible.
	if n == 0 || nrhs == 0 {
		return
	}

	if len(ab) < (n-1)*ldab+kd {
		panic(shortAB)
	}
	if len(b) < (n-1)*ldb+nrhs {
		panic(shortB)
	}

	ldabConv := n
	abConv := make([]float64, (kd+1)*ldabConv)
	bandTriToLapacke(uplo, n, kd, ab, ldab, abConv, ldabConv)
	lapacke.Dpbtrs(byte(uplo), n, kd, nrhs, abConv, ldabConv, b, ldb)
}

// Dpotrf computes the Cholesky decomposition of the symmetric positive definite
// matrix a. If ul == blas.Upper, then a is stored as an upper-triangular matrix,
// and a = U U^T is stored in place into a. If ul == blas.Lower, then a = L L^T
// is computed and stored in-place into a. If a is not positive definite, false
// is returned. This is the blocked version of the algorithm.
func (impl Implementation) Dpotrf(ul blas.Uplo, n int, a []float64, lda int) (ok bool) {
	switch {
	case ul != blas.Upper && ul != blas.Lower:
		panic(badUplo)
	case n < 0:
		panic(nLT0)
	case lda < max(1, n):
		panic(badLdA)
	}

	// Quick return if possible.
	if n == 0 {
		return true
	}

	if len(a) < (n-1)*lda+n {
		panic(shortA)
	}

	return lapacke.Dpotrf(byte(ul), n, a, lda)
}

// Dpotri computes the inverse of a real symmetric positive definite matrix A
// using its Cholesky factorization.
//
// On entry, a contains the triangular factor U or L from the Cholesky
// factorization A = U^T*U or A = L*L^T, as computed by Dpotrf.
// On return, a contains the upper or lower triangle of the (symmetric)
// inverse of A, overwriting the input factor U or L.
func (impl Implementation) Dpotri(uplo blas.Uplo, n int, a []float64, lda int) (ok bool) {
	switch {
	case uplo != blas.Upper && uplo != blas.Lower:
		panic(badUplo)
	case n < 0:
		panic(nLT0)
	case lda < max(1, n):
		panic(badLdA)
	}

	// Quick return if possible.
	if n == 0 {
		return true
	}

	if len(a) < (n-1)*lda+n {
		panic(shortA)
	}

	return lapacke.Dpotri(byte(uplo), n, a, lda)
}

// Dpotrs solves a system of n linear equations A*X = B where A is an n×n
// symmetric positive definite matrix and B is an n×nrhs matrix. The matrix A is
// represented by its Cholesky factorization
//
//	A = U^T*U  if uplo == blas.Upper
//	A = L*L^T  if uplo == blas.Lower
//
// as computed by Dpotrf. On entry, B contains the right-hand side matrix B, on
// return it contains the solution matrix X.
func (Implementation) Dpotrs(uplo blas.Uplo, n, nrhs int, a []float64, lda int, b []float64, ldb int) {
	switch {
	case uplo != blas.Upper && uplo != blas.Lower:
		panic(badUplo)
	case n < 0:
		panic(nLT0)
	case nrhs < 0:
		panic(nrhsLT0)
	case lda < max(1, n):
		panic(badLdA)
	case ldb < max(1, nrhs):
		panic(badLdB)
	}

	// Quick return if possible.
	if n == 0 || nrhs == 0 {
		return
	}

	switch {
	case len(a) < (n-1)*lda+n:
		panic(shortA)
	case len(b) < (n-1)*ldb+nrhs:
		panic(shortB)
	}

	lapacke.Dpotrs(byte(uplo), n, nrhs, a, lda, b, ldb)
}

// Dpstrf computes the Cholesky factorization with complete pivoting of an n×n
// symmetric positive semidefinite matrix A.
//
// The factorization has the form
//
//	Pᵀ * A * P = Uᵀ * U ,  if uplo = blas.Upper,
//	Pᵀ * A * P = L  * Lᵀ,  if uplo = blas.Lower,
//
// where U is an upper triangular matrix, L is lower triangular, and P is a
// permutation matrix.
//
// tol is a user-defined tolerance. The algorithm terminates if the pivot is
// less than or equal to tol. If tol is negative, then n*eps*max(A[k,k]) will be
// used instead.
//
// On return, A contains the factor U or L from the Cholesky factorization and
// piv contains P stored such that P[piv[k],k] = 1.
//
// Dpstrf returns the computed rank of A and whether the factorization can be
// used to solve a system. Dpstrf does not attempt to check that A is positive
// semi-definite, so if ok is false, the matrix A is either rank deficient or is
// not positive semidefinite.
//
// The length of piv must be n and the length of work must be at least 2*n,
// otherwise Dpstrf will panic.
//
// Dpstrf is an internal routine. It is exported for testing purposes.
func (impl Implementation) Dpstrf(uplo blas.Uplo, n int, a []float64, lda int, piv []int, tol float64, work []float64) (rank int, ok bool) {
	switch {
	case uplo != blas.Upper && uplo != blas.Lower:
		panic(badUplo)
	case n < 0:
		panic(nLT0)
	case lda < max(1, n):
		panic(badLdA)
	}

	// Quick return if possible.
	if n == 0 {
		return 0, true
	}

	switch {
	case len(a) < (n-1)*lda+n:
		panic(shortA)
	case len(piv) != n:
		panic(badLenPiv)
	case len(work) < 2*n:
		panic(shortWork)
	}

	piv32 := make([]int32, n)
	rank32 := make([]int32, 1)
	ok = lapacke.Dpstrf(byte(uplo), n, a, lda, piv32, rank32, tol, work)
	for i, v := range piv32 {
		piv[i] = int(v) - 1 // Transform to zero-based indices.
	}
	return int(rank32[0]), ok
}

// Dgebal balances an n×n matrix A. Balancing consists of two stages, permuting
// and scaling. Both steps are optional and depend on the value of job.
//
// Permuting consists of applying a permutation matrix P such that the matrix
// that results from P^T*A*P takes the upper block triangular form
//
//	          [ T1  X  Y  ]
//	P^T A P = [  0  B  Z  ],
//	          [  0  0  T2 ]
//
// where T1 and T2 are upper triangular matrices and B contains at least one
// nonzero off-diagonal element in each row and column. The indices ilo and ihi
// mark the starting and ending columns of the submatrix B. The eigenvalues of A
// isolated in the first 0 to ilo-1 and last ihi+1 to n-1 elements on the
// diagonal can be read off without any roundoff error.
//
// Scaling consists of applying a diagonal similarity transformation D such that
// D^{-1}*B*D has the 1-norm of each row and its corresponding column nearly
// equal. The output matrix is
//
//	[ T1     X*D          Y    ]
//	[  0  inv(D)*B*D  inv(D)*Z ].
//	[  0      0           T2   ]
//
// Scaling may reduce the 1-norm of the matrix, and improve the accuracy of
// the computed eigenvalues and/or eigenvectors.
//
// job specifies the operations that will be performed on A.
// If job is lapack.None, Dgebal sets scale[i] = 1 for all i and returns ilo=0, ihi=n-1.
// If job is lapack.Permute, only permuting will be done.
// If job is lapack.Scale, only scaling will be done.
// If job is lapack.PermuteScale, both permuting and scaling will be done.
//
// On return, if job is lapack.Permute or lapack.PermuteScale, it will hold that
//
//	A[i,j] == 0,   for i > j and j ∈ {0, ..., ilo-1, ihi+1, ..., n-1}.
//
// If job is lapack.None or lapack.Scale, or if n == 0, it will hold that
//
//	ilo == 0 and ihi == n-1.
//
// On return, scale will contain information about the permutations and scaling
// factors applied to A. If π(j) denotes the index of the column interchanged
// with column j, and D[j,j] denotes the scaling factor applied to column j,
// then
//
//	scale[j] == π(j),     for j ∈ {0, ..., ilo-1, ihi+1, ..., n-1},
//	         == D[j,j],   for j ∈ {ilo, ..., ihi}.
//
// scale must have length equal to n, otherwise Dgebal will panic.
//
// Dgebal is an internal routine. It is exported for testing purposes.
func (impl Implementation) Dgebal(job lapack.BalanceJob, n int, a []float64, lda int, scale []float64) (ilo, ihi int) {
	switch {
	case job != lapack.BalanceNone && job != lapack.Permute && job != lapack.Scale && job != lapack.PermuteScale:
		panic(badBalanceJob)
	case n < 0:
		panic(nLT0)
	case lda < max(1, n):
		panic(badLdA)
	}

	ilo = 0
	ihi = n - 1

	if n == 0 {
		return ilo, ihi
	}

	switch {
	case len(scale) != n:
		panic(shortScale)
	case len(a) < (n-1)*lda+n:
		panic(shortA)
	}

	ilo32 := []int32{0}
	ihi32 := []int32{0}
	lapacke.Dgebal(byte(job), n, a, lda, ilo32, ihi32, scale)
	ilo = int(ilo32[0]) - 1
	ihi = int(ihi32[0]) - 1
	for j := 0; j < ilo; j++ {
		scale[j]--
	}
	for j := ihi + 1; j < n; j++ {
		scale[j]--
	}
	return ilo, ihi
}

// Dgebak transforms an n×m matrix V as
//
//	V = P D V,        if side == blas.Right,
//	V = P D^{-1} V,   if side == blas.Left,
//
// where P and D are n×n permutation and scaling matrices, respectively,
// implicitly represented by job, scale, ilo and ihi as returned by Dgebal.
//
// Typically, columns of the matrix V contain the right or left (determined by
// side) eigenvectors of the balanced matrix output by Dgebal, and Dgebak forms
// the eigenvectors of the original matrix.
//
// Dgebak is an internal routine. It is exported for testing purposes.
func (impl Implementation) Dgebak(job lapack.BalanceJob, side lapack.EVSide, n, ilo, ihi int, scale []float64, m int, v []float64, ldv int) {
	switch {
	case job != lapack.BalanceNone && job != lapack.Permute && job != lapack.Scale && job != lapack.PermuteScale:
		panic(badBalanceJob)
	case side != lapack.EVLeft && side != lapack.EVRight:
		panic(badEVSide)
	case n < 0:
		panic(nLT0)
	case ilo < 0 || max(0, n-1) < ilo:
		panic(badIlo)
	case ihi < min(ilo, n-1) || n <= ihi:
		panic(badIhi)
	case m < 0:
		panic(mLT0)
	case ldv < max(1, m):
		panic(badLdV)
	}

	// Quick return if possible.
	if n == 0 || m == 0 {
		return
	}

	switch {
	case len(scale) < n:
		panic(shortScale)
	case len(v) < (n-1)*ldv+m:
		panic(shortV)
	}

	// Quick return if possible.
	if job == lapack.BalanceNone {
		return
	}

	// Convert permutation indices to 1-based.
	for j := 0; j < ilo; j++ {
		scale[j]++
	}
	for j := ihi + 1; j < n; j++ {
		scale[j]++
	}
	lapacke.Dgebak(byte(job), byte(side), n, ilo+1, ihi+1, scale, m, v, ldv)
	// Convert permutation indices back to 0-based.
	for j := 0; j < ilo; j++ {
		scale[j]--
	}
	for j := ihi + 1; j < n; j++ {
		scale[j]--
	}
}

// Dbdsqr performs a singular value decomposition of a real n×n bidiagonal matrix.
//
// The SVD of the bidiagonal matrix B is
//
//	B = Q * S * P^T
//
// where S is a diagonal matrix of singular values, Q is an orthogonal matrix of
// left singular vectors, and P is an orthogonal matrix of right singular vectors.
//
// Q and P are only computed if requested. If left singular vectors are requested,
// this routine returns U * Q instead of Q, and if right singular vectors are
// requested P^T * VT is returned instead of P^T.
//
// Frequently Dbdsqr is used in conjunction with Dgebrd which reduces a general
// matrix A into bidiagonal form. In this case, the SVD of A is
//
//	A = (U * Q) * S * (P^T * VT)
//
// This routine may also compute Q^T * C.
//
// d and e contain the elements of the bidiagonal matrix b. d must have length at
// least n, and e must have length at least n-1. Dbdsqr will panic if there is
// insufficient length. On exit, D contains the singular values of B in decreasing
// order.
//
// VT is a matrix of size n×ncvt whose elements are stored in vt. The elements
// of vt are modified to contain P^T * VT on exit. VT is not used if ncvt == 0.
//
// U is a matrix of size nru×n whose elements are stored in u. The elements
// of u are modified to contain U * Q on exit. U is not used if nru == 0.
//
// C is a matrix of size n×ncc whose elements are stored in c. The elements
// of c are modified to contain Q^T * C on exit. C is not used if ncc == 0.
//
// work contains temporary storage and must have length at least 4*(n-1). Dbdsqr
// will panic if there is insufficient working memory.
//
// Dbdsqr returns whether the decomposition was successful.
func (impl Implementation) Dbdsqr(uplo blas.Uplo, n, ncvt, nru, ncc int, d, e, vt []float64, ldvt int, u []float64, ldu int, c []float64, ldc int, work []float64) (ok bool) {
	switch {
	case uplo != blas.Upper && uplo != blas.Lower:
		panic(badUplo)
	case n < 0:
		panic(nLT0)
	case ncvt < 0:
		panic(ncvtLT0)
	case nru < 0:
		panic(nruLT0)
	case ncc < 0:
		panic(nccLT0)
	case ldvt < max(1, ncvt):
		panic(badLdVT)
	case (ldu < max(1, n) && nru > 0) || (ldu < 1 && nru == 0):
		panic(badLdU)
	case ldc < max(1, ncc):
		panic(badLdC)
	}

	// Quick return if possible.
	if n == 0 {
		return true
	}

	if len(vt) < (n-1)*ldvt+ncvt && ncvt != 0 {
		panic(shortVT)
	}
	if len(u) < (nru-1)*ldu+n && nru != 0 {
		panic(shortU)
	}
	if len(c) < (n-1)*ldc+ncc && ncc != 0 {
		panic(shortC)
	}
	if len(d) < n {
		panic(shortD)
	}
	if len(e) < n-1 {
		panic(shortE)
	}
	if len(work) < 4*(n-1) {
		panic(shortWork)
	}

	return lapacke.Dbdsqr(byte(uplo), n, ncvt, nru, ncc, d, e, vt, ldvt, u, ldu, c, ldc, work)
}

// Dgebrd reduces a general m×n matrix A to upper or lower bidiagonal form B by
// an orthogonal transformation:
//
//	Q^T * A * P = B.
//
// The diagonal elements of B are stored in d and the off-diagonal elements are
// stored in e. These are additionally stored along the diagonal of A and the
// off-diagonal of A. If m >= n B is an upper-bidiagonal matrix, and if m < n B
// is a lower-bidiagonal matrix.
//
// The remaining elements of A store the data needed to construct Q and P.
// The matrices Q and P are products of elementary reflectors
//
//	if m >= n, Q = H_0 * H_1 * ... * H_{n-1},
//	           P = G_0 * G_1 * ... * G_{n-2},
//	if m < n,  Q = H_0 * H_1 * ... * H_{m-2},
//	           P = G_0 * G_1 * ... * G_{m-1},
//
// where
//
//	H_i = I - tauQ[i] * v_i * v_i^T,
//	G_i = I - tauP[i] * u_i * u_i^T.
//
// As an example, on exit the entries of A when m = 6, and n = 5
//
//	[ d   e  u1  u1  u1]
//	[v1   d   e  u2  u2]
//	[v1  v2   d   e  u3]
//	[v1  v2  v3   d   e]
//	[v1  v2  v3  v4   d]
//	[v1  v2  v3  v4  v5]
//
// and when m = 5, n = 6
//
//	[ d  u1  u1  u1  u1  u1]
//	[ e   d  u2  u2  u2  u2]
//	[v1   e   d  u3  u3  u3]
//	[v1  v2   e   d  u4  u4]
//	[v1  v2  v3   e   d  u5]
//
// d, tauQ, and tauP must all have length at least min(m,n), and e must have
// length min(m,n) - 1, unless lwork is -1 when there is no check except for
// work which must have a length of at least one.
//
// work is temporary storage, and lwork specifies the usable memory length.
// At minimum, lwork >= max(1,m,n) or be -1 and this function will panic otherwise.
// Dgebrd is blocked decomposition, but the block size is limited
// by the temporary space available. If lwork == -1, instead of performing Dgebrd,
// the optimal work length will be stored into work[0].
//
// Dgebrd is an internal routine. It is exported for testing purposes.
func (impl Implementation) Dgebrd(m, n int, a []float64, lda int, d, e, tauQ, tauP, work []float64, lwork int) {
	switch {
	case m < 0:
		panic(mLT0)
	case n < 0:
		panic(nLT0)
	case lda < max(1, n):
		panic(badLdA)
	case lwork < max(1, max(m, n)) && lwork != -1:
		panic(badLWork)
	case len(work) < max(1, lwork):
		panic(shortWork)
	}

	// Quick return if possible.
	minmn := min(m, n)
	if minmn == 0 {
		work[0] = 1
		return
	}

	if lwork == -1 {
		lapacke.Dgebrd(m, n, a, lda, d, e, tauQ, tauP, work, -1)
		return
	}

	switch {
	case len(a) < (m-1)*lda+n:
		panic(shortA)
	case len(d) < minmn:
		panic(shortD)
	case len(e) < minmn-1:
		panic(shortE)
	case len(tauQ) < minmn:
		panic(shortTauQ)
	case len(tauP) < minmn:
		panic(shortTauP)
	}

	lapacke.Dgebrd(m, n, a, lda, d, e, tauQ, tauP, work, lwork)
}

// Dgecon estimates the reciprocal of the condition number of the n×n matrix A
// given the LU decomposition of the matrix. The condition number computed may
// be based on the 1-norm or the ∞-norm.
//
// The slice a contains the result of the LU decomposition of A as computed by Dgetrf.
//
// anorm is the corresponding 1-norm or ∞-norm of the original matrix A.
//
// work is a temporary data slice of length at least 4*n and Dgecon will panic otherwise.
//
// iwork is a temporary data slice of length at least n and Dgecon will panic otherwise.
func (impl Implementation) Dgecon(norm lapack.MatrixNorm, n int, a []float64, lda int, anorm float64, work []float64, iwork []int) float64 {
	switch {
	case norm != lapack.MaxColumnSum && norm != lapack.MaxRowSum:
		panic(badNorm)
	case n < 0:
		panic(nLT0)
	case lda < max(1, n):
		panic(badLdA)
	}

	// Quick return if possible.
	if n == 0 {
		return 1
	}

	switch {
	case len(a) < (n-1)*lda+n:
		panic(shortA)
	case len(work) < 4*n:
		panic(shortWork)
	case len(iwork) < n:
		panic(shortIWork)
	}

	rcond := []float64{0}
	_iwork := make([]int32, n)
	lapacke.Dgecon(byte(norm), n, a, lda, anorm, rcond, work, _iwork)
	return rcond[0]
}

// Dgelq2 computes the LQ factorization of the m×n matrix A.
//
// In an LQ factorization, L is a lower triangular m×n matrix, and Q is an n×n
// orthonormal matrix.
//
// a is modified to contain the information to construct L and Q.
// The lower triangle of a contains the matrix L. The upper triangular elements
// (not including the diagonal) contain the elementary reflectors. Tau is modified
// to contain the reflector scales. tau must have length of at least k = min(m,n)
// and this function will panic otherwise.
//
// See Dgeqr2 for a description of the elementary reflectors and orthonormal
// matrix Q. Q is constructed as a product of these elementary reflectors,
//
//	Q = H_{k-1} * ... * H_1 * H_0,
//
// where k = min(m,n).
//
// Work is temporary storage of length at least m and this function will panic otherwise.
func (impl Implementation) Dgelq2(m, n int, a []float64, lda int, tau, work []float64) {
	switch {
	case m < 0:
		panic(mLT0)
	case n < 0:
		panic(nLT0)
	case lda < max(1, n):
		panic(badLdA)
	}

	// Quick return if possible.
	k := min(m, n)
	if k == 0 {
		return
	}

	switch {
	case len(a) < (m-1)*lda+n:
		panic(shortA)
	case len(tau) < k:
		panic(shortTau)
	case len(work) < m:
		panic(shortWork)
	}

	lapacke.Dgelq2(m, n, a, lda, tau, work)
}

// Dgelqf computes the LQ factorization of the m×n matrix A using a blocked
// algorithm. See the documentation for Dgelq2 for a description of the
// parameters at entry and exit.
//
// work is temporary storage, and lwork specifies the usable memory length.
// At minimum, lwork >= m, and this function will panic otherwise.
// Dgelqf is a blocked LQ factorization, but the block size is limited
// by the temporary space available. If lwork == -1, instead of performing Dgelqf,
// the optimal work length will be stored into work[0].
//
// tau must have length at least min(m,n), and this function will panic otherwise.
func (impl Implementation) Dgelqf(m, n int, a []float64, lda int, tau, work []float64, lwork int) {
	switch {
	case m < 0:
		panic(mLT0)
	case n < 0:
		panic(nLT0)
	case lda < max(1, n):
		panic(badLdA)
	case lwork < max(1, m) && lwork != -1:
		panic(badLWork)
	case len(work) < max(1, lwork):
		panic(shortWork)
	}

	k := min(m, n)
	if k == 0 {
		work[0] = 1
		return
	}

	if lwork == -1 {
		lapacke.Dgelqf(m, n, a, lda, tau, work, -1)
		return
	}

	switch {
	case len(a) < (m-1)*lda+n:
		panic(shortA)
	case len(tau) < k:
		panic(shortTau)
	}

	lapacke.Dgelqf(m, n, a, lda, tau, work, lwork)
}

// Dgeqr2 computes a QR factorization of the m×n matrix A.
//
// In a QR factorization, Q is an m×m orthonormal matrix, and R is an
// upper triangular m×n matrix.
//
// A is modified to contain the information to construct Q and R.
// The upper triangle of a contains the matrix R. The lower triangular elements
// (not including the diagonal) contain the elementary reflectors. Tau is modified
// to contain the reflector scales. tau must have length at least min(m,n), and
// this function will panic otherwise.
//
// The ith elementary reflector can be explicitly constructed by first extracting
// the
//
//	v[j] = 0           j < i
//	v[j] = 1           j == i
//	v[j] = a[j*lda+i]  j > i
//
// and computing H_i = I - tau[i] * v * v^T.
//
// The orthonormal matrix Q can be constucted from a product of these elementary
// reflectors, Q = H_0 * H_1 * ... * H_{k-1}, where k = min(m,n).
//
// Work is temporary storage of length at least n and this function will panic otherwise.
func (impl Implementation) Dgeqr2(m, n int, a []float64, lda int, tau, work []float64) {
	switch {
	case m < 0:
		panic(mLT0)
	case n < 0:
		panic(nLT0)
	case lda < max(1, n):
		panic(badLdA)
	case len(work) < n:
		panic(shortWork)
	}

	// Quick return if possible.
	k := min(m, n)
	if k == 0 {
		return
	}

	switch {
	case len(a) < (m-1)*lda+n:
		panic(shortA)
	case len(tau) < k:
		panic(shortTau)
	}

	lapacke.Dgeqr2(m, n, a, lda, tau, work)
}

// Dgeqrf computes the QR factorization of the m×n matrix A using a blocked
// algorithm. See the documentation for Dgeqr2 for a description of the
// parameters at entry and exit.
//
// work is temporary storage, and lwork specifies the usable memory length.
// The length of work must be at least max(1, lwork) and lwork must be -1
// or at least n, otherwise this function will panic.
// Dgeqrf is a blocked QR factorization, but the block size is limited
// by the temporary space available. If lwork == -1, instead of performing Dgeqrf,
// the optimal work length will be stored into work[0].
//
// tau must have length at least min(m,n), and this function will panic otherwise.
func (impl Implementation) Dgeqrf(m, n int, a []float64, lda int, tau, work []float64, lwork int) {
	switch {
	case m < 0:
		panic(mLT0)
	case n < 0:
		panic(nLT0)
	case lda < max(1, n):
		panic(badLdA)
	case lwork < max(1, n) && lwork != -1:
		panic(badLWork)
	case len(work) < max(1, lwork):
		panic(shortWork)
	}

	// Quick return if possible.
	k := min(m, n)
	if k == 0 {
		work[0] = 1
		return
	}

	if lwork == -1 {
		lapacke.Dgeqrf(m, n, a, lda, tau, work, -1)
		return
	}

	switch {
	case len(a) < (m-1)*lda+n:
		panic(shortA)
	case len(tau) < k:
		panic(shortTau)
	}

	lapacke.Dgeqrf(m, n, a, lda, tau, work, lwork)
}

// Dgehrd reduces a block of a real n×n general matrix A to upper Hessenberg
// form H by an orthogonal similarity transformation Q^T * A * Q = H.
//
// The matrix Q is represented as a product of (ihi-ilo) elementary
// reflectors
//
//	Q = H_{ilo} H_{ilo+1} ... H_{ihi-1}.
//
// Each H_i has the form
//
//	H_i = I - tau[i] * v * v^T
//
// where v is a real vector with v[0:i+1] = 0, v[i+1] = 1 and v[ihi+1:n] = 0.
// v[i+2:ihi+1] is stored on exit in A[i+2:ihi+1,i].
//
// On entry, a contains the n×n general matrix to be reduced. On return, the
// upper triangle and the first subdiagonal of A will be overwritten with the
// upper Hessenberg matrix H, and the elements below the first subdiagonal, with
// the slice tau, represent the orthogonal matrix Q as a product of elementary
// reflectors.
//
// The contents of a are illustrated by the following example, with n = 7, ilo =
// 1 and ihi = 5.
// On entry,
//
//	[ a   a   a   a   a   a   a ]
//	[     a   a   a   a   a   a ]
//	[     a   a   a   a   a   a ]
//	[     a   a   a   a   a   a ]
//	[     a   a   a   a   a   a ]
//	[     a   a   a   a   a   a ]
//	[                         a ]
//
// on return,
//
//	[ a   a   h   h   h   h   a ]
//	[     a   h   h   h   h   a ]
//	[     h   h   h   h   h   h ]
//	[     v1  h   h   h   h   h ]
//	[     v1  v2  h   h   h   h ]
//	[     v1  v2  v3  h   h   h ]
//	[                         a ]
//
// where a denotes an element of the original matrix A, h denotes a
// modified element of the upper Hessenberg matrix H, and vi denotes an
// element of the vector defining H_i.
//
// ilo and ihi determine the block of A that will be reduced to upper Hessenberg
// form. It must hold that 0 <= ilo <= ihi < n if n > 0, and ilo == 0 and ihi ==
// -1 if n == 0, otherwise Dgehrd will panic.
//
// On return, tau will contain the scalar factors of the elementary reflectors.
// Elements tau[:ilo] and tau[ihi:] will be set to zero. tau must have length
// equal to n-1 if n > 0, otherwise Dgehrd will panic.
//
// work must have length at least lwork and lwork must be at least max(1,n),
// otherwise Dgehrd will panic. On return, work[0] contains the optimal value of
// lwork.
//
// If lwork == -1, instead of performing Dgehrd, only the optimal value of lwork
// will be stored in work[0].
//
// Dgehrd is an internal routine. It is exported for testing purposes.
func (impl Implementation) Dgehrd(n, ilo, ihi int, a []float64, lda int, tau, work []float64, lwork int) {
	switch {
	case n < 0:
		panic(nLT0)
	case ilo < 0 || max(0, n-1) < ilo:
		panic(badIlo)
	case ihi < min(ilo, n-1) || n <= ihi:
		panic(badIhi)
	case lda < max(1, n):
		panic(badLdA)
	case lwork < max(1, n) && lwork != -1:
		panic(badLWork)
	case len(work) < lwork:
		panic(shortWork)
	}

	// Quick return if possible.
	if n == 0 {
		work[0] = 1
		return
	}

	if lwork == -1 {
		lapacke.Dgehrd(n, ilo+1, ihi+1, a, lda, tau, work, -1)
		return
	}

	switch {
	case len(a) < (n-1)*lda+n:
		panic(shortA)
	case len(tau) != n-1:
		panic(badLenTau)
	}

	lapacke.Dgehrd(n, ilo+1, ihi+1, a, lda, tau, work, lwork)
}

// Dgels finds a minimum-norm solution based on the matrices A and B using the
// QR or LQ factorization. Dgels returns false if the matrix
// A is singular, and true if this solution was successfully found.
//
// The minimization problem solved depends on the input parameters.
//
//  1. If m >= n and trans == blas.NoTrans, Dgels finds X such that || A*X - B||_2
//     is minimized.
//  2. If m < n and trans == blas.NoTrans, Dgels finds the minimum norm solution of
//     A * X = B.
//  3. If m >= n and trans == blas.Trans, Dgels finds the minimum norm solution of
//     A^T * X = B.
//  4. If m < n and trans == blas.Trans, Dgels finds X such that || A*X - B||_2
//     is minimized.
//
// Note that the least-squares solutions (cases 1 and 3) perform the minimization
// per column of B. This is not the same as finding the minimum-norm matrix.
//
// The matrix A is a general matrix of size m×n and is modified during this call.
// The input matrix B is of size max(m,n)×nrhs, and serves two purposes. On entry,
// the elements of b specify the input matrix B. B has size m×nrhs if
// trans == blas.NoTrans, and n×nrhs if trans == blas.Trans. On exit, the
// leading submatrix of b contains the solution vectors X. If trans == blas.NoTrans,
// this submatrix is of size n×nrhs, and of size m×nrhs otherwise.
//
// work is temporary storage, and lwork specifies the usable memory length.
// At minimum, lwork >= max(m,n) + max(m,n,nrhs), and this function will panic
// otherwise. A longer work will enable blocked algorithms to be called.
// In the special case that lwork == -1, work[0] will be set to the optimal working
// length.
func (impl Implementation) Dgels(trans blas.Transpose, m, n, nrhs int, a []float64, lda int, b []float64, ldb int, work []float64, lwork int) bool {
	mn := min(m, n)
	minwrk := mn + max(mn, nrhs)
	switch {
	case trans != blas.NoTrans && trans != blas.Trans && trans != blas.ConjTrans:
		panic(badTrans)
	case m < 0:
		panic(mLT0)
	case n < 0:
		panic(nLT0)
	case nrhs < 0:
		panic(nrhsLT0)
	case lda < max(1, n):
		panic(badLdA)
	case ldb < max(1, nrhs):
		panic(badLdB)
	case lwork < max(1, minwrk) && lwork != -1:
		panic(badLWork)
	case len(work) < max(1, lwork):
		panic(shortWork)
	}

	// Quick return if possible.
	if mn == 0 || nrhs == 0 {
		impl.Dlaset(blas.All, max(m, n), nrhs, 0, 0, b, ldb)
		work[0] = 1
		return true
	}

	if lwork == -1 {
		return lapacke.Dgels(byte(trans), m, n, nrhs, a, lda, b, ldb, work, -1)
	}

	switch {
	case len(a) < (m-1)*lda+n:
		panic(shortA)
	case len(b) < (max(m, n)-1)*ldb+nrhs:
		panic(shortB)
	}

	return lapacke.Dgels(byte(trans), m, n, nrhs, a, lda, b, ldb, work, lwork)
}

// Dgesvd computes the singular value decomposition of the input matrix A.
//
// The singular value decomposition is
//
//	A = U * Sigma * V^T
//
// where Sigma is an m×n diagonal matrix containing the singular values of A,
// U is an m×m orthogonal matrix and V is an n×n orthogonal matrix. The first
// min(m,n) columns of U and V are the left and right singular vectors of A
// respectively.
//
// jobU and jobVT are options for computing the singular vectors. The behavior
// is as follows
//
//	jobU == lapack.SVDAll       All m columns of U are returned in u
//	jobU == lapack.SVDStore     The first min(m,n) columns are returned in u
//	jobU == lapack.SVDOverwrite The first min(m,n) columns of U are written into a
//	jobU == lapack.SVDNone      The columns of U are not computed.
//
// The behavior is the same for jobVT and the rows of V^T. At most one of jobU
// and jobVT can equal lapack.SVDOverwrite, and Dgesvd will panic otherwise.
//
// On entry, a contains the data for the m×n matrix A. During the call to Dgesvd
// the data is overwritten. On exit, A contains the appropriate singular vectors
// if either job is lapack.SVDOverwrite.
//
// s is a slice of length at least min(m,n) and on exit contains the singular
// values in decreasing order.
//
// u contains the left singular vectors on exit, stored column-wise. If
// jobU == lapack.SVDAll, u is of size m×m. If jobU == lapack.SVDStore u is
// of size m×min(m,n). If jobU == lapack.SVDOverwrite or lapack.SVDNone, u is
// not used.
//
// vt contains the left singular vectors on exit, stored rowwise. If
// jobV == lapack.SVDAll, vt is of size n×m. If jobVT == lapack.SVDStore vt is
// of size min(m,n)×n. If jobVT == lapack.SVDOverwrite or lapack.SVDNone, vt is
// not used.
//
// work is a slice for storing temporary memory, and lwork is the usable size of
// the slice. lwork must be at least max(5*min(m,n), 3*min(m,n)+max(m,n)).
// If lwork == -1, instead of performing Dgesvd, the optimal work length will be
// stored into work[0]. Dgesvd will panic if the working memory has insufficient
// storage.
//
// Dgesvd returns whether the decomposition successfully completed.
func (impl Implementation) Dgesvd(jobU, jobVT lapack.SVDJob, m, n int, a []float64, lda int, s, u []float64, ldu int, vt []float64, ldvt int, work []float64, lwork int) (ok bool) {
	wantua := jobU == lapack.SVDAll
	wantus := jobU == lapack.SVDStore
	wantuo := jobU == lapack.SVDOverwrite
	wantun := jobU == lapack.SVDNone
	if !(wantua || wantus || wantuo || wantun) {
		panic(badSVDJob)
	}

	wantva := jobVT == lapack.SVDAll
	wantvs := jobVT == lapack.SVDStore
	wantvas := wantva || wantvs
	wantvo := jobVT == lapack.SVDOverwrite
	wantvn := jobVT == lapack.SVDNone
	if !(wantva || wantvs || wantvo || wantvn) {
		panic(badSVDJob)
	}

	if wantuo && wantvo {
		panic(bothSVDOver)
	}

	minmn := min(m, n)
	minwork := 1
	if minmn > 0 {
		minwork = max(3*minmn+max(m, n), 5*minmn)
	}
	switch {
	case m < 0:
		panic(mLT0)
	case n < 0:
		panic(nLT0)
	case lda < max(1, n):
		panic(badLdA)
	case ldu < 1, wantua && ldu < m, wantus && ldu < minmn:
		panic(badLdU)
	case ldvt < 1 || (wantvas && ldvt < n):
		panic(badLdVT)
	case lwork < minwork && lwork != -1:
		panic(badLWork)
	case len(work) < max(1, lwork):
		panic(shortWork)
	}

	// Quick return if possible.
	if minmn == 0 {
		work[0] = 1
		return true
	}

	if lwork == -1 {
		return lapacke.Dgesvd(byte(jobU), byte(jobVT), m, n, a, lda, s, u, ldu, vt, ldvt, work, -1)
	}

	switch {
	case len(a) < (m-1)*lda+n:
		panic(shortA)
	case len(s) < minmn:
		panic(shortS)
	case (len(u) < (m-1)*ldu+m && wantua) || (len(u) < (m-1)*ldu+minmn && wantus):
		panic(shortU)
	case (len(vt) < (n-1)*ldvt+n && wantva) || (len(vt) < (minmn-1)*ldvt+n && wantvs):
		panic(shortVT)
	}

	return lapacke.Dgesvd(byte(jobU), byte(jobVT), m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork)
}

// Dgetf2 computes the LU decomposition of the m×n matrix A.
// The LU decomposition is a factorization of a into
//
//	A = P * L * U
//
// where P is a permutation matrix, L is a unit lower triangular matrix, and
// U is a (usually) non-unit upper triangular matrix. On exit, L and U are stored
// in place into a.
//
// ipiv is a permutation vector. It indicates that row i of the matrix was
// changed with ipiv[i]. ipiv must have length at least min(m,n), and will panic
// otherwise. ipiv is zero-indexed.
//
// Dgetf2 returns whether the matrix A is singular. The LU decomposition will
// be computed regardless of the singularity of A, but division by zero
// will occur if the false is returned and the result is used to solve a
// system of equations.
func (Implementation) Dgetf2(m, n int, a []float64, lda int, ipiv []int) (ok bool) {
	mn := min(m, n)
	switch {
	case m < 0:
		panic(mLT0)
	case n < 0:
		panic(nLT0)
	case lda < max(1, n):
		panic(badLdA)
	}

	// Quick return if possible.
	if mn == 0 {
		return true
	}

	switch {
	case len(a) < (m-1)*lda+n:
		panic(shortA)
	case len(ipiv) != mn:
		panic(badLenIpiv)
	}

	ipiv32 := make([]int32, mn)
	ok = lapacke.Dgetf2(m, n, a, lda, ipiv32)
	for i, v := range ipiv32 {
		ipiv[i] = int(v) - 1 // Transform to zero-indexed.
	}
	return ok
}

// Dgetrf computes the LU decomposition of the m×n matrix A.
// The LU decomposition is a factorization of A into
//
//	A = P * L * U
//
// where P is a permutation matrix, L is a unit lower triangular matrix, and
// U is a (usually) non-unit upper triangular matrix. On exit, L and U are stored
// in place into a.
//
// ipiv is a permutation vector. It indicates that row i of the matrix was
// changed with ipiv[i]. ipiv must have length at least min(m,n), and will panic
// otherwise. ipiv is zero-indexed.
//
// Dgetrf is the blocked version of the algorithm.
//
// Dgetrf returns whether the matrix A is singular. The LU decomposition will
// be computed regardless of the singularity of A, but division by zero
// will occur if the false is returned and the result is used to solve a
// system of equations.
func (impl Implementation) Dgetrf(m, n int, a []float64, lda int, ipiv []int) (ok bool) {
	mn := min(m, n)
	switch {
	case m < 0:
		panic(mLT0)
	case n < 0:
		panic(nLT0)
	case lda < max(1, n):
		panic(badLdA)
	}

	// Quick return if possible.
	if mn == 0 {
		return true
	}

	switch {
	case len(a) < (m-1)*lda+n:
		panic(shortA)
	case len(ipiv) != mn:
		panic(badLenIpiv)
	}

	ipiv32 := make([]int32, mn)
	ok = lapacke.Dgetrf(m, n, a, lda, ipiv32)
	for i, v := range ipiv32 {
		ipiv[i] = int(v) - 1 // Transform to zero-indexed.
	}
	return ok
}

// Dgetri computes the inverse of the matrix A using the LU factorization computed
// by Dgetrf. On entry, a contains the PLU decomposition of A as computed by
// Dgetrf and on exit contains the reciprocal of the original matrix.
//
// Dgetri will not perform the inversion if the matrix is singular, and returns
// a boolean indicating whether the inversion was successful.
//
// work is temporary storage, and lwork specifies the usable memory length.
// At minimum, lwork >= n and this function will panic otherwise.
// Dgetri is a blocked inversion, but the block size is limited
// by the temporary space available. If lwork == -1, instead of performing Dgetri,
// the optimal work length will be stored into work[0].
func (impl Implementation) Dgetri(n int, a []float64, lda int, ipiv []int, work []float64, lwork int) (ok bool) {
	iws := max(1, n)
	switch {
	case n < 0:
		panic(nLT0)
	case lda < max(1, n):
		panic(badLdA)
	case lwork < iws && lwork != -1:
		panic(badLWork)
	case len(work) < max(1, lwork):
		panic(shortWork)
	}

	if n == 0 {
		work[0] = 1
		return true
	}

	if lwork == -1 {
		return lapacke.Dgetri(n, a, lda, nil, work, -1)
	}

	switch {
	case len(a) < (n-1)*lda+n:
		panic(shortA)
	case len(ipiv) != n:
		panic(badLenIpiv)
	}

	ipiv32 := make([]int32, n)
	for i, v := range ipiv {
		v++ // Transform to one-indexed.
		if v != int(int32(v)) {
			panic("lapack: ipiv element out of range")
		}
		ipiv32[i] = int32(v)
	}
	return lapacke.Dgetri(n, a, lda, ipiv32, work, lwork)
}

// Dgetrs solves a system of equations using an LU factorization.
// The system of equations solved is
//
//	A * X = B if trans == blas.Trans
//	A^T * X = B if trans == blas.NoTrans
//
// A is a general n×n matrix with stride lda. B is a general matrix of size n×nrhs.
//
// On entry b contains the elements of the matrix B. On exit, b contains the
// elements of X, the solution to the system of equations.
//
// a and ipiv contain the LU factorization of A and the permutation indices as
// computed by Dgetrf. ipiv is zero-indexed.
func (impl Implementation) Dgetrs(trans blas.Transpose, n, nrhs int, a []float64, lda int, ipiv []int, b []float64, ldb int) {
	switch {
	case trans != blas.NoTrans && trans != blas.Trans && trans != blas.ConjTrans:
		panic(badTrans)
	case n < 0:
		panic(nLT0)
	case nrhs < 0:
		panic(nrhsLT0)
	case lda < max(1, n):
		panic(badLdA)
	case ldb < max(1, nrhs):
		panic(badLdB)
	}

	// Quick return if possible.
	if n == 0 || nrhs == 0 {
		return
	}

	switch {
	case len(a) < (n-1)*lda+n:
		panic(shortA)
	case len(b) < (n-1)*ldb+nrhs:
		panic(shortB)
	case len(ipiv) != n:
		panic(badLenIpiv)
	}

	ipiv32 := make([]int32, n)
	for i, v := range ipiv {
		v++ // Transform to one-indexed.
		if v != int(int32(v)) {
			panic("lapack: ipiv element out of range")
		}
		ipiv32[i] = int32(v)
	}
	lapacke.Dgetrs(byte(trans), n, nrhs, a, lda, ipiv32, b, ldb)
}

// Dggsvd3 computes the generalized singular value decomposition (GSVD)
// of an m×n matrix A and p×n matrix B:
//
//	U^T*A*Q = D1*[ 0 R ]
//
//	V^T*B*Q = D2*[ 0 R ]
//
// where U, V and Q are orthogonal matrices.
//
// Dggsvd3 returns k and l, the dimensions of the sub-blocks. k+l
// is the effective numerical rank of the (m+p)×n matrix [ A^T B^T ]^T.
// R is a (k+l)×(k+l) nonsingular upper triangular matrix, D1 and
// D2 are m×(k+l) and p×(k+l) diagonal matrices and of the following
// structures, respectively:
//
// If m-k-l >= 0,
//
//	                  k  l
//	     D1 =     k [ I  0 ]
//	              l [ 0  C ]
//	          m-k-l [ 0  0 ]
//
//	                k  l
//	     D2 = l   [ 0  S ]
//	          p-l [ 0  0 ]
//
//	             n-k-l  k    l
//	[ 0 R ] = k [  0   R11  R12 ] k
//	          l [  0    0   R22 ] l
//
// where
//
//	C = diag( alpha_k, ... , alpha_{k+l} ),
//	S = diag( beta_k,  ... , beta_{k+l} ),
//	C^2 + S^2 = I.
//
// R is stored in
//
//	A[0:k+l, n-k-l:n]
//
// on exit.
//
// If m-k-l < 0,
//
//	               k m-k k+l-m
//	    D1 =   k [ I  0    0  ]
//	         m-k [ 0  C    0  ]
//
//	                 k m-k k+l-m
//	    D2 =   m-k [ 0  S    0  ]
//	         k+l-m [ 0  0    I  ]
//	           p-l [ 0  0    0  ]
//
//	               n-k-l  k   m-k  k+l-m
//	[ 0 R ] =    k [ 0    R11  R12  R13 ]
//	           m-k [ 0     0   R22  R23 ]
//	         k+l-m [ 0     0    0   R33 ]
//
// where
//
//	C = diag( alpha_k, ... , alpha_m ),
//	S = diag( beta_k,  ... , beta_m ),
//	C^2 + S^2 = I.
//
//	R = [ R11 R12 R13 ] is stored in A[1:m, n-k-l+1:n]
//	    [  0  R22 R23 ]
//
// and R33 is stored in
//
//	B[m-k:l, n+m-k-l:n] on exit.
//
// Dggsvd3 computes C, S, R, and optionally the orthogonal transformation
// matrices U, V and Q.
//
// jobU, jobV and jobQ are options for computing the orthogonal matrices. The behavior
// is as follows
//
//	jobU == lapack.GSVDU        Compute orthogonal matrix U
//	jobU == lapack.GSVDNone     Do not compute orthogonal matrix.
//
// The behavior is the same for jobV and jobQ with the exception that instead of
// lapack.GSVDU these accept lapack.GSVDV and lapack.GSVDQ respectively.
// The matrices U, V and Q must be m×m, p×p and n×n respectively unless the
// relevant job parameter is lapack.GSVDNone.
//
// alpha and beta must have length n or Dggsvd3 will panic. On exit, alpha and
// beta contain the generalized singular value pairs of A and B
//
//	alpha[0:k] = 1,
//	beta[0:k]  = 0,
//
// if m-k-l >= 0,
//
//	alpha[k:k+l] = diag(C),
//	beta[k:k+l]  = diag(S),
//
// if m-k-l < 0,
//
//	alpha[k:m]= C, alpha[m:k+l]= 0
//	beta[k:m] = S, beta[m:k+l] = 1.
//
// if k+l < n,
//
//	alpha[k+l:n] = 0 and
//	beta[k+l:n]  = 0.
//
// On exit, iwork contains the permutation required to sort alpha descending.
//
// iwork must have length n, work must have length at least max(1, lwork), and
// lwork must be -1 or greater than n, otherwise Dggsvd3 will panic. If
// lwork is -1, work[0] holds the optimal lwork on return, but Dggsvd3 does
// not perform the GSVD.
func (impl Implementation) Dggsvd3(jobU, jobV, jobQ lapack.GSVDJob, m, n, p int, a []float64, lda int, b []float64, ldb int, alpha, beta, u []float64, ldu int, v []float64, ldv int, q []float64, ldq int, work []float64, lwork int, iwork []int) (k, l int, ok bool) {
	wantu := jobU == lapack.GSVDU
	wantv := jobV == lapack.GSVDV
	wantq := jobQ == lapack.GSVDQ
	switch {
	case !wantu && jobU != lapack.GSVDNone:
		panic(badGSVDJob + "U")
	case !wantv && jobV != lapack.GSVDNone:
		panic(badGSVDJob + "V")
	case !wantq && jobQ != lapack.GSVDNone:
		panic(badGSVDJob + "Q")
	case m < 0:
		panic(mLT0)
	case n < 0:
		panic(nLT0)
	case p < 0:
		panic(pLT0)
	case lda < max(1, n):
		panic(badLdA)
	case ldb < max(1, n):
		panic(badLdB)
	case ldu < 1, wantu && ldu < m:
		panic(badLdU)
	case ldv < 1, wantv && ldv < p:
		panic(badLdV)
	case ldq < 1, wantq && ldq < n:
		panic(badLdQ)
	case len(iwork) < n:
		panic(shortWork)
	case lwork < 1 && lwork != -1:
		panic(badLWork)
	case len(work) < max(1, lwork):
		panic(shortWork)
	}

	// Determine optimal work length.
	if lwork == -1 {
		lapacke.Dggsvd3(byte(jobU), byte(jobV), byte(jobQ), m, n, p, nil, nil, a, lda, b, ldb, alpha, beta, u, ldu, v, ldv, q, ldq, work, -1, nil)
		return 0, 0, true
	}

	switch {
	case len(a) < (m-1)*lda+n:
		panic(shortA)
	case len(b) < (p-1)*ldb+n:
		panic(shortB)
	case wantu && len(u) < (m-1)*ldu+m:
		panic(shortU)
	case wantv && len(v) < (p-1)*ldv+p:
		panic(shortV)
	case wantq && len(q) < (n-1)*ldq+n:
		panic(shortQ)
	case len(alpha) != n:
		panic(badLenAlpha)
	case len(beta) != n:
		panic(badLenBeta)
	}

	_k := []int32{0}
	_l := []int32{0}
	_iwork := make([]int32, n)
	ok = lapacke.Dggsvd3(byte(jobU), byte(jobV), byte(jobQ), m, n, p, _k, _l, a, lda, b, ldb, alpha, beta, u, ldu, v, ldv, q, ldq, work, lwork, _iwork)
	for i, v := range _iwork {
		iwork[i] = int(v - 1)
	}

	return int(_k[0]), int(_l[0]), ok
}

// Dggsvp3 computes orthogonal matrices U, V and Q such that
//
//	                n-k-l  k    l
//	U^T*A*Q =    k [ 0    A12  A13 ] if m-k-l >= 0;
//	             l [ 0     0   A23 ]
//	         m-k-l [ 0     0    0  ]
//
//	                n-k-l  k    l
//	U^T*A*Q =    k [ 0    A12  A13 ] if m-k-l < 0;
//	           m-k [ 0     0   A23 ]
//
//	                n-k-l  k    l
//	V^T*B*Q =    l [ 0     0   B13 ]
//	           p-l [ 0     0    0  ]
//
// where the k×k matrix A12 and l×l matrix B13 are non-singular
// upper triangular. A23 is l×l upper triangular if m-k-l >= 0,
// otherwise A23 is (m-k)×l upper trapezoidal.
//
// Dggsvp3 returns k and l, the dimensions of the sub-blocks. k+l
// is the effective numerical rank of the (m+p)×n matrix [ A^T B^T ]^T.
//
// jobU, jobV and jobQ are options for computing the orthogonal matrices. The behavior
// is as follows
//
//	jobU == lapack.GSVDU        Compute orthogonal matrix U
//	jobU == lapack.GSVDNone     Do not compute orthogonal matrix.
//
// The behavior is the same for jobV and jobQ with the exception that instead of
// lapack.GSVDU these accept lapack.GSVDV and lapack.GSVDQ respectively.
// The matrices U, V and Q must be m×m, p×p and n×n respectively unless the
// relevant job parameter is lapack.GSVDNone.
//
// tola and tolb are the convergence criteria for the Jacobi-Kogbetliantz
// iteration procedure. Generally, they are the same as used in the preprocessing
// step, for example,
//
//	tola = max(m, n)*norm(A)*eps,
//	tolb = max(p, n)*norm(B)*eps.
//
// Where eps is the machine epsilon.
//
// iwork must have length n, work must have length at least max(1, lwork), and
// lwork must be -1 or greater than zero, otherwise Dggsvp3 will panic.
//
// Dggsvp3 is an internal routine. It is exported for testing purposes.
func (impl Implementation) Dggsvp3(jobU, jobV, jobQ lapack.GSVDJob, m, p, n int, a []float64, lda int, b []float64, ldb int, tola, tolb float64, u []float64, ldu int, v []float64, ldv int, q []float64, ldq int, iwork []int, tau, work []float64, lwork int) (k, l int) {
	wantu := jobU == lapack.GSVDU
	wantv := jobV == lapack.GSVDV
	wantq := jobQ == lapack.GSVDQ
	switch {
	case !wantu && jobU != lapack.GSVDNone:
		panic(badGSVDJob + "U")
	case !wantv && jobV != lapack.GSVDNone:
		panic(badGSVDJob + "V")
	case !wantq && jobQ != lapack.GSVDNone:
		panic(badGSVDJob + "Q")
	case m < 0:
		panic(mLT0)
	case p < 0:
		panic(pLT0)
	case n < 0:
		panic(nLT0)
	case lda < max(1, n):
		panic(badLdA)
	case ldb < max(1, n):
		panic(badLdB)
	case ldu < 1, wantu && ldu < m:
		panic(badLdU)
	case ldv < 1, wantv && ldv < p:
		panic(badLdV)
	case ldq < 1, wantq && ldq < n:
		panic(badLdQ)
	case len(iwork) != n:
		panic(shortWork)
	case lwork < 1 && lwork != -1:
		panic(badLWork)
	case len(work) < max(1, lwork):
		panic(shortWork)
	}

	if lwork == -1 {
		lapacke.Dggsvp3(byte(jobU), byte(jobV), byte(jobQ), m, p, n, a, lda, b, ldb, tola, tolb, nil, nil, u, ldu, v, ldv, q, ldq, nil, tau, work, -1)
		return 0, 0
	}

	switch {
	case len(a) < (m-1)*lda+n:
		panic(shortA)
	case len(b) < (p-1)*ldb+n:
		panic(shortB)
	case wantu && len(u) < (m-1)*ldu+m:
		panic(shortU)
	case wantv && len(v) < (p-1)*ldv+p:
		panic(shortV)
	case wantq && len(q) < (n-1)*ldq+n:
		panic(shortQ)
	case len(tau) < n:
		// tau check must come after lwkopt query since
		// the Dggsvd3 call for lwkopt query may have
		// lwork == -1, and tau is provided by work.
		panic(shortTau)
	}

	_k := []int32{0}
	_l := []int32{0}
	_iwork := make([]int32, n)
	lapacke.Dggsvp3(byte(jobU), byte(jobV), byte(jobQ), m, p, n, a, lda, b, ldb, tola, tolb, _k, _l, u, ldu, v, ldv, q, ldq, _iwork, tau, work, lwork)
	return int(_k[0]), int(_l[0])
}

// Dorgbr generates one of the matrices Q or P^T computed by Dgebrd.
// See Dgebrd for the description of Q and P^T.
//
// If vect == lapack.ApplyQ, then a is assumed to have been an m×k matrix and
// Q is of order m. If m >= k, then Dorgbr returns the first n columns of Q
// where m >= n >= k. If m < k, then Dorgbr returns Q as an m×m matrix.
//
// If vect == lapack.ApplyP, then A is assumed to have been a k×n matrix, and
// P^T is of order n. If k < n, then Dorgbr returns the first m rows of P^T,
// where n >= m >= k. If k >= n, then Dorgbr returns P^T as an n×n matrix.
func (impl Implementation) Dorgbr(vect lapack.GenOrtho, m, n, k int, a []float64, lda int, tau, work []float64, lwork int) {
	wantq := vect == lapack.GenerateQ
	mn := min(m, n)
	switch {
	case vect != lapack.GenerateQ && vect != lapack.GeneratePT:
		panic(badGenOrtho)
	case m < 0:
		panic(mLT0)
	case n < 0:
		panic(nLT0)
	case k < 0:
		panic(kLT0)
	case wantq && n > m:
		panic(nGTM)
	case wantq && n < min(m, k):
		panic("lapack: n < min(m,k)")
	case !wantq && m > n:
		panic(mGTN)
	case !wantq && m < min(n, k):
		panic("lapack: m < min(n,k)")
	case lda < max(1, n):
		panic(badLdA)
	case lwork < max(1, mn) && lwork != -1:
		panic(badLWork)
	case len(work) < max(1, lwork):
		panic(shortWork)
	}

	// Quick return if possible.
	if m == 0 || n == 0 {
		work[0] = 1
		return
	}

	if lwork == -1 {
		lapacke.Dorgbr(byte(vect), m, n, k, a, lda, tau, work, -1)
		return
	}

	switch {
	case len(a) < (m-1)*lda+n:
		panic(shortA)
	case wantq && len(tau) < min(m, k):
		panic(shortTau)
	case !wantq && len(tau) < min(n, k):
		panic(shortTau)
	}

	lapacke.Dorgbr(byte(vect), m, n, k, a, lda, tau, work, lwork)
}

// Dorghr generates an n×n orthogonal matrix Q which is defined as the product
// of ihi-ilo elementary reflectors:
//
//	Q = H_{ilo} H_{ilo+1} ... H_{ihi-1}.
//
// a and lda represent an n×n matrix that contains the elementary reflectors, as
// returned by Dgehrd. On return, a is overwritten by the n×n orthogonal matrix
// Q. Q will be equal to the identity matrix except in the submatrix
// Q[ilo+1:ihi+1,ilo+1:ihi+1].
//
// ilo and ihi must have the same values as in the previous call of Dgehrd. It
// must hold that
//
//	0 <= ilo <= ihi < n,  if n > 0,
//	ilo = 0, ihi = -1,    if n == 0.
//
// tau contains the scalar factors of the elementary reflectors, as returned by
// Dgehrd. tau must have length n-1.
//
// work must have length at least max(1,lwork) and lwork must be at least
// ihi-ilo. For optimum performance lwork must be at least (ihi-ilo)*nb where nb
// is the optimal blocksize. On return, work[0] will contain the optimal value
// of lwork.
//
// If lwork == -1, instead of performing Dorghr, only the optimal value of lwork
// will be stored into work[0].
//
// If any requirement on input sizes is not met, Dorghr will panic.
//
// Dorghr is an internal routine. It is exported for testing purposes.
func (impl Implementation) Dorghr(n, ilo, ihi int, a []float64, lda int, tau, work []float64, lwork int) {
	nh := ihi - ilo
	switch {
	case ilo < 0 || max(1, n) <= ilo:
		panic(badIlo)
	case ihi < min(ilo, n-1) || n <= ihi:
		panic(badIhi)
	case lda < max(1, n):
		panic(badLdA)
	case lwork < max(1, nh) && lwork != -1:
		panic(badLWork)
	case len(work) < max(1, lwork):
		panic(shortWork)
	}

	// Quick return if possible.
	if n == 0 {
		work[0] = 1
		return
	}

	if lwork == -1 {
		lapacke.Dorghr(n, ilo+1, ihi+1, a, lda, tau, work, -1)
		return
	}

	switch {
	case len(a) < (n-1)*lda+n:
		panic(shortA)
	case len(tau) < n-1:
		panic(shortTau)
	}

	lapacke.Dorghr(n, ilo+1, ihi+1, a, lda, tau, work, lwork)
}

// Dorglq generates an m×n matrix Q with orthonormal rows defined by the product
// of elementary reflectors
//
//	Q = H_{k-1} * ... * H_1 * H_0
//
// as computed by Dgelqf. Dorglq is the blocked version of Dorgl2 that makes
// greater use of level-3 BLAS routines.
//
// len(tau) >= k, 0 <= k <= n, and 0 <= m <= n.
//
// work is temporary storage, and lwork specifies the usable memory length. At minimum,
// lwork >= m, and the amount of blocking is limited by the usable length.
// If lwork == -1, instead of computing Dorglq the optimal work length is stored
// into work[0].
//
// Dorglq will panic if the conditions on input values are not met.
//
// Dorglq is an internal routine. It is exported for testing purposes.
func (impl Implementation) Dorglq(m, n, k int, a []float64, lda int, tau, work []float64, lwork int) {
	switch {
	case m < 0:
		panic(mLT0)
	case n < m:
		panic(nLTM)
	case k < 0:
		panic(kLT0)
	case k > m:
		panic(kGTM)
	case lda < max(1, n):
		panic(badLdA)
	case lwork < max(1, m) && lwork != -1:
		panic(badLWork)
	case len(work) < max(1, lwork):
		panic(shortWork)
	}

	if m == 0 {
		work[0] = 1
		return
	}

	if lwork == -1 {
		lapacke.Dorglq(m, n, k, a, lda, tau, work, -1)
		return
	}

	switch {
	case len(a) < (m-1)*lda+n:
		panic(shortA)
	case len(tau) < k:
		panic(shortTau)
	}

	lapacke.Dorglq(m, n, k, a, lda, tau, work, lwork)
}

// Dorgql generates the m×n matrix Q with orthonormal columns defined as the
// last n columns of a product of k elementary reflectors of order m
//
//	Q = H_{k-1} * ... * H_1 * H_0.
//
// It must hold that
//
//	0 <= k <= n <= m,
//
// and Dorgql will panic otherwise.
//
// On entry, the (n-k+i)-th column of A must contain the vector which defines
// the elementary reflector H_i, for i=0,...,k-1, and tau[i] must contain its
// scalar factor. On return, a contains the m×n matrix Q.
//
// tau must have length at least k, and Dorgql will panic otherwise.
//
// work must have length at least max(1,lwork), and lwork must be at least
// max(1,n), otherwise Dorgql will panic. For optimum performance lwork must
// be a sufficiently large multiple of n.
//
// If lwork == -1, instead of computing Dorgql the optimal work length is stored
// into work[0].
//
// Dorgql is an internal routine. It is exported for testing purposes.
func (impl Implementation) Dorgql(m, n, k int, a []float64, lda int, tau, work []float64, lwork int) {
	switch {
	case m < 0:
		panic(mLT0)
	case n < 0:
		panic(nLT0)
	case n > m:
		panic(nGTM)
	case k < 0:
		panic(kLT0)
	case k > n:
		panic(kGTN)
	case lda < max(1, n):
		panic(badLdA)
	case lwork < max(1, n) && lwork != -1:
		panic(badLWork)
	case len(work) < max(1, lwork):
		panic(shortWork)
	}

	// Quick return if possible.
	if n == 0 {
		work[0] = 1
		return
	}

	if lwork == -1 {
		lapacke.Dorgql(m, n, k, a, lda, tau, work, -1)
		return
	}

	switch {
	case len(a) < (m-1)*lda+n:
		panic(shortA)
	case len(tau) < k:
		panic(shortTau)
	}

	lapacke.Dorgql(m, n, k, a, lda, tau, work, lwork)
}

// Dorgqr generates an m×n matrix Q with orthonormal columns defined by the
// product of elementary reflectors
//
//	Q = H_0 * H_1 * ... * H_{k-1}
//
// as computed by Dgeqrf. Dorgqr is the blocked version of Dorg2r that makes
// greater use of level-3 BLAS routines.
//
// The length of tau must be at least k, and the length of work must be at least n.
// It also must be that 0 <= k <= n and 0 <= n <= m.
//
// work is temporary storage, and lwork specifies the usable memory length. At
// minimum, lwork >= n, and the amount of blocking is limited by the usable
// length. If lwork == -1, instead of computing Dorgqr the optimal work length
// is stored into work[0].
//
// Dorgqr will panic if the conditions on input values are not met.
//
// Dorgqr is an internal routine. It is exported for testing purposes.
func (impl Implementation) Dorgqr(m, n, k int, a []float64, lda int, tau, work []float64, lwork int) {
	switch {
	case m < 0:
		panic(mLT0)
	case n < 0:
		panic(nLT0)
	case n > m:
		panic(nGTM)
	case k < 0:
		panic(kLT0)
	case k > n:
		panic(kGTN)
	case lda < max(1, n):
		panic(badLdA)
	case lwork < max(1, n) && lwork != -1:
		panic(badLWork)
	case len(work) < max(1, lwork):
		panic(shortWork)
	}

	if n == 0 {
		work[0] = 1
		return
	}

	if lwork == -1 {
		lapacke.Dorgqr(m, n, k, a, lda, tau, work, -1)
		return
	}

	switch {
	case len(a) < (m-1)*lda+n:
		panic(shortA)
	case len(tau) < k:
		panic(shortTau)
	}

	lapacke.Dorgqr(m, n, k, a, lda, tau, work, lwork)
}

// Dorgtr generates a real orthogonal matrix Q which is defined as the product
// of n-1 elementary reflectors of order n as returned by Dsytrd.
//
// The construction of Q depends on the value of uplo:
//
//	Q = H_{n-1} * ... * H_1 * H_0  if uplo == blas.Upper
//	Q = H_0 * H_1 * ... * H_{n-1}  if uplo == blas.Lower
//
// where H_i is constructed from the elementary reflectors as computed by Dsytrd.
// See the documentation for Dsytrd for more information.
//
// tau must have length at least n-1, and Dorgtr will panic otherwise.
//
// work is temporary storage, and lwork specifies the usable memory length. At
// minimum, lwork >= max(1,n-1), and Dorgtr will panic otherwise. The amount of blocking
// is limited by the usable length.
// If lwork == -1, instead of computing Dorgtr the optimal work length is stored
// into work[0].
//
// Dorgtr is an internal routine. It is exported for testing purposes.
func (impl Implementation) Dorgtr(uplo blas.Uplo, n int, a []float64, lda int, tau, work []float64, lwork int) {
	switch {
	case uplo != blas.Upper && uplo != blas.Lower:
		panic(badUplo)
	case n < 0:
		panic(nLT0)
	case lda < max(1, n):
		panic(badLdA)
	case lwork < max(1, n-1) && lwork != -1:
		panic(badLWork)
	case len(work) < max(1, lwork):
		panic(shortWork)
	}

	if n == 0 {
		work[0] = 1
		return
	}

	if lwork == -1 {
		lapacke.Dorgtr(byte(uplo), n, a, lda, tau, work, -1)
		return
	}

	switch {
	case len(a) < (n-1)*lda+n:
		panic(shortA)
	case len(tau) < n-1:
		panic(shortTau)
	}

	lapacke.Dorgtr(byte(uplo), n, a, lda, tau, work, lwork)
}

// Dormbr applies a multiplicative update to the matrix C based on a
// decomposition computed by Dgebrd.
//
// Dormbr overwrites the m×n matrix C with
//
//	Q * C   if vect == lapack.ApplyQ, side == blas.Left, and trans == blas.NoTrans
//	C * Q   if vect == lapack.ApplyQ, side == blas.Right, and trans == blas.NoTrans
//	Q^T * C if vect == lapack.ApplyQ, side == blas.Left, and trans == blas.Trans
//	C * Q^T if vect == lapack.ApplyQ, side == blas.Right, and trans == blas.Trans
//
//	P * C   if vect == lapack.ApplyP, side == blas.Left, and trans == blas.NoTrans
//	C * P   if vect == lapack.ApplyP, side == blas.Right, and trans == blas.NoTrans
//	P^T * C if vect == lapack.ApplyP, side == blas.Left, and trans == blas.Trans
//	C * P^T if vect == lapack.ApplyP, side == blas.Right, and trans == blas.Trans
//
// where P and Q are the orthogonal matrices determined by Dgebrd when reducing
// a matrix A to bidiagonal form: A = Q * B * P^T. See Dgebrd for the
// definitions of Q and P.
//
// If vect == lapack.ApplyQ, A is assumed to have been an nq×k matrix, while if
// vect == lapack.ApplyP, A is assumed to have been a k×nq matrix. nq = m if
// side == blas.Left, while nq = n if side == blas.Right.
//
// tau must have length min(nq,k), and Dormbr will panic otherwise. tau contains
// the elementary reflectors to construct Q or P depending on the value of
// vect.
//
// work must have length at least max(1,lwork), and lwork must be either -1 or
// at least max(1,n) if side == blas.Left, and at least max(1,m) if side ==
// blas.Right. For optimum performance lwork should be at least n*nb if side ==
// blas.Left, and at least m*nb if side == blas.Right, where nb is the optimal
// block size. On return, work[0] will contain the optimal value of lwork.
//
// If lwork == -1, the function only calculates the optimal value of lwork and
// returns it in work[0].
//
// Dormbr is an internal routine. It is exported for testing purposes.
func (impl Implementation) Dormbr(vect lapack.ApplyOrtho, side blas.Side, trans blas.Transpose, m, n, k int, a []float64, lda int, tau, c []float64, ldc int, work []float64, lwork int) {
	nq := n
	nw := m
	if side == blas.Left {
		nq = m
		nw = n
	}
	applyQ := vect == lapack.ApplyQ
	switch {
	case !applyQ && vect != lapack.ApplyP:
		panic(badApplyOrtho)
	case side != blas.Left && side != blas.Right:
		panic(badSide)
	case trans != blas.NoTrans && trans != blas.Trans:
		panic(badTrans)
	case m < 0:
		panic(mLT0)
	case n < 0:
		panic(nLT0)
	case k < 0:
		panic(kLT0)
	case applyQ && lda < max(1, min(nq, k)):
		panic(badLdA)
	case !applyQ && lda < max(1, nq):
		panic(badLdA)
	case ldc < max(1, n):
		panic(badLdC)
	case lwork < max(1, nw) && lwork != -1:
		panic(badLWork)
	case len(work) < max(1, lwork):
		panic(shortWork)
	}

	// Quick return if possible.
	if m == 0 || n == 0 {
		work[0] = 1
		return
	}

	if lwork == -1 {
		lapacke.Dormbr(byte(vect), byte(side), byte(trans), m, n, k, a, lda, tau, c, ldc, work, -1)
		return
	}

	minnqk := min(nq, k)
	switch {
	case applyQ && len(a) < (nq-1)*lda+minnqk:
		panic(shortA)
	case !applyQ && len(a) < (minnqk-1)*lda+nq:
		panic(shortA)
	case len(tau) < minnqk:
		panic(shortTau)
	case len(c) < (m-1)*ldc+n:
		panic(shortC)
	}

	lapacke.Dormbr(byte(vect), byte(side), byte(trans), m, n, k, a, lda, tau, c, ldc, work, lwork)
}

// Dormhr multiplies an m×n general matrix C with an nq×nq orthogonal matrix Q
//
//	Q * C,    if side == blas.Left and trans == blas.NoTrans,
//	Q^T * C,  if side == blas.Left and trans == blas.Trans,
//	C * Q,    if side == blas.Right and trans == blas.NoTrans,
//	C * Q^T,  if side == blas.Right and trans == blas.Trans,
//
// where nq == m if side == blas.Left and nq == n if side == blas.Right.
//
// Q is defined implicitly as the product of ihi-ilo elementary reflectors, as
// returned by Dgehrd:
//
//	Q = H_{ilo} H_{ilo+1} ... H_{ihi-1}.
//
// Q is equal to the identity matrix except in the submatrix
// Q[ilo+1:ihi+1,ilo+1:ihi+1].
//
// ilo and ihi must have the same values as in the previous call of Dgehrd. It
// must hold that
//
//	0 <= ilo <= ihi < m,   if m > 0 and side == blas.Left,
//	ilo = 0 and ihi = -1,  if m = 0 and side == blas.Left,
//	0 <= ilo <= ihi < n,   if n > 0 and side == blas.Right,
//	ilo = 0 and ihi = -1,  if n = 0 and side == blas.Right.
//
// a and lda represent an m×m matrix if side == blas.Left and an n×n matrix if
// side == blas.Right. The matrix contains vectors which define the elementary
// reflectors, as returned by Dgehrd.
//
// tau contains the scalar factors of the elementary reflectors, as returned by
// Dgehrd. tau must have length m-1 if side == blas.Left and n-1 if side ==
// blas.Right.
//
// c and ldc represent the m×n matrix C. On return, c is overwritten by the
// product with Q.
//
// work must have length at least max(1,lwork), and lwork must be at least
// max(1,n), if side == blas.Left, and max(1,m), if side == blas.Right. For
// optimum performance lwork should be at least n*nb if side == blas.Left and
// m*nb if side == blas.Right, where nb is the optimal block size. On return,
// work[0] will contain the optimal value of lwork.
//
// If lwork == -1, instead of performing Dormhr, only the optimal value of lwork
// will be stored in work[0].
//
// If any requirement on input sizes is not met, Dormhr will panic.
//
// Dormhr is an internal routine. It is exported for testing purposes.
func (impl Implementation) Dormhr(side blas.Side, trans blas.Transpose, m, n, ilo, ihi int, a []float64, lda int, tau, c []float64, ldc int, work []float64, lwork int) {
	nq := n // The order of Q.
	nw := m // The minimum length of work.
	if side == blas.Left {
		nq = m
		nw = n
	}
	switch {
	case side != blas.Left && side != blas.Right:
		panic(badSide)
	case trans != blas.NoTrans && trans != blas.Trans:
		panic(badTrans)
	case m < 0:
		panic(mLT0)
	case n < 0:
		panic(nLT0)
	case ilo < 0 || max(1, nq) <= ilo:
		panic(badIlo)
	case ihi < min(ilo, nq-1) || nq <= ihi:
		panic(badIhi)
	case lda < max(1, nq):
		panic(badLdA)
	case lwork < max(1, nw) && lwork != -1:
		panic(badLWork)
	case len(work) < max(1, lwork):
		panic(shortWork)
	}

	// Quick return if possible.
	if m == 0 || n == 0 {
		work[0] = 1
		return
	}

	if lwork == -1 {
		lapacke.Dormhr(byte(side), byte(trans), m, n, ilo+1, ihi+1, a, lda, tau, c, ldc, work, -1)
		return
	}

	switch {
	case len(a) < (nq-1)*lda+nq:
		panic(shortA)
	case len(c) < (m-1)*ldc+n:
		panic(shortC)
	case len(tau) != nq-1:
		panic(badLenTau)
	}

	lapacke.Dormhr(byte(side), byte(trans), m, n, ilo+1, ihi+1, a, lda, tau, c, ldc, work, lwork)
}

// Dormlq multiplies the matrix C by the orthogonal matrix Q defined by the
// slices a and tau. A and tau are as returned from Dgelqf.
//
//	C = Q * C    if side == blas.Left and trans == blas.NoTrans
//	C = Q^T * C  if side == blas.Left and trans == blas.Trans
//	C = C * Q    if side == blas.Right and trans == blas.NoTrans
//	C = C * Q^T  if side == blas.Right and trans == blas.Trans
//
// If side == blas.Left, A is a matrix of side k×m, and if side == blas.Right
// A is of size k×n. This uses a blocked algorithm.
//
// Work is temporary storage, and lwork specifies the usable memory length.
// At minimum, lwork >= m if side == blas.Left and lwork >= n if side == blas.Right,
// and this function will panic otherwise.
// Dormlq uses a block algorithm, but the block size is limited
// by the temporary space available. If lwork == -1, instead of performing Dormlq,
// the optimal work length will be stored into work[0].
//
// tau contains the Householder scales and must have length at least k, and
// this function will panic otherwise.
func (impl Implementation) Dormlq(side blas.Side, trans blas.Transpose, m, n, k int, a []float64, lda int, tau, c []float64, ldc int, work []float64, lwork int) {
	left := side == blas.Left
	nw := m
	if left {
		nw = n
	}
	switch {
	case !left && side != blas.Right:
		panic(badSide)
	case trans != blas.Trans && trans != blas.NoTrans:
		panic(badTrans)
	case m < 0:
		panic(mLT0)
	case n < 0:
		panic(nLT0)
	case k < 0:
		panic(kLT0)
	case left && k > m:
		panic(kGTM)
	case !left && k > n:
		panic(kGTN)
	case left && lda < max(1, m):
		panic(badLdA)
	case !left && lda < max(1, n):
		panic(badLdA)
	case lwork < max(1, nw) && lwork != -1:
		panic(badLWork)
	case len(work) < max(1, lwork):
		panic(shortWork)
	}

	// Quick return if possible.
	if m == 0 || n == 0 || k == 0 {
		work[0] = 1
		return
	}

	if lwork == -1 {
		lapacke.Dormlq(byte(side), byte(trans), m, n, k, a, lda, tau, c, ldc, work, -1)
		return
	}

	switch {
	case left && len(a) < (k-1)*lda+m:
		panic(shortA)
	case !left && len(a) < (k-1)*lda+n:
		panic(shortA)
	case len(tau) < k:
		panic(shortTau)
	case len(c) < (m-1)*ldc+n:
		panic(shortC)
	}

	lapacke.Dormlq(byte(side), byte(trans), m, n, k, a, lda, tau, c, ldc, work, lwork)
}

// Dormqr multiplies an m×n matrix C by an orthogonal matrix Q as
//
//	C = Q * C,    if side == blas.Left  and trans == blas.NoTrans,
//	C = Q^T * C,  if side == blas.Left  and trans == blas.Trans,
//	C = C * Q,    if side == blas.Right and trans == blas.NoTrans,
//	C = C * Q^T,  if side == blas.Right and trans == blas.Trans,
//
// where Q is defined as the product of k elementary reflectors
//
//	Q = H_0 * H_1 * ... * H_{k-1}.
//
// If side == blas.Left, A is an m×k matrix and 0 <= k <= m.
// If side == blas.Right, A is an n×k matrix and 0 <= k <= n.
// The ith column of A contains the vector which defines the elementary
// reflector H_i and tau[i] contains its scalar factor. tau must have length k
// and Dormqr will panic otherwise. Dgeqrf returns A and tau in the required
// form.
//
// work must have length at least max(1,lwork), and lwork must be at least n if
// side == blas.Left and at least m if side == blas.Right, otherwise Dormqr will
// panic.
//
// work is temporary storage, and lwork specifies the usable memory length. At
// minimum, lwork >= m if side == blas.Left and lwork >= n if side ==
// blas.Right, and this function will panic otherwise. Larger values of lwork
// will generally give better performance. On return, work[0] will contain the
// optimal value of lwork.
//
// If lwork is -1, instead of performing Dormqr, the optimal workspace size will
// be stored into work[0].
func (impl Implementation) Dormqr(side blas.Side, trans blas.Transpose, m, n, k int, a []float64, lda int, tau, c []float64, ldc int, work []float64, lwork int) {
	left := side == blas.Left
	nq := n
	nw := m
	if left {
		nq = m
		nw = n
	}
	switch {
	case !left && side != blas.Right:
		panic(badSide)
	case trans != blas.NoTrans && trans != blas.Trans:
		panic(badTrans)
	case m < 0:
		panic(mLT0)
	case n < 0:
		panic(nLT0)
	case k < 0:
		panic(kLT0)
	case left && k > m:
		panic(kGTM)
	case !left && k > n:
		panic(kGTN)
	case lda < max(1, k):
		panic(badLdA)
	case ldc < max(1, n):
		panic(badLdC)
	case lwork < max(1, nw) && lwork != -1:
		panic(badLWork)
	case len(work) < max(1, lwork):
		panic(shortWork)
	}

	// Quick return if possible.
	if m == 0 || n == 0 || k == 0 {
		work[0] = 1
		return
	}

	if lwork == -1 {
		lapacke.Dormqr(byte(side), byte(trans), m, n, k, a, lda, tau, c, ldc, work, -1)
		return
	}

	switch {
	case len(a) < (nq-1)*lda+k:
		panic(shortA)
	case len(tau) != k:
		panic(badLenTau)
	case len(c) < (m-1)*ldc+n:
		panic(shortC)
	}

	lapacke.Dormqr(byte(side), byte(trans), m, n, k, a, lda, tau, c, ldc, work, lwork)
}

// Dpocon estimates the reciprocal of the condition number of a positive-definite
// matrix A given the Cholesky decomposition of A. The condition number computed
// is based on the 1-norm and the ∞-norm.
//
// anorm is the 1-norm and the ∞-norm of the original matrix A.
//
// work is a temporary data slice of length at least 3*n and Dpocon will panic otherwise.
//
// iwork is a temporary data slice of length at least n and Dpocon will panic otherwise.
func (impl Implementation) Dpocon(uplo blas.Uplo, n int, a []float64, lda int, anorm float64, work []float64, iwork []int) float64 {
	switch {
	case uplo != blas.Upper && uplo != blas.Lower:
		panic(badUplo)
	case n < 0:
		panic(nLT0)
	case lda < max(1, n):
		panic(badLdA)
	case anorm < 0:
		panic(negANorm)
	}

	// Quick return if possible.
	if n == 0 {
		return 1
	}

	switch {
	case len(a) < (n-1)*lda+n:
		panic(shortA)
	case len(work) < 3*n:
		panic(shortWork)
	case len(iwork) < n:
		panic(shortIWork)
	}

	rcond := []float64{0}
	_iwork := make([]int32, n)
	lapacke.Dpocon(byte(uplo), n, a, lda, anorm, rcond, work, _iwork)
	return rcond[0]
}

// Dsteqr computes the eigenvalues and optionally the eigenvectors of a symmetric
// tridiagonal matrix using the implicit QL or QR method. The eigenvectors of a
// full or band symmetric matrix can also be found if Dsytrd, Dsptrd, or Dsbtrd
// have been used to reduce this matrix to tridiagonal form.
//
// d, on entry, contains the diagonal elements of the tridiagonal matrix. On exit,
// d contains the eigenvalues in ascending order. d must have length n and
// Dsteqr will panic otherwise.
//
// e, on entry, contains the off-diagonal elements of the tridiagonal matrix on
// entry, and is overwritten during the call to Dsteqr. e must have length n-1 and
// Dsteqr will panic otherwise.
//
// z, on entry, contains the n×n orthogonal matrix used in the reduction to
// tridiagonal form if compz == lapack.OriginalEV. On exit, if
// compz == lapack.OriginalEV, z contains the orthonormal eigenvectors of the
// original symmetric matrix, and if compz == lapack.TridiagEV, z contains the
// orthonormal eigenvectors of the symmetric tridiagonal matrix. z is not used
// if compz == lapack.None.
//
// work must have length at least max(1, 2*n-2) if the eigenvectors are computed,
// and Dsteqr will panic otherwise.
//
// Dsteqr is an internal routine. It is exported for testing purposes.
func (impl Implementation) Dsteqr(compz lapack.EVComp, n int, d, e, z []float64, ldz int, work []float64) (ok bool) {
	switch {
	case compz != lapack.EVCompNone && compz != lapack.EVTridiag && compz != lapack.EVOrig:
		panic(badEVComp)
	case n < 0:
		panic(nLT0)
	case ldz < 1, compz != lapack.EVCompNone && ldz < n:
		panic(badLdZ)
	}

	// Quick return if possible.
	if n == 0 {
		return true
	}

	switch {
	case len(d) < n:
		panic(shortD)
	case len(e) < n-1:
		panic(shortE)
	case compz != lapack.EVCompNone && len(z) < (n-1)*ldz+n:
		panic(shortZ)
	case compz != lapack.EVCompNone && len(work) < max(1, 2*n-2):
		panic(shortWork)
	}

	return lapacke.Dsteqr(byte(compz), n, d, e, z, ldz, work)
}

// Dsterf computes all eigenvalues of a symmetric tridiagonal matrix using the
// Pal-Walker-Kahan variant of the QL or QR algorithm.
//
// d contains the diagonal elements of the tridiagonal matrix on entry, and
// contains the eigenvalues in ascending order on exit. d must have length at
// least n, or Dsterf will panic.
//
// e contains the off-diagonal elements of the tridiagonal matrix on entry, and is
// overwritten during the call to Dsterf. e must have length of at least n-1 or
// Dsterf will panic.
//
// Dsterf is an internal routine. It is exported for testing purposes.
func (impl Implementation) Dsterf(n int, d, e []float64) (ok bool) {
	if n < 0 {
		panic(nLT0)
	}

	// Quick return if possible.
	if n == 0 {
		return true
	}

	switch {
	case len(d) < n:
		panic(shortD)
	case len(e) < n-1:
		panic(shortE)
	}

	return lapacke.Dsterf(n, d, e)
}

// Dsyev computes all eigenvalues and, optionally, the eigenvectors of a real
// symmetric matrix A.
//
// w contains the eigenvalues in ascending order upon return. w must have length
// at least n, and Dsyev will panic otherwise.
//
// On entry, a contains the elements of the symmetric matrix A in the triangular
// portion specified by uplo. If jobz == lapack.ComputeEV a contains the
// orthonormal eigenvectors of A on exit, otherwise on exit the specified
// triangular region is overwritten.
//
// work is temporary storage, and lwork specifies the usable memory length. At minimum,
// lwork >= 3*n-1, and Dsyev will panic otherwise. The amount of blocking is
// limited by the usable length. If lwork == -1, instead of computing Dsyev the
// optimal work length is stored into work[0].
func (impl Implementation) Dsyev(jobz lapack.EVJob, uplo blas.Uplo, n int, a []float64, lda int, w, work []float64, lwork int) (ok bool) {
	switch {
	case jobz != lapack.EVNone && jobz != lapack.EVCompute:
		panic(badEVJob)
	case uplo != blas.Upper && uplo != blas.Lower:
		panic(badUplo)
	case n < 0:
		panic(nLT0)
	case lda < max(1, n):
		panic(badLdA)
	case lwork < max(1, 3*n-1) && lwork != -1:
		panic(badLWork)
	case len(work) < max(1, lwork):
		panic(shortWork)
	}

	// Quick return if possible.
	if n == 0 {
		return true
	}

	if lwork == -1 {
		return lapacke.Dsyev(byte(jobz), byte(uplo), n, a, lda, w, work, -1)
	}

	switch {
	case len(a) < (n-1)*lda+n:
		panic(shortA)
	case len(w) < n:
		panic(shortW)
	}

	return lapacke.Dsyev(byte(jobz), byte(uplo), n, a, lda, w, work, lwork)
}

// Dsytrd reduces a symmetric n×n matrix A to symmetric tridiagonal form by an
// orthogonal similarity transformation
//
//	Q^T * A * Q = T
//
// where Q is an orthonormal matrix and T is symmetric and tridiagonal.
//
// On entry, a contains the elements of the input matrix in the triangle specified
// by uplo. On exit, the diagonal and sub/super-diagonal are overwritten by the
// corresponding elements of the tridiagonal matrix T. The remaining elements in
// the triangle, along with the array tau, contain the data to construct Q as
// the product of elementary reflectors.
//
// If uplo == blas.Upper, Q is constructed with
//
//	Q = H_{n-2} * ... * H_1 * H_0
//
// where
//
//	H_i = I - tau_i * v * v^T
//
// v is constructed as v[i+1:n] = 0, v[i] = 1, v[0:i-1] is stored in A[0:i-1, i+1].
// The elements of A are
//
//	[ d   e  v1  v2  v3]
//	[     d   e  v2  v3]
//	[         d   e  v3]
//	[             d   e]
//	[                 e]
//
// If uplo == blas.Lower, Q is constructed with
//
//	Q = H_0 * H_1 * ... * H_{n-2}
//
// where
//
//	H_i = I - tau_i * v * v^T
//
// v is constructed as v[0:i+1] = 0, v[i+1] = 1, v[i+2:n] is stored in A[i+2:n, i].
// The elements of A are
//
//	[ d                ]
//	[ e   d            ]
//	[v0   e   d        ]
//	[v0  v1   e   d    ]
//	[v0  v1  v2   e   d]
//
// d must have length n, and e and tau must have length n-1. Dsytrd will panic if
// these conditions are not met.
//
// work is temporary storage, and lwork specifies the usable memory length. At minimum,
// lwork >= 1, and Dsytrd will panic otherwise. The amount of blocking is
// limited by the usable length.
// If lwork == -1, instead of computing Dsytrd the optimal work length is stored
// into work[0].
//
// Dsytrd is an internal routine. It is exported for testing purposes.
func (impl Implementation) Dsytrd(uplo blas.Uplo, n int, a []float64, lda int, d, e, tau, work []float64, lwork int) {
	switch {
	case uplo != blas.Upper && uplo != blas.Lower:
		panic(badUplo)
	case n < 0:
		panic(nLT0)
	case lda < max(1, n):
		panic(badLdA)
	case lwork < 1 && lwork != -1:
		panic(badLWork)
	case len(work) < max(1, lwork):
		panic(shortWork)
	}

	// Quick return if possible.
	if n == 0 {
		work[0] = 1
		return
	}

	if lwork == -1 {
		lapacke.Dsytrd(byte(uplo), n, a, lda, d, e, tau, work, -1)
		return
	}

	switch {
	case len(a) < (n-1)*lda+n:
		panic(shortA)
	case len(d) < n:
		panic(shortD)
	case len(e) < n-1:
		panic(shortE)
	case len(tau) < n-1:
		panic(shortTau)
	}

	lapacke.Dsytrd(byte(uplo), n, a, lda, d, e, tau, work, lwork)
}

// Dtbtrs solves a triangular system of the form
//
//	A * X = B   if trans == blas.NoTrans
//	Aᵀ * X = B  if trans == blas.Trans or blas.ConjTrans
//
// where A is an n×n triangular band matrix with kd super- or subdiagonals, and
// B is an n×nrhs matrix.
//
// Dtbtrs returns whether A is non-singular. If A is singular, no solution X is
// computed.
func (impl Implementation) Dtbtrs(uplo blas.Uplo, trans blas.Transpose, diag blas.Diag, n, kd, nrhs int, a []float64, lda int, b []float64, ldb int) (ok bool) {
	switch {
	case uplo != blas.Upper && uplo != blas.Lower:
		panic(badUplo)
	case trans != blas.NoTrans && trans != blas.Trans && trans != blas.ConjTrans:
		panic(badTrans)
	case diag != blas.NonUnit && diag != blas.Unit:
		panic(badDiag)
	case n < 0:
		panic(nLT0)
	case kd < 0:
		panic(kdLT0)
	case nrhs < 0:
		panic(nrhsLT0)
	case lda < kd+1:
		panic(badLdA)
	case ldb < max(1, nrhs):
		panic(badLdB)
	}

	// Quick return if possible.
	if n == 0 {
		return true
	}

	switch {
	case len(a) < (n-1)*lda+kd+1:
		panic(shortA)
	case len(b) < (n-1)*ldb+nrhs:
		panic(shortB)
	}

	ldaConv := n
	aConv := make([]float64, (kd+1)*ldaConv)
	bandTriToLapacke(uplo, n, kd, a, lda, aConv, ldaConv)
	return lapacke.Dtbtrs(byte(uplo), byte(trans), byte(diag), n, kd, nrhs, aConv, ldaConv, b, ldb)
}

// Dtrcon estimates the reciprocal of the condition number of a triangular matrix A.
// The condition number computed may be based on the 1-norm or the ∞-norm.
//
// work is a temporary data slice of length at least 3*n and Dtrcon will panic otherwise.
//
// iwork is a temporary data slice of length at least n and Dtrcon will panic otherwise.
func (impl Implementation) Dtrcon(norm lapack.MatrixNorm, uplo blas.Uplo, diag blas.Diag, n int, a []float64, lda int, work []float64, iwork []int) float64 {
	switch {
	case norm != lapack.MaxColumnSum && norm != lapack.MaxRowSum:
		panic(badNorm)
	case uplo != blas.Upper && uplo != blas.Lower:
		panic(badUplo)
	case diag != blas.NonUnit && diag != blas.Unit:
		panic(badDiag)
	case n < 0:
		panic(nLT0)
	case lda < max(1, n):
		panic(badLdA)
	}

	if n == 0 {
		return 1
	}

	switch {
	case len(a) < (n-1)*lda+n:
		panic(shortA)
	case len(work) < 3*n:
		panic(shortWork)
	case len(iwork) < n:
		panic(shortIWork)
	}

	rcond := []float64{0}
	_iwork := make([]int32, n)
	lapacke.Dtrcon(byte(norm), byte(uplo), byte(diag), n, a, lda, rcond, work, _iwork)
	return rcond[0]
}

// Dtrexc reorders the real Schur factorization of a n×n real matrix
//
//	A = Q*T*Q^T
//
// so that the diagonal block of T with row index ifst is moved to row ilst.
//
// On entry, T must be in Schur canonical form, that is, block upper triangular
// with 1×1 and 2×2 diagonal blocks; each 2×2 diagonal block has its diagonal
// elements equal and its off-diagonal elements of opposite sign.
//
// On return, T will be reordered by an orthogonal similarity transformation Z
// as Z^T*T*Z, and will be again in Schur canonical form.
//
// If compq is lapack.UpdateSchur, on return the matrix Q of Schur vectors will be
// updated by postmultiplying it with Z.
// If compq is lapack.None, the matrix Q is not referenced and will not be
// updated.
// For other values of compq Dtrexc will panic.
//
// ifst and ilst specify the reordering of the diagonal blocks of T. The block
// with row index ifst is moved to row ilst, by a sequence of transpositions
// between adjacent blocks.
//
// If ifst points to the second row of a 2×2 block, ifstOut will point to the
// first row, otherwise it will be equal to ifst.
//
// ilstOut will point to the first row of the block in its final position. If ok
// is true, ilstOut may differ from ilst by +1 or -1.
//
// It must hold that
//
//	0 <= ifst < n, and  0 <= ilst < n,
//
// otherwise Dtrexc will panic.
//
// If ok is false, two adjacent blocks were too close to swap because the
// problem is very ill-conditioned. T may have been partially reordered, and
// ilstOut will point to the first row of the block at the position to which it
// has been moved.
//
// work must have length at least n, otherwise Dtrexc will panic.
//
// Dtrexc is an internal routine. It is exported for testing purposes.
func (impl Implementation) Dtrexc(compq lapack.UpdateSchurComp, n int, t []float64, ldt int, q []float64, ldq int, ifst, ilst int, work []float64) (ifstOut, ilstOut int, ok bool) {
	switch {
	case compq != lapack.UpdateSchur && compq != lapack.UpdateSchurNone:
		panic(badUpdateSchurComp)
	case n < 0:
		panic(nLT0)
	case ldt < max(1, n):
		panic(badLdT)
	case ldq < 1, compq == lapack.UpdateSchur && ldq < n:
		panic(badLdQ)
	case (ifst < 0 || n <= ifst) && n > 0:
		panic(badIfst)
	case (ilst < 0 || n <= ilst) && n > 0:
		panic(badIlst)
	}

	// Quick return if possible.
	if n == 0 {
		return ifst, ilst, true
	}

	switch {
	case len(t) < (n-1)*ldt+n:
		panic(shortT)
	case compq == lapack.UpdateSchur && len(q) < (n-1)*ldq+n:
		panic(shortQ)
	case len(work) < n:
		panic(shortWork)
	}

	// Quick return if possible.
	if n == 1 {
		return ifst, ilst, true
	}

	ifst32 := []int32{int32(ifst + 1)}
	ilst32 := []int32{int32(ilst + 1)}
	ok = lapacke.Dtrexc(byte(compq), n, t, ldt, q, ldq, ifst32, ilst32, work)
	ifst = int(ifst32[0] - 1)
	ilst = int(ilst32[0] - 1)
	return ifst, ilst, ok
}

// Dtrtri computes the inverse of a triangular matrix, storing the result in place
// into a. This is the BLAS level 3 version of the algorithm which builds upon
// Dtrti2 to operate on matrix blocks instead of only individual columns.
//
// Dtrtri returns whether the matrix a is singular.
// If the matrix is singular, the inversion is not performed.
func (impl Implementation) Dtrtri(uplo blas.Uplo, diag blas.Diag, n int, a []float64, lda int) (ok bool) {
	switch {
	case uplo != blas.Upper && uplo != blas.Lower:
		panic(badUplo)
	case diag != blas.NonUnit && diag != blas.Unit:
		panic(badDiag)
	case n < 0:
		panic(nLT0)
	case lda < max(1, n):
		panic(badLdA)
	}

	if n == 0 {
		return true
	}

	if len(a) < (n-1)*lda+n {
		panic(shortA)
	}

	return lapacke.Dtrtri(byte(uplo), byte(diag), n, a, lda)
}

// Dtrtrs solves a triangular system of the form A * X = B or A^T * X = B.
// Dtrtrs returns whether the solve completed successfully.
// If A is singular, no solve is performed.
func (impl Implementation) Dtrtrs(uplo blas.Uplo, trans blas.Transpose, diag blas.Diag, n, nrhs int, a []float64, lda int, b []float64, ldb int) (ok bool) {
	switch {
	case uplo != blas.Upper && uplo != blas.Lower:
		panic(badUplo)
	case trans != blas.NoTrans && trans != blas.Trans && trans != blas.ConjTrans:
		panic(badTrans)
	case diag != blas.NonUnit && diag != blas.Unit:
		panic(badDiag)
	case n < 0:
		panic(nLT0)
	case nrhs < 0:
		panic(nrhsLT0)
	case lda < max(1, n):
		panic(badLdA)
	case ldb < max(1, nrhs):
		panic(badLdB)
	}

	if n == 0 {
		return true
	}

	switch {
	case len(a) < (n-1)*lda+n:
		panic(shortA)
	case len(b) < (n-1)*ldb+nrhs:
		panic(shortB)
	}

	return lapacke.Dtrtrs(byte(uplo), byte(trans), byte(diag), n, nrhs, a, lda, b, ldb)
}

// Dhseqr computes the eigenvalues of an n×n Hessenberg matrix H and,
// optionally, the matrices T and Z from the Schur decomposition
//
//	H = Z T Z^T,
//
// where T is an n×n upper quasi-triangular matrix (the Schur form), and Z is
// the n×n orthogonal matrix of Schur vectors.
//
// Optionally Z may be postmultiplied into an input orthogonal matrix Q so that
// this routine can give the Schur factorization of a matrix A which has been
// reduced to the Hessenberg form H by the orthogonal matrix Q:
//
//	A = Q H Q^T = (QZ) T (QZ)^T.
//
// If job == lapack.EigenvaluesOnly, only the eigenvalues will be computed.
// If job == lapack.EigenvaluesAndSchur, the eigenvalues and the Schur form T will
// be computed.
// For other values of job Dhseqr will panic.
//
// If compz == lapack.None, no Schur vectors will be computed and Z will not be
// referenced.
// If compz == lapack.HessEV, on return Z will contain the matrix of Schur
// vectors of H.
// If compz == lapack.OriginalEV, on entry z is assumed to contain the orthogonal
// matrix Q that is the identity except for the submatrix
// Q[ilo:ihi+1,ilo:ihi+1]. On return z will be updated to the product Q*Z.
//
// ilo and ihi determine the block of H on which Dhseqr operates. It is assumed
// that H is already upper triangular in rows and columns [0:ilo] and [ihi+1:n],
// although it will be only checked that the block is isolated, that is,
//
//	ilo == 0   or H[ilo,ilo-1] == 0,
//	ihi == n-1 or H[ihi+1,ihi] == 0,
//
// and Dhseqr will panic otherwise. ilo and ihi are typically set by a previous
// call to Dgebal, otherwise they should be set to 0 and n-1, respectively. It
// must hold that
//
//	0 <= ilo <= ihi < n,     if n > 0,
//	ilo == 0 and ihi == -1,  if n == 0.
//
// wr and wi must have length n.
//
// work must have length at least lwork and lwork must be at least max(1,n)
// otherwise Dhseqr will panic. The minimum lwork delivers very good and
// sometimes optimal performance, although lwork as large as 11*n may be
// required. On return, work[0] will contain the optimal value of lwork.
//
// If lwork is -1, instead of performing Dhseqr, the function only estimates the
// optimal workspace size and stores it into work[0]. Neither h nor z are
// accessed.
//
// unconverged indicates whether Dhseqr computed all the eigenvalues.
//
// If unconverged == 0, all the eigenvalues have been computed and their real
// and imaginary parts will be stored on return in wr and wi, respectively. If
// two eigenvalues are computed as a complex conjugate pair, they are stored in
// consecutive elements of wr and wi, say the i-th and (i+1)th, with wi[i] > 0
// and wi[i+1] < 0.
//
// If unconverged == 0 and job == lapack.EigenvaluesAndSchur, on return H will
// contain the upper quasi-triangular matrix T from the Schur decomposition (the
// Schur form). 2×2 diagonal blocks (corresponding to complex conjugate pairs of
// eigenvalues) will be returned in standard form, with
//
//	H[i,i] == H[i+1,i+1],
//
// and
//
//	H[i+1,i]*H[i,i+1] < 0.
//
// The eigenvalues will be stored in wr and wi in the same order as on the
// diagonal of the Schur form returned in H, with
//
//	wr[i] = H[i,i],
//
// and, if H[i:i+2,i:i+2] is a 2×2 diagonal block,
//
//	wi[i]   = sqrt(-H[i+1,i]*H[i,i+1]),
//	wi[i+1] = -wi[i].
//
// If unconverged == 0 and job == lapack.EigenvaluesOnly, the contents of h
// on return is unspecified.
//
// If unconverged > 0, some eigenvalues have not converged, and the blocks
// [0:ilo] and [unconverged:n] of wr and wi will contain those eigenvalues which
// have been successfully computed. Failures are rare.
//
// If unconverged > 0 and job == lapack.EigenvaluesOnly, on return the
// remaining unconverged eigenvalues are the eigenvalues of the upper Hessenberg
// matrix H[ilo:unconverged,ilo:unconverged].
//
// If unconverged > 0 and job == lapack.EigenvaluesAndSchur, then on
// return
//
//	(initial H) U = U (final H),   (*)
//
// where U is an orthogonal matrix. The final H is upper Hessenberg and
// H[unconverged:ihi+1,unconverged:ihi+1] is upper quasi-triangular.
//
// If unconverged > 0 and compz == lapack.OriginalEV, then on return
//
//	(final Z) = (initial Z) U,
//
// where U is the orthogonal matrix in (*) regardless of the value of job.
//
// If unconverged > 0 and compz == lapack.InitZ, then on return
//
//	(final Z) = U,
//
// where U is the orthogonal matrix in (*) regardless of the value of job.
//
// References:
//
//	[1] R. Byers. LAPACK 3.1 xHSEQR: Tuning and Implementation Notes on the
//	    Small Bulge Multi-Shift QR Algorithm with Aggressive Early Deflation.
//	    LAPACK Working Note 187 (2007)
//	    URL: http://www.netlib.org/lapack/lawnspdf/lawn187.pdf
//	[2] K. Braman, R. Byers, R. Mathias. The Multishift QR Algorithm. Part I:
//	    Maintaining Well-Focused Shifts and Level 3 Performance. SIAM J. Matrix
//	    Anal. Appl. 23(4) (2002), pp. 929—947
//	    URL: http://dx.doi.org/10.1137/S0895479801384573
//	[3] K. Braman, R. Byers, R. Mathias. The Multishift QR Algorithm. Part II:
//	    Aggressive Early Deflation. SIAM J. Matrix Anal. Appl. 23(4) (2002), pp. 948—973
//	    URL: http://dx.doi.org/10.1137/S0895479801384585
//
// Dhseqr is an internal routine. It is exported for testing purposes.
func (impl Implementation) Dhseqr(job lapack.SchurJob, compz lapack.SchurComp, n, ilo, ihi int, h []float64, ldh int, wr, wi []float64, z []float64, ldz int, work []float64, lwork int) (unconverged int) {
	wantz := compz == lapack.SchurHess || compz == lapack.SchurOrig

	switch {
	case job != lapack.EigenvaluesOnly && job != lapack.EigenvaluesAndSchur:
		panic(badSchurJob)
	case compz != lapack.SchurNone && compz != lapack.SchurHess && compz != lapack.SchurOrig:
		panic(badSchurComp)
	case n < 0:
		panic(nLT0)
	case ilo < 0 || max(0, n-1) < ilo:
		panic(badIlo)
	case ihi < min(ilo, n-1) || n <= ihi:
		panic(badIhi)
	case ldh < max(1, n):
		panic(badLdH)
	case ldz < 1, wantz && ldz < n:
		panic(badLdZ)
	case lwork < max(1, n) && lwork != -1:
		panic(badLWork)
	case len(work) < max(1, lwork):
		panic(shortWork)
	}

	// Quick return if possible.
	if n == 0 {
		work[0] = 1
		return 0
	}

	// Quick return in case of a workspace query.
	if lwork == -1 {
		return lapacke.Dhseqr(byte(job), byte(compz), n, ilo+1, ihi+1, h, ldh, wr, wi, z, ldz, work, -1)
	}

	switch {
	case len(h) < (n-1)*ldh+n:
		panic(shortH)
	case wantz && len(z) < (n-1)*ldz+n:
		panic(shortZ)
	case len(wr) < n:
		panic(shortWr)
	case len(wi) < n:
		panic(shortWi)
	}

	return lapacke.Dhseqr(byte(job), byte(compz), n, ilo+1, ihi+1, h, ldh, wr, wi, z, ldz, work, lwork)
}

// Dgeev computes the eigenvalues and, optionally, the left and/or right
// eigenvectors for an n×n real nonsymmetric matrix A.
//
// The right eigenvector v_j of A corresponding to an eigenvalue λ_j
// is defined by
//
//	A v_j = λ_j v_j,
//
// and the left eigenvector u_j corresponding to an eigenvalue λ_j is defined by
//
//	u_j^H A = λ_j u_j^H,
//
// where u_j^H is the conjugate transpose of u_j.
//
// On return, A will be overwritten and the left and right eigenvectors will be
// stored, respectively, in the columns of the n×n matrices VL and VR in the
// same order as their eigenvalues. If the j-th eigenvalue is real, then
//
//	u_j = VL[:,j],
//	v_j = VR[:,j],
//
// and if it is not real, then j and j+1 form a complex conjugate pair and the
// eigenvectors can be recovered as
//
//	u_j     = VL[:,j] + i*VL[:,j+1],
//	u_{j+1} = VL[:,j] - i*VL[:,j+1],
//	v_j     = VR[:,j] + i*VR[:,j+1],
//	v_{j+1} = VR[:,j] - i*VR[:,j+1].
//
// where i is the imaginary unit. The computed eigenvectors are normalized to
// have Euclidean norm equal to 1 and largest component real.
//
// Left eigenvectors will be computed only if jobvl == lapack.ComputeLeftEV,
// otherwise jobvl must be lapack.None. Right eigenvectors will be computed
// only if jobvr == lapack.ComputeRightEV, otherwise jobvr must be lapack.None.
// For other values of jobvl and jobvr Dgeev will panic.
//
// wr and wi contain the real and imaginary parts, respectively, of the computed
// eigenvalues. Complex conjugate pairs of eigenvalues appear consecutively with
// the eigenvalue having the positive imaginary part first.
// wr and wi must have length n, and Dgeev will panic otherwise.
//
// work must have length at least lwork and lwork must be at least max(1,4*n) if
// the left or right eigenvectors are computed, and at least max(1,3*n) if no
// eigenvectors are computed. For good performance, lwork must generally be
// larger.  On return, optimal value of lwork will be stored in work[0].
//
// If lwork == -1, instead of performing Dgeev, the function only calculates the
// optimal vaule of lwork and stores it into work[0].
//
// On return, first is the index of the first valid eigenvalue. If first == 0,
// all eigenvalues and eigenvectors have been computed. If first is positive,
// Dgeev failed to compute all the eigenvalues, no eigenvectors have been
// computed and wr[first:] and wi[first:] contain those eigenvalues which have
// converged.
func (impl Implementation) Dgeev(jobvl lapack.LeftEVJob, jobvr lapack.RightEVJob, n int, a []float64, lda int, wr, wi []float64, vl []float64, ldvl int, vr []float64, ldvr int, work []float64, lwork int) (first int) {
	wantvl := jobvl == lapack.LeftEVCompute
	wantvr := jobvr == lapack.RightEVCompute
	var minwrk int
	if wantvl || wantvr {
		minwrk = max(1, 4*n)
	} else {
		minwrk = max(1, 3*n)
	}
	switch {
	case jobvl != lapack.LeftEVCompute && jobvl != lapack.LeftEVNone:
		panic(badLeftEVJob)
	case jobvr != lapack.RightEVCompute && jobvr != lapack.RightEVNone:
		panic(badRightEVJob)
	case n < 0:
		panic(nLT0)
	case lda < max(1, n):
		panic(badLdA)
	case ldvl < 1 || (ldvl < n && wantvl):
		panic(badLdVL)
	case ldvr < 1 || (ldvr < n && wantvr):
		panic(badLdVR)
	case lwork < minwrk && lwork != -1:
		panic(badLWork)
	case len(work) < lwork:
		panic(shortWork)
	}

	// Quick return if possible.
	if n == 0 {
		work[0] = 1
		return 0
	}

	// TODO(vladimir-ch): The calls to lapacke.Dgeev below require max(n,ldvl) and
	// max(n,ldvr) because the leading dimension checks in
	// LAPACKE_dgeev_work are too strict. This has been reported in
	//  https://github.com/Reference-LAPACK/lapack/issues/327
	// Remove the calls to max if and when the upstream fixes this.

	if lwork == -1 {
		lapacke.Dgeev(byte(jobvl), byte(jobvr), n, a, lda, wr, wi, vl, max(n, ldvl), vr, max(n, ldvr), work, -1)
		return 0
	}

	switch {
	case len(a) < (n-1)*lda+n:
		panic(shortA)
	case len(wr) != n:
		panic(badLenWr)
	case len(wi) != n:
		panic(badLenWi)
	case len(vl) < (n-1)*ldvl+n && wantvl:
		panic(shortVL)
	case len(vr) < (n-1)*ldvr+n && wantvr:
		panic(shortVR)
	}

	return lapacke.Dgeev(byte(jobvl), byte(jobvr), n, a, lda, wr, wi, vl, max(n, ldvl), vr, max(n, ldvr), work, lwork)
}

// Dtgsja computes the generalized singular value decomposition (GSVD)
// of two real upper triangular or trapezoidal matrices A and B.
//
// A and B have the following forms, which may be obtained by the
// preprocessing subroutine Dggsvp from a general m×n matrix A and p×n
// matrix B:
//
//	          n-k-l  k    l
//	A =    k [  0   A12  A13 ] if m-k-l >= 0;
//	       l [  0    0   A23 ]
//	   m-k-l [  0    0    0  ]
//
//	          n-k-l  k    l
//	A =    k [  0   A12  A13 ] if m-k-l < 0;
//	     m-k [  0    0   A23 ]
//
//	          n-k-l  k    l
//	B =    l [  0    0   B13 ]
//	     p-l [  0    0    0  ]
//
// where the k×k matrix A12 and l×l matrix B13 are non-singular
// upper triangular. A23 is l×l upper triangular if m-k-l >= 0,
// otherwise A23 is (m-k)×l upper trapezoidal.
//
// On exit,
//
//	U^T*A*Q = D1*[ 0 R ], V^T*B*Q = D2*[ 0 R ],
//
// where U, V and Q are orthogonal matrices.
// R is a non-singular upper triangular matrix, and D1 and D2 are
// diagonal matrices, which are of the following structures:
//
// If m-k-l >= 0,
//
//	                  k  l
//	     D1 =     k [ I  0 ]
//	              l [ 0  C ]
//	          m-k-l [ 0  0 ]
//
//	                k  l
//	     D2 = l   [ 0  S ]
//	          p-l [ 0  0 ]
//
//	             n-k-l  k    l
//	[ 0 R ] = k [  0   R11  R12 ] k
//	          l [  0    0   R22 ] l
//
// where
//
//	C = diag( alpha_k, ... , alpha_{k+l} ),
//	S = diag( beta_k,  ... , beta_{k+l} ),
//	C^2 + S^2 = I.
//
// R is stored in
//
//	A[0:k+l, n-k-l:n]
//
// on exit.
//
// If m-k-l < 0,
//
//	               k m-k k+l-m
//	    D1 =   k [ I  0    0  ]
//	         m-k [ 0  C    0  ]
//
//	                 k m-k k+l-m
//	    D2 =   m-k [ 0  S    0  ]
//	         k+l-m [ 0  0    I  ]
//	           p-l [ 0  0    0  ]
//
//	               n-k-l  k   m-k  k+l-m
//	[ 0 R ] =    k [ 0    R11  R12  R13 ]
//	           m-k [ 0     0   R22  R23 ]
//	         k+l-m [ 0     0    0   R33 ]
//
// where
//
//	C = diag( alpha_k, ... , alpha_m ),
//	S = diag( beta_k,  ... , beta_m ),
//	C^2 + S^2 = I.
//
//	R = [ R11 R12 R13 ] is stored in A[1:m, n-k-l+1:n]
//	    [  0  R22 R23 ]
//
// and R33 is stored in
//
//	B[m-k:l, n+m-k-l:n] on exit.
//
// The computation of the orthogonal transformation matrices U, V or Q
// is optional. These matrices may either be formed explicitly, or they
// may be post-multiplied into input matrices U1, V1, or Q1.
//
// Dtgsja essentially uses a variant of Kogbetliantz algorithm to reduce
// min(l,m-k)×l triangular or trapezoidal matrix A23 and l×l
// matrix B13 to the form:
//
//	U1^T*A13*Q1 = C1*R1; V1^T*B13*Q1 = S1*R1,
//
// where U1, V1 and Q1 are orthogonal matrices. C1 and S1 are diagonal
// matrices satisfying
//
//	C1^2 + S1^2 = I,
//
// and R1 is an l×l non-singular upper triangular matrix.
//
// jobU, jobV and jobQ are options for computing the orthogonal matrices. The behavior
// is as follows
//
//	jobU == lapack.GSVDU        Compute orthogonal matrix U
//	jobU == lapack.GSVDUnit     Use unit-initialized matrix
//	jobU == lapack.GSVDNone     Do not compute orthogonal matrix.
//
// The behavior is the same for jobV and jobQ with the exception that instead of
// lapack.GSVDU these accept lapack.GSVDV and lapack.GSVDQ respectively.
// The matrices U, V and Q must be m×m, p×p and n×n respectively unless the
// relevant job parameter is lapack.GSVDNone.
//
// k and l specify the sub-blocks in the input matrices A and B:
//
//	A23 = A[k:min(k+l,m), n-l:n) and B13 = B[0:l, n-l:n]
//
// of A and B, whose GSVD is going to be computed by Dtgsja.
//
// tola and tolb are the convergence criteria for the Jacobi-Kogbetliantz
// iteration procedure. Generally, they are the same as used in the preprocessing
// step, for example,
//
//	tola = max(m, n)*norm(A)*eps,
//	tolb = max(p, n)*norm(B)*eps,
//
// where eps is the machine epsilon.
//
// work must have length at least 2*n, otherwise Dtgsja will panic.
//
// alpha and beta must have length n or Dtgsja will panic. On exit, alpha and
// beta contain the generalized singular value pairs of A and B
//
//	alpha[0:k] = 1,
//	beta[0:k]  = 0,
//
// if m-k-l >= 0,
//
//	alpha[k:k+l] = diag(C),
//	beta[k:k+l]  = diag(S),
//
// if m-k-l < 0,
//
//	alpha[k:m]= C, alpha[m:k+l]= 0
//	beta[k:m] = S, beta[m:k+l] = 1.
//
// if k+l < n,
//
//	alpha[k+l:n] = 0 and
//	beta[k+l:n]  = 0.
//
// On exit, A[n-k:n, 0:min(k+l,m)] contains the triangular matrix R or part of R
// and if necessary, B[m-k:l, n+m-k-l:n] contains a part of R.
//
// Dtgsja returns whether the routine converged and the number of iteration cycles
// that were run.
//
// Dtgsja is an internal routine. It is exported for testing purposes.
func (impl Implementation) Dtgsja(jobU, jobV, jobQ lapack.GSVDJob, m, p, n, k, l int, a []float64, lda int, b []float64, ldb int, tola, tolb float64, alpha, beta, u []float64, ldu int, v []float64, ldv int, q []float64, ldq int, work []float64) (cycles int, ok bool) {
	initu := jobU == lapack.GSVDUnit
	wantu := initu || jobU == lapack.GSVDU

	initv := jobV == lapack.GSVDUnit
	wantv := initv || jobV == lapack.GSVDV

	initq := jobQ == lapack.GSVDUnit
	wantq := initq || jobQ == lapack.GSVDQ

	switch {
	case !initu && !wantu && jobU != lapack.GSVDNone:
		panic(badGSVDJob + "U")
	case !initv && !wantv && jobV != lapack.GSVDNone:
		panic(badGSVDJob + "V")
	case !initq && !wantq && jobQ != lapack.GSVDNone:
		panic(badGSVDJob + "Q")
	case m < 0:
		panic(mLT0)
	case p < 0:
		panic(pLT0)
	case n < 0:
		panic(nLT0)

	case lda < max(1, n):
		panic(badLdA)
	case len(a) < (m-1)*lda+n:
		panic(shortA)

	case ldb < max(1, n):
		panic(badLdB)
	case len(b) < (p-1)*ldb+n:
		panic(shortB)

	case len(alpha) != n:
		panic(badLenAlpha)
	case len(beta) != n:
		panic(badLenBeta)

	case ldu < 1, wantu && ldu < m:
		panic(badLdU)
	case wantu && len(u) < (m-1)*ldu+m:
		panic(shortU)

	case ldv < 1, wantv && ldv < p:
		panic(badLdV)
	case wantv && len(v) < (p-1)*ldv+p:
		panic(shortV)

	case ldq < 1, wantq && ldq < n:
		panic(badLdQ)
	case wantq && len(q) < (n-1)*ldq+n:
		panic(shortQ)

	case len(work) < 2*n:
		panic(shortWork)
	}

	ncycle := []int32{0}
	ok = lapacke.Dtgsja(byte(jobU), byte(jobV), byte(jobQ), m, p, n, k, l, a, lda, b, ldb, tola, tolb, alpha, beta, u, ldu, v, ldv, q, ldq, work, ncycle)
	return int(ncycle[0]), ok
}

func min(m, n int) int {
	if m < n {
		return m
	}
	return n
}

func max(m, n int) int {
	if m < n {
		return n
	}
	return m
}
