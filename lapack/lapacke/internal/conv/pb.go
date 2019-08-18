// Copyright ©2019 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package conv

// DpbToColMajor converts a symmetric band matrix A in CBLAS row-major layout
// to FORTRAN column-major layout and stores the result in B.
//
// For example, when n = 6, kd = 2 and uplo == 'U', DpbToColMajor
// converts
//  A = a00  a01  a02
//      a11  a12  a13
//      a22  a23  a24
//      a33  a34  a35
//      a44  a45   *
//      a55   *    *
// stored in a slice as
//  a = [a00 a01 a02 a11 a12 a13 a22 a23 a24 a33 a34 a35 a44 a45 * a55 * *]
// to
//  B =  *   *  a02 a13 a24 a35
//       *  a01 a12 a23 a34 a45
//      a00 a11 a22 a33 a44 a55
// stored in a slice as
//  b = [* * a00 * a01 a11 a02 a12 a22 a13 a23 a33 a24 a34 a44 a35 a45 a55]
//
// When n = 6, kd = 2 and uplo == 'L', DpbToColMajor converts
//  A =  *    *   a00
//       *   a10  a11
//      a20  a21  a22
//      a31  a32  a33
//      a42  a43  a44
//      a53  a54  a55
// stored in a slice as
//  a = [* * a00 * a10 a11 a20 a21 a22 a31 a32 a33 a42 a43 a44 a53 a54 a55]
// to
//  B = a00 a11 a22 a33 a44 a55
//      a10 a21 a32 a43 a54  *
//      a20 a31 a42 a53  *   *
// stored in a slice as
//  b = [a00 a10 a20 a11 a21 a31 a22 a32 a42 a33 a43 a53 a44 a54 * a55 * *]
//
// In these example elements marked as * are not referenced.
func DpbToColMajor(uplo byte, n, kd int, a []float64, lda int, b []float64, ldb int) {
	if uplo == 'U' {
		for i := 0; i < n; i++ {
			for jb := 0; jb < min(n-i, kd+1); jb++ {
				j := i + jb // Column index in the full matrix
				b[kd-jb+j*ldb] = a[i*lda+jb]
			}
		}
	} else {
		for i := 0; i < n; i++ {
			for jb := max(0, kd-i); jb < kd+1; jb++ {
				j := i - kd + jb // Column index in the full matrix
				b[kd-jb+j*ldb] = a[i*lda+jb]
			}
		}
	}
}

// DpbToRowMajor converts a symmetric band matrix A in FORTRAN column-major
// layout to CBLAS row-major layout and stores the result in B. In other words,
// it performs the inverse conversion to DpbToColMajor.
func DpbToRowMajor(uplo byte, n, kd int, a []float64, lda int, b []float64, ldb int) {
	if uplo == 'U' {
		for j := 0; j < n; j++ {
			for ib := max(0, kd-j); ib < kd+1; ib++ {
				i := j - kd + ib // Row index in the full matrix
				b[i*ldb+kd-ib] = a[ib+j*lda]
			}
		}
	} else {
		for j := 0; j < n; j++ {
			for ib := 0; ib < min(n-j, kd+1); ib++ {
				i := j + ib // Row index in the full matrix
				b[i*ldb+kd-ib] = a[ib+j*lda]
			}
		}
	}
}
