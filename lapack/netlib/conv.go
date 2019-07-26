// Copyright ©2019 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package netlib

import "gonum.org/v1/gonum/blas"

// convDpbToLapacke converts a symmetric band matrix A in CBLAS row-major layout
// to LAPACKE row-major layout and stores the result in B.
//
// For example, when n = 6, kd = 2 and uplo == 'U', convDpbToLapacke converts
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
//  b = [* * a02 a13 a24 a35 * a01 a12 a23 a34 a45 a00 a11 a22 a33 a44 a55]
//
// When n = 6, kd = 2 and uplo == 'L', convDpbToLapacke converts
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
//  b = [a00 a11 a22 a33 a44 a55 a10 a21 a32 a43 a54 * a20 a31 a42 a53 * * ]
//
// In these example elements marked as * are not referenced.
func convDpbToLapacke(uplo blas.Uplo, n, kd int, a []float64, lda int, b []float64, ldb int) {
	if uplo == blas.Upper {
		for i := 0; i < n; i++ {
			for jb := 0; jb < min(n-i, kd+1); jb++ {
				j := i + jb // Column index in the full matrix
				b[(kd-jb)*ldb+j] = a[i*lda+jb]
			}
		}
	} else {
		for i := 0; i < n; i++ {
			for jb := max(0, kd-i); jb < kd+1; jb++ {
				j := i - kd + jb // Column index in the full matrix
				b[(kd-jb)*ldb+j] = a[i*lda+jb]
			}
		}
	}
}

// convDpbToGonum converts a symmetric band matrix A in LAPACKE row-major layout
// to CBLAS row-major layout and stores the result in B. In other words, it
// performs the inverse conversion to convDpbToLapacke.
func convDpbToGonum(uplo blas.Uplo, n, kd int, a []float64, lda int, b []float64, ldb int) {
	if uplo == blas.Upper {
		for j := 0; j < n; j++ {
			for ib := max(0, kd-j); ib < kd+1; ib++ {
				i := j - kd + ib // Row index in the full matrix
				b[i*ldb+kd-ib] = a[ib*lda+j]
			}
		}
	} else {
		for j := 0; j < n; j++ {
			for ib := 0; ib < min(n-j, kd+1); ib++ {
				i := j + ib // Row index in the full matrix
				b[i*ldb+kd-ib] = a[ib*lda+j]
			}
		}
	}
}
