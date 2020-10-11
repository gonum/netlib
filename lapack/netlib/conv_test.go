// Copyright ©2019 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package netlib

import (
	"fmt"
	"testing"

	"golang.org/x/exp/rand"

	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/floats"
)

func TestConvBandTri(t *testing.T) {
	for ti, test := range []struct {
		uplo  blas.Uplo
		n, kd int
		a, b  []float64
	}{
		{
			uplo: blas.Upper,
			n:    6,
			kd:   2,
			a: []float64{
				1, 2, 3, // 1. row
				4, 5, 6,
				7, 8, 9,
				10, 11, 12,
				13, 14, -1,
				15, -1, -1, // 6. row
			},
			b: []float64{
				-1, -1, 3, 6, 9, 12, // 2. super-diagonal
				-1, 2, 5, 8, 11, 14,
				1, 4, 7, 10, 13, 15, // main diagonal
			},
		},
		{
			uplo: blas.Lower,
			n:    6,
			kd:   2,
			a: []float64{
				-1, -1, 1, // 1. row
				-1, 2, 3,
				4, 5, 6,
				7, 8, 9,
				10, 11, 12,
				13, 14, 15, // 6. row
			},
			b: []float64{
				1, 3, 6, 9, 12, 15, // main diagonal
				2, 5, 8, 11, 14, -1,
				4, 7, 10, 13, -1, -1, // 2. sub-diagonal
			},
		},
	} {
		uplo := test.uplo
		n := test.n
		kd := test.kd
		name := fmt.Sprintf("Case %v (uplo=%c,n=%v,kd=%v)", ti, uplo, n, kd)

		a := make([]float64, len(test.a))
		copy(a, test.a)
		lda := kd + 1

		got := make([]float64, len(test.b))
		for i := range got {
			got[i] = -1
		}
		ldb := max(1, n)

		bandTriToLapacke(uplo, n, kd, a, lda, got, ldb)
		if !floats.Equal(test.a, a) {
			t.Errorf("%v: unexpected modification of A in conversion to LAPACKE row-major", name)
		}
		if !floats.Equal(test.b, got) {
			t.Errorf("%v: unexpected conversion to LAPACKE row-major;\ngot  %v\nwant %v", name, got, test.b)
		}

		b := make([]float64, len(test.b))
		copy(b, test.b)

		got = make([]float64, len(test.a))
		for i := range got {
			got[i] = -1
		}

		bandTriToGonum(uplo, n, kd, b, ldb, got, lda)
		if !floats.Equal(test.b, b) {
			t.Errorf("%v: unexpected modification of B in conversion to Gonum row-major", name)
		}
		if !floats.Equal(test.a, got) {
			t.Errorf("%v: unexpected conversion to Gonum row-major;\ngot  %v\nwant %v", name, got, test.b)
		}
	}

	rnd := rand.New(rand.NewSource(1))
	for _, n := range []int{0, 1, 2, 3, 4, 5, 10} {
		for _, kd := range []int{0, (n + 1) / 4, (3*n - 1) / 4, (5*n + 1) / 4} {
			for _, uplo := range []blas.Uplo{blas.Upper, blas.Lower} {
				for _, ldextra := range []int{0, 3} {
					name := fmt.Sprintf("uplo=%c,n=%v,kd=%v", uplo, n, kd)

					lda := kd + 1 + ldextra
					a := make([]float64, n*lda)
					for i := range a {
						a[i] = rnd.NormFloat64()
					}
					aCopy := make([]float64, len(a))
					copy(aCopy, a)

					ldb := max(1, n) + ldextra
					b := make([]float64, (kd+1)*ldb)
					for i := range b {
						b[i] = rnd.NormFloat64()
					}

					bandTriToLapacke(uplo, n, kd, a, lda, b, ldb)
					bandTriToGonum(uplo, n, kd, b, ldb, a, lda)

					if !floats.Equal(a, aCopy) {
						t.Errorf("%v: conversion does not roundtrip", name)
					}
				}
			}
		}
	}
}
