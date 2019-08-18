// Copyright ©2019 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package conv

import (
	"testing"

	"golang.org/x/exp/rand"

	"gonum.org/v1/gonum/floats"
)

func TestDpb(t *testing.T) {
	for ti, test := range []struct {
		uplo  byte
		n, kd int
		a, b  []float64
	}{
		{
			uplo: 'U',
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
				-1, -1, 1, // 1. column
				-1, 2, 4,
				3, 5, 7,
				6, 8, 10,
				9, 11, 13,
				12, 14, 15, // 6. column
			},
		},
		{
			uplo: 'L',
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
				1, 2, 4, // 1. column
				3, 5, 7,
				6, 8, 10,
				9, 11, 13,
				12, 14, -1,
				15, -1, -1, // 6. column
			},
		},
	} {
		uplo := test.uplo
		n := test.n
		kd := test.kd
		lda := kd + 1

		a := make([]float64, len(test.a))
		copy(a, test.a)

		b := make([]float64, len(test.b))
		copy(b, test.b)

		got := make([]float64, len(a))
		for i := range got {
			got[i] = -1
		}
		DpbToColMajor(uplo, n, kd, a, lda, got, lda)
		if !floats.Equal(test.b, got) {
			t.Errorf("Case %v (uplo=%v,n=%v,kd=%v): unexpected conversion to column-major;\ngot  %v\nwant %v",
				ti, string(uplo), n, kd, got, test.b)
		}

		for i := range got {
			got[i] = -1
		}
		DpbToRowMajor(uplo, n, kd, b, lda, got, lda)
		if !floats.Equal(test.a, got) {
			t.Errorf("Case %v (uplo=%v,n=%v,kd=%v): unexpected conversion to row-major;\ngot  %v\nwant %v",
				ti, string(uplo), n, kd, got, test.b)
		}
	}

	rnd := rand.New(rand.NewSource(1))
	for _, n := range []int{0, 1, 2, 3, 4, 5, 10} {
		for _, kd := range []int{0, (n + 1) / 4, (3*n - 1) / 4, (5*n + 1) / 4} {
			for _, uplo := range []byte{'U', 'L'} {
				for _, lda := range []int{kd + 1, kd + 1 + 7} {
					a := make([]float64, n*lda)
					for i := range a {
						a[i] = rnd.NormFloat64()
					}
					aCopy := make([]float64, len(a))
					copy(aCopy, a)

					ldb := lda
					b := make([]float64, ldb*n)
					for i := range b {
						b[i] = rnd.NormFloat64()
					}

					DpbToColMajor(uplo, n, kd, a, lda, b, ldb)
					DpbToRowMajor(uplo, n, kd, b, ldb, a, lda)

					if !floats.Equal(a, aCopy) {
						t.Errorf("uplo=%v,n=%v,kd=%v: conversion does not roundtrip", string(uplo), n, kd)
					}
				}
			}
		}
	}
}
