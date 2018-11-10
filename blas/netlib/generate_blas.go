// Copyright ©2016 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build ignore

// generate_blas creates a blas.go file from the provided C header file
// with optionally added documentation from the documentation package.
package main

import (
	"bytes"
	"fmt"
	"go/ast"
	"go/format"
	"io/ioutil"
	"log"
	"strings"
	"text/template"

	"github.com/cznic/cc"

	"gonum.org/v1/netlib/internal/binding"
)

const (
	header        = "cblas.h"
	documentation = "../../../gonum/blas/gonum"
	target        = "blas.go"

	typ = "Implementation"

	prefix = "cblas_"

	warning = "Float32 implementations are autogenerated and not directly tested."
)

const (
	cribDocs      = true
	elideRepeat   = true
	noteOrigin    = true
	separateFuncs = false
)

var skip = map[string]bool{
	"cblas_errprn":    true,
	"cblas_srotg":     true,
	"cblas_srotmg":    true,
	"cblas_srotm":     true,
	"cblas_drotg":     true,
	"cblas_drotmg":    true,
	"cblas_drotm":     true,
	"cblas_crotg":     true,
	"cblas_zrotg":     true,
	"cblas_cdotu_sub": true,
	"cblas_cdotc_sub": true,
	"cblas_zdotu_sub": true,
	"cblas_zdotc_sub": true,

	// ATLAS extensions.
	"cblas_csrot": true,
	"cblas_zdrot": true,
}

var cToGoType = map[string]string{
	"int":    "int",
	"float":  "float32",
	"double": "float64",
}

var blasEnums = map[string]*template.Template{
	"CBLAS_ORDER":     template.Must(template.New("order").Parse("order")),
	"CBLAS_DIAG":      template.Must(template.New("diag").Parse("blas.Diag")),
	"CBLAS_TRANSPOSE": template.Must(template.New("trans").Parse("blas.Transpose")),
	"CBLAS_UPLO":      template.Must(template.New("uplo").Parse("blas.Uplo")),
	"CBLAS_SIDE":      template.Must(template.New("side").Parse("blas.Side")),
}

var cgoEnums = map[string]*template.Template{
	"CBLAS_ORDER":     template.Must(template.New("order").Parse("C.enum_CBLAS_ORDER(rowMajor)")),
	"CBLAS_DIAG":      template.Must(template.New("diag").Parse("C.enum_CBLAS_DIAG({{.}})")),
	"CBLAS_TRANSPOSE": template.Must(template.New("trans").Parse("C.enum_CBLAS_TRANSPOSE({{.}})")),
	"CBLAS_UPLO":      template.Must(template.New("uplo").Parse("C.enum_CBLAS_UPLO({{.}})")),
	"CBLAS_SIDE":      template.Must(template.New("side").Parse("C.enum_CBLAS_SIDE({{.}})")),
}

var cgoTypes = map[binding.TypeKey]*template.Template{
	{Kind: cc.Float, IsPointer: true}: template.Must(template.New("float*").Parse(
		`(*C.float)({{if eq . "alpha" "beta"}}&{{else}}_{{end}}{{.}})`,
	)),
	{Kind: cc.Double, IsPointer: true}: template.Must(template.New("double*").Parse(
		`(*C.double)({{if eq . "alpha" "beta"}}&{{else}}_{{end}}{{.}})`,
	)),
	{Kind: cc.Void, IsPointer: true}: template.Must(template.New("void*").Parse(
		`unsafe.Pointer({{if eq . "alpha" "beta"}}&{{else}}_{{end}}{{.}})`,
	)),
}

var (
	complex64Type = map[binding.TypeKey]*template.Template{
		{Kind: cc.Void, IsPointer: true}: template.Must(template.New("void*").Parse(
			`{{if eq . "alpha" "beta"}}complex64{{else}}[]complex64{{end}}`,
		))}

	complex128Type = map[binding.TypeKey]*template.Template{
		{Kind: cc.Void, IsPointer: true}: template.Must(template.New("void*").Parse(
			`{{if eq . "alpha" "beta"}}complex128{{else}}[]complex128{{end}}`,
		))}
)

var names = map[string]string{
	"uplo":   "ul",
	"trans":  "t",
	"transA": "tA",
	"transB": "tB",
	"side":   "s",
	"diag":   "d",
}

func shorten(n string) string {
	s, ok := names[n]
	if ok {
		return s
	}
	return n
}

func main() {
	decls, err := binding.Declarations(header)
	if err != nil {
		log.Fatal(err)
	}
	var docs map[string]map[string][]*ast.Comment
	if cribDocs {
		docs, err = binding.DocComments(documentation)
		if err != nil {
			log.Fatal(err)
		}
	}

	var buf bytes.Buffer

	h, err := template.New("handwritten").Parse(handwritten)
	if err != nil {
		log.Fatal(err)
	}
	err = h.Execute(&buf, header)
	if err != nil {
		log.Fatal(err)
	}

	var n int
	for _, d := range decls {
		if !strings.HasPrefix(d.Name, prefix) || skip[d.Name] {
			continue
		}
		if n != 0 && (separateFuncs || cribDocs) {
			buf.WriteByte('\n')
		}
		n++
		goSignature(&buf, d, docs[typ])
		if noteOrigin {
			fmt.Fprintf(&buf, "\t// declared at %s %s %s ...\n\n", d.Position(), d.Return, d.Name)
		}
		parameterChecks(&buf, d, parameterCheckRules)
		buf.WriteByte('\t')
		cgoCall(&buf, d)
		buf.WriteString("}\n")
	}

	b, err := format.Source(buf.Bytes())
	if err != nil {
		log.Fatal(err)
	}
	err = ioutil.WriteFile(target, b, 0664)
	if err != nil {
		log.Fatal(err)
	}
}

func goSignature(buf *bytes.Buffer, d binding.Declaration, docs map[string][]*ast.Comment) {
	blasName := strings.TrimPrefix(d.Name, prefix)
	goName := binding.UpperCaseFirst(blasName)

	if docs != nil {
		if doc, ok := docs[goName]; ok {
			if strings.Contains(doc[len(doc)-1].Text, warning) {
				doc = doc[:len(doc)-2]
			}
			for _, c := range doc {
				buf.WriteString(c.Text)
				buf.WriteByte('\n')
			}
		}
	}

	parameters := d.Parameters()

	var voidPtrType map[binding.TypeKey]*template.Template
	for _, p := range parameters {
		if p.Kind() == cc.Ptr && p.Elem().Kind() == cc.Void {
			switch {
			case blasName[0] == 'c', blasName[1] == 'c' && blasName[0] != 'z':
				voidPtrType = complex64Type
			case blasName[0] == 'z', blasName[1] == 'z':
				voidPtrType = complex128Type
			}
			break
		}
	}

	fmt.Fprintf(buf, "func (%s) %s(", typ, goName)
	c := 0
	for i, p := range parameters {
		if p.Kind() == cc.Enum && binding.GoTypeForEnum(p.Type(), "", blasEnums) == "order" {
			continue
		}
		if c != 0 {
			buf.WriteString(", ")
		}
		c++

		n := shorten(binding.LowerCaseFirst(p.Name()))
		var this, next string

		if p.Kind() == cc.Enum {
			this = binding.GoTypeForEnum(p.Type(), n, blasEnums)
		} else {
			this = binding.GoTypeFor(p.Type(), n, voidPtrType)
		}

		if elideRepeat && i < len(parameters)-1 && p.Type().Kind() == parameters[i+1].Type().Kind() {
			p := parameters[i+1]
			n := shorten(binding.LowerCaseFirst(p.Name()))
			if p.Kind() == cc.Enum {
				next = binding.GoTypeForEnum(p.Type(), n, blasEnums)
			} else {
				next = binding.GoTypeFor(p.Type(), n, voidPtrType)
			}
		}
		if next == this {
			buf.WriteString(n)
		} else {
			fmt.Fprintf(buf, "%s %s", n, this)
		}
	}
	if d.Return.Kind() != cc.Void {
		fmt.Fprintf(buf, ") %s {\n", cToGoType[d.Return.String()])
	} else {
		buf.WriteString(") {\n")
	}
}

func parameterChecks(buf *bytes.Buffer, d binding.Declaration, rules []func(*bytes.Buffer, binding.Declaration, binding.Parameter) bool) {
	done := make(map[int]bool)
	for i, r := range rules {
		for _, p := range d.Parameters() {
			if done[i] {
				continue
			}
			done[i] = r(buf, d, p)
		}
	}
}

func cgoCall(buf *bytes.Buffer, d binding.Declaration) {
	if d.Return.Kind() != cc.Void {
		fmt.Fprintf(buf, "return %s(", cToGoType[d.Return.String()])
	}
	fmt.Fprintf(buf, "C.%s(", d.Name)
	for i, p := range d.Parameters() {
		if i != 0 {
			buf.WriteString(", ")
		}
		if p.Type().Kind() == cc.Enum {
			buf.WriteString(binding.CgoConversionForEnum(shorten(binding.LowerCaseFirst(p.Name())), p.Type(), cgoEnums))
		} else {
			buf.WriteString(binding.CgoConversionFor(shorten(binding.LowerCaseFirst(p.Name())), p.Type(), cgoTypes))
		}
	}
	if d.Return.Kind() != cc.Void {
		buf.WriteString(")")
	}
	buf.WriteString(")\n")
}

var parameterCheckRules = []func(*bytes.Buffer, binding.Declaration, binding.Parameter) bool{
	trans,
	uplo,
	diag,
	side,
	shape,
	leadingDim,
	zeroInc,

	noWork,

	apShape,
	sidedShape,
	mvShape,
	rkShape,
	gemmShape,
	scalShape,
	amaxShape,
	nrmSumShape,
	vectorShape,
	othersShape,

	address,
}

func amaxShape(buf *bytes.Buffer, d binding.Declaration, p binding.Parameter) bool {
	switch d.Name {
	case "cblas_isamax", "cblas_idamax", "cblas_icamax", "cblas_izamax":
	default:
		return true
	}

	if d.CParameters[len(d.CParameters)-1] != p.Parameter {
		return false // Come back later.
	}

	fmt.Fprint(buf, `	if (n-1)*incX >= len(x) {
		panic("blas: x index out of range")
	}
`)
	return true
}

func apShape(buf *bytes.Buffer, _ binding.Declaration, p binding.Parameter) bool {
	n := binding.LowerCaseFirst(p.Name())
	if n != "ap" {
		return false
	}
	fmt.Fprint(buf, `	if n*(n+1)/2 > len(ap) {
		panic("blas: index of ap out of range")
	}
`)
	return true
}

func diag(buf *bytes.Buffer, _ binding.Declaration, p binding.Parameter) bool {
	if p.Name() != "Diag" {
		return false
	}
	fmt.Fprint(buf, `	switch d {
	case blas.NonUnit:
		d = C.CblasNonUnit
	case blas.Unit:
		d = C.CblasUnit
	default:
		panic("blas: illegal diagonal")
	}
`)
	return true
}

func gemmShape(buf *bytes.Buffer, d binding.Declaration, p binding.Parameter) bool {
	switch d.Name {
	case "cblas_sgemm", "cblas_dgemm", "cblas_cgemm", "cblas_zgemm":
	default:
		return true
	}

	if d.CParameters[len(d.CParameters)-1] != p.Parameter {
		return false // Come back later.
	}

	fmt.Fprint(buf, `	if lda*(rowA-1)+colA > len(a) {
		panic("blas: index of a out of range")
	}
	if ldb*(rowB-1)+colB > len(b) {
		panic("blas: index of b out of range")
	}
	if ldc*(m-1)+n > len(c) {
		panic("blas: index of c out of range")
	}
`)
	return true
}

func mvShape(buf *bytes.Buffer, d binding.Declaration, p binding.Parameter) bool {
	switch d.Name {
	case "cblas_sgbmv", "cblas_dgbmv", "cblas_cgbmv", "cblas_zgbmv",
		"cblas_sgemv", "cblas_dgemv", "cblas_cgemv", "cblas_zgemv":
	default:
		return true
	}

	if d.CParameters[len(d.CParameters)-1] != p.Parameter {
		return false // Come back later.
	}

	fmt.Fprint(buf, `	var lenX, lenY int
	if tA == C.CblasNoTrans {
		lenX, lenY = n, m
	} else {
		lenX, lenY = m, n
	}
	if (incX > 0 && (lenX-1)*incX >= len(x)) || (incX < 0 && (1-lenX)*incX >= len(x)) {
		panic("blas: x index out of range")
	}
	if (incY > 0 && (lenY-1)*incY >= len(y)) || (incY < 0 && (1-lenY)*incY >= len(y)) {
		panic("blas: y index out of range")
	}
`)
	return true
}

func noWork(buf *bytes.Buffer, d binding.Declaration, p binding.Parameter) bool {
	if d.CParameters[len(d.CParameters)-1] != p.Parameter {
		return false // Come back later.
	}

	switch d.Name {
	case "cblas_snrm2", "cblas_dnrm2", "cblas_scnrm2", "cblas_dznrm2",
		"cblas_sasum", "cblas_dasum", "cblas_scasum", "cblas_dzasum":
		fmt.Fprint(buf, `	if n == 0 || incX < 0 {
		return 0
	}
`)
		return true

	case "cblas_sscal", "cblas_dscal", "cblas_cscal", "cblas_zscal", "cblas_csscal", "cblas_zdscal":
		fmt.Fprint(buf, `	if n == 0 || incX < 0 {
		return
	}
`)
		return true

	case "cblas_isamax", "cblas_idamax", "cblas_icamax", "cblas_izamax":
		fmt.Fprint(buf, `	if n == 0 || incX < 0 {
		return -1
	}
`)
		return true
	}

	var value string
	switch d.Return.String() {
	case "float", "double":
		value = " 0"
	}
	var hasM bool
	for _, p := range d.Parameters() {
		if shorten(binding.LowerCaseFirst(p.Name())) == "m" {
			hasM = true
		}
	}
	if !hasM {
		fmt.Fprintf(buf, `	if n == 0 {
		return%s
	}
`, value)
	} else {
		fmt.Fprintf(buf, `	if m == 0 || n == 0 {
		return
	}
`)
	}

	return true
}

func nrmSumShape(buf *bytes.Buffer, d binding.Declaration, p binding.Parameter) bool {
	switch d.Name {
	case "cblas_snrm2", "cblas_dnrm2", "cblas_scnrm2", "cblas_dznrm2",
		"cblas_sasum", "cblas_dasum", "cblas_scasum", "cblas_dzasum":
	default:
		return true
	}

	if d.CParameters[len(d.CParameters)-1] != p.Parameter {
		return false // Come back later.
	}

	fmt.Fprint(buf, `	if (n-1)*incX >= len(x) {
		panic("blas: x index out of range")
	}
`)
	return true
}

func rkShape(buf *bytes.Buffer, d binding.Declaration, p binding.Parameter) bool {
	switch d.Name {
	case "cblas_ssyrk", "cblas_dsyrk", "cblas_csyrk", "cblas_zsyrk",
		"cblas_ssyr2k", "cblas_dsyr2k", "cblas_csyr2k", "cblas_zsyr2k",
		"cblas_cherk", "cblas_zherk", "cblas_cher2k", "cblas_zher2k":
	default:
		return true
	}

	if d.CParameters[len(d.CParameters)-1] != p.Parameter {
		return false // Come back later.
	}

	has := make(map[string]bool)
	for _, p := range d.Parameters() {
		if p.Kind() != cc.Ptr {
			continue
		}
		has[shorten(binding.LowerCaseFirst(p.Name()))] = true
	}
	for _, label := range []string{"a", "b"} {
		if has[label] {
			fmt.Fprintf(buf, `	if ld%[1]s*(row-1)+col > len(%[1]s) {
		panic("blas: index of %[1]s out of range")
	}
`, label)
		}
	}
	if has["c"] {
		fmt.Fprint(buf, `	if ldc*(n-1)+n > len(c) {
		panic("blas: index of c out of range")
	}
`)
	}

	return true
}

func scalShape(buf *bytes.Buffer, d binding.Declaration, p binding.Parameter) bool {
	switch d.Name {
	case "cblas_sscal", "cblas_dscal", "cblas_cscal", "cblas_zscal", "cblas_csscal", "cblas_zdscal":
	default:
		return true
	}

	if d.CParameters[len(d.CParameters)-1] != p.Parameter {
		return false // Come back later.
	}

	fmt.Fprint(buf, `	if (n-1)*incX >= len(x) {
		panic("blas: x index out of range")
	}
`)
	return true
}

func shape(buf *bytes.Buffer, _ binding.Declaration, p binding.Parameter) bool {
	switch n := binding.LowerCaseFirst(p.Name()); n {
	case "m", "n", "k", "kL", "kU":
		fmt.Fprintf(buf, `	if %[1]s < 0 {
		panic("blas: %[1]s < 0")
	}
`, n)
		return false
	}
	return false
}

func side(buf *bytes.Buffer, _ binding.Declaration, p binding.Parameter) bool {
	if p.Name() != "Side" {
		return false
	}
	fmt.Fprint(buf, `	switch s {
	case blas.Left:
		s = C.CblasLeft
	case blas.Right:
		s = C.CblasRight
	default:
		panic("blas: illegal side")
	}
`)
	return true
}

func sidedShape(buf *bytes.Buffer, d binding.Declaration, p binding.Parameter) bool {
	var hasS, hasA, hasB, hasC bool
	for _, p := range d.Parameters() {
		switch shorten(binding.LowerCaseFirst(p.Name())) {
		case "s":
			hasS = true
		case "a":
			hasA = true
		case "b":
			hasB = true
		case "c":
			hasC = true
		}
	}
	if !hasS {
		return true
	}

	if d.CParameters[len(d.CParameters)-1] != p.Parameter {
		return false // Come back later.
	}

	if hasA && hasB {
		fmt.Fprint(buf, `	if lda*(k-1)+k > len(a) {
		panic("blas: index of a out of range")
	}
	if ldb*(m-1)+n > len(b) {
		panic("blas: index of b out of range")
	}
`)
	} else {
		return true
	}
	if hasC {
		fmt.Fprint(buf, `	if ldc*(m-1)+n > len(c) {
		panic("blas: index of c out of range")
	}
`)
	}

	return true
}

func trans(buf *bytes.Buffer, d binding.Declaration, p binding.Parameter) bool {
	switch n := shorten(binding.LowerCaseFirst(p.Name())); n {
	case "t", "tA", "tB":
		switch {
		case strings.HasPrefix(d.Name, "cblas_ch"), strings.HasPrefix(d.Name, "cblas_zh"):
			fmt.Fprintf(buf, `	switch %[1]s {
	case blas.NoTrans:
		%[1]s = C.CblasNoTrans
	case blas.ConjTrans:
		%[1]s = C.CblasConjTrans
	default:
		panic("blas: illegal transpose")
	}
`, n)
		case strings.HasPrefix(d.Name, "cblas_cs"), strings.HasPrefix(d.Name, "cblas_zs"):
			fmt.Fprintf(buf, `	switch %[1]s {
	case blas.NoTrans:
		%[1]s = C.CblasNoTrans
	case blas.Trans:
		%[1]s = C.CblasTrans
	default:
		panic("blas: illegal transpose")
	}
`, n)
		default:
			fmt.Fprintf(buf, `	switch %[1]s {
	case blas.NoTrans:
		%[1]s = C.CblasNoTrans
	case blas.Trans:
		%[1]s = C.CblasTrans
	case blas.ConjTrans:
		%[1]s = C.CblasConjTrans
	default:
		panic("blas: illegal transpose")
	}
`, n)
		}
	}
	return false
}

func uplo(buf *bytes.Buffer, _ binding.Declaration, p binding.Parameter) bool {
	if p.Name() != "Uplo" {
		return false
	}
	fmt.Fprint(buf, `	switch ul {
	case blas.Upper:
		ul = C.CblasUpper
	case blas.Lower:
		ul = C.CblasLower
	default:
		panic("blas: illegal triangle")
	}
`)
	return true
}

func vectorShape(buf *bytes.Buffer, d binding.Declaration, p binding.Parameter) bool {
	switch d.Name {
	case "cblas_sgbmv", "cblas_dgbmv", "cblas_cgbmv", "cblas_zgbmv",
		"cblas_sgemv", "cblas_dgemv", "cblas_cgemv", "cblas_zgemv",
		"cblas_sscal", "cblas_dscal", "cblas_cscal", "cblas_zscal", "cblas_csscal", "cblas_zdscal",
		"cblas_isamax", "cblas_idamax", "cblas_icamax", "cblas_izamax",
		"cblas_snrm2", "cblas_dnrm2", "cblas_scnrm2", "cblas_dznrm2",
		"cblas_sasum", "cblas_dasum", "cblas_scasum", "cblas_dzasum":
		return true
	}

	var hasN, hasM, hasIncX, hasIncY bool
	for _, p := range d.Parameters() {
		switch shorten(binding.LowerCaseFirst(p.Name())) {
		case "n":
			hasN = true
		case "m":
			hasM = true
		case "incX":
			hasIncX = true
		case "incY":
			hasIncY = true
		}
	}
	if !hasN && !hasM {
		return true
	}

	if d.CParameters[len(d.CParameters)-1] != p.Parameter {
		return false // Come back later.
	}

	var label string
	if hasM {
		label = "m"
	} else {
		label = "n"
	}
	if hasIncX {
		fmt.Fprintf(buf, `	if (incX > 0 && (%[1]s-1)*incX >= len(x)) || (incX < 0 && (1-%[1]s)*incX >= len(x)) {
		panic("blas: x index out of range")
	}
`, label)
	}
	if hasIncY {
		fmt.Fprint(buf, `	if (incY > 0 && (n-1)*incY >= len(y)) || (incY < 0 && (1-n)*incY >= len(y)) {
		panic("blas: y index out of range")
	}
`)
	}
	return true
}

func leadingDim(buf *bytes.Buffer, d binding.Declaration, p binding.Parameter) bool {
	pname := binding.LowerCaseFirst(p.Name())
	if !strings.HasPrefix(pname, "ld") {
		return false
	}

	if pname == "ldc" {
		// C matrix has always n columns.
		fmt.Fprintf(buf, `	if ldc < max(1, n) {
		panic("blas: bad ldc")
	}
`)
		return false
	}

	has := make(map[string]bool)
	for _, p := range d.Parameters() {
		has[shorten(binding.LowerCaseFirst(p.Name()))] = true
	}

	switch d.Name {
	case "cblas_sgemm", "cblas_dgemm", "cblas_cgemm", "cblas_zgemm":
		if pname == "lda" {
			fmt.Fprint(buf, `	var rowA, colA, rowB, colB int
	if tA == C.CblasNoTrans {
		rowA, colA = m, k
	} else {
		rowA, colA = k, m
	}
	if tB == C.CblasNoTrans {
		rowB, colB = k, n
	} else {
		rowB, colB = n, k
	}
	if lda < max(1, colA) {
		panic("blas: bad lda")
	}
`)
		} else {
			fmt.Fprint(buf, `	if ldb < max(1, colB) {
		panic("blas: bad ldb")
	}
`)
		}
		return false

	case "cblas_ssyrk", "cblas_dsyrk", "cblas_csyrk", "cblas_zsyrk",
		"cblas_ssyr2k", "cblas_dsyr2k", "cblas_csyr2k", "cblas_zsyr2k",
		"cblas_cherk", "cblas_zherk", "cblas_cher2k", "cblas_zher2k":
		if pname == "lda" {
			fmt.Fprint(buf, `	var row, col int
	if t == C.CblasNoTrans {
		row, col = n, k
	} else {
		row, col = k, n
	}
`)
		}
		fmt.Fprintf(buf, `	if %[1]s < max(1, col) {
		panic("blas: bad %[1]s")
	}
`, pname)
		return false

	case "cblas_sgbmv", "cblas_dgbmv", "cblas_cgbmv", "cblas_zgbmv":
		fmt.Fprintf(buf, `	if lda < kL+kU+1 {
		panic("blas: bad lda")
	}
`)
		return false
	}

	switch {
	case has["k"]:
		// cblas_stbmv cblas_dtbmv cblas_ctbmv cblas_ztbmv
		// cblas_stbsv cblas_dtbsv cblas_ctbsv cblas_ztbsv
		// cblas_ssbmv cblas_dsbmv cblas_chbmv cblas_zhbmv
		fmt.Fprintf(buf, `	if lda < k+1 {
		panic("blas: bad lda")
	}
`)
	case has["s"] && pname == "lda":
		// cblas_ssymm cblas_dsymm cblas_csymm cblas_zsymm
		// cblas_strmm cblas_dtrmm cblas_ctrmm cblas_ztrmm
		// cblas_strsm cblas_dtrsm cblas_ctrsm cblas_ztrsm
		// cblas_chemm cblas_zhemm
		fmt.Fprintf(buf, `	var k int
	if s == C.CblasLeft {
		k = m
	} else {
		k = n
	}
	if lda < max(1, k) {
		panic("blas: bad lda")
	}
`)
	default:
		fmt.Fprintf(buf, `	if %[1]s < max(1, n) {
		panic("blas: bad %[1]s")
	}
`, pname)
	}
	return false
}

func zeroInc(buf *bytes.Buffer, _ binding.Declaration, p binding.Parameter) bool {
	switch n := binding.LowerCaseFirst(p.Name()); n {
	case "incX":
		fmt.Fprintf(buf, `	if incX == 0 {
		panic("blas: zero x index increment")
	}
`)
	case "incY":
		fmt.Fprintf(buf, `	if incY == 0 {
		panic("blas: zero y index increment")
	}
`)
	}
	return false
}

func othersShape(buf *bytes.Buffer, d binding.Declaration, p binding.Parameter) bool {
	switch d.Name {
	case "cblas_sgemm", "cblas_dgemm", "cblas_cgemm", "cblas_zgemm",
		"cblas_ssyrk", "cblas_dsyrk", "cblas_csyrk", "cblas_zsyrk",
		"cblas_ssyr2k", "cblas_dsyr2k", "cblas_csyr2k", "cblas_zsyr2k",
		"cblas_cherk", "cblas_zherk", "cblas_cher2k", "cblas_zher2k":
		return true
	}

	has := make(map[string]bool)
	for _, p := range d.Parameters() {
		has[shorten(binding.LowerCaseFirst(p.Name()))] = true
	}
	if !has["a"] || has["s"] {
		return true
	}

	if d.CParameters[len(d.CParameters)-1] != p.Parameter {
		return false // Come back later.
	}

	switch {
	case has["kL"] && has["kU"]:
		fmt.Fprintf(buf, `	if lda*(min(m, n+kL)-1)+kL+kU+1 > len(a) {
		panic("blas: index of a out of range")
	}
`)
	case has["m"]:
		fmt.Fprintf(buf, `	if lda*(m-1)+n > len(a) {
		panic("blas: index of a out of range")
	}
`)
	case has["k"]:
		fmt.Fprintf(buf, `	if lda*(n-1)+k+1 > len(a) {
		panic("blas: index of a out of range")
	}
`)
	default:
		fmt.Fprintf(buf, `	if lda*(n-1)+n > len(a) {
		panic("blas: index of a out of range")
	}
`)
	}

	return true
}

var addrTypes = map[string]string{
	"char":   "byte",
	"int":    "int32",
	"float":  "float32",
	"double": "float64",
}

func address(buf *bytes.Buffer, d binding.Declaration, p binding.Parameter) bool {
	n := shorten(binding.LowerCaseFirst(p.Name()))
	blasName := strings.TrimPrefix(d.Name, prefix)
	switch n {
	case "a", "b", "c", "ap", "x", "y":
	default:
		return false
	}
	if p.Type().Kind() == cc.Ptr {
		t := addrTypes[strings.TrimPrefix(p.Type().Element().String(), "const ")]
		if t == "" {
			switch {
			case blasName[0] == 'c', blasName[1] == 'c' && blasName[0] != 'z':
				t = "complex64"
			case blasName[0] == 'z', blasName[1] == 'z':
				t = "complex128"
			}
		}
		fmt.Fprintf(buf, `	var _%[1]s *%[2]s
        if len(%[1]s) > 0 {
                _%[1]s = &%[1]s[0]
        }
`, n, t)
	}
	return false
}

const handwritten = `// Code generated by "go generate gonum.org/v1/netlib/blas/netlib" from {{.}}; DO NOT EDIT.

// Copyright ©2014 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package netlib

/*
#cgo CFLAGS: -g -O2
#include "{{.}}"
*/
import "C"

import (
	"unsafe"

	"gonum.org/v1/gonum/blas"
)

// Type check assertions:
var (
	_ blas.Float32    = Implementation{}
	_ blas.Float64    = Implementation{}
	_ blas.Complex64  = Implementation{}
	_ blas.Complex128 = Implementation{}
)

// Type order is used to specify the matrix storage format. We still interact with
// an API that allows client calls to specify order, so this is here to document that fact.
type order int

const rowMajor order = C.CblasRowMajor

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

type Implementation struct{}

// Special cases...

type srotmParams struct {
	flag float32
	h    [4]float32
}

type drotmParams struct {
	flag float64
	h    [4]float64
}

func (Implementation) Srotg(a float32, b float32) (c float32, s float32, r float32, z float32) {
	C.cblas_srotg((*C.float)(&a), (*C.float)(&b), (*C.float)(&c), (*C.float)(&s))
	return c, s, a, b
}
func (Implementation) Srotmg(d1 float32, d2 float32, b1 float32, b2 float32) (p blas.SrotmParams, rd1 float32, rd2 float32, rb1 float32) {
	var pi srotmParams
	C.cblas_srotmg((*C.float)(&d1), (*C.float)(&d2), (*C.float)(&b1), C.float(b2), (*C.float)(unsafe.Pointer(&pi)))
	return blas.SrotmParams{Flag: blas.Flag(pi.flag), H: pi.h}, d1, d2, b1
}
func (Implementation) Srotm(n int, x []float32, incX int, y []float32, incY int, p blas.SrotmParams) {
	if n < 0 {
		panic("blas: n < 0")
	}
        var _x *float32
	if len(x) > 0 {
		_x = &x[0]
	}
	if incX == 0 {
		panic("blas: zero x index increment")
	}
        var _y *float32
	if len(y) > 0 {
		_y = &y[0]
	}
	if incY == 0 {
		panic("blas: zero y index increment")
	}
	if (incX > 0 && (n-1)*incX >= len(x)) || (incX < 0 && (1-n)*incX >= len(x)) {
		panic("blas: x index out of range")
	}
	if (incY > 0 && (n-1)*incY >= len(y)) || (incY < 0 && (1-n)*incY >= len(y)) {
		panic("blas: y index out of range")
	}
	if p.Flag < blas.Identity || p.Flag > blas.Diagonal {
		panic("blas: illegal blas.Flag value")
	}
	if n == 0 {
		return
	}
	pi := srotmParams{
		flag: float32(p.Flag),
		h:    p.H,
	}
	C.cblas_srotm(C.int(n), (*C.float)(_x), C.int(incX), (*C.float)(_y), C.int(incY), (*C.float)(unsafe.Pointer(&pi)))
}
func (Implementation) Drotg(a float64, b float64) (c float64, s float64, r float64, z float64) {
	C.cblas_drotg((*C.double)(&a), (*C.double)(&b), (*C.double)(&c), (*C.double)(&s))
	return c, s, a, b
}
func (Implementation) Drotmg(d1 float64, d2 float64, b1 float64, b2 float64) (p blas.DrotmParams, rd1 float64, rd2 float64, rb1 float64) {
	var pi drotmParams
	C.cblas_drotmg((*C.double)(&d1), (*C.double)(&d2), (*C.double)(&b1), C.double(b2), (*C.double)(unsafe.Pointer(&pi)))
	return blas.DrotmParams{Flag: blas.Flag(pi.flag), H: pi.h}, d1, d2, b1
}
func (Implementation) Drotm(n int, x []float64, incX int, y []float64, incY int, p blas.DrotmParams) {
	if n < 0 {
		panic("blas: n < 0")
	}
        var _x *float64
	if len(x) > 0 {
		_x = &x[0]
	}
	if incX == 0 {
		panic("blas: zero x index increment")
	}
        var _y *float64
	if len(y) > 0 {
		_y = &y[0]
	}
	if incY == 0 {
		panic("blas: zero y index increment")
	}
	if (incX > 0 && (n-1)*incX >= len(x)) || (incX < 0 && (1-n)*incX >= len(x)) {
		panic("blas: x index out of range")
	}
	if (incY > 0 && (n-1)*incY >= len(y)) || (incY < 0 && (1-n)*incY >= len(y)) {
		panic("blas: y index out of range")
	}
	if p.Flag < blas.Identity || p.Flag > blas.Diagonal {
		panic("blas: illegal blas.Flag value")
	}
	if n == 0 {
		return
	}
	pi := drotmParams{
		flag: float64(p.Flag),
		h:    p.H,
	}
	C.cblas_drotm(C.int(n), (*C.double)(_x), C.int(incX), (*C.double)(_y), C.int(incY), (*C.double)(unsafe.Pointer(&pi)))
}
func (Implementation) Cdotu(n int, x []complex64, incX int, y []complex64, incY int) (dotu complex64) {
	if n < 0 {
		panic("blas: n < 0")
	}
        var _x *complex64
	if len(x) > 0 {
		_x = &x[0]
	}
	if incX == 0 {
		panic("blas: zero x index increment")
	}
        var _y *complex64
	if len(y) > 0 {
		_y = &y[0]
	}
	if incY == 0 {
		panic("blas: zero y index increment")
	}
	if (incX > 0 && (n-1)*incX >= len(x)) || (incX < 0 && (1-n)*incX >= len(x)) {
		panic("blas: x index out of range")
	}
	if (incY > 0 && (n-1)*incY >= len(y)) || (incY < 0 && (1-n)*incY >= len(y)) {
		panic("blas: y index out of range")
	}
	if n == 0 {
		return 0
	}
	C.cblas_cdotu_sub(C.int(n), unsafe.Pointer(_x), C.int(incX), unsafe.Pointer(_y), C.int(incY), unsafe.Pointer(&dotu))
	return dotu
}
func (Implementation) Cdotc(n int, x []complex64, incX int, y []complex64, incY int) (dotc complex64) {
	if n < 0 {
		panic("blas: n < 0")
	}
        var _x *complex64
	if len(x) > 0 {
		_x = &x[0]
	}
	if incX == 0 {
		panic("blas: zero x index increment")
	}
        var _y *complex64
	if len(y) > 0 {
		_y = &y[0]
	}
	if incY == 0 {
		panic("blas: zero y index increment")
	}
	if (incX > 0 && (n-1)*incX >= len(x)) || (incX < 0 && (1-n)*incX >= len(x)) {
		panic("blas: x index out of range")
	}
	if (incY > 0 && (n-1)*incY >= len(y)) || (incY < 0 && (1-n)*incY >= len(y)) {
		panic("blas: y index out of range")
	}
	if n == 0 {
		return 0
	}
	C.cblas_cdotc_sub(C.int(n), unsafe.Pointer(_x), C.int(incX), unsafe.Pointer(_y), C.int(incY), unsafe.Pointer(&dotc))
	return dotc
}
func (Implementation) Zdotu(n int, x []complex128, incX int, y []complex128, incY int) (dotu complex128) {
	if n < 0 {
		panic("blas: n < 0")
	}
        var _x *complex128
	if len(x) > 0 {
		_x = &x[0]
	}
	if incX == 0 {
		panic("blas: zero x index increment")
	}
        var _y *complex128
	if len(y) > 0 {
		_y = &y[0]
	}
	if incY == 0 {
		panic("blas: zero y index increment")
	}
	if (incX > 0 && (n-1)*incX >= len(x)) || (incX < 0 && (1-n)*incX >= len(x)) {
		panic("blas: x index out of range")
	}
	if (incY > 0 && (n-1)*incY >= len(y)) || (incY < 0 && (1-n)*incY >= len(y)) {
		panic("blas: y index out of range")
	}
	if n == 0 {
		return 0
	}
	C.cblas_zdotu_sub(C.int(n), unsafe.Pointer(_x), C.int(incX), unsafe.Pointer(_y), C.int(incY), unsafe.Pointer(&dotu))
	return dotu
}
func (Implementation) Zdotc(n int, x []complex128, incX int, y []complex128, incY int) (dotc complex128) {
	if n < 0 {
		panic("blas: n < 0")
	}
        var _x *complex128
	if len(x) > 0 {
		_x = &x[0]
	}
	if incX == 0 {
		panic("blas: zero x index increment")
	}
        var _y *complex128
	if len(y) > 0 {
		_y = &y[0]
	}
	if incY == 0 {
		panic("blas: zero y index increment")
	}
	if (incX > 0 && (n-1)*incX >= len(x)) || (incX < 0 && (1-n)*incX >= len(x)) {
		panic("blas: x index out of range")
	}
	if (incY > 0 && (n-1)*incY >= len(y)) || (incY < 0 && (1-n)*incY >= len(y)) {
		panic("blas: y index out of range")
	}
	if n == 0 {
		return 0
	}
	C.cblas_zdotc_sub(C.int(n), unsafe.Pointer(_x), C.int(incX), unsafe.Pointer(_y), C.int(incY), unsafe.Pointer(&dotc))
	return dotc
}

// Generated cases ...

`
