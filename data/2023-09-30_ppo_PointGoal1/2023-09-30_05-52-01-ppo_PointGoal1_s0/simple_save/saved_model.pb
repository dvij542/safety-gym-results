Şź
Ó$§$
:
Add
x"T
y"T
z"T"
Ttype:
2	
W
AddN
inputs"T*N
sum"T"
Nint(0"!
Ttype:
2	
î
	ApplyAdam
var"T	
m"T	
v"T
beta1_power"T
beta2_power"T
lr"T

beta1"T

beta2"T
epsilon"T	
grad"T
out"T" 
Ttype:
2	"
use_lockingbool( "
use_nesterovbool( 
x
Assign
ref"T

value"T

output_ref"T"	
Ttype"
validate_shapebool("
use_lockingbool(
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
~
BiasAddGrad
out_backprop"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
R
BroadcastGradientArgs
s0"T
s1"T
r0"T
r1"T"
Ttype0:
2	
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
S
DynamicStitch
indices*N
data"T*N
merged"T"
Nint(0"	
Ttype
,
Exp
x"T
y"T"
Ttype:

2
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
?
FloorDiv
x"T
y"T
z"T"
Ttype:
2	
9
FloorMod
x"T
y"T
z"T"
Ttype:

2	
=
Greater
x"T
y"T
z
"
Ttype:
2	
.
Identity

input"T
output"T"	
Ttype
?
	LessEqual
x"T
y"T
z
"
Ttype:
2	
,
Log
x"T
y"T"
Ttype:

2
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
;
Maximum
x"T
y"T
z"T"
Ttype:

2	

Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
;
Minimum
x"T
y"T
z"T"
Ttype:

2	
=
Mul
x"T
y"T
z"T"
Ttype:
2	
.
Neg
x"T
y"T"
Ttype:

2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
X
PlaceholderWithDefault
input"dtype
output"dtype"
dtypetype"
shapeshape
6
Pow
x"T
y"T
z"T"
Ttype:

2	

Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
PyFunc
input2Tin
output2Tout"
tokenstring"
Tin
list(type)("
Tout
list(type)(

RandomStandardNormal

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	
a
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx"
Tidxtype0:	
2	
>
RealDiv
x"T
y"T
z"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename

SplitV

value"T
size_splits"Tlen
	split_dim
output"T*	num_split"
	num_splitint(0"	
Ttype"
Tlentype0	:
2	
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
:
Sub
x"T
y"T
z"T"
Ttype:
2	

Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
-
Tanh
x"T
y"T"
Ttype:

2
:
TanhGrad
y"T
dy"T
z"T"
Ttype:

2
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	
s

VariableV2
ref"dtype"
shapeshape"
dtypetype"
	containerstring "
shared_namestring 
&
	ZerosLike
x"T
y"T"	
Ttype"serve*1.13.12b'v1.13.1-0-g6612da8951'ďÉ
n
PlaceholderPlaceholder*
dtype0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙<*
shape:˙˙˙˙˙˙˙˙˙<
p
Placeholder_1Placeholder*
dtype0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
shape:˙˙˙˙˙˙˙˙˙
h
Placeholder_2Placeholder*
shape:˙˙˙˙˙˙˙˙˙*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0
h
Placeholder_3Placeholder*
dtype0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
shape:˙˙˙˙˙˙˙˙˙
h
Placeholder_4Placeholder*
dtype0*
shape:˙˙˙˙˙˙˙˙˙*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
h
Placeholder_5Placeholder*
shape:˙˙˙˙˙˙˙˙˙*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0
h
Placeholder_6Placeholder*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
shape:˙˙˙˙˙˙˙˙˙*
dtype0
N
Placeholder_7Placeholder*
shape: *
dtype0*
_output_shapes
: 
N
Placeholder_8Placeholder*
dtype0*
_output_shapes
: *
shape: 
Ľ
0pi/dense/kernel/Initializer/random_uniform/shapeConst*
_output_shapes
:*
valueB"<      *
dtype0*"
_class
loc:@pi/dense/kernel

.pi/dense/kernel/Initializer/random_uniform/minConst*"
_class
loc:@pi/dense/kernel*
dtype0*
_output_shapes
: *
valueB
 *ž

.pi/dense/kernel/Initializer/random_uniform/maxConst*
_output_shapes
: *"
_class
loc:@pi/dense/kernel*
valueB
 *>*
dtype0
ď
8pi/dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform0pi/dense/kernel/Initializer/random_uniform/shape*
_output_shapes
:	<*
seed2*

seed *"
_class
loc:@pi/dense/kernel*
dtype0*
T0
Ú
.pi/dense/kernel/Initializer/random_uniform/subSub.pi/dense/kernel/Initializer/random_uniform/max.pi/dense/kernel/Initializer/random_uniform/min*
_output_shapes
: *"
_class
loc:@pi/dense/kernel*
T0
í
.pi/dense/kernel/Initializer/random_uniform/mulMul8pi/dense/kernel/Initializer/random_uniform/RandomUniform.pi/dense/kernel/Initializer/random_uniform/sub*"
_class
loc:@pi/dense/kernel*
T0*
_output_shapes
:	<
ß
*pi/dense/kernel/Initializer/random_uniformAdd.pi/dense/kernel/Initializer/random_uniform/mul.pi/dense/kernel/Initializer/random_uniform/min*
_output_shapes
:	<*"
_class
loc:@pi/dense/kernel*
T0
Š
pi/dense/kernel
VariableV2*
dtype0*
shared_name *
_output_shapes
:	<*
shape:	<*
	container *"
_class
loc:@pi/dense/kernel
Ô
pi/dense/kernel/AssignAssignpi/dense/kernel*pi/dense/kernel/Initializer/random_uniform*
use_locking(*"
_class
loc:@pi/dense/kernel*
validate_shape(*
_output_shapes
:	<*
T0

pi/dense/kernel/readIdentitypi/dense/kernel*
_output_shapes
:	<*"
_class
loc:@pi/dense/kernel*
T0

pi/dense/bias/Initializer/zerosConst*
dtype0*
valueB*    *
_output_shapes	
:* 
_class
loc:@pi/dense/bias

pi/dense/bias
VariableV2* 
_class
loc:@pi/dense/bias*
shared_name *
dtype0*
	container *
_output_shapes	
:*
shape:
ż
pi/dense/bias/AssignAssignpi/dense/biaspi/dense/bias/Initializer/zeros*
use_locking(* 
_class
loc:@pi/dense/bias*
T0*
_output_shapes	
:*
validate_shape(
u
pi/dense/bias/readIdentitypi/dense/bias*
T0*
_output_shapes	
:* 
_class
loc:@pi/dense/bias

pi/dense/MatMulMatMulPlaceholderpi/dense/kernel/read*
transpose_a( *
transpose_b( *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

pi/dense/BiasAddBiasAddpi/dense/MatMulpi/dense/bias/read*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
data_formatNHWC
Z
pi/dense/TanhTanhpi/dense/BiasAdd*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Š
2pi/dense_1/kernel/Initializer/random_uniform/shapeConst*
_output_shapes
:*
dtype0*$
_class
loc:@pi/dense_1/kernel*
valueB"      

0pi/dense_1/kernel/Initializer/random_uniform/minConst*
dtype0*$
_class
loc:@pi/dense_1/kernel*
valueB
 *×łÝ˝*
_output_shapes
: 

0pi/dense_1/kernel/Initializer/random_uniform/maxConst*$
_class
loc:@pi/dense_1/kernel*
valueB
 *×łÝ=*
_output_shapes
: *
dtype0
ö
:pi/dense_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform2pi/dense_1/kernel/Initializer/random_uniform/shape* 
_output_shapes
:
*

seed *$
_class
loc:@pi/dense_1/kernel*
seed2*
T0*
dtype0
â
0pi/dense_1/kernel/Initializer/random_uniform/subSub0pi/dense_1/kernel/Initializer/random_uniform/max0pi/dense_1/kernel/Initializer/random_uniform/min*
_output_shapes
: *$
_class
loc:@pi/dense_1/kernel*
T0
ö
0pi/dense_1/kernel/Initializer/random_uniform/mulMul:pi/dense_1/kernel/Initializer/random_uniform/RandomUniform0pi/dense_1/kernel/Initializer/random_uniform/sub*
T0*$
_class
loc:@pi/dense_1/kernel* 
_output_shapes
:

č
,pi/dense_1/kernel/Initializer/random_uniformAdd0pi/dense_1/kernel/Initializer/random_uniform/mul0pi/dense_1/kernel/Initializer/random_uniform/min*$
_class
loc:@pi/dense_1/kernel*
T0* 
_output_shapes
:

Ż
pi/dense_1/kernel
VariableV2*
	container *
shape:
*$
_class
loc:@pi/dense_1/kernel*
shared_name * 
_output_shapes
:
*
dtype0
Ý
pi/dense_1/kernel/AssignAssignpi/dense_1/kernel,pi/dense_1/kernel/Initializer/random_uniform* 
_output_shapes
:
*
validate_shape(*$
_class
loc:@pi/dense_1/kernel*
use_locking(*
T0

pi/dense_1/kernel/readIdentitypi/dense_1/kernel*
T0* 
_output_shapes
:
*$
_class
loc:@pi/dense_1/kernel

!pi/dense_1/bias/Initializer/zerosConst*"
_class
loc:@pi/dense_1/bias*
valueB*    *
_output_shapes	
:*
dtype0
Ą
pi/dense_1/bias
VariableV2*
dtype0*
shape:*
shared_name *"
_class
loc:@pi/dense_1/bias*
	container *
_output_shapes	
:
Ç
pi/dense_1/bias/AssignAssignpi/dense_1/bias!pi/dense_1/bias/Initializer/zeros*"
_class
loc:@pi/dense_1/bias*
use_locking(*
_output_shapes	
:*
validate_shape(*
T0
{
pi/dense_1/bias/readIdentitypi/dense_1/bias*
_output_shapes	
:*
T0*"
_class
loc:@pi/dense_1/bias

pi/dense_1/MatMulMatMulpi/dense/Tanhpi/dense_1/kernel/read*
transpose_a( *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_b( *
T0

pi/dense_1/BiasAddBiasAddpi/dense_1/MatMulpi/dense_1/bias/read*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
data_formatNHWC
^
pi/dense_1/TanhTanhpi/dense_1/BiasAdd*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Š
2pi/dense_2/kernel/Initializer/random_uniform/shapeConst*
valueB"      *
dtype0*$
_class
loc:@pi/dense_2/kernel*
_output_shapes
:

0pi/dense_2/kernel/Initializer/random_uniform/minConst*$
_class
loc:@pi/dense_2/kernel*
dtype0*
valueB
 *(ž*
_output_shapes
: 

0pi/dense_2/kernel/Initializer/random_uniform/maxConst*$
_class
loc:@pi/dense_2/kernel*
valueB
 *(>*
_output_shapes
: *
dtype0
ő
:pi/dense_2/kernel/Initializer/random_uniform/RandomUniformRandomUniform2pi/dense_2/kernel/Initializer/random_uniform/shape*

seed *
_output_shapes
:	*
dtype0*
seed2.*$
_class
loc:@pi/dense_2/kernel*
T0
â
0pi/dense_2/kernel/Initializer/random_uniform/subSub0pi/dense_2/kernel/Initializer/random_uniform/max0pi/dense_2/kernel/Initializer/random_uniform/min*$
_class
loc:@pi/dense_2/kernel*
T0*
_output_shapes
: 
ő
0pi/dense_2/kernel/Initializer/random_uniform/mulMul:pi/dense_2/kernel/Initializer/random_uniform/RandomUniform0pi/dense_2/kernel/Initializer/random_uniform/sub*
T0*$
_class
loc:@pi/dense_2/kernel*
_output_shapes
:	
ç
,pi/dense_2/kernel/Initializer/random_uniformAdd0pi/dense_2/kernel/Initializer/random_uniform/mul0pi/dense_2/kernel/Initializer/random_uniform/min*
_output_shapes
:	*$
_class
loc:@pi/dense_2/kernel*
T0
­
pi/dense_2/kernel
VariableV2*
dtype0*
shape:	*
shared_name *
	container *
_output_shapes
:	*$
_class
loc:@pi/dense_2/kernel
Ü
pi/dense_2/kernel/AssignAssignpi/dense_2/kernel,pi/dense_2/kernel/Initializer/random_uniform*$
_class
loc:@pi/dense_2/kernel*
use_locking(*
_output_shapes
:	*
validate_shape(*
T0

pi/dense_2/kernel/readIdentitypi/dense_2/kernel*
T0*$
_class
loc:@pi/dense_2/kernel*
_output_shapes
:	

!pi/dense_2/bias/Initializer/zerosConst*"
_class
loc:@pi/dense_2/bias*
dtype0*
valueB*    *
_output_shapes
:

pi/dense_2/bias
VariableV2*
shared_name *"
_class
loc:@pi/dense_2/bias*
	container *
dtype0*
shape:*
_output_shapes
:
Ć
pi/dense_2/bias/AssignAssignpi/dense_2/bias!pi/dense_2/bias/Initializer/zeros*
T0*
validate_shape(*
_output_shapes
:*
use_locking(*"
_class
loc:@pi/dense_2/bias
z
pi/dense_2/bias/readIdentitypi/dense_2/bias*
_output_shapes
:*
T0*"
_class
loc:@pi/dense_2/bias

pi/dense_2/MatMulMatMulpi/dense_1/Tanhpi/dense_2/kernel/read*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b( 

pi/dense_2/BiasAddBiasAddpi/dense_2/MatMulpi/dense_2/bias/read*
data_formatNHWC*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
i
pi/log_std/initial_valueConst*
dtype0*
_output_shapes
:*
valueB"   ż   ż
v

pi/log_std
VariableV2*
shape:*
dtype0*
_output_shapes
:*
	container *
shared_name 
Ž
pi/log_std/AssignAssign
pi/log_stdpi/log_std/initial_value*
T0*
_output_shapes
:*
_class
loc:@pi/log_std*
use_locking(*
validate_shape(
k
pi/log_std/readIdentity
pi/log_std*
_output_shapes
:*
T0*
_class
loc:@pi/log_std
C
pi/ExpExppi/log_std/read*
_output_shapes
:*
T0
Z
pi/ShapeShapepi/dense_2/BiasAdd*
out_type0*
T0*
_output_shapes
:
Z
pi/random_normal/meanConst*
_output_shapes
: *
valueB
 *    *
dtype0
\
pi/random_normal/stddevConst*
_output_shapes
: *
valueB
 *  ?*
dtype0

%pi/random_normal/RandomStandardNormalRandomStandardNormalpi/Shape*
seed2C*

seed *'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
dtype0

pi/random_normal/mulMul%pi/random_normal/RandomStandardNormalpi/random_normal/stddev*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
v
pi/random_normalAddpi/random_normal/mulpi/random_normal/mean*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Y
pi/mulMulpi/random_normalpi/Exp*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
[
pi/addAddpi/dense_2/BiasAddpi/mul*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
b
pi/subSubPlaceholder_1pi/dense_2/BiasAdd*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
E
pi/Exp_1Exppi/log_std/read*
T0*
_output_shapes
:
O

pi/add_1/yConst*
valueB
 *wĚ+2*
_output_shapes
: *
dtype0
J
pi/add_1Addpi/Exp_1
pi/add_1/y*
_output_shapes
:*
T0
Y

pi/truedivRealDivpi/subpi/add_1*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
M
pi/pow/yConst*
dtype0*
_output_shapes
: *
valueB
 *   @
U
pi/powPow
pi/truedivpi/pow/y*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
O

pi/mul_1/xConst*
dtype0*
_output_shapes
: *
valueB
 *   @
Q
pi/mul_1Mul
pi/mul_1/xpi/log_std/read*
T0*
_output_shapes
:
S
pi/add_2Addpi/powpi/mul_1*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
O

pi/add_3/yConst*
valueB
 *?ë?*
dtype0*
_output_shapes
: 
W
pi/add_3Addpi/add_2
pi/add_3/y*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
O

pi/mul_2/xConst*
valueB
 *   ż*
dtype0*
_output_shapes
: 
W
pi/mul_2Mul
pi/mul_2/xpi/add_3*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Z
pi/Sum/reduction_indicesConst*
value	B :*
dtype0*
_output_shapes
: 
|
pi/SumSumpi/mul_2pi/Sum/reduction_indices*
T0*

Tidx0*
	keep_dims( *#
_output_shapes
:˙˙˙˙˙˙˙˙˙
]
pi/sub_1Subpi/addpi/dense_2/BiasAdd*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
E
pi/Exp_2Exppi/log_std/read*
T0*
_output_shapes
:
O

pi/add_4/yConst*
valueB
 *wĚ+2*
_output_shapes
: *
dtype0
J
pi/add_4Addpi/Exp_2
pi/add_4/y*
_output_shapes
:*
T0
]
pi/truediv_1RealDivpi/sub_1pi/add_4*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
O

pi/pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @
[
pi/pow_1Powpi/truediv_1
pi/pow_1/y*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
O

pi/mul_3/xConst*
_output_shapes
: *
valueB
 *   @*
dtype0
Q
pi/mul_3Mul
pi/mul_3/xpi/log_std/read*
T0*
_output_shapes
:
U
pi/add_5Addpi/pow_1pi/mul_3*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
O

pi/add_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *?ë?
W
pi/add_6Addpi/add_5
pi/add_6/y*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
O

pi/mul_4/xConst*
valueB
 *   ż*
dtype0*
_output_shapes
: 
W
pi/mul_4Mul
pi/mul_4/xpi/add_6*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
\
pi/Sum_1/reduction_indicesConst*
_output_shapes
: *
value	B :*
dtype0

pi/Sum_1Sumpi/mul_4pi/Sum_1/reduction_indices*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*

Tidx0*
	keep_dims( 
q
pi/PlaceholderPlaceholder*
shape:˙˙˙˙˙˙˙˙˙*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0
s
pi/Placeholder_1Placeholder*
shape:˙˙˙˙˙˙˙˙˙*
dtype0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
O

pi/mul_5/xConst*
dtype0*
valueB
 *   @*
_output_shapes
: 
Q
pi/mul_5Mul
pi/mul_5/xpi/log_std/read*
T0*
_output_shapes
:
>
pi/Exp_3Exppi/mul_5*
T0*
_output_shapes
:
O

pi/mul_6/xConst*
valueB
 *   @*
_output_shapes
: *
dtype0
_
pi/mul_6Mul
pi/mul_6/xpi/Placeholder_1*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
K
pi/Exp_4Exppi/mul_6*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
e
pi/sub_2Subpi/Placeholderpi/dense_2/BiasAdd*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
O

pi/pow_2/yConst*
valueB
 *   @*
_output_shapes
: *
dtype0
W
pi/pow_2Powpi/sub_2
pi/pow_2/y*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
U
pi/add_7Addpi/pow_2pi/Exp_3*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
O

pi/add_8/yConst*
_output_shapes
: *
valueB
 *wĚ+2*
dtype0
W
pi/add_8Addpi/Exp_4
pi/add_8/y*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
]
pi/truediv_2RealDivpi/add_7pi/add_8*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
O

pi/sub_3/yConst*
dtype0*
valueB
 *  ?*
_output_shapes
: 
[
pi/sub_3Subpi/truediv_2
pi/sub_3/y*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
O

pi/mul_7/xConst*
valueB
 *   ?*
_output_shapes
: *
dtype0
W
pi/mul_7Mul
pi/mul_7/xpi/sub_3*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
]
pi/add_9Addpi/mul_7pi/Placeholder_1*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
\
pi/sub_4Subpi/add_9pi/log_std/read*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
\
pi/Sum_2/reduction_indicesConst*
_output_shapes
: *
value	B :*
dtype0

pi/Sum_2Sumpi/sub_4pi/Sum_2/reduction_indices*
	keep_dims( *
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*

Tidx0
R
pi/ConstConst*
valueB: *
_output_shapes
:*
dtype0
a
pi/MeanMeanpi/Sum_2pi/Const*
	keep_dims( *

Tidx0*
_output_shapes
: *
T0
P
pi/add_10/yConst*
valueB
 *Çľ?*
_output_shapes
: *
dtype0
S
	pi/add_10Addpi/log_std/readpi/add_10/y*
_output_shapes
:*
T0
e
pi/Sum_3/reduction_indicesConst*
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙*
_output_shapes
: 
t
pi/Sum_3Sum	pi/add_10pi/Sum_3/reduction_indices*

Tidx0*
_output_shapes
: *
	keep_dims( *
T0
M

pi/Const_1Const*
_output_shapes
: *
valueB *
dtype0
e
	pi/Mean_1Meanpi/Sum_3
pi/Const_1*

Tidx0*
_output_shapes
: *
	keep_dims( *
T0
Ľ
0vf/dense/kernel/Initializer/random_uniform/shapeConst*
dtype0*
valueB"<      *"
_class
loc:@vf/dense/kernel*
_output_shapes
:

.vf/dense/kernel/Initializer/random_uniform/minConst*
_output_shapes
: *
valueB
 *ž*
dtype0*"
_class
loc:@vf/dense/kernel

.vf/dense/kernel/Initializer/random_uniform/maxConst*
valueB
 *>*
_output_shapes
: *"
_class
loc:@vf/dense/kernel*
dtype0
đ
8vf/dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform0vf/dense/kernel/Initializer/random_uniform/shape*"
_class
loc:@vf/dense/kernel*
dtype0*

seed *
T0*
_output_shapes
:	<*
seed2
Ú
.vf/dense/kernel/Initializer/random_uniform/subSub.vf/dense/kernel/Initializer/random_uniform/max.vf/dense/kernel/Initializer/random_uniform/min*"
_class
loc:@vf/dense/kernel*
T0*
_output_shapes
: 
í
.vf/dense/kernel/Initializer/random_uniform/mulMul8vf/dense/kernel/Initializer/random_uniform/RandomUniform.vf/dense/kernel/Initializer/random_uniform/sub*
T0*"
_class
loc:@vf/dense/kernel*
_output_shapes
:	<
ß
*vf/dense/kernel/Initializer/random_uniformAdd.vf/dense/kernel/Initializer/random_uniform/mul.vf/dense/kernel/Initializer/random_uniform/min*
_output_shapes
:	<*"
_class
loc:@vf/dense/kernel*
T0
Š
vf/dense/kernel
VariableV2*
shared_name *"
_class
loc:@vf/dense/kernel*
	container *
_output_shapes
:	<*
dtype0*
shape:	<
Ô
vf/dense/kernel/AssignAssignvf/dense/kernel*vf/dense/kernel/Initializer/random_uniform*
_output_shapes
:	<*
T0*
use_locking(*
validate_shape(*"
_class
loc:@vf/dense/kernel

vf/dense/kernel/readIdentityvf/dense/kernel*
T0*"
_class
loc:@vf/dense/kernel*
_output_shapes
:	<

vf/dense/bias/Initializer/zerosConst*
dtype0*
valueB*    *
_output_shapes	
:* 
_class
loc:@vf/dense/bias

vf/dense/bias
VariableV2*
shared_name *
dtype0*
	container * 
_class
loc:@vf/dense/bias*
shape:*
_output_shapes	
:
ż
vf/dense/bias/AssignAssignvf/dense/biasvf/dense/bias/Initializer/zeros*
validate_shape(*
_output_shapes	
:*
use_locking(* 
_class
loc:@vf/dense/bias*
T0
u
vf/dense/bias/readIdentityvf/dense/bias* 
_class
loc:@vf/dense/bias*
_output_shapes	
:*
T0

vf/dense/MatMulMatMulPlaceholdervf/dense/kernel/read*
transpose_b( *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
T0

vf/dense/BiasAddBiasAddvf/dense/MatMulvf/dense/bias/read*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
data_formatNHWC
Z
vf/dense/TanhTanhvf/dense/BiasAdd*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Š
2vf/dense_1/kernel/Initializer/random_uniform/shapeConst*
_output_shapes
:*
valueB"      *
dtype0*$
_class
loc:@vf/dense_1/kernel

0vf/dense_1/kernel/Initializer/random_uniform/minConst*
_output_shapes
: *
dtype0*$
_class
loc:@vf/dense_1/kernel*
valueB
 *×łÝ˝

0vf/dense_1/kernel/Initializer/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *×łÝ=*$
_class
loc:@vf/dense_1/kernel
÷
:vf/dense_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform2vf/dense_1/kernel/Initializer/random_uniform/shape*
seed2*
dtype0*
T0* 
_output_shapes
:
*$
_class
loc:@vf/dense_1/kernel*

seed 
â
0vf/dense_1/kernel/Initializer/random_uniform/subSub0vf/dense_1/kernel/Initializer/random_uniform/max0vf/dense_1/kernel/Initializer/random_uniform/min*$
_class
loc:@vf/dense_1/kernel*
_output_shapes
: *
T0
ö
0vf/dense_1/kernel/Initializer/random_uniform/mulMul:vf/dense_1/kernel/Initializer/random_uniform/RandomUniform0vf/dense_1/kernel/Initializer/random_uniform/sub*$
_class
loc:@vf/dense_1/kernel* 
_output_shapes
:
*
T0
č
,vf/dense_1/kernel/Initializer/random_uniformAdd0vf/dense_1/kernel/Initializer/random_uniform/mul0vf/dense_1/kernel/Initializer/random_uniform/min*$
_class
loc:@vf/dense_1/kernel* 
_output_shapes
:
*
T0
Ż
vf/dense_1/kernel
VariableV2*$
_class
loc:@vf/dense_1/kernel*
dtype0* 
_output_shapes
:
*
shared_name *
	container *
shape:

Ý
vf/dense_1/kernel/AssignAssignvf/dense_1/kernel,vf/dense_1/kernel/Initializer/random_uniform*
validate_shape(*
T0*$
_class
loc:@vf/dense_1/kernel* 
_output_shapes
:
*
use_locking(

vf/dense_1/kernel/readIdentityvf/dense_1/kernel*
T0*$
_class
loc:@vf/dense_1/kernel* 
_output_shapes
:


!vf/dense_1/bias/Initializer/zerosConst*
_output_shapes	
:*"
_class
loc:@vf/dense_1/bias*
dtype0*
valueB*    
Ą
vf/dense_1/bias
VariableV2*
_output_shapes	
:*
dtype0*
shape:*
shared_name *
	container *"
_class
loc:@vf/dense_1/bias
Ç
vf/dense_1/bias/AssignAssignvf/dense_1/bias!vf/dense_1/bias/Initializer/zeros*
_output_shapes	
:*
validate_shape(*"
_class
loc:@vf/dense_1/bias*
T0*
use_locking(
{
vf/dense_1/bias/readIdentityvf/dense_1/bias*
T0*"
_class
loc:@vf/dense_1/bias*
_output_shapes	
:

vf/dense_1/MatMulMatMulvf/dense/Tanhvf/dense_1/kernel/read*
T0*
transpose_a( *
transpose_b( *(
_output_shapes
:˙˙˙˙˙˙˙˙˙

vf/dense_1/BiasAddBiasAddvf/dense_1/MatMulvf/dense_1/bias/read*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
data_formatNHWC
^
vf/dense_1/TanhTanhvf/dense_1/BiasAdd*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Š
2vf/dense_2/kernel/Initializer/random_uniform/shapeConst*
valueB"      *
_output_shapes
:*$
_class
loc:@vf/dense_2/kernel*
dtype0

0vf/dense_2/kernel/Initializer/random_uniform/minConst*
valueB
 *Ivž*$
_class
loc:@vf/dense_2/kernel*
dtype0*
_output_shapes
: 

0vf/dense_2/kernel/Initializer/random_uniform/maxConst*
_output_shapes
: *$
_class
loc:@vf/dense_2/kernel*
dtype0*
valueB
 *Iv>
ö
:vf/dense_2/kernel/Initializer/random_uniform/RandomUniformRandomUniform2vf/dense_2/kernel/Initializer/random_uniform/shape*
_output_shapes
:	*
dtype0*
seed2Ź*

seed *$
_class
loc:@vf/dense_2/kernel*
T0
â
0vf/dense_2/kernel/Initializer/random_uniform/subSub0vf/dense_2/kernel/Initializer/random_uniform/max0vf/dense_2/kernel/Initializer/random_uniform/min*$
_class
loc:@vf/dense_2/kernel*
T0*
_output_shapes
: 
ő
0vf/dense_2/kernel/Initializer/random_uniform/mulMul:vf/dense_2/kernel/Initializer/random_uniform/RandomUniform0vf/dense_2/kernel/Initializer/random_uniform/sub*
T0*
_output_shapes
:	*$
_class
loc:@vf/dense_2/kernel
ç
,vf/dense_2/kernel/Initializer/random_uniformAdd0vf/dense_2/kernel/Initializer/random_uniform/mul0vf/dense_2/kernel/Initializer/random_uniform/min*
T0*$
_class
loc:@vf/dense_2/kernel*
_output_shapes
:	
­
vf/dense_2/kernel
VariableV2*
dtype0*
shared_name *
_output_shapes
:	*$
_class
loc:@vf/dense_2/kernel*
	container *
shape:	
Ü
vf/dense_2/kernel/AssignAssignvf/dense_2/kernel,vf/dense_2/kernel/Initializer/random_uniform*$
_class
loc:@vf/dense_2/kernel*
T0*
validate_shape(*
use_locking(*
_output_shapes
:	

vf/dense_2/kernel/readIdentityvf/dense_2/kernel*$
_class
loc:@vf/dense_2/kernel*
_output_shapes
:	*
T0

!vf/dense_2/bias/Initializer/zerosConst*
dtype0*"
_class
loc:@vf/dense_2/bias*
_output_shapes
:*
valueB*    

vf/dense_2/bias
VariableV2*
shape:*
shared_name *
_output_shapes
:*"
_class
loc:@vf/dense_2/bias*
	container *
dtype0
Ć
vf/dense_2/bias/AssignAssignvf/dense_2/bias!vf/dense_2/bias/Initializer/zeros*
use_locking(*
validate_shape(*
T0*"
_class
loc:@vf/dense_2/bias*
_output_shapes
:
z
vf/dense_2/bias/readIdentityvf/dense_2/bias*"
_class
loc:@vf/dense_2/bias*
T0*
_output_shapes
:

vf/dense_2/MatMulMatMulvf/dense_1/Tanhvf/dense_2/kernel/read*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_b( *
transpose_a( *
T0

vf/dense_2/BiasAddBiasAddvf/dense_2/MatMulvf/dense_2/bias/read*
data_formatNHWC*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
n

vf/SqueezeSqueezevf/dense_2/BiasAdd*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
squeeze_dims
*
T0
Ľ
0vc/dense/kernel/Initializer/random_uniform/shapeConst*
valueB"<      *"
_class
loc:@vc/dense/kernel*
dtype0*
_output_shapes
:

.vc/dense/kernel/Initializer/random_uniform/minConst*
valueB
 *ž*"
_class
loc:@vc/dense/kernel*
dtype0*
_output_shapes
: 

.vc/dense/kernel/Initializer/random_uniform/maxConst*
dtype0*"
_class
loc:@vc/dense/kernel*
_output_shapes
: *
valueB
 *>
đ
8vc/dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform0vc/dense/kernel/Initializer/random_uniform/shape*
T0*
_output_shapes
:	<*
dtype0*
seed2˝*

seed *"
_class
loc:@vc/dense/kernel
Ú
.vc/dense/kernel/Initializer/random_uniform/subSub.vc/dense/kernel/Initializer/random_uniform/max.vc/dense/kernel/Initializer/random_uniform/min*"
_class
loc:@vc/dense/kernel*
_output_shapes
: *
T0
í
.vc/dense/kernel/Initializer/random_uniform/mulMul8vc/dense/kernel/Initializer/random_uniform/RandomUniform.vc/dense/kernel/Initializer/random_uniform/sub*
T0*
_output_shapes
:	<*"
_class
loc:@vc/dense/kernel
ß
*vc/dense/kernel/Initializer/random_uniformAdd.vc/dense/kernel/Initializer/random_uniform/mul.vc/dense/kernel/Initializer/random_uniform/min*"
_class
loc:@vc/dense/kernel*
_output_shapes
:	<*
T0
Š
vc/dense/kernel
VariableV2*
shape:	<*"
_class
loc:@vc/dense/kernel*
dtype0*
shared_name *
_output_shapes
:	<*
	container 
Ô
vc/dense/kernel/AssignAssignvc/dense/kernel*vc/dense/kernel/Initializer/random_uniform*
validate_shape(*
_output_shapes
:	<*"
_class
loc:@vc/dense/kernel*
use_locking(*
T0

vc/dense/kernel/readIdentityvc/dense/kernel*"
_class
loc:@vc/dense/kernel*
T0*
_output_shapes
:	<

vc/dense/bias/Initializer/zerosConst*
dtype0*
_output_shapes	
:* 
_class
loc:@vc/dense/bias*
valueB*    

vc/dense/bias
VariableV2*
shape:*
	container *
_output_shapes	
:*
dtype0* 
_class
loc:@vc/dense/bias*
shared_name 
ż
vc/dense/bias/AssignAssignvc/dense/biasvc/dense/bias/Initializer/zeros*
use_locking(*
_output_shapes	
:*
validate_shape(*
T0* 
_class
loc:@vc/dense/bias
u
vc/dense/bias/readIdentityvc/dense/bias*
T0*
_output_shapes	
:* 
_class
loc:@vc/dense/bias

vc/dense/MatMulMatMulPlaceholdervc/dense/kernel/read*
transpose_a( *
transpose_b( *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

vc/dense/BiasAddBiasAddvc/dense/MatMulvc/dense/bias/read*
data_formatNHWC*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Z
vc/dense/TanhTanhvc/dense/BiasAdd*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Š
2vc/dense_1/kernel/Initializer/random_uniform/shapeConst*
valueB"      *
_output_shapes
:*
dtype0*$
_class
loc:@vc/dense_1/kernel

0vc/dense_1/kernel/Initializer/random_uniform/minConst*$
_class
loc:@vc/dense_1/kernel*
valueB
 *×łÝ˝*
_output_shapes
: *
dtype0

0vc/dense_1/kernel/Initializer/random_uniform/maxConst*
valueB
 *×łÝ=*
dtype0*
_output_shapes
: *$
_class
loc:@vc/dense_1/kernel
÷
:vc/dense_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform2vc/dense_1/kernel/Initializer/random_uniform/shape*
dtype0*$
_class
loc:@vc/dense_1/kernel*

seed * 
_output_shapes
:
*
seed2Î*
T0
â
0vc/dense_1/kernel/Initializer/random_uniform/subSub0vc/dense_1/kernel/Initializer/random_uniform/max0vc/dense_1/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*$
_class
loc:@vc/dense_1/kernel
ö
0vc/dense_1/kernel/Initializer/random_uniform/mulMul:vc/dense_1/kernel/Initializer/random_uniform/RandomUniform0vc/dense_1/kernel/Initializer/random_uniform/sub* 
_output_shapes
:
*$
_class
loc:@vc/dense_1/kernel*
T0
č
,vc/dense_1/kernel/Initializer/random_uniformAdd0vc/dense_1/kernel/Initializer/random_uniform/mul0vc/dense_1/kernel/Initializer/random_uniform/min* 
_output_shapes
:
*$
_class
loc:@vc/dense_1/kernel*
T0
Ż
vc/dense_1/kernel
VariableV2*
	container * 
_output_shapes
:
*
dtype0*
shape:
*$
_class
loc:@vc/dense_1/kernel*
shared_name 
Ý
vc/dense_1/kernel/AssignAssignvc/dense_1/kernel,vc/dense_1/kernel/Initializer/random_uniform*
T0*
validate_shape(*
use_locking(*$
_class
loc:@vc/dense_1/kernel* 
_output_shapes
:


vc/dense_1/kernel/readIdentityvc/dense_1/kernel* 
_output_shapes
:
*$
_class
loc:@vc/dense_1/kernel*
T0

!vc/dense_1/bias/Initializer/zerosConst*
_output_shapes	
:*
dtype0*"
_class
loc:@vc/dense_1/bias*
valueB*    
Ą
vc/dense_1/bias
VariableV2*
	container *
_output_shapes	
:*
shared_name *
dtype0*
shape:*"
_class
loc:@vc/dense_1/bias
Ç
vc/dense_1/bias/AssignAssignvc/dense_1/bias!vc/dense_1/bias/Initializer/zeros*
use_locking(*
validate_shape(*
_output_shapes	
:*"
_class
loc:@vc/dense_1/bias*
T0
{
vc/dense_1/bias/readIdentityvc/dense_1/bias*
_output_shapes	
:*
T0*"
_class
loc:@vc/dense_1/bias

vc/dense_1/MatMulMatMulvc/dense/Tanhvc/dense_1/kernel/read*
transpose_b( *
transpose_a( *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

vc/dense_1/BiasAddBiasAddvc/dense_1/MatMulvc/dense_1/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
^
vc/dense_1/TanhTanhvc/dense_1/BiasAdd*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Š
2vc/dense_2/kernel/Initializer/random_uniform/shapeConst*
_output_shapes
:*
dtype0*$
_class
loc:@vc/dense_2/kernel*
valueB"      

0vc/dense_2/kernel/Initializer/random_uniform/minConst*
dtype0*$
_class
loc:@vc/dense_2/kernel*
valueB
 *Ivž*
_output_shapes
: 

0vc/dense_2/kernel/Initializer/random_uniform/maxConst*
valueB
 *Iv>*
_output_shapes
: *
dtype0*$
_class
loc:@vc/dense_2/kernel
ö
:vc/dense_2/kernel/Initializer/random_uniform/RandomUniformRandomUniform2vc/dense_2/kernel/Initializer/random_uniform/shape*

seed *$
_class
loc:@vc/dense_2/kernel*
seed2ß*
_output_shapes
:	*
dtype0*
T0
â
0vc/dense_2/kernel/Initializer/random_uniform/subSub0vc/dense_2/kernel/Initializer/random_uniform/max0vc/dense_2/kernel/Initializer/random_uniform/min*$
_class
loc:@vc/dense_2/kernel*
T0*
_output_shapes
: 
ő
0vc/dense_2/kernel/Initializer/random_uniform/mulMul:vc/dense_2/kernel/Initializer/random_uniform/RandomUniform0vc/dense_2/kernel/Initializer/random_uniform/sub*
T0*
_output_shapes
:	*$
_class
loc:@vc/dense_2/kernel
ç
,vc/dense_2/kernel/Initializer/random_uniformAdd0vc/dense_2/kernel/Initializer/random_uniform/mul0vc/dense_2/kernel/Initializer/random_uniform/min*
_output_shapes
:	*$
_class
loc:@vc/dense_2/kernel*
T0
­
vc/dense_2/kernel
VariableV2*
dtype0*
shape:	*
_output_shapes
:	*
	container *$
_class
loc:@vc/dense_2/kernel*
shared_name 
Ü
vc/dense_2/kernel/AssignAssignvc/dense_2/kernel,vc/dense_2/kernel/Initializer/random_uniform*
use_locking(*$
_class
loc:@vc/dense_2/kernel*
validate_shape(*
_output_shapes
:	*
T0

vc/dense_2/kernel/readIdentityvc/dense_2/kernel*
_output_shapes
:	*
T0*$
_class
loc:@vc/dense_2/kernel

!vc/dense_2/bias/Initializer/zerosConst*
_output_shapes
:*
valueB*    *"
_class
loc:@vc/dense_2/bias*
dtype0

vc/dense_2/bias
VariableV2*
	container *
shape:*"
_class
loc:@vc/dense_2/bias*
dtype0*
_output_shapes
:*
shared_name 
Ć
vc/dense_2/bias/AssignAssignvc/dense_2/bias!vc/dense_2/bias/Initializer/zeros*
use_locking(*"
_class
loc:@vc/dense_2/bias*
_output_shapes
:*
validate_shape(*
T0
z
vc/dense_2/bias/readIdentityvc/dense_2/bias*"
_class
loc:@vc/dense_2/bias*
T0*
_output_shapes
:

vc/dense_2/MatMulMatMulvc/dense_1/Tanhvc/dense_2/kernel/read*
transpose_a( *
transpose_b( *'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

vc/dense_2/BiasAddBiasAddvc/dense_2/MatMulvc/dense_2/bias/read*
data_formatNHWC*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
n

vc/SqueezeSqueezevc/dense_2/BiasAdd*
squeeze_dims
*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
@
NegNegpi/Sum*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
O
ConstConst*
_output_shapes
:*
dtype0*
valueB: 
V
MeanMeanNegConst*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
O
subSubpi/SumPlaceholder_6*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
=
ExpExpsub*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
N
	Greater/yConst*
valueB
 *    *
_output_shapes
: *
dtype0
Z
GreaterGreaterPlaceholder_2	Greater/y*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
J
mul/xConst*
_output_shapes
: *
valueB
 *?*
dtype0
N
mulMulmul/xPlaceholder_2*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
L
mul_1/xConst*
dtype0*
valueB
 *ÍĚL?*
_output_shapes
: 
R
mul_1Mulmul_1/xPlaceholder_2*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
S
SelectSelectGreatermulmul_1*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
N
mul_2MulExpPlaceholder_2*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
O
MinimumMinimummul_2Select*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
Q
Const_1Const*
valueB: *
dtype0*
_output_shapes
:
^
Mean_1MeanMinimumConst_1*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
N
mul_3MulExpPlaceholder_3*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Q
Const_2Const*
valueB: *
dtype0*
_output_shapes
:
\
Mean_2Meanmul_3Const_2*
T0*

Tidx0*
	keep_dims( *
_output_shapes
: 
L
mul_4/xConst*
valueB
 *    *
dtype0*
_output_shapes
: 
A
mul_4Mulmul_4/x	pi/Mean_1*
T0*
_output_shapes
: 
:
addAddMean_1mul_4*
_output_shapes
: *
T0
2
Neg_1Negadd*
_output_shapes
: *
T0
R
gradients/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
X
gradients/grad_ys_0Const*
valueB
 *  ?*
_output_shapes
: *
dtype0
o
gradients/FillFillgradients/Shapegradients/grad_ys_0*
T0*

index_type0*
_output_shapes
: 
P
gradients/Neg_1_grad/NegNeggradients/Fill*
_output_shapes
: *
T0
F
#gradients/add_grad/tuple/group_depsNoOp^gradients/Neg_1_grad/Neg
Ĺ
+gradients/add_grad/tuple/control_dependencyIdentitygradients/Neg_1_grad/Neg$^gradients/add_grad/tuple/group_deps*
T0*+
_class!
loc:@gradients/Neg_1_grad/Neg*
_output_shapes
: 
Ç
-gradients/add_grad/tuple/control_dependency_1Identitygradients/Neg_1_grad/Neg$^gradients/add_grad/tuple/group_deps*
_output_shapes
: *
T0*+
_class!
loc:@gradients/Neg_1_grad/Neg
m
#gradients/Mean_1_grad/Reshape/shapeConst*
_output_shapes
:*
valueB:*
dtype0
­
gradients/Mean_1_grad/ReshapeReshape+gradients/add_grad/tuple/control_dependency#gradients/Mean_1_grad/Reshape/shape*
T0*
_output_shapes
:*
Tshape0
b
gradients/Mean_1_grad/ShapeShapeMinimum*
out_type0*
_output_shapes
:*
T0

gradients/Mean_1_grad/TileTilegradients/Mean_1_grad/Reshapegradients/Mean_1_grad/Shape*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*

Tmultiples0*
T0
d
gradients/Mean_1_grad/Shape_1ShapeMinimum*
T0*
out_type0*
_output_shapes
:
`
gradients/Mean_1_grad/Shape_2Const*
_output_shapes
: *
valueB *
dtype0
e
gradients/Mean_1_grad/ConstConst*
_output_shapes
:*
dtype0*
valueB: 

gradients/Mean_1_grad/ProdProdgradients/Mean_1_grad/Shape_1gradients/Mean_1_grad/Const*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
g
gradients/Mean_1_grad/Const_1Const*
dtype0*
_output_shapes
:*
valueB: 
 
gradients/Mean_1_grad/Prod_1Prodgradients/Mean_1_grad/Shape_2gradients/Mean_1_grad/Const_1*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
a
gradients/Mean_1_grad/Maximum/yConst*
dtype0*
_output_shapes
: *
value	B :

gradients/Mean_1_grad/MaximumMaximumgradients/Mean_1_grad/Prod_1gradients/Mean_1_grad/Maximum/y*
_output_shapes
: *
T0

gradients/Mean_1_grad/floordivFloorDivgradients/Mean_1_grad/Prodgradients/Mean_1_grad/Maximum*
T0*
_output_shapes
: 

gradients/Mean_1_grad/CastCastgradients/Mean_1_grad/floordiv*

SrcT0*
_output_shapes
: *

DstT0*
Truncate( 

gradients/Mean_1_grad/truedivRealDivgradients/Mean_1_grad/Tilegradients/Mean_1_grad/Cast*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
z
gradients/mul_4_grad/MulMul-gradients/add_grad/tuple/control_dependency_1	pi/Mean_1*
T0*
_output_shapes
: 
z
gradients/mul_4_grad/Mul_1Mul-gradients/add_grad/tuple/control_dependency_1mul_4/x*
T0*
_output_shapes
: 
e
%gradients/mul_4_grad/tuple/group_depsNoOp^gradients/mul_4_grad/Mul^gradients/mul_4_grad/Mul_1
É
-gradients/mul_4_grad/tuple/control_dependencyIdentitygradients/mul_4_grad/Mul&^gradients/mul_4_grad/tuple/group_deps*
_output_shapes
: *+
_class!
loc:@gradients/mul_4_grad/Mul*
T0
Ď
/gradients/mul_4_grad/tuple/control_dependency_1Identitygradients/mul_4_grad/Mul_1&^gradients/mul_4_grad/tuple/group_deps*-
_class#
!loc:@gradients/mul_4_grad/Mul_1*
_output_shapes
: *
T0
a
gradients/Minimum_grad/ShapeShapemul_2*
_output_shapes
:*
T0*
out_type0
d
gradients/Minimum_grad/Shape_1ShapeSelect*
_output_shapes
:*
T0*
out_type0
{
gradients/Minimum_grad/Shape_2Shapegradients/Mean_1_grad/truediv*
_output_shapes
:*
out_type0*
T0
g
"gradients/Minimum_grad/zeros/ConstConst*
valueB
 *    *
_output_shapes
: *
dtype0
¨
gradients/Minimum_grad/zerosFillgradients/Minimum_grad/Shape_2"gradients/Minimum_grad/zeros/Const*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*

index_type0
j
 gradients/Minimum_grad/LessEqual	LessEqualmul_2Select*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ŕ
,gradients/Minimum_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/Minimum_grad/Shapegradients/Minimum_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0
´
gradients/Minimum_grad/SelectSelect gradients/Minimum_grad/LessEqualgradients/Mean_1_grad/truedivgradients/Minimum_grad/zeros*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
ś
gradients/Minimum_grad/Select_1Select gradients/Minimum_grad/LessEqualgradients/Minimum_grad/zerosgradients/Mean_1_grad/truediv*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ž
gradients/Minimum_grad/SumSumgradients/Minimum_grad/Select,gradients/Minimum_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
_output_shapes
:*
T0

gradients/Minimum_grad/ReshapeReshapegradients/Minimum_grad/Sumgradients/Minimum_grad/Shape*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
Tshape0
´
gradients/Minimum_grad/Sum_1Sumgradients/Minimum_grad/Select_1.gradients/Minimum_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*

Tidx0*
	keep_dims( 
Ľ
 gradients/Minimum_grad/Reshape_1Reshapegradients/Minimum_grad/Sum_1gradients/Minimum_grad/Shape_1*
Tshape0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
s
'gradients/Minimum_grad/tuple/group_depsNoOp^gradients/Minimum_grad/Reshape!^gradients/Minimum_grad/Reshape_1
ć
/gradients/Minimum_grad/tuple/control_dependencyIdentitygradients/Minimum_grad/Reshape(^gradients/Minimum_grad/tuple/group_deps*1
_class'
%#loc:@gradients/Minimum_grad/Reshape*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
ě
1gradients/Minimum_grad/tuple/control_dependency_1Identity gradients/Minimum_grad/Reshape_1(^gradients/Minimum_grad/tuple/group_deps*3
_class)
'%loc:@gradients/Minimum_grad/Reshape_1*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
i
&gradients/pi/Mean_1_grad/Reshape/shapeConst*
_output_shapes
: *
dtype0*
valueB 
ł
 gradients/pi/Mean_1_grad/ReshapeReshape/gradients/mul_4_grad/tuple/control_dependency_1&gradients/pi/Mean_1_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
: 
a
gradients/pi/Mean_1_grad/ConstConst*
valueB *
_output_shapes
: *
dtype0

gradients/pi/Mean_1_grad/TileTile gradients/pi/Mean_1_grad/Reshapegradients/pi/Mean_1_grad/Const*
_output_shapes
: *
T0*

Tmultiples0
e
 gradients/pi/Mean_1_grad/Const_1Const*
dtype0*
valueB
 *  ?*
_output_shapes
: 

 gradients/pi/Mean_1_grad/truedivRealDivgradients/pi/Mean_1_grad/Tile gradients/pi/Mean_1_grad/Const_1*
_output_shapes
: *
T0
]
gradients/mul_2_grad/ShapeShapeExp*
T0*
_output_shapes
:*
out_type0
i
gradients/mul_2_grad/Shape_1ShapePlaceholder_2*
_output_shapes
:*
T0*
out_type0
ş
*gradients/mul_2_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/mul_2_grad/Shapegradients/mul_2_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0

gradients/mul_2_grad/MulMul/gradients/Minimum_grad/tuple/control_dependencyPlaceholder_2*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ľ
gradients/mul_2_grad/SumSumgradients/mul_2_grad/Mul*gradients/mul_2_grad/BroadcastGradientArgs*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 

gradients/mul_2_grad/ReshapeReshapegradients/mul_2_grad/Sumgradients/mul_2_grad/Shape*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
Tshape0

gradients/mul_2_grad/Mul_1MulExp/gradients/Minimum_grad/tuple/control_dependency*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ť
gradients/mul_2_grad/Sum_1Sumgradients/mul_2_grad/Mul_1,gradients/mul_2_grad/BroadcastGradientArgs:1*
T0*
	keep_dims( *

Tidx0*
_output_shapes
:

gradients/mul_2_grad/Reshape_1Reshapegradients/mul_2_grad/Sum_1gradients/mul_2_grad/Shape_1*
T0*
Tshape0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
m
%gradients/mul_2_grad/tuple/group_depsNoOp^gradients/mul_2_grad/Reshape^gradients/mul_2_grad/Reshape_1
Ţ
-gradients/mul_2_grad/tuple/control_dependencyIdentitygradients/mul_2_grad/Reshape&^gradients/mul_2_grad/tuple/group_deps*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*/
_class%
#!loc:@gradients/mul_2_grad/Reshape
ä
/gradients/mul_2_grad/tuple/control_dependency_1Identitygradients/mul_2_grad/Reshape_1&^gradients/mul_2_grad/tuple/group_deps*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*1
_class'
%#loc:@gradients/mul_2_grad/Reshape_1*
T0
g
gradients/pi/Sum_3_grad/ShapeConst*
valueB:*
_output_shapes
:*
dtype0

gradients/pi/Sum_3_grad/SizeConst*
dtype0*
_output_shapes
: *
value	B :*0
_class&
$"loc:@gradients/pi/Sum_3_grad/Shape
Ż
gradients/pi/Sum_3_grad/addAddpi/Sum_3/reduction_indicesgradients/pi/Sum_3_grad/Size*0
_class&
$"loc:@gradients/pi/Sum_3_grad/Shape*
T0*
_output_shapes
: 
ľ
gradients/pi/Sum_3_grad/modFloorModgradients/pi/Sum_3_grad/addgradients/pi/Sum_3_grad/Size*0
_class&
$"loc:@gradients/pi/Sum_3_grad/Shape*
_output_shapes
: *
T0

gradients/pi/Sum_3_grad/Shape_1Const*
valueB *
_output_shapes
: *0
_class&
$"loc:@gradients/pi/Sum_3_grad/Shape*
dtype0

#gradients/pi/Sum_3_grad/range/startConst*0
_class&
$"loc:@gradients/pi/Sum_3_grad/Shape*
_output_shapes
: *
dtype0*
value	B : 

#gradients/pi/Sum_3_grad/range/deltaConst*
dtype0*
_output_shapes
: *
value	B :*0
_class&
$"loc:@gradients/pi/Sum_3_grad/Shape
č
gradients/pi/Sum_3_grad/rangeRange#gradients/pi/Sum_3_grad/range/startgradients/pi/Sum_3_grad/Size#gradients/pi/Sum_3_grad/range/delta*

Tidx0*
_output_shapes
:*0
_class&
$"loc:@gradients/pi/Sum_3_grad/Shape

"gradients/pi/Sum_3_grad/Fill/valueConst*
_output_shapes
: *0
_class&
$"loc:@gradients/pi/Sum_3_grad/Shape*
dtype0*
value	B :
Î
gradients/pi/Sum_3_grad/FillFillgradients/pi/Sum_3_grad/Shape_1"gradients/pi/Sum_3_grad/Fill/value*0
_class&
$"loc:@gradients/pi/Sum_3_grad/Shape*
_output_shapes
: *
T0*

index_type0

%gradients/pi/Sum_3_grad/DynamicStitchDynamicStitchgradients/pi/Sum_3_grad/rangegradients/pi/Sum_3_grad/modgradients/pi/Sum_3_grad/Shapegradients/pi/Sum_3_grad/Fill*
_output_shapes
:*
T0*
N*0
_class&
$"loc:@gradients/pi/Sum_3_grad/Shape

!gradients/pi/Sum_3_grad/Maximum/yConst*
value	B :*
dtype0*0
_class&
$"loc:@gradients/pi/Sum_3_grad/Shape*
_output_shapes
: 
Ë
gradients/pi/Sum_3_grad/MaximumMaximum%gradients/pi/Sum_3_grad/DynamicStitch!gradients/pi/Sum_3_grad/Maximum/y*0
_class&
$"loc:@gradients/pi/Sum_3_grad/Shape*
_output_shapes
:*
T0
Ă
 gradients/pi/Sum_3_grad/floordivFloorDivgradients/pi/Sum_3_grad/Shapegradients/pi/Sum_3_grad/Maximum*0
_class&
$"loc:@gradients/pi/Sum_3_grad/Shape*
T0*
_output_shapes
:
Ś
gradients/pi/Sum_3_grad/ReshapeReshape gradients/pi/Mean_1_grad/truediv%gradients/pi/Sum_3_grad/DynamicStitch*
T0*
_output_shapes
:*
Tshape0

gradients/pi/Sum_3_grad/TileTilegradients/pi/Sum_3_grad/Reshape gradients/pi/Sum_3_grad/floordiv*
T0*
_output_shapes
:*

Tmultiples0

gradients/Exp_grad/mulMul-gradients/mul_2_grad/tuple/control_dependencyExp*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
h
gradients/pi/add_10_grad/ShapeConst*
valueB:*
_output_shapes
:*
dtype0
c
 gradients/pi/add_10_grad/Shape_1Const*
_output_shapes
: *
dtype0*
valueB 
Ć
.gradients/pi/add_10_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/pi/add_10_grad/Shape gradients/pi/add_10_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0
ł
gradients/pi/add_10_grad/SumSumgradients/pi/Sum_3_grad/Tile.gradients/pi/add_10_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 

 gradients/pi/add_10_grad/ReshapeReshapegradients/pi/add_10_grad/Sumgradients/pi/add_10_grad/Shape*
Tshape0*
T0*
_output_shapes
:
ł
gradients/pi/add_10_grad/Sum_1Sumgradients/pi/Sum_3_grad/Tile0gradients/pi/add_10_grad/BroadcastGradientArgs:1*

Tidx0*
T0*
	keep_dims( *
_output_shapes
: 

"gradients/pi/add_10_grad/Reshape_1Reshapegradients/pi/add_10_grad/Sum_1 gradients/pi/add_10_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
y
)gradients/pi/add_10_grad/tuple/group_depsNoOp!^gradients/pi/add_10_grad/Reshape#^gradients/pi/add_10_grad/Reshape_1
ĺ
1gradients/pi/add_10_grad/tuple/control_dependencyIdentity gradients/pi/add_10_grad/Reshape*^gradients/pi/add_10_grad/tuple/group_deps*3
_class)
'%loc:@gradients/pi/add_10_grad/Reshape*
T0*
_output_shapes
:
ç
3gradients/pi/add_10_grad/tuple/control_dependency_1Identity"gradients/pi/add_10_grad/Reshape_1*^gradients/pi/add_10_grad/tuple/group_deps*
T0*
_output_shapes
: *5
_class+
)'loc:@gradients/pi/add_10_grad/Reshape_1
^
gradients/sub_grad/ShapeShapepi/Sum*
_output_shapes
:*
T0*
out_type0
g
gradients/sub_grad/Shape_1ShapePlaceholder_6*
out_type0*
_output_shapes
:*
T0
´
(gradients/sub_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/sub_grad/Shapegradients/sub_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙

gradients/sub_grad/SumSumgradients/Exp_grad/mul(gradients/sub_grad/BroadcastGradientArgs*

Tidx0*
T0*
_output_shapes
:*
	keep_dims( 

gradients/sub_grad/ReshapeReshapegradients/sub_grad/Sumgradients/sub_grad/Shape*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
Tshape0
Ł
gradients/sub_grad/Sum_1Sumgradients/Exp_grad/mul*gradients/sub_grad/BroadcastGradientArgs:1*

Tidx0*
_output_shapes
:*
T0*
	keep_dims( 
Z
gradients/sub_grad/NegNeggradients/sub_grad/Sum_1*
_output_shapes
:*
T0

gradients/sub_grad/Reshape_1Reshapegradients/sub_grad/Neggradients/sub_grad/Shape_1*
Tshape0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
g
#gradients/sub_grad/tuple/group_depsNoOp^gradients/sub_grad/Reshape^gradients/sub_grad/Reshape_1
Ö
+gradients/sub_grad/tuple/control_dependencyIdentitygradients/sub_grad/Reshape$^gradients/sub_grad/tuple/group_deps*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*-
_class#
!loc:@gradients/sub_grad/Reshape
Ü
-gradients/sub_grad/tuple/control_dependency_1Identitygradients/sub_grad/Reshape_1$^gradients/sub_grad/tuple/group_deps*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*/
_class%
#!loc:@gradients/sub_grad/Reshape_1
c
gradients/pi/Sum_grad/ShapeShapepi/mul_2*
_output_shapes
:*
out_type0*
T0

gradients/pi/Sum_grad/SizeConst*
value	B :*
_output_shapes
: *.
_class$
" loc:@gradients/pi/Sum_grad/Shape*
dtype0
§
gradients/pi/Sum_grad/addAddpi/Sum/reduction_indicesgradients/pi/Sum_grad/Size*
T0*.
_class$
" loc:@gradients/pi/Sum_grad/Shape*
_output_shapes
: 
­
gradients/pi/Sum_grad/modFloorModgradients/pi/Sum_grad/addgradients/pi/Sum_grad/Size*
_output_shapes
: *.
_class$
" loc:@gradients/pi/Sum_grad/Shape*
T0

gradients/pi/Sum_grad/Shape_1Const*.
_class$
" loc:@gradients/pi/Sum_grad/Shape*
_output_shapes
: *
valueB *
dtype0

!gradients/pi/Sum_grad/range/startConst*
value	B : *
_output_shapes
: *
dtype0*.
_class$
" loc:@gradients/pi/Sum_grad/Shape

!gradients/pi/Sum_grad/range/deltaConst*
_output_shapes
: *
value	B :*
dtype0*.
_class$
" loc:@gradients/pi/Sum_grad/Shape
Ţ
gradients/pi/Sum_grad/rangeRange!gradients/pi/Sum_grad/range/startgradients/pi/Sum_grad/Size!gradients/pi/Sum_grad/range/delta*.
_class$
" loc:@gradients/pi/Sum_grad/Shape*

Tidx0*
_output_shapes
:

 gradients/pi/Sum_grad/Fill/valueConst*
dtype0*.
_class$
" loc:@gradients/pi/Sum_grad/Shape*
_output_shapes
: *
value	B :
Ć
gradients/pi/Sum_grad/FillFillgradients/pi/Sum_grad/Shape_1 gradients/pi/Sum_grad/Fill/value*
T0*

index_type0*.
_class$
" loc:@gradients/pi/Sum_grad/Shape*
_output_shapes
: 

#gradients/pi/Sum_grad/DynamicStitchDynamicStitchgradients/pi/Sum_grad/rangegradients/pi/Sum_grad/modgradients/pi/Sum_grad/Shapegradients/pi/Sum_grad/Fill*
T0*.
_class$
" loc:@gradients/pi/Sum_grad/Shape*
N*
_output_shapes
:

gradients/pi/Sum_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: *.
_class$
" loc:@gradients/pi/Sum_grad/Shape
Ă
gradients/pi/Sum_grad/MaximumMaximum#gradients/pi/Sum_grad/DynamicStitchgradients/pi/Sum_grad/Maximum/y*
_output_shapes
:*.
_class$
" loc:@gradients/pi/Sum_grad/Shape*
T0
ť
gradients/pi/Sum_grad/floordivFloorDivgradients/pi/Sum_grad/Shapegradients/pi/Sum_grad/Maximum*
T0*.
_class$
" loc:@gradients/pi/Sum_grad/Shape*
_output_shapes
:
Ă
gradients/pi/Sum_grad/ReshapeReshape+gradients/sub_grad/tuple/control_dependency#gradients/pi/Sum_grad/DynamicStitch*
Tshape0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
Ľ
gradients/pi/Sum_grad/TileTilegradients/pi/Sum_grad/Reshapegradients/pi/Sum_grad/floordiv*

Tmultiples0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
`
gradients/pi/mul_2_grad/ShapeConst*
_output_shapes
: *
dtype0*
valueB 
g
gradients/pi/mul_2_grad/Shape_1Shapepi/add_3*
T0*
out_type0*
_output_shapes
:
Ă
-gradients/pi/mul_2_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/pi/mul_2_grad/Shapegradients/pi/mul_2_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
z
gradients/pi/mul_2_grad/MulMulgradients/pi/Sum_grad/Tilepi/add_3*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ž
gradients/pi/mul_2_grad/SumSumgradients/pi/mul_2_grad/Mul-gradients/pi/mul_2_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0

gradients/pi/mul_2_grad/ReshapeReshapegradients/pi/mul_2_grad/Sumgradients/pi/mul_2_grad/Shape*
Tshape0*
_output_shapes
: *
T0
~
gradients/pi/mul_2_grad/Mul_1Mul
pi/mul_2/xgradients/pi/Sum_grad/Tile*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
´
gradients/pi/mul_2_grad/Sum_1Sumgradients/pi/mul_2_grad/Mul_1/gradients/pi/mul_2_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *
T0*

Tidx0
Ź
!gradients/pi/mul_2_grad/Reshape_1Reshapegradients/pi/mul_2_grad/Sum_1gradients/pi/mul_2_grad/Shape_1*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
Tshape0
v
(gradients/pi/mul_2_grad/tuple/group_depsNoOp ^gradients/pi/mul_2_grad/Reshape"^gradients/pi/mul_2_grad/Reshape_1
Ý
0gradients/pi/mul_2_grad/tuple/control_dependencyIdentitygradients/pi/mul_2_grad/Reshape)^gradients/pi/mul_2_grad/tuple/group_deps*2
_class(
&$loc:@gradients/pi/mul_2_grad/Reshape*
_output_shapes
: *
T0
ô
2gradients/pi/mul_2_grad/tuple/control_dependency_1Identity!gradients/pi/mul_2_grad/Reshape_1)^gradients/pi/mul_2_grad/tuple/group_deps*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*4
_class*
(&loc:@gradients/pi/mul_2_grad/Reshape_1*
T0
e
gradients/pi/add_3_grad/ShapeShapepi/add_2*
out_type0*
T0*
_output_shapes
:
b
gradients/pi/add_3_grad/Shape_1Const*
dtype0*
valueB *
_output_shapes
: 
Ă
-gradients/pi/add_3_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/pi/add_3_grad/Shapegradients/pi/add_3_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
Ĺ
gradients/pi/add_3_grad/SumSum2gradients/pi/mul_2_grad/tuple/control_dependency_1-gradients/pi/add_3_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
Ś
gradients/pi/add_3_grad/ReshapeReshapegradients/pi/add_3_grad/Sumgradients/pi/add_3_grad/Shape*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
Tshape0
É
gradients/pi/add_3_grad/Sum_1Sum2gradients/pi/mul_2_grad/tuple/control_dependency_1/gradients/pi/add_3_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *
T0*

Tidx0

!gradients/pi/add_3_grad/Reshape_1Reshapegradients/pi/add_3_grad/Sum_1gradients/pi/add_3_grad/Shape_1*
Tshape0*
T0*
_output_shapes
: 
v
(gradients/pi/add_3_grad/tuple/group_depsNoOp ^gradients/pi/add_3_grad/Reshape"^gradients/pi/add_3_grad/Reshape_1
î
0gradients/pi/add_3_grad/tuple/control_dependencyIdentitygradients/pi/add_3_grad/Reshape)^gradients/pi/add_3_grad/tuple/group_deps*2
_class(
&$loc:@gradients/pi/add_3_grad/Reshape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
ă
2gradients/pi/add_3_grad/tuple/control_dependency_1Identity!gradients/pi/add_3_grad/Reshape_1)^gradients/pi/add_3_grad/tuple/group_deps*
_output_shapes
: *4
_class*
(&loc:@gradients/pi/add_3_grad/Reshape_1*
T0
c
gradients/pi/add_2_grad/ShapeShapepi/pow*
_output_shapes
:*
T0*
out_type0
i
gradients/pi/add_2_grad/Shape_1Const*
dtype0*
valueB:*
_output_shapes
:
Ă
-gradients/pi/add_2_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/pi/add_2_grad/Shapegradients/pi/add_2_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0
Ă
gradients/pi/add_2_grad/SumSum0gradients/pi/add_3_grad/tuple/control_dependency-gradients/pi/add_2_grad/BroadcastGradientArgs*
	keep_dims( *
T0*
_output_shapes
:*

Tidx0
Ś
gradients/pi/add_2_grad/ReshapeReshapegradients/pi/add_2_grad/Sumgradients/pi/add_2_grad/Shape*
Tshape0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ç
gradients/pi/add_2_grad/Sum_1Sum0gradients/pi/add_3_grad/tuple/control_dependency/gradients/pi/add_2_grad/BroadcastGradientArgs:1*

Tidx0*
_output_shapes
:*
T0*
	keep_dims( 

!gradients/pi/add_2_grad/Reshape_1Reshapegradients/pi/add_2_grad/Sum_1gradients/pi/add_2_grad/Shape_1*
Tshape0*
T0*
_output_shapes
:
v
(gradients/pi/add_2_grad/tuple/group_depsNoOp ^gradients/pi/add_2_grad/Reshape"^gradients/pi/add_2_grad/Reshape_1
î
0gradients/pi/add_2_grad/tuple/control_dependencyIdentitygradients/pi/add_2_grad/Reshape)^gradients/pi/add_2_grad/tuple/group_deps*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*2
_class(
&$loc:@gradients/pi/add_2_grad/Reshape*
T0
ç
2gradients/pi/add_2_grad/tuple/control_dependency_1Identity!gradients/pi/add_2_grad/Reshape_1)^gradients/pi/add_2_grad/tuple/group_deps*
T0*4
_class*
(&loc:@gradients/pi/add_2_grad/Reshape_1*
_output_shapes
:
e
gradients/pi/pow_grad/ShapeShape
pi/truediv*
_output_shapes
:*
T0*
out_type0
`
gradients/pi/pow_grad/Shape_1Const*
dtype0*
valueB *
_output_shapes
: 
˝
+gradients/pi/pow_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/pi/pow_grad/Shapegradients/pi/pow_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙

gradients/pi/pow_grad/mulMul0gradients/pi/add_2_grad/tuple/control_dependencypi/pow/y*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
`
gradients/pi/pow_grad/sub/yConst*
dtype0*
valueB
 *  ?*
_output_shapes
: 
h
gradients/pi/pow_grad/subSubpi/pow/ygradients/pi/pow_grad/sub/y*
T0*
_output_shapes
: 
y
gradients/pi/pow_grad/PowPow
pi/truedivgradients/pi/pow_grad/sub*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

gradients/pi/pow_grad/mul_1Mulgradients/pi/pow_grad/mulgradients/pi/pow_grad/Pow*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ş
gradients/pi/pow_grad/SumSumgradients/pi/pow_grad/mul_1+gradients/pi/pow_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*

Tidx0*
	keep_dims( 
 
gradients/pi/pow_grad/ReshapeReshapegradients/pi/pow_grad/Sumgradients/pi/pow_grad/Shape*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
Tshape0
d
gradients/pi/pow_grad/Greater/yConst*
dtype0*
valueB
 *    *
_output_shapes
: 

gradients/pi/pow_grad/GreaterGreater
pi/truedivgradients/pi/pow_grad/Greater/y*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
o
%gradients/pi/pow_grad/ones_like/ShapeShape
pi/truediv*
_output_shapes
:*
T0*
out_type0
j
%gradients/pi/pow_grad/ones_like/ConstConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
š
gradients/pi/pow_grad/ones_likeFill%gradients/pi/pow_grad/ones_like/Shape%gradients/pi/pow_grad/ones_like/Const*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*

index_type0
¤
gradients/pi/pow_grad/SelectSelectgradients/pi/pow_grad/Greater
pi/truedivgradients/pi/pow_grad/ones_like*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
p
gradients/pi/pow_grad/LogLoggradients/pi/pow_grad/Select*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
k
 gradients/pi/pow_grad/zeros_like	ZerosLike
pi/truediv*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
ś
gradients/pi/pow_grad/Select_1Selectgradients/pi/pow_grad/Greatergradients/pi/pow_grad/Log gradients/pi/pow_grad/zeros_like*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

gradients/pi/pow_grad/mul_2Mul0gradients/pi/add_2_grad/tuple/control_dependencypi/pow*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

gradients/pi/pow_grad/mul_3Mulgradients/pi/pow_grad/mul_2gradients/pi/pow_grad/Select_1*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ž
gradients/pi/pow_grad/Sum_1Sumgradients/pi/pow_grad/mul_3-gradients/pi/pow_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*
_output_shapes
:*

Tidx0

gradients/pi/pow_grad/Reshape_1Reshapegradients/pi/pow_grad/Sum_1gradients/pi/pow_grad/Shape_1*
Tshape0*
_output_shapes
: *
T0
p
&gradients/pi/pow_grad/tuple/group_depsNoOp^gradients/pi/pow_grad/Reshape ^gradients/pi/pow_grad/Reshape_1
ć
.gradients/pi/pow_grad/tuple/control_dependencyIdentitygradients/pi/pow_grad/Reshape'^gradients/pi/pow_grad/tuple/group_deps*
T0*0
_class&
$"loc:@gradients/pi/pow_grad/Reshape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ű
0gradients/pi/pow_grad/tuple/control_dependency_1Identitygradients/pi/pow_grad/Reshape_1'^gradients/pi/pow_grad/tuple/group_deps*2
_class(
&$loc:@gradients/pi/pow_grad/Reshape_1*
T0*
_output_shapes
: 
`
gradients/pi/mul_1_grad/ShapeConst*
_output_shapes
: *
valueB *
dtype0
i
gradients/pi/mul_1_grad/Shape_1Const*
_output_shapes
:*
valueB:*
dtype0
Ă
-gradients/pi/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/pi/mul_1_grad/Shapegradients/pi/mul_1_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0

gradients/pi/mul_1_grad/MulMul2gradients/pi/add_2_grad/tuple/control_dependency_1pi/log_std/read*
_output_shapes
:*
T0
Ź
gradients/pi/mul_1_grad/SumSumgradients/pi/mul_1_grad/Mul-gradients/pi/mul_1_grad/BroadcastGradientArgs*

Tidx0*
T0*
	keep_dims( *
_output_shapes
: 

gradients/pi/mul_1_grad/ReshapeReshapegradients/pi/mul_1_grad/Sumgradients/pi/mul_1_grad/Shape*
_output_shapes
: *
Tshape0*
T0

gradients/pi/mul_1_grad/Mul_1Mul
pi/mul_1/x2gradients/pi/add_2_grad/tuple/control_dependency_1*
T0*
_output_shapes
:
ś
gradients/pi/mul_1_grad/Sum_1Sumgradients/pi/mul_1_grad/Mul_1/gradients/pi/mul_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0

!gradients/pi/mul_1_grad/Reshape_1Reshapegradients/pi/mul_1_grad/Sum_1gradients/pi/mul_1_grad/Shape_1*
T0*
_output_shapes
:*
Tshape0
v
(gradients/pi/mul_1_grad/tuple/group_depsNoOp ^gradients/pi/mul_1_grad/Reshape"^gradients/pi/mul_1_grad/Reshape_1
Ý
0gradients/pi/mul_1_grad/tuple/control_dependencyIdentitygradients/pi/mul_1_grad/Reshape)^gradients/pi/mul_1_grad/tuple/group_deps*
_output_shapes
: *2
_class(
&$loc:@gradients/pi/mul_1_grad/Reshape*
T0
ç
2gradients/pi/mul_1_grad/tuple/control_dependency_1Identity!gradients/pi/mul_1_grad/Reshape_1)^gradients/pi/mul_1_grad/tuple/group_deps*
T0*4
_class*
(&loc:@gradients/pi/mul_1_grad/Reshape_1*
_output_shapes
:
e
gradients/pi/truediv_grad/ShapeShapepi/sub*
_output_shapes
:*
out_type0*
T0
k
!gradients/pi/truediv_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:
É
/gradients/pi/truediv_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/pi/truediv_grad/Shape!gradients/pi/truediv_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙

!gradients/pi/truediv_grad/RealDivRealDiv.gradients/pi/pow_grad/tuple/control_dependencypi/add_1*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
¸
gradients/pi/truediv_grad/SumSum!gradients/pi/truediv_grad/RealDiv/gradients/pi/truediv_grad/BroadcastGradientArgs*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
Ź
!gradients/pi/truediv_grad/ReshapeReshapegradients/pi/truediv_grad/Sumgradients/pi/truediv_grad/Shape*
Tshape0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
^
gradients/pi/truediv_grad/NegNegpi/sub*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

#gradients/pi/truediv_grad/RealDiv_1RealDivgradients/pi/truediv_grad/Negpi/add_1*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

#gradients/pi/truediv_grad/RealDiv_2RealDiv#gradients/pi/truediv_grad/RealDiv_1pi/add_1*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ť
gradients/pi/truediv_grad/mulMul.gradients/pi/pow_grad/tuple/control_dependency#gradients/pi/truediv_grad/RealDiv_2*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
¸
gradients/pi/truediv_grad/Sum_1Sumgradients/pi/truediv_grad/mul1gradients/pi/truediv_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( *
_output_shapes
:
Ľ
#gradients/pi/truediv_grad/Reshape_1Reshapegradients/pi/truediv_grad/Sum_1!gradients/pi/truediv_grad/Shape_1*
_output_shapes
:*
T0*
Tshape0
|
*gradients/pi/truediv_grad/tuple/group_depsNoOp"^gradients/pi/truediv_grad/Reshape$^gradients/pi/truediv_grad/Reshape_1
ö
2gradients/pi/truediv_grad/tuple/control_dependencyIdentity!gradients/pi/truediv_grad/Reshape+^gradients/pi/truediv_grad/tuple/group_deps*4
_class*
(&loc:@gradients/pi/truediv_grad/Reshape*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
ď
4gradients/pi/truediv_grad/tuple/control_dependency_1Identity#gradients/pi/truediv_grad/Reshape_1+^gradients/pi/truediv_grad/tuple/group_deps*
_output_shapes
:*
T0*6
_class,
*(loc:@gradients/pi/truediv_grad/Reshape_1
h
gradients/pi/sub_grad/ShapeShapePlaceholder_1*
_output_shapes
:*
out_type0*
T0
o
gradients/pi/sub_grad/Shape_1Shapepi/dense_2/BiasAdd*
out_type0*
_output_shapes
:*
T0
˝
+gradients/pi/sub_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/pi/sub_grad/Shapegradients/pi/sub_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0
Á
gradients/pi/sub_grad/SumSum2gradients/pi/truediv_grad/tuple/control_dependency+gradients/pi/sub_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
 
gradients/pi/sub_grad/ReshapeReshapegradients/pi/sub_grad/Sumgradients/pi/sub_grad/Shape*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
Tshape0
Ĺ
gradients/pi/sub_grad/Sum_1Sum2gradients/pi/truediv_grad/tuple/control_dependency-gradients/pi/sub_grad/BroadcastGradientArgs:1*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
`
gradients/pi/sub_grad/NegNeggradients/pi/sub_grad/Sum_1*
_output_shapes
:*
T0
¤
gradients/pi/sub_grad/Reshape_1Reshapegradients/pi/sub_grad/Neggradients/pi/sub_grad/Shape_1*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
Tshape0
p
&gradients/pi/sub_grad/tuple/group_depsNoOp^gradients/pi/sub_grad/Reshape ^gradients/pi/sub_grad/Reshape_1
ć
.gradients/pi/sub_grad/tuple/control_dependencyIdentitygradients/pi/sub_grad/Reshape'^gradients/pi/sub_grad/tuple/group_deps*
T0*0
_class&
$"loc:@gradients/pi/sub_grad/Reshape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
ě
0gradients/pi/sub_grad/tuple/control_dependency_1Identitygradients/pi/sub_grad/Reshape_1'^gradients/pi/sub_grad/tuple/group_deps*2
_class(
&$loc:@gradients/pi/sub_grad/Reshape_1*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
g
gradients/pi/add_1_grad/ShapeConst*
_output_shapes
:*
valueB:*
dtype0
b
gradients/pi/add_1_grad/Shape_1Const*
_output_shapes
: *
dtype0*
valueB 
Ă
-gradients/pi/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/pi/add_1_grad/Shapegradients/pi/add_1_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0
É
gradients/pi/add_1_grad/SumSum4gradients/pi/truediv_grad/tuple/control_dependency_1-gradients/pi/add_1_grad/BroadcastGradientArgs*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:

gradients/pi/add_1_grad/ReshapeReshapegradients/pi/add_1_grad/Sumgradients/pi/add_1_grad/Shape*
Tshape0*
T0*
_output_shapes
:
É
gradients/pi/add_1_grad/Sum_1Sum4gradients/pi/truediv_grad/tuple/control_dependency_1/gradients/pi/add_1_grad/BroadcastGradientArgs:1*

Tidx0*
_output_shapes
: *
T0*
	keep_dims( 

!gradients/pi/add_1_grad/Reshape_1Reshapegradients/pi/add_1_grad/Sum_1gradients/pi/add_1_grad/Shape_1*
_output_shapes
: *
Tshape0*
T0
v
(gradients/pi/add_1_grad/tuple/group_depsNoOp ^gradients/pi/add_1_grad/Reshape"^gradients/pi/add_1_grad/Reshape_1
á
0gradients/pi/add_1_grad/tuple/control_dependencyIdentitygradients/pi/add_1_grad/Reshape)^gradients/pi/add_1_grad/tuple/group_deps*2
_class(
&$loc:@gradients/pi/add_1_grad/Reshape*
_output_shapes
:*
T0
ă
2gradients/pi/add_1_grad/tuple/control_dependency_1Identity!gradients/pi/add_1_grad/Reshape_1)^gradients/pi/add_1_grad/tuple/group_deps*
T0*
_output_shapes
: *4
_class*
(&loc:@gradients/pi/add_1_grad/Reshape_1
Ş
-gradients/pi/dense_2/BiasAdd_grad/BiasAddGradBiasAddGrad0gradients/pi/sub_grad/tuple/control_dependency_1*
data_formatNHWC*
_output_shapes
:*
T0

2gradients/pi/dense_2/BiasAdd_grad/tuple/group_depsNoOp.^gradients/pi/dense_2/BiasAdd_grad/BiasAddGrad1^gradients/pi/sub_grad/tuple/control_dependency_1

:gradients/pi/dense_2/BiasAdd_grad/tuple/control_dependencyIdentity0gradients/pi/sub_grad/tuple/control_dependency_13^gradients/pi/dense_2/BiasAdd_grad/tuple/group_deps*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*2
_class(
&$loc:@gradients/pi/sub_grad/Reshape_1

<gradients/pi/dense_2/BiasAdd_grad/tuple/control_dependency_1Identity-gradients/pi/dense_2/BiasAdd_grad/BiasAddGrad3^gradients/pi/dense_2/BiasAdd_grad/tuple/group_deps*@
_class6
42loc:@gradients/pi/dense_2/BiasAdd_grad/BiasAddGrad*
_output_shapes
:*
T0

gradients/pi/Exp_1_grad/mulMul0gradients/pi/add_1_grad/tuple/control_dependencypi/Exp_1*
_output_shapes
:*
T0
Ţ
'gradients/pi/dense_2/MatMul_grad/MatMulMatMul:gradients/pi/dense_2/BiasAdd_grad/tuple/control_dependencypi/dense_2/kernel/read*
transpose_b(*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
transpose_a( 
Đ
)gradients/pi/dense_2/MatMul_grad/MatMul_1MatMulpi/dense_1/Tanh:gradients/pi/dense_2/BiasAdd_grad/tuple/control_dependency*
T0*
_output_shapes
:	*
transpose_b( *
transpose_a(

1gradients/pi/dense_2/MatMul_grad/tuple/group_depsNoOp(^gradients/pi/dense_2/MatMul_grad/MatMul*^gradients/pi/dense_2/MatMul_grad/MatMul_1

9gradients/pi/dense_2/MatMul_grad/tuple/control_dependencyIdentity'gradients/pi/dense_2/MatMul_grad/MatMul2^gradients/pi/dense_2/MatMul_grad/tuple/group_deps*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*:
_class0
.,loc:@gradients/pi/dense_2/MatMul_grad/MatMul

;gradients/pi/dense_2/MatMul_grad/tuple/control_dependency_1Identity)gradients/pi/dense_2/MatMul_grad/MatMul_12^gradients/pi/dense_2/MatMul_grad/tuple/group_deps*
T0*<
_class2
0.loc:@gradients/pi/dense_2/MatMul_grad/MatMul_1*
_output_shapes
:	
ý
gradients/AddNAddN1gradients/pi/add_10_grad/tuple/control_dependency2gradients/pi/mul_1_grad/tuple/control_dependency_1gradients/pi/Exp_1_grad/mul*3
_class)
'%loc:@gradients/pi/add_10_grad/Reshape*
N*
_output_shapes
:*
T0
˛
'gradients/pi/dense_1/Tanh_grad/TanhGradTanhGradpi/dense_1/Tanh9gradients/pi/dense_2/MatMul_grad/tuple/control_dependency*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
˘
-gradients/pi/dense_1/BiasAdd_grad/BiasAddGradBiasAddGrad'gradients/pi/dense_1/Tanh_grad/TanhGrad*
data_formatNHWC*
_output_shapes	
:*
T0

2gradients/pi/dense_1/BiasAdd_grad/tuple/group_depsNoOp.^gradients/pi/dense_1/BiasAdd_grad/BiasAddGrad(^gradients/pi/dense_1/Tanh_grad/TanhGrad

:gradients/pi/dense_1/BiasAdd_grad/tuple/control_dependencyIdentity'gradients/pi/dense_1/Tanh_grad/TanhGrad3^gradients/pi/dense_1/BiasAdd_grad/tuple/group_deps*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*:
_class0
.,loc:@gradients/pi/dense_1/Tanh_grad/TanhGrad

<gradients/pi/dense_1/BiasAdd_grad/tuple/control_dependency_1Identity-gradients/pi/dense_1/BiasAdd_grad/BiasAddGrad3^gradients/pi/dense_1/BiasAdd_grad/tuple/group_deps*
T0*
_output_shapes	
:*@
_class6
42loc:@gradients/pi/dense_1/BiasAdd_grad/BiasAddGrad
Ţ
'gradients/pi/dense_1/MatMul_grad/MatMulMatMul:gradients/pi/dense_1/BiasAdd_grad/tuple/control_dependencypi/dense_1/kernel/read*
transpose_b(*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( 
Ď
)gradients/pi/dense_1/MatMul_grad/MatMul_1MatMulpi/dense/Tanh:gradients/pi/dense_1/BiasAdd_grad/tuple/control_dependency*
transpose_a(* 
_output_shapes
:
*
T0*
transpose_b( 

1gradients/pi/dense_1/MatMul_grad/tuple/group_depsNoOp(^gradients/pi/dense_1/MatMul_grad/MatMul*^gradients/pi/dense_1/MatMul_grad/MatMul_1

9gradients/pi/dense_1/MatMul_grad/tuple/control_dependencyIdentity'gradients/pi/dense_1/MatMul_grad/MatMul2^gradients/pi/dense_1/MatMul_grad/tuple/group_deps*
T0*:
_class0
.,loc:@gradients/pi/dense_1/MatMul_grad/MatMul*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

;gradients/pi/dense_1/MatMul_grad/tuple/control_dependency_1Identity)gradients/pi/dense_1/MatMul_grad/MatMul_12^gradients/pi/dense_1/MatMul_grad/tuple/group_deps*
T0* 
_output_shapes
:
*<
_class2
0.loc:@gradients/pi/dense_1/MatMul_grad/MatMul_1
Ž
%gradients/pi/dense/Tanh_grad/TanhGradTanhGradpi/dense/Tanh9gradients/pi/dense_1/MatMul_grad/tuple/control_dependency*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

+gradients/pi/dense/BiasAdd_grad/BiasAddGradBiasAddGrad%gradients/pi/dense/Tanh_grad/TanhGrad*
_output_shapes	
:*
data_formatNHWC*
T0

0gradients/pi/dense/BiasAdd_grad/tuple/group_depsNoOp,^gradients/pi/dense/BiasAdd_grad/BiasAddGrad&^gradients/pi/dense/Tanh_grad/TanhGrad

8gradients/pi/dense/BiasAdd_grad/tuple/control_dependencyIdentity%gradients/pi/dense/Tanh_grad/TanhGrad1^gradients/pi/dense/BiasAdd_grad/tuple/group_deps*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*8
_class.
,*loc:@gradients/pi/dense/Tanh_grad/TanhGrad

:gradients/pi/dense/BiasAdd_grad/tuple/control_dependency_1Identity+gradients/pi/dense/BiasAdd_grad/BiasAddGrad1^gradients/pi/dense/BiasAdd_grad/tuple/group_deps*
T0*>
_class4
20loc:@gradients/pi/dense/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:
×
%gradients/pi/dense/MatMul_grad/MatMulMatMul8gradients/pi/dense/BiasAdd_grad/tuple/control_dependencypi/dense/kernel/read*
transpose_a( *'
_output_shapes
:˙˙˙˙˙˙˙˙˙<*
T0*
transpose_b(
Č
'gradients/pi/dense/MatMul_grad/MatMul_1MatMulPlaceholder8gradients/pi/dense/BiasAdd_grad/tuple/control_dependency*
_output_shapes
:	<*
transpose_a(*
T0*
transpose_b( 

/gradients/pi/dense/MatMul_grad/tuple/group_depsNoOp&^gradients/pi/dense/MatMul_grad/MatMul(^gradients/pi/dense/MatMul_grad/MatMul_1

7gradients/pi/dense/MatMul_grad/tuple/control_dependencyIdentity%gradients/pi/dense/MatMul_grad/MatMul0^gradients/pi/dense/MatMul_grad/tuple/group_deps*8
_class.
,*loc:@gradients/pi/dense/MatMul_grad/MatMul*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙<

9gradients/pi/dense/MatMul_grad/tuple/control_dependency_1Identity'gradients/pi/dense/MatMul_grad/MatMul_10^gradients/pi/dense/MatMul_grad/tuple/group_deps*
T0*:
_class0
.,loc:@gradients/pi/dense/MatMul_grad/MatMul_1*
_output_shapes
:	<
`
Reshape/shapeConst*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙*
_output_shapes
:

ReshapeReshape9gradients/pi/dense/MatMul_grad/tuple/control_dependency_1Reshape/shape*
T0*
_output_shapes	
:x*
Tshape0
b
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙

	Reshape_1Reshape:gradients/pi/dense/BiasAdd_grad/tuple/control_dependency_1Reshape_1/shape*
_output_shapes	
:*
T0*
Tshape0
b
Reshape_2/shapeConst*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙*
_output_shapes
:

	Reshape_2Reshape;gradients/pi/dense_1/MatMul_grad/tuple/control_dependency_1Reshape_2/shape*
T0*
Tshape0*
_output_shapes

:
b
Reshape_3/shapeConst*
dtype0*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙

	Reshape_3Reshape<gradients/pi/dense_1/BiasAdd_grad/tuple/control_dependency_1Reshape_3/shape*
T0*
Tshape0*
_output_shapes	
:
b
Reshape_4/shapeConst*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙*
_output_shapes
:

	Reshape_4Reshape;gradients/pi/dense_2/MatMul_grad/tuple/control_dependency_1Reshape_4/shape*
T0*
Tshape0*
_output_shapes	
:
b
Reshape_5/shapeConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙

	Reshape_5Reshape<gradients/pi/dense_2/BiasAdd_grad/tuple/control_dependency_1Reshape_5/shape*
T0*
Tshape0*
_output_shapes
:
b
Reshape_6/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
h
	Reshape_6Reshapegradients/AddNReshape_6/shape*
Tshape0*
T0*
_output_shapes
:
M
concat/axisConst*
dtype0*
value	B : *
_output_shapes
: 
Ś
concatConcatV2Reshape	Reshape_1	Reshape_2	Reshape_3	Reshape_4	Reshape_5	Reshape_6concat/axis*
N*
_output_shapes

:*
T0*

Tidx0
h
PyFuncPyFuncconcat*
token
pyfunc_0*
_output_shapes

:*
Tin
2*
Tout
2
l
Const_3Const*1
value(B&" <                    *
_output_shapes
:*
dtype0
Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 

splitSplitVPyFuncConst_3split/split_dim*

Tlen0*
	num_split*D
_output_shapes2
0:x::::::*
T0
`
Reshape_7/shapeConst*
dtype0*
_output_shapes
:*
valueB"<      
d
	Reshape_7ReshapesplitReshape_7/shape*
_output_shapes
:	<*
Tshape0*
T0
Z
Reshape_8/shapeConst*
_output_shapes
:*
valueB:*
dtype0
b
	Reshape_8Reshapesplit:1Reshape_8/shape*
Tshape0*
_output_shapes	
:*
T0
`
Reshape_9/shapeConst*
dtype0*
_output_shapes
:*
valueB"      
g
	Reshape_9Reshapesplit:2Reshape_9/shape*
T0*
Tshape0* 
_output_shapes
:

[
Reshape_10/shapeConst*
dtype0*
_output_shapes
:*
valueB:
d

Reshape_10Reshapesplit:3Reshape_10/shape*
_output_shapes	
:*
Tshape0*
T0
a
Reshape_11/shapeConst*
_output_shapes
:*
valueB"      *
dtype0
h

Reshape_11Reshapesplit:4Reshape_11/shape*
T0*
Tshape0*
_output_shapes
:	
Z
Reshape_12/shapeConst*
dtype0*
valueB:*
_output_shapes
:
c

Reshape_12Reshapesplit:5Reshape_12/shape*
Tshape0*
_output_shapes
:*
T0
Z
Reshape_13/shapeConst*
_output_shapes
:*
valueB:*
dtype0
c

Reshape_13Reshapesplit:6Reshape_13/shape*
Tshape0*
_output_shapes
:*
T0

beta1_power/initial_valueConst*
_output_shapes
: *
valueB
 *fff?*
dtype0* 
_class
loc:@pi/dense/bias

beta1_power
VariableV2* 
_class
loc:@pi/dense/bias*
shape: *
dtype0*
	container *
shared_name *
_output_shapes
: 
°
beta1_power/AssignAssignbeta1_powerbeta1_power/initial_value*
validate_shape(*
use_locking(*
_output_shapes
: *
T0* 
_class
loc:@pi/dense/bias
l
beta1_power/readIdentitybeta1_power*
T0* 
_class
loc:@pi/dense/bias*
_output_shapes
: 

beta2_power/initial_valueConst*
_output_shapes
: *
dtype0*
valueB
 *wž?* 
_class
loc:@pi/dense/bias

beta2_power
VariableV2*
shape: * 
_class
loc:@pi/dense/bias*
	container *
_output_shapes
: *
dtype0*
shared_name 
°
beta2_power/AssignAssignbeta2_powerbeta2_power/initial_value* 
_class
loc:@pi/dense/bias*
_output_shapes
: *
use_locking(*
T0*
validate_shape(
l
beta2_power/readIdentitybeta2_power*
T0* 
_class
loc:@pi/dense/bias*
_output_shapes
: 
Ť
6pi/dense/kernel/Adam/Initializer/zeros/shape_as_tensorConst*"
_class
loc:@pi/dense/kernel*
dtype0*
valueB"<      *
_output_shapes
:

,pi/dense/kernel/Adam/Initializer/zeros/ConstConst*
_output_shapes
: *"
_class
loc:@pi/dense/kernel*
dtype0*
valueB
 *    
ô
&pi/dense/kernel/Adam/Initializer/zerosFill6pi/dense/kernel/Adam/Initializer/zeros/shape_as_tensor,pi/dense/kernel/Adam/Initializer/zeros/Const*
_output_shapes
:	<*
T0*

index_type0*"
_class
loc:@pi/dense/kernel
Ž
pi/dense/kernel/Adam
VariableV2*"
_class
loc:@pi/dense/kernel*
dtype0*
_output_shapes
:	<*
shape:	<*
	container *
shared_name 
Ú
pi/dense/kernel/Adam/AssignAssignpi/dense/kernel/Adam&pi/dense/kernel/Adam/Initializer/zeros*
_output_shapes
:	<*
use_locking(*"
_class
loc:@pi/dense/kernel*
validate_shape(*
T0

pi/dense/kernel/Adam/readIdentitypi/dense/kernel/Adam*"
_class
loc:@pi/dense/kernel*
T0*
_output_shapes
:	<
­
8pi/dense/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*
valueB"<      *
dtype0*"
_class
loc:@pi/dense/kernel

.pi/dense/kernel/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    *
dtype0*"
_class
loc:@pi/dense/kernel*
_output_shapes
: 
ú
(pi/dense/kernel/Adam_1/Initializer/zerosFill8pi/dense/kernel/Adam_1/Initializer/zeros/shape_as_tensor.pi/dense/kernel/Adam_1/Initializer/zeros/Const*"
_class
loc:@pi/dense/kernel*

index_type0*
_output_shapes
:	<*
T0
°
pi/dense/kernel/Adam_1
VariableV2*
shared_name *
	container *"
_class
loc:@pi/dense/kernel*
dtype0*
shape:	<*
_output_shapes
:	<
ŕ
pi/dense/kernel/Adam_1/AssignAssignpi/dense/kernel/Adam_1(pi/dense/kernel/Adam_1/Initializer/zeros*
validate_shape(*"
_class
loc:@pi/dense/kernel*
_output_shapes
:	<*
T0*
use_locking(

pi/dense/kernel/Adam_1/readIdentitypi/dense/kernel/Adam_1*
_output_shapes
:	<*"
_class
loc:@pi/dense/kernel*
T0

$pi/dense/bias/Adam/Initializer/zerosConst*
_output_shapes	
:*
dtype0* 
_class
loc:@pi/dense/bias*
valueB*    
˘
pi/dense/bias/Adam
VariableV2*
shape:*
dtype0* 
_class
loc:@pi/dense/bias*
_output_shapes	
:*
shared_name *
	container 
Î
pi/dense/bias/Adam/AssignAssignpi/dense/bias/Adam$pi/dense/bias/Adam/Initializer/zeros*
T0*
use_locking(* 
_class
loc:@pi/dense/bias*
_output_shapes	
:*
validate_shape(

pi/dense/bias/Adam/readIdentitypi/dense/bias/Adam* 
_class
loc:@pi/dense/bias*
T0*
_output_shapes	
:

&pi/dense/bias/Adam_1/Initializer/zerosConst*
valueB*    *
_output_shapes	
:*
dtype0* 
_class
loc:@pi/dense/bias
¤
pi/dense/bias/Adam_1
VariableV2*
dtype0*
shape:*
_output_shapes	
:*
shared_name *
	container * 
_class
loc:@pi/dense/bias
Ô
pi/dense/bias/Adam_1/AssignAssignpi/dense/bias/Adam_1&pi/dense/bias/Adam_1/Initializer/zeros*
use_locking(*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes	
:

pi/dense/bias/Adam_1/readIdentitypi/dense/bias/Adam_1*
T0*
_output_shapes	
:* 
_class
loc:@pi/dense/bias
Ż
8pi/dense_1/kernel/Adam/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*
valueB"      *$
_class
loc:@pi/dense_1/kernel*
dtype0

.pi/dense_1/kernel/Adam/Initializer/zeros/ConstConst*
_output_shapes
: *
valueB
 *    *$
_class
loc:@pi/dense_1/kernel*
dtype0
ý
(pi/dense_1/kernel/Adam/Initializer/zerosFill8pi/dense_1/kernel/Adam/Initializer/zeros/shape_as_tensor.pi/dense_1/kernel/Adam/Initializer/zeros/Const* 
_output_shapes
:
*

index_type0*$
_class
loc:@pi/dense_1/kernel*
T0
´
pi/dense_1/kernel/Adam
VariableV2* 
_output_shapes
:
*
shape:
*
dtype0*
shared_name *$
_class
loc:@pi/dense_1/kernel*
	container 
ă
pi/dense_1/kernel/Adam/AssignAssignpi/dense_1/kernel/Adam(pi/dense_1/kernel/Adam/Initializer/zeros* 
_output_shapes
:
*
T0*$
_class
loc:@pi/dense_1/kernel*
use_locking(*
validate_shape(

pi/dense_1/kernel/Adam/readIdentitypi/dense_1/kernel/Adam*$
_class
loc:@pi/dense_1/kernel* 
_output_shapes
:
*
T0
ą
:pi/dense_1/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*$
_class
loc:@pi/dense_1/kernel*
dtype0*
_output_shapes
:*
valueB"      

0pi/dense_1/kernel/Adam_1/Initializer/zeros/ConstConst*$
_class
loc:@pi/dense_1/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 

*pi/dense_1/kernel/Adam_1/Initializer/zerosFill:pi/dense_1/kernel/Adam_1/Initializer/zeros/shape_as_tensor0pi/dense_1/kernel/Adam_1/Initializer/zeros/Const*$
_class
loc:@pi/dense_1/kernel* 
_output_shapes
:
*

index_type0*
T0
ś
pi/dense_1/kernel/Adam_1
VariableV2* 
_output_shapes
:
*
shape:
*
shared_name *$
_class
loc:@pi/dense_1/kernel*
	container *
dtype0
é
pi/dense_1/kernel/Adam_1/AssignAssignpi/dense_1/kernel/Adam_1*pi/dense_1/kernel/Adam_1/Initializer/zeros*
T0*
use_locking(*
validate_shape(* 
_output_shapes
:
*$
_class
loc:@pi/dense_1/kernel

pi/dense_1/kernel/Adam_1/readIdentitypi/dense_1/kernel/Adam_1*$
_class
loc:@pi/dense_1/kernel* 
_output_shapes
:
*
T0

&pi/dense_1/bias/Adam/Initializer/zerosConst*"
_class
loc:@pi/dense_1/bias*
dtype0*
valueB*    *
_output_shapes	
:
Ś
pi/dense_1/bias/Adam
VariableV2*
_output_shapes	
:*
shape:*
shared_name *
dtype0*
	container *"
_class
loc:@pi/dense_1/bias
Ö
pi/dense_1/bias/Adam/AssignAssignpi/dense_1/bias/Adam&pi/dense_1/bias/Adam/Initializer/zeros*
T0*"
_class
loc:@pi/dense_1/bias*
_output_shapes	
:*
validate_shape(*
use_locking(

pi/dense_1/bias/Adam/readIdentitypi/dense_1/bias/Adam*
_output_shapes	
:*
T0*"
_class
loc:@pi/dense_1/bias

(pi/dense_1/bias/Adam_1/Initializer/zerosConst*
_output_shapes	
:*"
_class
loc:@pi/dense_1/bias*
valueB*    *
dtype0
¨
pi/dense_1/bias/Adam_1
VariableV2*
	container *
_output_shapes	
:*
shape:*
shared_name *"
_class
loc:@pi/dense_1/bias*
dtype0
Ü
pi/dense_1/bias/Adam_1/AssignAssignpi/dense_1/bias/Adam_1(pi/dense_1/bias/Adam_1/Initializer/zeros*"
_class
loc:@pi/dense_1/bias*
_output_shapes	
:*
use_locking(*
T0*
validate_shape(

pi/dense_1/bias/Adam_1/readIdentitypi/dense_1/bias/Adam_1*"
_class
loc:@pi/dense_1/bias*
_output_shapes	
:*
T0
Ľ
(pi/dense_2/kernel/Adam/Initializer/zerosConst*
valueB	*    *
_output_shapes
:	*$
_class
loc:@pi/dense_2/kernel*
dtype0
˛
pi/dense_2/kernel/Adam
VariableV2*
_output_shapes
:	*
shared_name *
dtype0*$
_class
loc:@pi/dense_2/kernel*
	container *
shape:	
â
pi/dense_2/kernel/Adam/AssignAssignpi/dense_2/kernel/Adam(pi/dense_2/kernel/Adam/Initializer/zeros*
use_locking(*
T0*$
_class
loc:@pi/dense_2/kernel*
validate_shape(*
_output_shapes
:	

pi/dense_2/kernel/Adam/readIdentitypi/dense_2/kernel/Adam*
_output_shapes
:	*
T0*$
_class
loc:@pi/dense_2/kernel
§
*pi/dense_2/kernel/Adam_1/Initializer/zerosConst*$
_class
loc:@pi/dense_2/kernel*
valueB	*    *
_output_shapes
:	*
dtype0
´
pi/dense_2/kernel/Adam_1
VariableV2*
shape:	*
_output_shapes
:	*
	container *$
_class
loc:@pi/dense_2/kernel*
shared_name *
dtype0
č
pi/dense_2/kernel/Adam_1/AssignAssignpi/dense_2/kernel/Adam_1*pi/dense_2/kernel/Adam_1/Initializer/zeros*
_output_shapes
:	*
use_locking(*$
_class
loc:@pi/dense_2/kernel*
validate_shape(*
T0

pi/dense_2/kernel/Adam_1/readIdentitypi/dense_2/kernel/Adam_1*$
_class
loc:@pi/dense_2/kernel*
_output_shapes
:	*
T0

&pi/dense_2/bias/Adam/Initializer/zerosConst*
dtype0*
valueB*    *
_output_shapes
:*"
_class
loc:@pi/dense_2/bias
¤
pi/dense_2/bias/Adam
VariableV2*
shared_name *"
_class
loc:@pi/dense_2/bias*
shape:*
_output_shapes
:*
	container *
dtype0
Ő
pi/dense_2/bias/Adam/AssignAssignpi/dense_2/bias/Adam&pi/dense_2/bias/Adam/Initializer/zeros*
_output_shapes
:*"
_class
loc:@pi/dense_2/bias*
validate_shape(*
use_locking(*
T0

pi/dense_2/bias/Adam/readIdentitypi/dense_2/bias/Adam*"
_class
loc:@pi/dense_2/bias*
_output_shapes
:*
T0

(pi/dense_2/bias/Adam_1/Initializer/zerosConst*
valueB*    *
_output_shapes
:*"
_class
loc:@pi/dense_2/bias*
dtype0
Ś
pi/dense_2/bias/Adam_1
VariableV2*
dtype0*
shape:*"
_class
loc:@pi/dense_2/bias*
	container *
_output_shapes
:*
shared_name 
Ű
pi/dense_2/bias/Adam_1/AssignAssignpi/dense_2/bias/Adam_1(pi/dense_2/bias/Adam_1/Initializer/zeros*
T0*
validate_shape(*"
_class
loc:@pi/dense_2/bias*
_output_shapes
:*
use_locking(

pi/dense_2/bias/Adam_1/readIdentitypi/dense_2/bias/Adam_1*
T0*"
_class
loc:@pi/dense_2/bias*
_output_shapes
:

!pi/log_std/Adam/Initializer/zerosConst*
valueB*    *
_output_shapes
:*
_class
loc:@pi/log_std*
dtype0

pi/log_std/Adam
VariableV2*
shape:*
shared_name *
_class
loc:@pi/log_std*
dtype0*
	container *
_output_shapes
:
Á
pi/log_std/Adam/AssignAssignpi/log_std/Adam!pi/log_std/Adam/Initializer/zeros*
_class
loc:@pi/log_std*
_output_shapes
:*
T0*
use_locking(*
validate_shape(
u
pi/log_std/Adam/readIdentitypi/log_std/Adam*
_output_shapes
:*
T0*
_class
loc:@pi/log_std

#pi/log_std/Adam_1/Initializer/zerosConst*
_output_shapes
:*
dtype0*
_class
loc:@pi/log_std*
valueB*    

pi/log_std/Adam_1
VariableV2*
_output_shapes
:*
shared_name *
_class
loc:@pi/log_std*
dtype0*
	container *
shape:
Ç
pi/log_std/Adam_1/AssignAssignpi/log_std/Adam_1#pi/log_std/Adam_1/Initializer/zeros*
_class
loc:@pi/log_std*
_output_shapes
:*
validate_shape(*
use_locking(*
T0
y
pi/log_std/Adam_1/readIdentitypi/log_std/Adam_1*
_output_shapes
:*
T0*
_class
loc:@pi/log_std
W
Adam/learning_rateConst*
dtype0*
valueB
 *RI9*
_output_shapes
: 
O

Adam/beta1Const*
dtype0*
_output_shapes
: *
valueB
 *fff?
O

Adam/beta2Const*
_output_shapes
: *
dtype0*
valueB
 *wž?
Q
Adam/epsilonConst*
dtype0*
valueB
 *wĚ+2*
_output_shapes
: 
Ď
%Adam/update_pi/dense/kernel/ApplyAdam	ApplyAdampi/dense/kernelpi/dense/kernel/Adampi/dense/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon	Reshape_7*"
_class
loc:@pi/dense/kernel*
T0*
use_locking( *
use_nesterov( *
_output_shapes
:	<
Á
#Adam/update_pi/dense/bias/ApplyAdam	ApplyAdampi/dense/biaspi/dense/bias/Adampi/dense/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon	Reshape_8*
T0* 
_class
loc:@pi/dense/bias*
use_nesterov( *
_output_shapes	
:*
use_locking( 
Ú
'Adam/update_pi/dense_1/kernel/ApplyAdam	ApplyAdampi/dense_1/kernelpi/dense_1/kernel/Adampi/dense_1/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon	Reshape_9* 
_output_shapes
:
*
use_nesterov( *
T0*$
_class
loc:@pi/dense_1/kernel*
use_locking( 
Ě
%Adam/update_pi/dense_1/bias/ApplyAdam	ApplyAdampi/dense_1/biaspi/dense_1/bias/Adampi/dense_1/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon
Reshape_10*
use_nesterov( *
use_locking( *
T0*"
_class
loc:@pi/dense_1/bias*
_output_shapes	
:
Ú
'Adam/update_pi/dense_2/kernel/ApplyAdam	ApplyAdampi/dense_2/kernelpi/dense_2/kernel/Adampi/dense_2/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon
Reshape_11*
use_nesterov( *
T0*
_output_shapes
:	*$
_class
loc:@pi/dense_2/kernel*
use_locking( 
Ë
%Adam/update_pi/dense_2/bias/ApplyAdam	ApplyAdampi/dense_2/biaspi/dense_2/bias/Adampi/dense_2/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon
Reshape_12*
T0*
use_locking( *
use_nesterov( *"
_class
loc:@pi/dense_2/bias*
_output_shapes
:
˛
 Adam/update_pi/log_std/ApplyAdam	ApplyAdam
pi/log_stdpi/log_std/Adampi/log_std/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon
Reshape_13*
T0*
use_nesterov( *
use_locking( *
_class
loc:@pi/log_std*
_output_shapes
:

Adam/mulMulbeta1_power/read
Adam/beta1$^Adam/update_pi/dense/bias/ApplyAdam&^Adam/update_pi/dense/kernel/ApplyAdam&^Adam/update_pi/dense_1/bias/ApplyAdam(^Adam/update_pi/dense_1/kernel/ApplyAdam&^Adam/update_pi/dense_2/bias/ApplyAdam(^Adam/update_pi/dense_2/kernel/ApplyAdam!^Adam/update_pi/log_std/ApplyAdam*
_output_shapes
: *
T0* 
_class
loc:@pi/dense/bias

Adam/AssignAssignbeta1_powerAdam/mul*
use_locking( *
_output_shapes
: * 
_class
loc:@pi/dense/bias*
validate_shape(*
T0


Adam/mul_1Mulbeta2_power/read
Adam/beta2$^Adam/update_pi/dense/bias/ApplyAdam&^Adam/update_pi/dense/kernel/ApplyAdam&^Adam/update_pi/dense_1/bias/ApplyAdam(^Adam/update_pi/dense_1/kernel/ApplyAdam&^Adam/update_pi/dense_2/bias/ApplyAdam(^Adam/update_pi/dense_2/kernel/ApplyAdam!^Adam/update_pi/log_std/ApplyAdam*
_output_shapes
: * 
_class
loc:@pi/dense/bias*
T0

Adam/Assign_1Assignbeta2_power
Adam/mul_1*
T0*
use_locking( *
validate_shape(* 
_class
loc:@pi/dense/bias*
_output_shapes
: 
ż
AdamNoOp^Adam/Assign^Adam/Assign_1$^Adam/update_pi/dense/bias/ApplyAdam&^Adam/update_pi/dense/kernel/ApplyAdam&^Adam/update_pi/dense_1/bias/ApplyAdam(^Adam/update_pi/dense_1/kernel/ApplyAdam&^Adam/update_pi/dense_2/bias/ApplyAdam(^Adam/update_pi/dense_2/kernel/ApplyAdam!^Adam/update_pi/log_std/ApplyAdam
j
Reshape_14/shapeConst^Adam*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
q

Reshape_14Reshapepi/dense/kernel/readReshape_14/shape*
_output_shapes	
:x*
T0*
Tshape0
j
Reshape_15/shapeConst^Adam*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙*
_output_shapes
:
o

Reshape_15Reshapepi/dense/bias/readReshape_15/shape*
Tshape0*
_output_shapes	
:*
T0
j
Reshape_16/shapeConst^Adam*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
t

Reshape_16Reshapepi/dense_1/kernel/readReshape_16/shape*
_output_shapes

:*
Tshape0*
T0
j
Reshape_17/shapeConst^Adam*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0
q

Reshape_17Reshapepi/dense_1/bias/readReshape_17/shape*
T0*
Tshape0*
_output_shapes	
:
j
Reshape_18/shapeConst^Adam*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0
s

Reshape_18Reshapepi/dense_2/kernel/readReshape_18/shape*
Tshape0*
_output_shapes	
:*
T0
j
Reshape_19/shapeConst^Adam*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
p

Reshape_19Reshapepi/dense_2/bias/readReshape_19/shape*
T0*
_output_shapes
:*
Tshape0
j
Reshape_20/shapeConst^Adam*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0
k

Reshape_20Reshapepi/log_std/readReshape_20/shape*
Tshape0*
T0*
_output_shapes
:
V
concat_1/axisConst^Adam*
value	B : *
dtype0*
_output_shapes
: 
ł
concat_1ConcatV2
Reshape_14
Reshape_15
Reshape_16
Reshape_17
Reshape_18
Reshape_19
Reshape_20concat_1/axis*
T0*
_output_shapes

:*

Tidx0*
N
h
PyFunc_1PyFuncconcat_1*
Tout
2*
Tin
2*
_output_shapes
:*
token
pyfunc_1
s
Const_4Const^Adam*
_output_shapes
:*
dtype0*1
value(B&" <                    
Z
split_1/split_dimConst^Adam*
_output_shapes
: *
dtype0*
value	B : 

split_1SplitVPyFunc_1Const_4split_1/split_dim*
T0*

Tlen0*
	num_split*0
_output_shapes
:::::::
h
Reshape_21/shapeConst^Adam*
_output_shapes
:*
dtype0*
valueB"<      
h

Reshape_21Reshapesplit_1Reshape_21/shape*
_output_shapes
:	<*
T0*
Tshape0
b
Reshape_22/shapeConst^Adam*
valueB:*
dtype0*
_output_shapes
:
f

Reshape_22Reshape	split_1:1Reshape_22/shape*
Tshape0*
_output_shapes	
:*
T0
h
Reshape_23/shapeConst^Adam*
_output_shapes
:*
dtype0*
valueB"      
k

Reshape_23Reshape	split_1:2Reshape_23/shape*
Tshape0*
T0* 
_output_shapes
:

b
Reshape_24/shapeConst^Adam*
valueB:*
dtype0*
_output_shapes
:
f

Reshape_24Reshape	split_1:3Reshape_24/shape*
Tshape0*
T0*
_output_shapes	
:
h
Reshape_25/shapeConst^Adam*
valueB"      *
dtype0*
_output_shapes
:
j

Reshape_25Reshape	split_1:4Reshape_25/shape*
T0*
Tshape0*
_output_shapes
:	
a
Reshape_26/shapeConst^Adam*
valueB:*
dtype0*
_output_shapes
:
e

Reshape_26Reshape	split_1:5Reshape_26/shape*
_output_shapes
:*
Tshape0*
T0
a
Reshape_27/shapeConst^Adam*
valueB:*
_output_shapes
:*
dtype0
e

Reshape_27Reshape	split_1:6Reshape_27/shape*
Tshape0*
T0*
_output_shapes
:
¤
AssignAssignpi/dense/kernel
Reshape_21*
validate_shape(*"
_class
loc:@pi/dense/kernel*
_output_shapes
:	<*
use_locking(*
T0

Assign_1Assignpi/dense/bias
Reshape_22* 
_class
loc:@pi/dense/bias*
validate_shape(*
T0*
_output_shapes	
:*
use_locking(
Ť
Assign_2Assignpi/dense_1/kernel
Reshape_23*
T0* 
_output_shapes
:
*$
_class
loc:@pi/dense_1/kernel*
use_locking(*
validate_shape(
˘
Assign_3Assignpi/dense_1/bias
Reshape_24*
_output_shapes	
:*"
_class
loc:@pi/dense_1/bias*
use_locking(*
T0*
validate_shape(
Ş
Assign_4Assignpi/dense_2/kernel
Reshape_25*$
_class
loc:@pi/dense_2/kernel*
validate_shape(*
use_locking(*
_output_shapes
:	*
T0
Ą
Assign_5Assignpi/dense_2/bias
Reshape_26*
_output_shapes
:*
use_locking(*"
_class
loc:@pi/dense_2/bias*
validate_shape(*
T0

Assign_6Assign
pi/log_std
Reshape_27*
validate_shape(*
T0*
use_locking(*
_output_shapes
:*
_class
loc:@pi/log_std
d

group_depsNoOp^Adam^Assign	^Assign_1	^Assign_2	^Assign_3	^Assign_4	^Assign_5	^Assign_6
(
group_deps_1NoOp^Adam^group_deps
U
sub_1SubPlaceholder_4
vf/Squeeze*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
J
pow/yConst*
_output_shapes
: *
valueB
 *   @*
dtype0
F
powPowsub_1pow/y*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
Q
Const_5Const*
_output_shapes
:*
valueB: *
dtype0
Z
Mean_3MeanpowConst_5*
_output_shapes
: *
	keep_dims( *
T0*

Tidx0
U
sub_2SubPlaceholder_5
vc/Squeeze*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
L
pow_1/yConst*
valueB
 *   @*
_output_shapes
: *
dtype0
J
pow_1Powsub_2pow_1/y*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Q
Const_6Const*
dtype0*
valueB: *
_output_shapes
:
\
Mean_4Meanpow_1Const_6*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
=
add_1AddMean_3Mean_4*
T0*
_output_shapes
: 
T
gradients_1/ShapeConst*
valueB *
_output_shapes
: *
dtype0
Z
gradients_1/grad_ys_0Const*
_output_shapes
: *
valueB
 *  ?*
dtype0
u
gradients_1/FillFillgradients_1/Shapegradients_1/grad_ys_0*

index_type0*
T0*
_output_shapes
: 
B
'gradients_1/add_1_grad/tuple/group_depsNoOp^gradients_1/Fill
˝
/gradients_1/add_1_grad/tuple/control_dependencyIdentitygradients_1/Fill(^gradients_1/add_1_grad/tuple/group_deps*
T0*
_output_shapes
: *#
_class
loc:@gradients_1/Fill
ż
1gradients_1/add_1_grad/tuple/control_dependency_1Identitygradients_1/Fill(^gradients_1/add_1_grad/tuple/group_deps*
T0*
_output_shapes
: *#
_class
loc:@gradients_1/Fill
o
%gradients_1/Mean_3_grad/Reshape/shapeConst*
dtype0*
valueB:*
_output_shapes
:
ľ
gradients_1/Mean_3_grad/ReshapeReshape/gradients_1/add_1_grad/tuple/control_dependency%gradients_1/Mean_3_grad/Reshape/shape*
_output_shapes
:*
Tshape0*
T0
`
gradients_1/Mean_3_grad/ShapeShapepow*
T0*
out_type0*
_output_shapes
:
¤
gradients_1/Mean_3_grad/TileTilegradients_1/Mean_3_grad/Reshapegradients_1/Mean_3_grad/Shape*

Tmultiples0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
b
gradients_1/Mean_3_grad/Shape_1Shapepow*
_output_shapes
:*
out_type0*
T0
b
gradients_1/Mean_3_grad/Shape_2Const*
valueB *
_output_shapes
: *
dtype0
g
gradients_1/Mean_3_grad/ConstConst*
_output_shapes
:*
valueB: *
dtype0
˘
gradients_1/Mean_3_grad/ProdProdgradients_1/Mean_3_grad/Shape_1gradients_1/Mean_3_grad/Const*
	keep_dims( *
_output_shapes
: *

Tidx0*
T0
i
gradients_1/Mean_3_grad/Const_1Const*
dtype0*
valueB: *
_output_shapes
:
Ś
gradients_1/Mean_3_grad/Prod_1Prodgradients_1/Mean_3_grad/Shape_2gradients_1/Mean_3_grad/Const_1*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
c
!gradients_1/Mean_3_grad/Maximum/yConst*
dtype0*
value	B :*
_output_shapes
: 

gradients_1/Mean_3_grad/MaximumMaximumgradients_1/Mean_3_grad/Prod_1!gradients_1/Mean_3_grad/Maximum/y*
T0*
_output_shapes
: 

 gradients_1/Mean_3_grad/floordivFloorDivgradients_1/Mean_3_grad/Prodgradients_1/Mean_3_grad/Maximum*
_output_shapes
: *
T0

gradients_1/Mean_3_grad/CastCast gradients_1/Mean_3_grad/floordiv*

DstT0*
Truncate( *
_output_shapes
: *

SrcT0

gradients_1/Mean_3_grad/truedivRealDivgradients_1/Mean_3_grad/Tilegradients_1/Mean_3_grad/Cast*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
o
%gradients_1/Mean_4_grad/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
ˇ
gradients_1/Mean_4_grad/ReshapeReshape1gradients_1/add_1_grad/tuple/control_dependency_1%gradients_1/Mean_4_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
:
b
gradients_1/Mean_4_grad/ShapeShapepow_1*
_output_shapes
:*
T0*
out_type0
¤
gradients_1/Mean_4_grad/TileTilegradients_1/Mean_4_grad/Reshapegradients_1/Mean_4_grad/Shape*

Tmultiples0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
d
gradients_1/Mean_4_grad/Shape_1Shapepow_1*
out_type0*
_output_shapes
:*
T0
b
gradients_1/Mean_4_grad/Shape_2Const*
_output_shapes
: *
dtype0*
valueB 
g
gradients_1/Mean_4_grad/ConstConst*
valueB: *
_output_shapes
:*
dtype0
˘
gradients_1/Mean_4_grad/ProdProdgradients_1/Mean_4_grad/Shape_1gradients_1/Mean_4_grad/Const*

Tidx0*
T0*
_output_shapes
: *
	keep_dims( 
i
gradients_1/Mean_4_grad/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
Ś
gradients_1/Mean_4_grad/Prod_1Prodgradients_1/Mean_4_grad/Shape_2gradients_1/Mean_4_grad/Const_1*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
c
!gradients_1/Mean_4_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 

gradients_1/Mean_4_grad/MaximumMaximumgradients_1/Mean_4_grad/Prod_1!gradients_1/Mean_4_grad/Maximum/y*
T0*
_output_shapes
: 

 gradients_1/Mean_4_grad/floordivFloorDivgradients_1/Mean_4_grad/Prodgradients_1/Mean_4_grad/Maximum*
_output_shapes
: *
T0

gradients_1/Mean_4_grad/CastCast gradients_1/Mean_4_grad/floordiv*
Truncate( *
_output_shapes
: *

SrcT0*

DstT0

gradients_1/Mean_4_grad/truedivRealDivgradients_1/Mean_4_grad/Tilegradients_1/Mean_4_grad/Cast*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
_
gradients_1/pow_grad/ShapeShapesub_1*
_output_shapes
:*
out_type0*
T0
_
gradients_1/pow_grad/Shape_1Const*
_output_shapes
: *
valueB *
dtype0
ş
*gradients_1/pow_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/pow_grad/Shapegradients_1/pow_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
u
gradients_1/pow_grad/mulMulgradients_1/Mean_3_grad/truedivpow/y*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
_
gradients_1/pow_grad/sub/yConst*
valueB
 *  ?*
_output_shapes
: *
dtype0
c
gradients_1/pow_grad/subSubpow/ygradients_1/pow_grad/sub/y*
_output_shapes
: *
T0
n
gradients_1/pow_grad/PowPowsub_1gradients_1/pow_grad/sub*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

gradients_1/pow_grad/mul_1Mulgradients_1/pow_grad/mulgradients_1/pow_grad/Pow*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
§
gradients_1/pow_grad/SumSumgradients_1/pow_grad/mul_1*gradients_1/pow_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
_output_shapes
:*
T0

gradients_1/pow_grad/ReshapeReshapegradients_1/pow_grad/Sumgradients_1/pow_grad/Shape*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
Tshape0*
T0
c
gradients_1/pow_grad/Greater/yConst*
valueB
 *    *
_output_shapes
: *
dtype0
|
gradients_1/pow_grad/GreaterGreatersub_1gradients_1/pow_grad/Greater/y*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
i
$gradients_1/pow_grad/ones_like/ShapeShapesub_1*
T0*
out_type0*
_output_shapes
:
i
$gradients_1/pow_grad/ones_like/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *  ?
˛
gradients_1/pow_grad/ones_likeFill$gradients_1/pow_grad/ones_like/Shape$gradients_1/pow_grad/ones_like/Const*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*

index_type0

gradients_1/pow_grad/SelectSelectgradients_1/pow_grad/Greatersub_1gradients_1/pow_grad/ones_like*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
j
gradients_1/pow_grad/LogLoggradients_1/pow_grad/Select*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
a
gradients_1/pow_grad/zeros_like	ZerosLikesub_1*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ž
gradients_1/pow_grad/Select_1Selectgradients_1/pow_grad/Greatergradients_1/pow_grad/Loggradients_1/pow_grad/zeros_like*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
u
gradients_1/pow_grad/mul_2Mulgradients_1/Mean_3_grad/truedivpow*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙

gradients_1/pow_grad/mul_3Mulgradients_1/pow_grad/mul_2gradients_1/pow_grad/Select_1*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ť
gradients_1/pow_grad/Sum_1Sumgradients_1/pow_grad/mul_3,gradients_1/pow_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:

gradients_1/pow_grad/Reshape_1Reshapegradients_1/pow_grad/Sum_1gradients_1/pow_grad/Shape_1*
Tshape0*
_output_shapes
: *
T0
m
%gradients_1/pow_grad/tuple/group_depsNoOp^gradients_1/pow_grad/Reshape^gradients_1/pow_grad/Reshape_1
Ţ
-gradients_1/pow_grad/tuple/control_dependencyIdentitygradients_1/pow_grad/Reshape&^gradients_1/pow_grad/tuple/group_deps*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*/
_class%
#!loc:@gradients_1/pow_grad/Reshape*
T0
×
/gradients_1/pow_grad/tuple/control_dependency_1Identitygradients_1/pow_grad/Reshape_1&^gradients_1/pow_grad/tuple/group_deps*
T0*
_output_shapes
: *1
_class'
%#loc:@gradients_1/pow_grad/Reshape_1
a
gradients_1/pow_1_grad/ShapeShapesub_2*
T0*
out_type0*
_output_shapes
:
a
gradients_1/pow_1_grad/Shape_1Const*
_output_shapes
: *
dtype0*
valueB 
Ŕ
,gradients_1/pow_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/pow_1_grad/Shapegradients_1/pow_1_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0
y
gradients_1/pow_1_grad/mulMulgradients_1/Mean_4_grad/truedivpow_1/y*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
a
gradients_1/pow_1_grad/sub/yConst*
_output_shapes
: *
valueB
 *  ?*
dtype0
i
gradients_1/pow_1_grad/subSubpow_1/ygradients_1/pow_1_grad/sub/y*
_output_shapes
: *
T0
r
gradients_1/pow_1_grad/PowPowsub_2gradients_1/pow_1_grad/sub*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙

gradients_1/pow_1_grad/mul_1Mulgradients_1/pow_1_grad/mulgradients_1/pow_1_grad/Pow*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
­
gradients_1/pow_1_grad/SumSumgradients_1/pow_1_grad/mul_1,gradients_1/pow_1_grad/BroadcastGradientArgs*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:

gradients_1/pow_1_grad/ReshapeReshapegradients_1/pow_1_grad/Sumgradients_1/pow_1_grad/Shape*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
Tshape0
e
 gradients_1/pow_1_grad/Greater/yConst*
dtype0*
valueB
 *    *
_output_shapes
: 

gradients_1/pow_1_grad/GreaterGreatersub_2 gradients_1/pow_1_grad/Greater/y*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
k
&gradients_1/pow_1_grad/ones_like/ShapeShapesub_2*
_output_shapes
:*
out_type0*
T0
k
&gradients_1/pow_1_grad/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
¸
 gradients_1/pow_1_grad/ones_likeFill&gradients_1/pow_1_grad/ones_like/Shape&gradients_1/pow_1_grad/ones_like/Const*
T0*

index_type0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙

gradients_1/pow_1_grad/SelectSelectgradients_1/pow_1_grad/Greatersub_2 gradients_1/pow_1_grad/ones_like*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
n
gradients_1/pow_1_grad/LogLoggradients_1/pow_1_grad/Select*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
c
!gradients_1/pow_1_grad/zeros_like	ZerosLikesub_2*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
ś
gradients_1/pow_1_grad/Select_1Selectgradients_1/pow_1_grad/Greatergradients_1/pow_1_grad/Log!gradients_1/pow_1_grad/zeros_like*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
y
gradients_1/pow_1_grad/mul_2Mulgradients_1/Mean_4_grad/truedivpow_1*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

gradients_1/pow_1_grad/mul_3Mulgradients_1/pow_1_grad/mul_2gradients_1/pow_1_grad/Select_1*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
ą
gradients_1/pow_1_grad/Sum_1Sumgradients_1/pow_1_grad/mul_3.gradients_1/pow_1_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
_output_shapes
:*
T0

 gradients_1/pow_1_grad/Reshape_1Reshapegradients_1/pow_1_grad/Sum_1gradients_1/pow_1_grad/Shape_1*
T0*
_output_shapes
: *
Tshape0
s
'gradients_1/pow_1_grad/tuple/group_depsNoOp^gradients_1/pow_1_grad/Reshape!^gradients_1/pow_1_grad/Reshape_1
ć
/gradients_1/pow_1_grad/tuple/control_dependencyIdentitygradients_1/pow_1_grad/Reshape(^gradients_1/pow_1_grad/tuple/group_deps*1
_class'
%#loc:@gradients_1/pow_1_grad/Reshape*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
ß
1gradients_1/pow_1_grad/tuple/control_dependency_1Identity gradients_1/pow_1_grad/Reshape_1(^gradients_1/pow_1_grad/tuple/group_deps*
T0*3
_class)
'%loc:@gradients_1/pow_1_grad/Reshape_1*
_output_shapes
: 
i
gradients_1/sub_1_grad/ShapeShapePlaceholder_4*
out_type0*
T0*
_output_shapes
:
h
gradients_1/sub_1_grad/Shape_1Shape
vf/Squeeze*
T0*
out_type0*
_output_shapes
:
Ŕ
,gradients_1/sub_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/sub_1_grad/Shapegradients_1/sub_1_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
ž
gradients_1/sub_1_grad/SumSum-gradients_1/pow_grad/tuple/control_dependency,gradients_1/sub_1_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
_output_shapes
:*
T0

gradients_1/sub_1_grad/ReshapeReshapegradients_1/sub_1_grad/Sumgradients_1/sub_1_grad/Shape*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
Tshape0
Â
gradients_1/sub_1_grad/Sum_1Sum-gradients_1/pow_grad/tuple/control_dependency.gradients_1/sub_1_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
b
gradients_1/sub_1_grad/NegNeggradients_1/sub_1_grad/Sum_1*
_output_shapes
:*
T0
Ł
 gradients_1/sub_1_grad/Reshape_1Reshapegradients_1/sub_1_grad/Neggradients_1/sub_1_grad/Shape_1*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
Tshape0
s
'gradients_1/sub_1_grad/tuple/group_depsNoOp^gradients_1/sub_1_grad/Reshape!^gradients_1/sub_1_grad/Reshape_1
ć
/gradients_1/sub_1_grad/tuple/control_dependencyIdentitygradients_1/sub_1_grad/Reshape(^gradients_1/sub_1_grad/tuple/group_deps*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*1
_class'
%#loc:@gradients_1/sub_1_grad/Reshape
ě
1gradients_1/sub_1_grad/tuple/control_dependency_1Identity gradients_1/sub_1_grad/Reshape_1(^gradients_1/sub_1_grad/tuple/group_deps*
T0*3
_class)
'%loc:@gradients_1/sub_1_grad/Reshape_1*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
i
gradients_1/sub_2_grad/ShapeShapePlaceholder_5*
_output_shapes
:*
out_type0*
T0
h
gradients_1/sub_2_grad/Shape_1Shape
vc/Squeeze*
out_type0*
_output_shapes
:*
T0
Ŕ
,gradients_1/sub_2_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/sub_2_grad/Shapegradients_1/sub_2_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0
Ŕ
gradients_1/sub_2_grad/SumSum/gradients_1/pow_1_grad/tuple/control_dependency,gradients_1/sub_2_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
_output_shapes
:*
T0

gradients_1/sub_2_grad/ReshapeReshapegradients_1/sub_2_grad/Sumgradients_1/sub_2_grad/Shape*
T0*
Tshape0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ä
gradients_1/sub_2_grad/Sum_1Sum/gradients_1/pow_1_grad/tuple/control_dependency.gradients_1/sub_2_grad/BroadcastGradientArgs:1*
	keep_dims( *
_output_shapes
:*

Tidx0*
T0
b
gradients_1/sub_2_grad/NegNeggradients_1/sub_2_grad/Sum_1*
T0*
_output_shapes
:
Ł
 gradients_1/sub_2_grad/Reshape_1Reshapegradients_1/sub_2_grad/Neggradients_1/sub_2_grad/Shape_1*
T0*
Tshape0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
s
'gradients_1/sub_2_grad/tuple/group_depsNoOp^gradients_1/sub_2_grad/Reshape!^gradients_1/sub_2_grad/Reshape_1
ć
/gradients_1/sub_2_grad/tuple/control_dependencyIdentitygradients_1/sub_2_grad/Reshape(^gradients_1/sub_2_grad/tuple/group_deps*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*1
_class'
%#loc:@gradients_1/sub_2_grad/Reshape*
T0
ě
1gradients_1/sub_2_grad/tuple/control_dependency_1Identity gradients_1/sub_2_grad/Reshape_1(^gradients_1/sub_2_grad/tuple/group_deps*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*3
_class)
'%loc:@gradients_1/sub_2_grad/Reshape_1
s
!gradients_1/vf/Squeeze_grad/ShapeShapevf/dense_2/BiasAdd*
T0*
_output_shapes
:*
out_type0
Ä
#gradients_1/vf/Squeeze_grad/ReshapeReshape1gradients_1/sub_1_grad/tuple/control_dependency_1!gradients_1/vf/Squeeze_grad/Shape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
Tshape0*
T0
s
!gradients_1/vc/Squeeze_grad/ShapeShapevc/dense_2/BiasAdd*
out_type0*
_output_shapes
:*
T0
Ä
#gradients_1/vc/Squeeze_grad/ReshapeReshape1gradients_1/sub_2_grad/tuple/control_dependency_1!gradients_1/vc/Squeeze_grad/Shape*
T0*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

/gradients_1/vf/dense_2/BiasAdd_grad/BiasAddGradBiasAddGrad#gradients_1/vf/Squeeze_grad/Reshape*
_output_shapes
:*
data_formatNHWC*
T0

4gradients_1/vf/dense_2/BiasAdd_grad/tuple/group_depsNoOp$^gradients_1/vf/Squeeze_grad/Reshape0^gradients_1/vf/dense_2/BiasAdd_grad/BiasAddGrad

<gradients_1/vf/dense_2/BiasAdd_grad/tuple/control_dependencyIdentity#gradients_1/vf/Squeeze_grad/Reshape5^gradients_1/vf/dense_2/BiasAdd_grad/tuple/group_deps*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*6
_class,
*(loc:@gradients_1/vf/Squeeze_grad/Reshape

>gradients_1/vf/dense_2/BiasAdd_grad/tuple/control_dependency_1Identity/gradients_1/vf/dense_2/BiasAdd_grad/BiasAddGrad5^gradients_1/vf/dense_2/BiasAdd_grad/tuple/group_deps*
T0*B
_class8
64loc:@gradients_1/vf/dense_2/BiasAdd_grad/BiasAddGrad*
_output_shapes
:

/gradients_1/vc/dense_2/BiasAdd_grad/BiasAddGradBiasAddGrad#gradients_1/vc/Squeeze_grad/Reshape*
data_formatNHWC*
T0*
_output_shapes
:

4gradients_1/vc/dense_2/BiasAdd_grad/tuple/group_depsNoOp$^gradients_1/vc/Squeeze_grad/Reshape0^gradients_1/vc/dense_2/BiasAdd_grad/BiasAddGrad

<gradients_1/vc/dense_2/BiasAdd_grad/tuple/control_dependencyIdentity#gradients_1/vc/Squeeze_grad/Reshape5^gradients_1/vc/dense_2/BiasAdd_grad/tuple/group_deps*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*6
_class,
*(loc:@gradients_1/vc/Squeeze_grad/Reshape

>gradients_1/vc/dense_2/BiasAdd_grad/tuple/control_dependency_1Identity/gradients_1/vc/dense_2/BiasAdd_grad/BiasAddGrad5^gradients_1/vc/dense_2/BiasAdd_grad/tuple/group_deps*
_output_shapes
:*
T0*B
_class8
64loc:@gradients_1/vc/dense_2/BiasAdd_grad/BiasAddGrad
â
)gradients_1/vf/dense_2/MatMul_grad/MatMulMatMul<gradients_1/vf/dense_2/BiasAdd_grad/tuple/control_dependencyvf/dense_2/kernel/read*
transpose_b(*
transpose_a( *
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ô
+gradients_1/vf/dense_2/MatMul_grad/MatMul_1MatMulvf/dense_1/Tanh<gradients_1/vf/dense_2/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
_output_shapes
:	*
T0*
transpose_a(

3gradients_1/vf/dense_2/MatMul_grad/tuple/group_depsNoOp*^gradients_1/vf/dense_2/MatMul_grad/MatMul,^gradients_1/vf/dense_2/MatMul_grad/MatMul_1

;gradients_1/vf/dense_2/MatMul_grad/tuple/control_dependencyIdentity)gradients_1/vf/dense_2/MatMul_grad/MatMul4^gradients_1/vf/dense_2/MatMul_grad/tuple/group_deps*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*<
_class2
0.loc:@gradients_1/vf/dense_2/MatMul_grad/MatMul*
T0

=gradients_1/vf/dense_2/MatMul_grad/tuple/control_dependency_1Identity+gradients_1/vf/dense_2/MatMul_grad/MatMul_14^gradients_1/vf/dense_2/MatMul_grad/tuple/group_deps*>
_class4
20loc:@gradients_1/vf/dense_2/MatMul_grad/MatMul_1*
_output_shapes
:	*
T0
â
)gradients_1/vc/dense_2/MatMul_grad/MatMulMatMul<gradients_1/vc/dense_2/BiasAdd_grad/tuple/control_dependencyvc/dense_2/kernel/read*
transpose_a( *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_b(*
T0
Ô
+gradients_1/vc/dense_2/MatMul_grad/MatMul_1MatMulvc/dense_1/Tanh<gradients_1/vc/dense_2/BiasAdd_grad/tuple/control_dependency*
T0*
transpose_b( *
transpose_a(*
_output_shapes
:	

3gradients_1/vc/dense_2/MatMul_grad/tuple/group_depsNoOp*^gradients_1/vc/dense_2/MatMul_grad/MatMul,^gradients_1/vc/dense_2/MatMul_grad/MatMul_1

;gradients_1/vc/dense_2/MatMul_grad/tuple/control_dependencyIdentity)gradients_1/vc/dense_2/MatMul_grad/MatMul4^gradients_1/vc/dense_2/MatMul_grad/tuple/group_deps*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*<
_class2
0.loc:@gradients_1/vc/dense_2/MatMul_grad/MatMul

=gradients_1/vc/dense_2/MatMul_grad/tuple/control_dependency_1Identity+gradients_1/vc/dense_2/MatMul_grad/MatMul_14^gradients_1/vc/dense_2/MatMul_grad/tuple/group_deps*>
_class4
20loc:@gradients_1/vc/dense_2/MatMul_grad/MatMul_1*
_output_shapes
:	*
T0
ś
)gradients_1/vf/dense_1/Tanh_grad/TanhGradTanhGradvf/dense_1/Tanh;gradients_1/vf/dense_2/MatMul_grad/tuple/control_dependency*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ś
)gradients_1/vc/dense_1/Tanh_grad/TanhGradTanhGradvc/dense_1/Tanh;gradients_1/vc/dense_2/MatMul_grad/tuple/control_dependency*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ś
/gradients_1/vf/dense_1/BiasAdd_grad/BiasAddGradBiasAddGrad)gradients_1/vf/dense_1/Tanh_grad/TanhGrad*
_output_shapes	
:*
data_formatNHWC*
T0

4gradients_1/vf/dense_1/BiasAdd_grad/tuple/group_depsNoOp0^gradients_1/vf/dense_1/BiasAdd_grad/BiasAddGrad*^gradients_1/vf/dense_1/Tanh_grad/TanhGrad

<gradients_1/vf/dense_1/BiasAdd_grad/tuple/control_dependencyIdentity)gradients_1/vf/dense_1/Tanh_grad/TanhGrad5^gradients_1/vf/dense_1/BiasAdd_grad/tuple/group_deps*<
_class2
0.loc:@gradients_1/vf/dense_1/Tanh_grad/TanhGrad*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

>gradients_1/vf/dense_1/BiasAdd_grad/tuple/control_dependency_1Identity/gradients_1/vf/dense_1/BiasAdd_grad/BiasAddGrad5^gradients_1/vf/dense_1/BiasAdd_grad/tuple/group_deps*
_output_shapes	
:*
T0*B
_class8
64loc:@gradients_1/vf/dense_1/BiasAdd_grad/BiasAddGrad
Ś
/gradients_1/vc/dense_1/BiasAdd_grad/BiasAddGradBiasAddGrad)gradients_1/vc/dense_1/Tanh_grad/TanhGrad*
T0*
_output_shapes	
:*
data_formatNHWC

4gradients_1/vc/dense_1/BiasAdd_grad/tuple/group_depsNoOp0^gradients_1/vc/dense_1/BiasAdd_grad/BiasAddGrad*^gradients_1/vc/dense_1/Tanh_grad/TanhGrad

<gradients_1/vc/dense_1/BiasAdd_grad/tuple/control_dependencyIdentity)gradients_1/vc/dense_1/Tanh_grad/TanhGrad5^gradients_1/vc/dense_1/BiasAdd_grad/tuple/group_deps*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*<
_class2
0.loc:@gradients_1/vc/dense_1/Tanh_grad/TanhGrad

>gradients_1/vc/dense_1/BiasAdd_grad/tuple/control_dependency_1Identity/gradients_1/vc/dense_1/BiasAdd_grad/BiasAddGrad5^gradients_1/vc/dense_1/BiasAdd_grad/tuple/group_deps*
T0*
_output_shapes	
:*B
_class8
64loc:@gradients_1/vc/dense_1/BiasAdd_grad/BiasAddGrad
â
)gradients_1/vf/dense_1/MatMul_grad/MatMulMatMul<gradients_1/vf/dense_1/BiasAdd_grad/tuple/control_dependencyvf/dense_1/kernel/read*
transpose_a( *
transpose_b(*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ó
+gradients_1/vf/dense_1/MatMul_grad/MatMul_1MatMulvf/dense/Tanh<gradients_1/vf/dense_1/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
transpose_a(*
T0* 
_output_shapes
:


3gradients_1/vf/dense_1/MatMul_grad/tuple/group_depsNoOp*^gradients_1/vf/dense_1/MatMul_grad/MatMul,^gradients_1/vf/dense_1/MatMul_grad/MatMul_1

;gradients_1/vf/dense_1/MatMul_grad/tuple/control_dependencyIdentity)gradients_1/vf/dense_1/MatMul_grad/MatMul4^gradients_1/vf/dense_1/MatMul_grad/tuple/group_deps*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*<
_class2
0.loc:@gradients_1/vf/dense_1/MatMul_grad/MatMul

=gradients_1/vf/dense_1/MatMul_grad/tuple/control_dependency_1Identity+gradients_1/vf/dense_1/MatMul_grad/MatMul_14^gradients_1/vf/dense_1/MatMul_grad/tuple/group_deps* 
_output_shapes
:
*
T0*>
_class4
20loc:@gradients_1/vf/dense_1/MatMul_grad/MatMul_1
â
)gradients_1/vc/dense_1/MatMul_grad/MatMulMatMul<gradients_1/vc/dense_1/BiasAdd_grad/tuple/control_dependencyvc/dense_1/kernel/read*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
transpose_a( *
transpose_b(
Ó
+gradients_1/vc/dense_1/MatMul_grad/MatMul_1MatMulvc/dense/Tanh<gradients_1/vc/dense_1/BiasAdd_grad/tuple/control_dependency*
transpose_a(*
T0* 
_output_shapes
:
*
transpose_b( 

3gradients_1/vc/dense_1/MatMul_grad/tuple/group_depsNoOp*^gradients_1/vc/dense_1/MatMul_grad/MatMul,^gradients_1/vc/dense_1/MatMul_grad/MatMul_1

;gradients_1/vc/dense_1/MatMul_grad/tuple/control_dependencyIdentity)gradients_1/vc/dense_1/MatMul_grad/MatMul4^gradients_1/vc/dense_1/MatMul_grad/tuple/group_deps*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*<
_class2
0.loc:@gradients_1/vc/dense_1/MatMul_grad/MatMul

=gradients_1/vc/dense_1/MatMul_grad/tuple/control_dependency_1Identity+gradients_1/vc/dense_1/MatMul_grad/MatMul_14^gradients_1/vc/dense_1/MatMul_grad/tuple/group_deps*>
_class4
20loc:@gradients_1/vc/dense_1/MatMul_grad/MatMul_1*
T0* 
_output_shapes
:

˛
'gradients_1/vf/dense/Tanh_grad/TanhGradTanhGradvf/dense/Tanh;gradients_1/vf/dense_1/MatMul_grad/tuple/control_dependency*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
˛
'gradients_1/vc/dense/Tanh_grad/TanhGradTanhGradvc/dense/Tanh;gradients_1/vc/dense_1/MatMul_grad/tuple/control_dependency*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
˘
-gradients_1/vf/dense/BiasAdd_grad/BiasAddGradBiasAddGrad'gradients_1/vf/dense/Tanh_grad/TanhGrad*
T0*
data_formatNHWC*
_output_shapes	
:

2gradients_1/vf/dense/BiasAdd_grad/tuple/group_depsNoOp.^gradients_1/vf/dense/BiasAdd_grad/BiasAddGrad(^gradients_1/vf/dense/Tanh_grad/TanhGrad

:gradients_1/vf/dense/BiasAdd_grad/tuple/control_dependencyIdentity'gradients_1/vf/dense/Tanh_grad/TanhGrad3^gradients_1/vf/dense/BiasAdd_grad/tuple/group_deps*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*:
_class0
.,loc:@gradients_1/vf/dense/Tanh_grad/TanhGrad

<gradients_1/vf/dense/BiasAdd_grad/tuple/control_dependency_1Identity-gradients_1/vf/dense/BiasAdd_grad/BiasAddGrad3^gradients_1/vf/dense/BiasAdd_grad/tuple/group_deps*
_output_shapes	
:*@
_class6
42loc:@gradients_1/vf/dense/BiasAdd_grad/BiasAddGrad*
T0
˘
-gradients_1/vc/dense/BiasAdd_grad/BiasAddGradBiasAddGrad'gradients_1/vc/dense/Tanh_grad/TanhGrad*
T0*
_output_shapes	
:*
data_formatNHWC

2gradients_1/vc/dense/BiasAdd_grad/tuple/group_depsNoOp.^gradients_1/vc/dense/BiasAdd_grad/BiasAddGrad(^gradients_1/vc/dense/Tanh_grad/TanhGrad

:gradients_1/vc/dense/BiasAdd_grad/tuple/control_dependencyIdentity'gradients_1/vc/dense/Tanh_grad/TanhGrad3^gradients_1/vc/dense/BiasAdd_grad/tuple/group_deps*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*:
_class0
.,loc:@gradients_1/vc/dense/Tanh_grad/TanhGrad*
T0

<gradients_1/vc/dense/BiasAdd_grad/tuple/control_dependency_1Identity-gradients_1/vc/dense/BiasAdd_grad/BiasAddGrad3^gradients_1/vc/dense/BiasAdd_grad/tuple/group_deps*
T0*
_output_shapes	
:*@
_class6
42loc:@gradients_1/vc/dense/BiasAdd_grad/BiasAddGrad
Ű
'gradients_1/vf/dense/MatMul_grad/MatMulMatMul:gradients_1/vf/dense/BiasAdd_grad/tuple/control_dependencyvf/dense/kernel/read*'
_output_shapes
:˙˙˙˙˙˙˙˙˙<*
T0*
transpose_a( *
transpose_b(
Ě
)gradients_1/vf/dense/MatMul_grad/MatMul_1MatMulPlaceholder:gradients_1/vf/dense/BiasAdd_grad/tuple/control_dependency*
transpose_a(*
transpose_b( *
T0*
_output_shapes
:	<

1gradients_1/vf/dense/MatMul_grad/tuple/group_depsNoOp(^gradients_1/vf/dense/MatMul_grad/MatMul*^gradients_1/vf/dense/MatMul_grad/MatMul_1

9gradients_1/vf/dense/MatMul_grad/tuple/control_dependencyIdentity'gradients_1/vf/dense/MatMul_grad/MatMul2^gradients_1/vf/dense/MatMul_grad/tuple/group_deps*'
_output_shapes
:˙˙˙˙˙˙˙˙˙<*:
_class0
.,loc:@gradients_1/vf/dense/MatMul_grad/MatMul*
T0

;gradients_1/vf/dense/MatMul_grad/tuple/control_dependency_1Identity)gradients_1/vf/dense/MatMul_grad/MatMul_12^gradients_1/vf/dense/MatMul_grad/tuple/group_deps*<
_class2
0.loc:@gradients_1/vf/dense/MatMul_grad/MatMul_1*
T0*
_output_shapes
:	<
Ű
'gradients_1/vc/dense/MatMul_grad/MatMulMatMul:gradients_1/vc/dense/BiasAdd_grad/tuple/control_dependencyvc/dense/kernel/read*'
_output_shapes
:˙˙˙˙˙˙˙˙˙<*
transpose_b(*
T0*
transpose_a( 
Ě
)gradients_1/vc/dense/MatMul_grad/MatMul_1MatMulPlaceholder:gradients_1/vc/dense/BiasAdd_grad/tuple/control_dependency*
_output_shapes
:	<*
T0*
transpose_a(*
transpose_b( 

1gradients_1/vc/dense/MatMul_grad/tuple/group_depsNoOp(^gradients_1/vc/dense/MatMul_grad/MatMul*^gradients_1/vc/dense/MatMul_grad/MatMul_1

9gradients_1/vc/dense/MatMul_grad/tuple/control_dependencyIdentity'gradients_1/vc/dense/MatMul_grad/MatMul2^gradients_1/vc/dense/MatMul_grad/tuple/group_deps*
T0*:
_class0
.,loc:@gradients_1/vc/dense/MatMul_grad/MatMul*'
_output_shapes
:˙˙˙˙˙˙˙˙˙<

;gradients_1/vc/dense/MatMul_grad/tuple/control_dependency_1Identity)gradients_1/vc/dense/MatMul_grad/MatMul_12^gradients_1/vc/dense/MatMul_grad/tuple/group_deps*
_output_shapes
:	<*
T0*<
_class2
0.loc:@gradients_1/vc/dense/MatMul_grad/MatMul_1
c
Reshape_28/shapeConst*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0


Reshape_28Reshape;gradients_1/vf/dense/MatMul_grad/tuple/control_dependency_1Reshape_28/shape*
_output_shapes	
:x*
Tshape0*
T0
c
Reshape_29/shapeConst*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙*
_output_shapes
:


Reshape_29Reshape<gradients_1/vf/dense/BiasAdd_grad/tuple/control_dependency_1Reshape_29/shape*
_output_shapes	
:*
Tshape0*
T0
c
Reshape_30/shapeConst*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙*
_output_shapes
:


Reshape_30Reshape=gradients_1/vf/dense_1/MatMul_grad/tuple/control_dependency_1Reshape_30/shape*
T0*
Tshape0*
_output_shapes

:
c
Reshape_31/shapeConst*
dtype0*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙


Reshape_31Reshape>gradients_1/vf/dense_1/BiasAdd_grad/tuple/control_dependency_1Reshape_31/shape*
Tshape0*
T0*
_output_shapes	
:
c
Reshape_32/shapeConst*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0


Reshape_32Reshape=gradients_1/vf/dense_2/MatMul_grad/tuple/control_dependency_1Reshape_32/shape*
_output_shapes	
:*
T0*
Tshape0
c
Reshape_33/shapeConst*
dtype0*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙


Reshape_33Reshape>gradients_1/vf/dense_2/BiasAdd_grad/tuple/control_dependency_1Reshape_33/shape*
Tshape0*
_output_shapes
:*
T0
c
Reshape_34/shapeConst*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙*
_output_shapes
:


Reshape_34Reshape;gradients_1/vc/dense/MatMul_grad/tuple/control_dependency_1Reshape_34/shape*
Tshape0*
_output_shapes	
:x*
T0
c
Reshape_35/shapeConst*
dtype0*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙


Reshape_35Reshape<gradients_1/vc/dense/BiasAdd_grad/tuple/control_dependency_1Reshape_35/shape*
Tshape0*
T0*
_output_shapes	
:
c
Reshape_36/shapeConst*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙*
_output_shapes
:


Reshape_36Reshape=gradients_1/vc/dense_1/MatMul_grad/tuple/control_dependency_1Reshape_36/shape*
Tshape0*
_output_shapes

:*
T0
c
Reshape_37/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
_output_shapes
:*
dtype0


Reshape_37Reshape>gradients_1/vc/dense_1/BiasAdd_grad/tuple/control_dependency_1Reshape_37/shape*
_output_shapes	
:*
Tshape0*
T0
c
Reshape_38/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
_output_shapes
:*
dtype0


Reshape_38Reshape=gradients_1/vc/dense_2/MatMul_grad/tuple/control_dependency_1Reshape_38/shape*
Tshape0*
T0*
_output_shapes	
:
c
Reshape_39/shapeConst*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0


Reshape_39Reshape>gradients_1/vc/dense_2/BiasAdd_grad/tuple/control_dependency_1Reshape_39/shape*
_output_shapes
:*
T0*
Tshape0
O
concat_2/axisConst*
dtype0*
_output_shapes
: *
value	B : 
ď
concat_2ConcatV2
Reshape_28
Reshape_29
Reshape_30
Reshape_31
Reshape_32
Reshape_33
Reshape_34
Reshape_35
Reshape_36
Reshape_37
Reshape_38
Reshape_39concat_2/axis*
_output_shapes

:ü	*
N*
T0*

Tidx0
l
PyFunc_2PyFuncconcat_2*
_output_shapes

:ü	*
Tout
2*
token
pyfunc_2*
Tin
2

Const_7Const*E
value<B:"0 <                  <                 *
dtype0*
_output_shapes
:
S
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
Ç
split_2SplitVPyFunc_2Const_7split_2/split_dim*
T0*
	num_split*

Tlen0*h
_output_shapesV
T:x::::::x:::::
a
Reshape_40/shapeConst*
_output_shapes
:*
dtype0*
valueB"<      
h

Reshape_40Reshapesplit_2Reshape_40/shape*
Tshape0*
T0*
_output_shapes
:	<
[
Reshape_41/shapeConst*
_output_shapes
:*
valueB:*
dtype0
f

Reshape_41Reshape	split_2:1Reshape_41/shape*
_output_shapes	
:*
T0*
Tshape0
a
Reshape_42/shapeConst*
dtype0*
valueB"      *
_output_shapes
:
k

Reshape_42Reshape	split_2:2Reshape_42/shape* 
_output_shapes
:
*
T0*
Tshape0
[
Reshape_43/shapeConst*
dtype0*
_output_shapes
:*
valueB:
f

Reshape_43Reshape	split_2:3Reshape_43/shape*
_output_shapes	
:*
T0*
Tshape0
a
Reshape_44/shapeConst*
valueB"      *
_output_shapes
:*
dtype0
j

Reshape_44Reshape	split_2:4Reshape_44/shape*
_output_shapes
:	*
Tshape0*
T0
Z
Reshape_45/shapeConst*
dtype0*
valueB:*
_output_shapes
:
e

Reshape_45Reshape	split_2:5Reshape_45/shape*
T0*
_output_shapes
:*
Tshape0
a
Reshape_46/shapeConst*
valueB"<      *
_output_shapes
:*
dtype0
j

Reshape_46Reshape	split_2:6Reshape_46/shape*
T0*
_output_shapes
:	<*
Tshape0
[
Reshape_47/shapeConst*
_output_shapes
:*
valueB:*
dtype0
f

Reshape_47Reshape	split_2:7Reshape_47/shape*
_output_shapes	
:*
Tshape0*
T0
a
Reshape_48/shapeConst*
_output_shapes
:*
valueB"      *
dtype0
k

Reshape_48Reshape	split_2:8Reshape_48/shape*
Tshape0*
T0* 
_output_shapes
:

[
Reshape_49/shapeConst*
dtype0*
valueB:*
_output_shapes
:
f

Reshape_49Reshape	split_2:9Reshape_49/shape*
T0*
_output_shapes	
:*
Tshape0
a
Reshape_50/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
k

Reshape_50Reshape
split_2:10Reshape_50/shape*
_output_shapes
:	*
T0*
Tshape0
Z
Reshape_51/shapeConst*
valueB:*
dtype0*
_output_shapes
:
f

Reshape_51Reshape
split_2:11Reshape_51/shape*
T0*
Tshape0*
_output_shapes
:

beta1_power_1/initial_valueConst* 
_class
loc:@vc/dense/bias*
valueB
 *fff?*
_output_shapes
: *
dtype0

beta1_power_1
VariableV2*
shared_name *
	container *
shape: *
dtype0*
_output_shapes
: * 
_class
loc:@vc/dense/bias
ś
beta1_power_1/AssignAssignbeta1_power_1beta1_power_1/initial_value*
T0*
_output_shapes
: * 
_class
loc:@vc/dense/bias*
use_locking(*
validate_shape(
p
beta1_power_1/readIdentitybeta1_power_1*
_output_shapes
: *
T0* 
_class
loc:@vc/dense/bias

beta2_power_1/initial_valueConst* 
_class
loc:@vc/dense/bias*
_output_shapes
: *
valueB
 *wž?*
dtype0

beta2_power_1
VariableV2*
_output_shapes
: *
shared_name *
	container * 
_class
loc:@vc/dense/bias*
dtype0*
shape: 
ś
beta2_power_1/AssignAssignbeta2_power_1beta2_power_1/initial_value*
T0*
validate_shape(*
use_locking(* 
_class
loc:@vc/dense/bias*
_output_shapes
: 
p
beta2_power_1/readIdentitybeta2_power_1*
T0* 
_class
loc:@vc/dense/bias*
_output_shapes
: 
Ť
6vf/dense/kernel/Adam/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB"<      *"
_class
loc:@vf/dense/kernel

,vf/dense/kernel/Adam/Initializer/zeros/ConstConst*
_output_shapes
: *
valueB
 *    *"
_class
loc:@vf/dense/kernel*
dtype0
ô
&vf/dense/kernel/Adam/Initializer/zerosFill6vf/dense/kernel/Adam/Initializer/zeros/shape_as_tensor,vf/dense/kernel/Adam/Initializer/zeros/Const*"
_class
loc:@vf/dense/kernel*
T0*
_output_shapes
:	<*

index_type0
Ž
vf/dense/kernel/Adam
VariableV2*
dtype0*
	container *
shape:	<*
_output_shapes
:	<*
shared_name *"
_class
loc:@vf/dense/kernel
Ú
vf/dense/kernel/Adam/AssignAssignvf/dense/kernel/Adam&vf/dense/kernel/Adam/Initializer/zeros*
use_locking(*
T0*
validate_shape(*
_output_shapes
:	<*"
_class
loc:@vf/dense/kernel

vf/dense/kernel/Adam/readIdentityvf/dense/kernel/Adam*"
_class
loc:@vf/dense/kernel*
T0*
_output_shapes
:	<
­
8vf/dense/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*"
_class
loc:@vf/dense/kernel*
valueB"<      *
dtype0*
_output_shapes
:

.vf/dense/kernel/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    *"
_class
loc:@vf/dense/kernel*
_output_shapes
: *
dtype0
ú
(vf/dense/kernel/Adam_1/Initializer/zerosFill8vf/dense/kernel/Adam_1/Initializer/zeros/shape_as_tensor.vf/dense/kernel/Adam_1/Initializer/zeros/Const*
_output_shapes
:	<*

index_type0*
T0*"
_class
loc:@vf/dense/kernel
°
vf/dense/kernel/Adam_1
VariableV2*
	container *
shared_name *
dtype0*
shape:	<*
_output_shapes
:	<*"
_class
loc:@vf/dense/kernel
ŕ
vf/dense/kernel/Adam_1/AssignAssignvf/dense/kernel/Adam_1(vf/dense/kernel/Adam_1/Initializer/zeros*"
_class
loc:@vf/dense/kernel*
use_locking(*
_output_shapes
:	<*
T0*
validate_shape(

vf/dense/kernel/Adam_1/readIdentityvf/dense/kernel/Adam_1*
T0*
_output_shapes
:	<*"
_class
loc:@vf/dense/kernel

$vf/dense/bias/Adam/Initializer/zerosConst* 
_class
loc:@vf/dense/bias*
_output_shapes	
:*
dtype0*
valueB*    
˘
vf/dense/bias/Adam
VariableV2*
_output_shapes	
:*
shape:*
	container *
dtype0* 
_class
loc:@vf/dense/bias*
shared_name 
Î
vf/dense/bias/Adam/AssignAssignvf/dense/bias/Adam$vf/dense/bias/Adam/Initializer/zeros*
use_locking(*
validate_shape(*
_output_shapes	
:*
T0* 
_class
loc:@vf/dense/bias

vf/dense/bias/Adam/readIdentityvf/dense/bias/Adam*
_output_shapes	
:*
T0* 
_class
loc:@vf/dense/bias

&vf/dense/bias/Adam_1/Initializer/zerosConst* 
_class
loc:@vf/dense/bias*
dtype0*
_output_shapes	
:*
valueB*    
¤
vf/dense/bias/Adam_1
VariableV2*
	container *
shape:* 
_class
loc:@vf/dense/bias*
shared_name *
_output_shapes	
:*
dtype0
Ô
vf/dense/bias/Adam_1/AssignAssignvf/dense/bias/Adam_1&vf/dense/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*
_output_shapes	
:* 
_class
loc:@vf/dense/bias*
validate_shape(

vf/dense/bias/Adam_1/readIdentityvf/dense/bias/Adam_1* 
_class
loc:@vf/dense/bias*
T0*
_output_shapes	
:
Ż
8vf/dense_1/kernel/Adam/Initializer/zeros/shape_as_tensorConst*$
_class
loc:@vf/dense_1/kernel*
dtype0*
valueB"      *
_output_shapes
:

.vf/dense_1/kernel/Adam/Initializer/zeros/ConstConst*
valueB
 *    *
_output_shapes
: *$
_class
loc:@vf/dense_1/kernel*
dtype0
ý
(vf/dense_1/kernel/Adam/Initializer/zerosFill8vf/dense_1/kernel/Adam/Initializer/zeros/shape_as_tensor.vf/dense_1/kernel/Adam/Initializer/zeros/Const* 
_output_shapes
:
*
T0*

index_type0*$
_class
loc:@vf/dense_1/kernel
´
vf/dense_1/kernel/Adam
VariableV2*
dtype0*
shape:
*$
_class
loc:@vf/dense_1/kernel* 
_output_shapes
:
*
shared_name *
	container 
ă
vf/dense_1/kernel/Adam/AssignAssignvf/dense_1/kernel/Adam(vf/dense_1/kernel/Adam/Initializer/zeros*
T0* 
_output_shapes
:
*
use_locking(*
validate_shape(*$
_class
loc:@vf/dense_1/kernel

vf/dense_1/kernel/Adam/readIdentityvf/dense_1/kernel/Adam*
T0*$
_class
loc:@vf/dense_1/kernel* 
_output_shapes
:

ą
:vf/dense_1/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*
dtype0*$
_class
loc:@vf/dense_1/kernel*
valueB"      

0vf/dense_1/kernel/Adam_1/Initializer/zeros/ConstConst*
_output_shapes
: *
valueB
 *    *$
_class
loc:@vf/dense_1/kernel*
dtype0

*vf/dense_1/kernel/Adam_1/Initializer/zerosFill:vf/dense_1/kernel/Adam_1/Initializer/zeros/shape_as_tensor0vf/dense_1/kernel/Adam_1/Initializer/zeros/Const* 
_output_shapes
:
*

index_type0*
T0*$
_class
loc:@vf/dense_1/kernel
ś
vf/dense_1/kernel/Adam_1
VariableV2*
shape:
* 
_output_shapes
:
*
	container *
dtype0*
shared_name *$
_class
loc:@vf/dense_1/kernel
é
vf/dense_1/kernel/Adam_1/AssignAssignvf/dense_1/kernel/Adam_1*vf/dense_1/kernel/Adam_1/Initializer/zeros* 
_output_shapes
:
*
validate_shape(*$
_class
loc:@vf/dense_1/kernel*
T0*
use_locking(

vf/dense_1/kernel/Adam_1/readIdentityvf/dense_1/kernel/Adam_1*
T0*$
_class
loc:@vf/dense_1/kernel* 
_output_shapes
:


&vf/dense_1/bias/Adam/Initializer/zerosConst*"
_class
loc:@vf/dense_1/bias*
_output_shapes	
:*
dtype0*
valueB*    
Ś
vf/dense_1/bias/Adam
VariableV2*"
_class
loc:@vf/dense_1/bias*
_output_shapes	
:*
shape:*
shared_name *
dtype0*
	container 
Ö
vf/dense_1/bias/Adam/AssignAssignvf/dense_1/bias/Adam&vf/dense_1/bias/Adam/Initializer/zeros*"
_class
loc:@vf/dense_1/bias*
_output_shapes	
:*
use_locking(*
validate_shape(*
T0

vf/dense_1/bias/Adam/readIdentityvf/dense_1/bias/Adam*
_output_shapes	
:*
T0*"
_class
loc:@vf/dense_1/bias

(vf/dense_1/bias/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes	
:*"
_class
loc:@vf/dense_1/bias*
valueB*    
¨
vf/dense_1/bias/Adam_1
VariableV2*
shape:*
shared_name *
dtype0*
_output_shapes	
:*"
_class
loc:@vf/dense_1/bias*
	container 
Ü
vf/dense_1/bias/Adam_1/AssignAssignvf/dense_1/bias/Adam_1(vf/dense_1/bias/Adam_1/Initializer/zeros*
_output_shapes	
:*
T0*"
_class
loc:@vf/dense_1/bias*
use_locking(*
validate_shape(

vf/dense_1/bias/Adam_1/readIdentityvf/dense_1/bias/Adam_1*
_output_shapes	
:*
T0*"
_class
loc:@vf/dense_1/bias
Ľ
(vf/dense_2/kernel/Adam/Initializer/zerosConst*
dtype0*
_output_shapes
:	*
valueB	*    *$
_class
loc:@vf/dense_2/kernel
˛
vf/dense_2/kernel/Adam
VariableV2*$
_class
loc:@vf/dense_2/kernel*
dtype0*
	container *
shared_name *
_output_shapes
:	*
shape:	
â
vf/dense_2/kernel/Adam/AssignAssignvf/dense_2/kernel/Adam(vf/dense_2/kernel/Adam/Initializer/zeros*
_output_shapes
:	*
T0*
validate_shape(*$
_class
loc:@vf/dense_2/kernel*
use_locking(

vf/dense_2/kernel/Adam/readIdentityvf/dense_2/kernel/Adam*
_output_shapes
:	*$
_class
loc:@vf/dense_2/kernel*
T0
§
*vf/dense_2/kernel/Adam_1/Initializer/zerosConst*$
_class
loc:@vf/dense_2/kernel*
dtype0*
_output_shapes
:	*
valueB	*    
´
vf/dense_2/kernel/Adam_1
VariableV2*
	container *
_output_shapes
:	*$
_class
loc:@vf/dense_2/kernel*
shared_name *
shape:	*
dtype0
č
vf/dense_2/kernel/Adam_1/AssignAssignvf/dense_2/kernel/Adam_1*vf/dense_2/kernel/Adam_1/Initializer/zeros*
validate_shape(*
T0*
use_locking(*$
_class
loc:@vf/dense_2/kernel*
_output_shapes
:	

vf/dense_2/kernel/Adam_1/readIdentityvf/dense_2/kernel/Adam_1*
T0*$
_class
loc:@vf/dense_2/kernel*
_output_shapes
:	

&vf/dense_2/bias/Adam/Initializer/zerosConst*
_output_shapes
:*
valueB*    *
dtype0*"
_class
loc:@vf/dense_2/bias
¤
vf/dense_2/bias/Adam
VariableV2*"
_class
loc:@vf/dense_2/bias*
dtype0*
_output_shapes
:*
	container *
shape:*
shared_name 
Ő
vf/dense_2/bias/Adam/AssignAssignvf/dense_2/bias/Adam&vf/dense_2/bias/Adam/Initializer/zeros*
use_locking(*
T0*
_output_shapes
:*
validate_shape(*"
_class
loc:@vf/dense_2/bias

vf/dense_2/bias/Adam/readIdentityvf/dense_2/bias/Adam*
T0*"
_class
loc:@vf/dense_2/bias*
_output_shapes
:

(vf/dense_2/bias/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes
:*"
_class
loc:@vf/dense_2/bias*
valueB*    
Ś
vf/dense_2/bias/Adam_1
VariableV2*
_output_shapes
:*
shape:*"
_class
loc:@vf/dense_2/bias*
dtype0*
shared_name *
	container 
Ű
vf/dense_2/bias/Adam_1/AssignAssignvf/dense_2/bias/Adam_1(vf/dense_2/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*
validate_shape(*"
_class
loc:@vf/dense_2/bias*
_output_shapes
:

vf/dense_2/bias/Adam_1/readIdentityvf/dense_2/bias/Adam_1*
_output_shapes
:*"
_class
loc:@vf/dense_2/bias*
T0
Ť
6vc/dense/kernel/Adam/Initializer/zeros/shape_as_tensorConst*
valueB"<      *"
_class
loc:@vc/dense/kernel*
_output_shapes
:*
dtype0

,vc/dense/kernel/Adam/Initializer/zeros/ConstConst*"
_class
loc:@vc/dense/kernel*
dtype0*
valueB
 *    *
_output_shapes
: 
ô
&vc/dense/kernel/Adam/Initializer/zerosFill6vc/dense/kernel/Adam/Initializer/zeros/shape_as_tensor,vc/dense/kernel/Adam/Initializer/zeros/Const*
T0*

index_type0*"
_class
loc:@vc/dense/kernel*
_output_shapes
:	<
Ž
vc/dense/kernel/Adam
VariableV2*
shape:	<*
shared_name *
_output_shapes
:	<*
dtype0*"
_class
loc:@vc/dense/kernel*
	container 
Ú
vc/dense/kernel/Adam/AssignAssignvc/dense/kernel/Adam&vc/dense/kernel/Adam/Initializer/zeros*
use_locking(*"
_class
loc:@vc/dense/kernel*
_output_shapes
:	<*
validate_shape(*
T0

vc/dense/kernel/Adam/readIdentityvc/dense/kernel/Adam*
T0*
_output_shapes
:	<*"
_class
loc:@vc/dense/kernel
­
8vc/dense/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
dtype0*
valueB"<      *"
_class
loc:@vc/dense/kernel*
_output_shapes
:

.vc/dense/kernel/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    *"
_class
loc:@vc/dense/kernel*
dtype0*
_output_shapes
: 
ú
(vc/dense/kernel/Adam_1/Initializer/zerosFill8vc/dense/kernel/Adam_1/Initializer/zeros/shape_as_tensor.vc/dense/kernel/Adam_1/Initializer/zeros/Const*

index_type0*"
_class
loc:@vc/dense/kernel*
_output_shapes
:	<*
T0
°
vc/dense/kernel/Adam_1
VariableV2*
dtype0*
shape:	<*
shared_name *
_output_shapes
:	<*"
_class
loc:@vc/dense/kernel*
	container 
ŕ
vc/dense/kernel/Adam_1/AssignAssignvc/dense/kernel/Adam_1(vc/dense/kernel/Adam_1/Initializer/zeros*"
_class
loc:@vc/dense/kernel*
use_locking(*
_output_shapes
:	<*
T0*
validate_shape(

vc/dense/kernel/Adam_1/readIdentityvc/dense/kernel/Adam_1*
T0*"
_class
loc:@vc/dense/kernel*
_output_shapes
:	<

$vc/dense/bias/Adam/Initializer/zerosConst*
dtype0*
valueB*    * 
_class
loc:@vc/dense/bias*
_output_shapes	
:
˘
vc/dense/bias/Adam
VariableV2*
dtype0*
shape:*
shared_name *
_output_shapes	
:*
	container * 
_class
loc:@vc/dense/bias
Î
vc/dense/bias/Adam/AssignAssignvc/dense/bias/Adam$vc/dense/bias/Adam/Initializer/zeros* 
_class
loc:@vc/dense/bias*
_output_shapes	
:*
use_locking(*
T0*
validate_shape(

vc/dense/bias/Adam/readIdentityvc/dense/bias/Adam* 
_class
loc:@vc/dense/bias*
T0*
_output_shapes	
:

&vc/dense/bias/Adam_1/Initializer/zerosConst*
_output_shapes	
:* 
_class
loc:@vc/dense/bias*
valueB*    *
dtype0
¤
vc/dense/bias/Adam_1
VariableV2*
	container *
dtype0*
_output_shapes	
:*
shape:*
shared_name * 
_class
loc:@vc/dense/bias
Ô
vc/dense/bias/Adam_1/AssignAssignvc/dense/bias/Adam_1&vc/dense/bias/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes	
:*
T0*
use_locking(* 
_class
loc:@vc/dense/bias

vc/dense/bias/Adam_1/readIdentityvc/dense/bias/Adam_1*
T0*
_output_shapes	
:* 
_class
loc:@vc/dense/bias
Ż
8vc/dense_1/kernel/Adam/Initializer/zeros/shape_as_tensorConst*$
_class
loc:@vc/dense_1/kernel*
valueB"      *
_output_shapes
:*
dtype0

.vc/dense_1/kernel/Adam/Initializer/zeros/ConstConst*
valueB
 *    *
_output_shapes
: *$
_class
loc:@vc/dense_1/kernel*
dtype0
ý
(vc/dense_1/kernel/Adam/Initializer/zerosFill8vc/dense_1/kernel/Adam/Initializer/zeros/shape_as_tensor.vc/dense_1/kernel/Adam/Initializer/zeros/Const* 
_output_shapes
:
*$
_class
loc:@vc/dense_1/kernel*
T0*

index_type0
´
vc/dense_1/kernel/Adam
VariableV2*
dtype0* 
_output_shapes
:
*$
_class
loc:@vc/dense_1/kernel*
shared_name *
shape:
*
	container 
ă
vc/dense_1/kernel/Adam/AssignAssignvc/dense_1/kernel/Adam(vc/dense_1/kernel/Adam/Initializer/zeros*
use_locking(*
validate_shape(* 
_output_shapes
:
*
T0*$
_class
loc:@vc/dense_1/kernel

vc/dense_1/kernel/Adam/readIdentityvc/dense_1/kernel/Adam*$
_class
loc:@vc/dense_1/kernel* 
_output_shapes
:
*
T0
ą
:vc/dense_1/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*$
_class
loc:@vc/dense_1/kernel*
_output_shapes
:*
valueB"      *
dtype0

0vc/dense_1/kernel/Adam_1/Initializer/zeros/ConstConst*
_output_shapes
: *
valueB
 *    *
dtype0*$
_class
loc:@vc/dense_1/kernel

*vc/dense_1/kernel/Adam_1/Initializer/zerosFill:vc/dense_1/kernel/Adam_1/Initializer/zeros/shape_as_tensor0vc/dense_1/kernel/Adam_1/Initializer/zeros/Const*$
_class
loc:@vc/dense_1/kernel*
T0*

index_type0* 
_output_shapes
:

ś
vc/dense_1/kernel/Adam_1
VariableV2* 
_output_shapes
:
*
dtype0*$
_class
loc:@vc/dense_1/kernel*
shared_name *
	container *
shape:

é
vc/dense_1/kernel/Adam_1/AssignAssignvc/dense_1/kernel/Adam_1*vc/dense_1/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0* 
_output_shapes
:
*
validate_shape(*$
_class
loc:@vc/dense_1/kernel

vc/dense_1/kernel/Adam_1/readIdentityvc/dense_1/kernel/Adam_1*$
_class
loc:@vc/dense_1/kernel*
T0* 
_output_shapes
:


&vc/dense_1/bias/Adam/Initializer/zerosConst*
dtype0*
valueB*    *
_output_shapes	
:*"
_class
loc:@vc/dense_1/bias
Ś
vc/dense_1/bias/Adam
VariableV2*
	container *
shape:*
shared_name *
dtype0*
_output_shapes	
:*"
_class
loc:@vc/dense_1/bias
Ö
vc/dense_1/bias/Adam/AssignAssignvc/dense_1/bias/Adam&vc/dense_1/bias/Adam/Initializer/zeros*"
_class
loc:@vc/dense_1/bias*
_output_shapes	
:*
use_locking(*
T0*
validate_shape(

vc/dense_1/bias/Adam/readIdentityvc/dense_1/bias/Adam*
T0*
_output_shapes	
:*"
_class
loc:@vc/dense_1/bias

(vc/dense_1/bias/Adam_1/Initializer/zerosConst*
dtype0*"
_class
loc:@vc/dense_1/bias*
valueB*    *
_output_shapes	
:
¨
vc/dense_1/bias/Adam_1
VariableV2*"
_class
loc:@vc/dense_1/bias*
shared_name *
_output_shapes	
:*
dtype0*
	container *
shape:
Ü
vc/dense_1/bias/Adam_1/AssignAssignvc/dense_1/bias/Adam_1(vc/dense_1/bias/Adam_1/Initializer/zeros*
use_locking(*
_output_shapes	
:*
T0*
validate_shape(*"
_class
loc:@vc/dense_1/bias

vc/dense_1/bias/Adam_1/readIdentityvc/dense_1/bias/Adam_1*
T0*"
_class
loc:@vc/dense_1/bias*
_output_shapes	
:
Ľ
(vc/dense_2/kernel/Adam/Initializer/zerosConst*
dtype0*
valueB	*    *
_output_shapes
:	*$
_class
loc:@vc/dense_2/kernel
˛
vc/dense_2/kernel/Adam
VariableV2*
shape:	*$
_class
loc:@vc/dense_2/kernel*
shared_name *
dtype0*
_output_shapes
:	*
	container 
â
vc/dense_2/kernel/Adam/AssignAssignvc/dense_2/kernel/Adam(vc/dense_2/kernel/Adam/Initializer/zeros*
validate_shape(*
_output_shapes
:	*
T0*
use_locking(*$
_class
loc:@vc/dense_2/kernel

vc/dense_2/kernel/Adam/readIdentityvc/dense_2/kernel/Adam*$
_class
loc:@vc/dense_2/kernel*
T0*
_output_shapes
:	
§
*vc/dense_2/kernel/Adam_1/Initializer/zerosConst*
dtype0*$
_class
loc:@vc/dense_2/kernel*
valueB	*    *
_output_shapes
:	
´
vc/dense_2/kernel/Adam_1
VariableV2*
_output_shapes
:	*
dtype0*
	container *
shared_name *
shape:	*$
_class
loc:@vc/dense_2/kernel
č
vc/dense_2/kernel/Adam_1/AssignAssignvc/dense_2/kernel/Adam_1*vc/dense_2/kernel/Adam_1/Initializer/zeros*$
_class
loc:@vc/dense_2/kernel*
_output_shapes
:	*
validate_shape(*
use_locking(*
T0

vc/dense_2/kernel/Adam_1/readIdentityvc/dense_2/kernel/Adam_1*$
_class
loc:@vc/dense_2/kernel*
_output_shapes
:	*
T0

&vc/dense_2/bias/Adam/Initializer/zerosConst*"
_class
loc:@vc/dense_2/bias*
valueB*    *
_output_shapes
:*
dtype0
¤
vc/dense_2/bias/Adam
VariableV2*
_output_shapes
:*
	container *
shared_name *"
_class
loc:@vc/dense_2/bias*
dtype0*
shape:
Ő
vc/dense_2/bias/Adam/AssignAssignvc/dense_2/bias/Adam&vc/dense_2/bias/Adam/Initializer/zeros*"
_class
loc:@vc/dense_2/bias*
use_locking(*
T0*
_output_shapes
:*
validate_shape(

vc/dense_2/bias/Adam/readIdentityvc/dense_2/bias/Adam*"
_class
loc:@vc/dense_2/bias*
T0*
_output_shapes
:

(vc/dense_2/bias/Adam_1/Initializer/zerosConst*
_output_shapes
:*"
_class
loc:@vc/dense_2/bias*
valueB*    *
dtype0
Ś
vc/dense_2/bias/Adam_1
VariableV2*
shape:*
shared_name *
dtype0*"
_class
loc:@vc/dense_2/bias*
_output_shapes
:*
	container 
Ű
vc/dense_2/bias/Adam_1/AssignAssignvc/dense_2/bias/Adam_1(vc/dense_2/bias/Adam_1/Initializer/zeros*"
_class
loc:@vc/dense_2/bias*
use_locking(*
validate_shape(*
T0*
_output_shapes
:

vc/dense_2/bias/Adam_1/readIdentityvc/dense_2/bias/Adam_1*
T0*"
_class
loc:@vc/dense_2/bias*
_output_shapes
:
Y
Adam_1/learning_rateConst*
dtype0*
valueB
 *o:*
_output_shapes
: 
Q
Adam_1/beta1Const*
dtype0*
valueB
 *fff?*
_output_shapes
: 
Q
Adam_1/beta2Const*
valueB
 *wž?*
_output_shapes
: *
dtype0
S
Adam_1/epsilonConst*
dtype0*
valueB
 *wĚ+2*
_output_shapes
: 
Ţ
'Adam_1/update_vf/dense/kernel/ApplyAdam	ApplyAdamvf/dense/kernelvf/dense/kernel/Adamvf/dense/kernel/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilon
Reshape_40*"
_class
loc:@vf/dense/kernel*
T0*
_output_shapes
:	<*
use_nesterov( *
use_locking( 
Đ
%Adam_1/update_vf/dense/bias/ApplyAdam	ApplyAdamvf/dense/biasvf/dense/bias/Adamvf/dense/bias/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilon
Reshape_41*
T0*
_output_shapes	
:*
use_locking( * 
_class
loc:@vf/dense/bias*
use_nesterov( 
é
)Adam_1/update_vf/dense_1/kernel/ApplyAdam	ApplyAdamvf/dense_1/kernelvf/dense_1/kernel/Adamvf/dense_1/kernel/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilon
Reshape_42*$
_class
loc:@vf/dense_1/kernel*
use_nesterov( * 
_output_shapes
:
*
T0*
use_locking( 
Ú
'Adam_1/update_vf/dense_1/bias/ApplyAdam	ApplyAdamvf/dense_1/biasvf/dense_1/bias/Adamvf/dense_1/bias/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilon
Reshape_43*
use_nesterov( *
T0*"
_class
loc:@vf/dense_1/bias*
use_locking( *
_output_shapes	
:
č
)Adam_1/update_vf/dense_2/kernel/ApplyAdam	ApplyAdamvf/dense_2/kernelvf/dense_2/kernel/Adamvf/dense_2/kernel/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilon
Reshape_44*
use_nesterov( *
T0*
_output_shapes
:	*
use_locking( *$
_class
loc:@vf/dense_2/kernel
Ů
'Adam_1/update_vf/dense_2/bias/ApplyAdam	ApplyAdamvf/dense_2/biasvf/dense_2/bias/Adamvf/dense_2/bias/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilon
Reshape_45*
use_nesterov( *
use_locking( *"
_class
loc:@vf/dense_2/bias*
T0*
_output_shapes
:
Ţ
'Adam_1/update_vc/dense/kernel/ApplyAdam	ApplyAdamvc/dense/kernelvc/dense/kernel/Adamvc/dense/kernel/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilon
Reshape_46*
use_locking( *"
_class
loc:@vc/dense/kernel*
use_nesterov( *
_output_shapes
:	<*
T0
Đ
%Adam_1/update_vc/dense/bias/ApplyAdam	ApplyAdamvc/dense/biasvc/dense/bias/Adamvc/dense/bias/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilon
Reshape_47*
use_nesterov( * 
_class
loc:@vc/dense/bias*
_output_shapes	
:*
use_locking( *
T0
é
)Adam_1/update_vc/dense_1/kernel/ApplyAdam	ApplyAdamvc/dense_1/kernelvc/dense_1/kernel/Adamvc/dense_1/kernel/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilon
Reshape_48*$
_class
loc:@vc/dense_1/kernel* 
_output_shapes
:
*
use_locking( *
use_nesterov( *
T0
Ú
'Adam_1/update_vc/dense_1/bias/ApplyAdam	ApplyAdamvc/dense_1/biasvc/dense_1/bias/Adamvc/dense_1/bias/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilon
Reshape_49*
T0*"
_class
loc:@vc/dense_1/bias*
use_locking( *
use_nesterov( *
_output_shapes	
:
č
)Adam_1/update_vc/dense_2/kernel/ApplyAdam	ApplyAdamvc/dense_2/kernelvc/dense_2/kernel/Adamvc/dense_2/kernel/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilon
Reshape_50*
use_locking( *$
_class
loc:@vc/dense_2/kernel*
T0*
_output_shapes
:	*
use_nesterov( 
Ů
'Adam_1/update_vc/dense_2/bias/ApplyAdam	ApplyAdamvc/dense_2/biasvc/dense_2/bias/Adamvc/dense_2/bias/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilon
Reshape_51*
use_nesterov( *
T0*
_output_shapes
:*"
_class
loc:@vc/dense_2/bias*
use_locking( 
ň

Adam_1/mulMulbeta1_power_1/readAdam_1/beta1&^Adam_1/update_vc/dense/bias/ApplyAdam(^Adam_1/update_vc/dense/kernel/ApplyAdam(^Adam_1/update_vc/dense_1/bias/ApplyAdam*^Adam_1/update_vc/dense_1/kernel/ApplyAdam(^Adam_1/update_vc/dense_2/bias/ApplyAdam*^Adam_1/update_vc/dense_2/kernel/ApplyAdam&^Adam_1/update_vf/dense/bias/ApplyAdam(^Adam_1/update_vf/dense/kernel/ApplyAdam(^Adam_1/update_vf/dense_1/bias/ApplyAdam*^Adam_1/update_vf/dense_1/kernel/ApplyAdam(^Adam_1/update_vf/dense_2/bias/ApplyAdam*^Adam_1/update_vf/dense_2/kernel/ApplyAdam*
_output_shapes
: *
T0* 
_class
loc:@vc/dense/bias

Adam_1/AssignAssignbeta1_power_1
Adam_1/mul*
T0*
validate_shape(*
_output_shapes
: * 
_class
loc:@vc/dense/bias*
use_locking( 
ô
Adam_1/mul_1Mulbeta2_power_1/readAdam_1/beta2&^Adam_1/update_vc/dense/bias/ApplyAdam(^Adam_1/update_vc/dense/kernel/ApplyAdam(^Adam_1/update_vc/dense_1/bias/ApplyAdam*^Adam_1/update_vc/dense_1/kernel/ApplyAdam(^Adam_1/update_vc/dense_2/bias/ApplyAdam*^Adam_1/update_vc/dense_2/kernel/ApplyAdam&^Adam_1/update_vf/dense/bias/ApplyAdam(^Adam_1/update_vf/dense/kernel/ApplyAdam(^Adam_1/update_vf/dense_1/bias/ApplyAdam*^Adam_1/update_vf/dense_1/kernel/ApplyAdam(^Adam_1/update_vf/dense_2/bias/ApplyAdam*^Adam_1/update_vf/dense_2/kernel/ApplyAdam* 
_class
loc:@vc/dense/bias*
_output_shapes
: *
T0
˘
Adam_1/Assign_1Assignbeta2_power_1Adam_1/mul_1*
_output_shapes
: *
validate_shape(*
use_locking( * 
_class
loc:@vc/dense/bias*
T0
Ź
Adam_1NoOp^Adam_1/Assign^Adam_1/Assign_1&^Adam_1/update_vc/dense/bias/ApplyAdam(^Adam_1/update_vc/dense/kernel/ApplyAdam(^Adam_1/update_vc/dense_1/bias/ApplyAdam*^Adam_1/update_vc/dense_1/kernel/ApplyAdam(^Adam_1/update_vc/dense_2/bias/ApplyAdam*^Adam_1/update_vc/dense_2/kernel/ApplyAdam&^Adam_1/update_vf/dense/bias/ApplyAdam(^Adam_1/update_vf/dense/kernel/ApplyAdam(^Adam_1/update_vf/dense_1/bias/ApplyAdam*^Adam_1/update_vf/dense_1/kernel/ApplyAdam(^Adam_1/update_vf/dense_2/bias/ApplyAdam*^Adam_1/update_vf/dense_2/kernel/ApplyAdam
l
Reshape_52/shapeConst^Adam_1*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0
q

Reshape_52Reshapevf/dense/kernel/readReshape_52/shape*
_output_shapes	
:x*
Tshape0*
T0
l
Reshape_53/shapeConst^Adam_1*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0
o

Reshape_53Reshapevf/dense/bias/readReshape_53/shape*
T0*
_output_shapes	
:*
Tshape0
l
Reshape_54/shapeConst^Adam_1*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙*
_output_shapes
:
t

Reshape_54Reshapevf/dense_1/kernel/readReshape_54/shape*
_output_shapes

:*
Tshape0*
T0
l
Reshape_55/shapeConst^Adam_1*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0
q

Reshape_55Reshapevf/dense_1/bias/readReshape_55/shape*
Tshape0*
T0*
_output_shapes	
:
l
Reshape_56/shapeConst^Adam_1*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
s

Reshape_56Reshapevf/dense_2/kernel/readReshape_56/shape*
Tshape0*
T0*
_output_shapes	
:
l
Reshape_57/shapeConst^Adam_1*
dtype0*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙
p

Reshape_57Reshapevf/dense_2/bias/readReshape_57/shape*
_output_shapes
:*
T0*
Tshape0
l
Reshape_58/shapeConst^Adam_1*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙
q

Reshape_58Reshapevc/dense/kernel/readReshape_58/shape*
Tshape0*
_output_shapes	
:x*
T0
l
Reshape_59/shapeConst^Adam_1*
valueB:
˙˙˙˙˙˙˙˙˙*
_output_shapes
:*
dtype0
o

Reshape_59Reshapevc/dense/bias/readReshape_59/shape*
Tshape0*
_output_shapes	
:*
T0
l
Reshape_60/shapeConst^Adam_1*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙
t

Reshape_60Reshapevc/dense_1/kernel/readReshape_60/shape*
Tshape0*
_output_shapes

:*
T0
l
Reshape_61/shapeConst^Adam_1*
valueB:
˙˙˙˙˙˙˙˙˙*
_output_shapes
:*
dtype0
q

Reshape_61Reshapevc/dense_1/bias/readReshape_61/shape*
Tshape0*
_output_shapes	
:*
T0
l
Reshape_62/shapeConst^Adam_1*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
s

Reshape_62Reshapevc/dense_2/kernel/readReshape_62/shape*
_output_shapes	
:*
Tshape0*
T0
l
Reshape_63/shapeConst^Adam_1*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
p

Reshape_63Reshapevc/dense_2/bias/readReshape_63/shape*
_output_shapes
:*
Tshape0*
T0
X
concat_3/axisConst^Adam_1*
_output_shapes
: *
dtype0*
value	B : 
ď
concat_3ConcatV2
Reshape_52
Reshape_53
Reshape_54
Reshape_55
Reshape_56
Reshape_57
Reshape_58
Reshape_59
Reshape_60
Reshape_61
Reshape_62
Reshape_63concat_3/axis*
T0*
N*
_output_shapes

:ü	*

Tidx0
h
PyFunc_3PyFuncconcat_3*
Tin
2*
Tout
2*
token
pyfunc_3*
_output_shapes
:

Const_8Const^Adam_1*
dtype0*
_output_shapes
:*E
value<B:"0 <                  <                 
\
split_3/split_dimConst^Adam_1*
dtype0*
_output_shapes
: *
value	B : 
Ł
split_3SplitVPyFunc_3Const_8split_3/split_dim*
T0*D
_output_shapes2
0::::::::::::*
	num_split*

Tlen0
j
Reshape_64/shapeConst^Adam_1*
_output_shapes
:*
valueB"<      *
dtype0
h

Reshape_64Reshapesplit_3Reshape_64/shape*
_output_shapes
:	<*
T0*
Tshape0
d
Reshape_65/shapeConst^Adam_1*
valueB:*
dtype0*
_output_shapes
:
f

Reshape_65Reshape	split_3:1Reshape_65/shape*
T0*
_output_shapes	
:*
Tshape0
j
Reshape_66/shapeConst^Adam_1*
dtype0*
_output_shapes
:*
valueB"      
k

Reshape_66Reshape	split_3:2Reshape_66/shape*
T0* 
_output_shapes
:
*
Tshape0
d
Reshape_67/shapeConst^Adam_1*
_output_shapes
:*
dtype0*
valueB:
f

Reshape_67Reshape	split_3:3Reshape_67/shape*
Tshape0*
T0*
_output_shapes	
:
j
Reshape_68/shapeConst^Adam_1*
_output_shapes
:*
valueB"      *
dtype0
j

Reshape_68Reshape	split_3:4Reshape_68/shape*
Tshape0*
_output_shapes
:	*
T0
c
Reshape_69/shapeConst^Adam_1*
valueB:*
_output_shapes
:*
dtype0
e

Reshape_69Reshape	split_3:5Reshape_69/shape*
T0*
Tshape0*
_output_shapes
:
j
Reshape_70/shapeConst^Adam_1*
_output_shapes
:*
dtype0*
valueB"<      
j

Reshape_70Reshape	split_3:6Reshape_70/shape*
T0*
_output_shapes
:	<*
Tshape0
d
Reshape_71/shapeConst^Adam_1*
_output_shapes
:*
valueB:*
dtype0
f

Reshape_71Reshape	split_3:7Reshape_71/shape*
_output_shapes	
:*
T0*
Tshape0
j
Reshape_72/shapeConst^Adam_1*
_output_shapes
:*
valueB"      *
dtype0
k

Reshape_72Reshape	split_3:8Reshape_72/shape*
T0*
Tshape0* 
_output_shapes
:

d
Reshape_73/shapeConst^Adam_1*
valueB:*
dtype0*
_output_shapes
:
f

Reshape_73Reshape	split_3:9Reshape_73/shape*
Tshape0*
_output_shapes	
:*
T0
j
Reshape_74/shapeConst^Adam_1*
valueB"      *
dtype0*
_output_shapes
:
k

Reshape_74Reshape
split_3:10Reshape_74/shape*
_output_shapes
:	*
Tshape0*
T0
c
Reshape_75/shapeConst^Adam_1*
_output_shapes
:*
valueB:*
dtype0
f

Reshape_75Reshape
split_3:11Reshape_75/shape*
_output_shapes
:*
Tshape0*
T0
Ś
Assign_7Assignvf/dense/kernel
Reshape_64*"
_class
loc:@vf/dense/kernel*
_output_shapes
:	<*
use_locking(*
validate_shape(*
T0

Assign_8Assignvf/dense/bias
Reshape_65*
validate_shape(*
use_locking(*
T0*
_output_shapes	
:* 
_class
loc:@vf/dense/bias
Ť
Assign_9Assignvf/dense_1/kernel
Reshape_66* 
_output_shapes
:
*$
_class
loc:@vf/dense_1/kernel*
use_locking(*
T0*
validate_shape(
Ł
	Assign_10Assignvf/dense_1/bias
Reshape_67*
T0*"
_class
loc:@vf/dense_1/bias*
use_locking(*
validate_shape(*
_output_shapes	
:
Ť
	Assign_11Assignvf/dense_2/kernel
Reshape_68*
_output_shapes
:	*
validate_shape(*
use_locking(*$
_class
loc:@vf/dense_2/kernel*
T0
˘
	Assign_12Assignvf/dense_2/bias
Reshape_69*
T0*"
_class
loc:@vf/dense_2/bias*
_output_shapes
:*
use_locking(*
validate_shape(
§
	Assign_13Assignvc/dense/kernel
Reshape_70*
validate_shape(*
_output_shapes
:	<*
T0*
use_locking(*"
_class
loc:@vc/dense/kernel

	Assign_14Assignvc/dense/bias
Reshape_71*
validate_shape(*
T0*
use_locking(* 
_class
loc:@vc/dense/bias*
_output_shapes	
:
Ź
	Assign_15Assignvc/dense_1/kernel
Reshape_72* 
_output_shapes
:
*
use_locking(*$
_class
loc:@vc/dense_1/kernel*
validate_shape(*
T0
Ł
	Assign_16Assignvc/dense_1/bias
Reshape_73*
_output_shapes	
:*
use_locking(*
T0*
validate_shape(*"
_class
loc:@vc/dense_1/bias
Ť
	Assign_17Assignvc/dense_2/kernel
Reshape_74*
use_locking(*
T0*
_output_shapes
:	*
validate_shape(*$
_class
loc:@vc/dense_2/kernel
˘
	Assign_18Assignvc/dense_2/bias
Reshape_75*
_output_shapes
:*
validate_shape(*"
_class
loc:@vc/dense_2/bias*
T0*
use_locking(
Ş
group_deps_2NoOp^Adam_1
^Assign_10
^Assign_11
^Assign_12
^Assign_13
^Assign_14
^Assign_15
^Assign_16
^Assign_17
^Assign_18	^Assign_7	^Assign_8	^Assign_9
,
group_deps_3NoOp^Adam_1^group_deps_2
Ü
initNoOp^beta1_power/Assign^beta1_power_1/Assign^beta2_power/Assign^beta2_power_1/Assign^pi/dense/bias/Adam/Assign^pi/dense/bias/Adam_1/Assign^pi/dense/bias/Assign^pi/dense/kernel/Adam/Assign^pi/dense/kernel/Adam_1/Assign^pi/dense/kernel/Assign^pi/dense_1/bias/Adam/Assign^pi/dense_1/bias/Adam_1/Assign^pi/dense_1/bias/Assign^pi/dense_1/kernel/Adam/Assign ^pi/dense_1/kernel/Adam_1/Assign^pi/dense_1/kernel/Assign^pi/dense_2/bias/Adam/Assign^pi/dense_2/bias/Adam_1/Assign^pi/dense_2/bias/Assign^pi/dense_2/kernel/Adam/Assign ^pi/dense_2/kernel/Adam_1/Assign^pi/dense_2/kernel/Assign^pi/log_std/Adam/Assign^pi/log_std/Adam_1/Assign^pi/log_std/Assign^vc/dense/bias/Adam/Assign^vc/dense/bias/Adam_1/Assign^vc/dense/bias/Assign^vc/dense/kernel/Adam/Assign^vc/dense/kernel/Adam_1/Assign^vc/dense/kernel/Assign^vc/dense_1/bias/Adam/Assign^vc/dense_1/bias/Adam_1/Assign^vc/dense_1/bias/Assign^vc/dense_1/kernel/Adam/Assign ^vc/dense_1/kernel/Adam_1/Assign^vc/dense_1/kernel/Assign^vc/dense_2/bias/Adam/Assign^vc/dense_2/bias/Adam_1/Assign^vc/dense_2/bias/Assign^vc/dense_2/kernel/Adam/Assign ^vc/dense_2/kernel/Adam_1/Assign^vc/dense_2/kernel/Assign^vf/dense/bias/Adam/Assign^vf/dense/bias/Adam_1/Assign^vf/dense/bias/Assign^vf/dense/kernel/Adam/Assign^vf/dense/kernel/Adam_1/Assign^vf/dense/kernel/Assign^vf/dense_1/bias/Adam/Assign^vf/dense_1/bias/Adam_1/Assign^vf/dense_1/bias/Assign^vf/dense_1/kernel/Adam/Assign ^vf/dense_1/kernel/Adam_1/Assign^vf/dense_1/kernel/Assign^vf/dense_2/bias/Adam/Assign^vf/dense_2/bias/Adam_1/Assign^vf/dense_2/bias/Assign^vf/dense_2/kernel/Adam/Assign ^vf/dense_2/kernel/Adam_1/Assign^vf/dense_2/kernel/Assign
c
Reshape_76/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
q

Reshape_76Reshapepi/dense/kernel/readReshape_76/shape*
_output_shapes	
:x*
T0*
Tshape0
c
Reshape_77/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
o

Reshape_77Reshapepi/dense/bias/readReshape_77/shape*
Tshape0*
T0*
_output_shapes	
:
c
Reshape_78/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
_output_shapes
:*
dtype0
t

Reshape_78Reshapepi/dense_1/kernel/readReshape_78/shape*
_output_shapes

:*
Tshape0*
T0
c
Reshape_79/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
_output_shapes
:*
dtype0
q

Reshape_79Reshapepi/dense_1/bias/readReshape_79/shape*
Tshape0*
_output_shapes	
:*
T0
c
Reshape_80/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
s

Reshape_80Reshapepi/dense_2/kernel/readReshape_80/shape*
_output_shapes	
:*
T0*
Tshape0
c
Reshape_81/shapeConst*
dtype0*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙
p

Reshape_81Reshapepi/dense_2/bias/readReshape_81/shape*
_output_shapes
:*
T0*
Tshape0
c
Reshape_82/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
_output_shapes
:*
dtype0
k

Reshape_82Reshapepi/log_std/readReshape_82/shape*
Tshape0*
T0*
_output_shapes
:
c
Reshape_83/shapeConst*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙*
_output_shapes
:
q

Reshape_83Reshapevf/dense/kernel/readReshape_83/shape*
T0*
_output_shapes	
:x*
Tshape0
c
Reshape_84/shapeConst*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙*
_output_shapes
:
o

Reshape_84Reshapevf/dense/bias/readReshape_84/shape*
Tshape0*
T0*
_output_shapes	
:
c
Reshape_85/shapeConst*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0
t

Reshape_85Reshapevf/dense_1/kernel/readReshape_85/shape*
T0*
Tshape0*
_output_shapes

:
c
Reshape_86/shapeConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙
q

Reshape_86Reshapevf/dense_1/bias/readReshape_86/shape*
_output_shapes	
:*
Tshape0*
T0
c
Reshape_87/shapeConst*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙*
_output_shapes
:
s

Reshape_87Reshapevf/dense_2/kernel/readReshape_87/shape*
Tshape0*
_output_shapes	
:*
T0
c
Reshape_88/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
_output_shapes
:*
dtype0
p

Reshape_88Reshapevf/dense_2/bias/readReshape_88/shape*
T0*
_output_shapes
:*
Tshape0
c
Reshape_89/shapeConst*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0
q

Reshape_89Reshapevc/dense/kernel/readReshape_89/shape*
_output_shapes	
:x*
T0*
Tshape0
c
Reshape_90/shapeConst*
dtype0*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙
o

Reshape_90Reshapevc/dense/bias/readReshape_90/shape*
Tshape0*
_output_shapes	
:*
T0
c
Reshape_91/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
_output_shapes
:*
dtype0
t

Reshape_91Reshapevc/dense_1/kernel/readReshape_91/shape*
Tshape0*
T0*
_output_shapes

:
c
Reshape_92/shapeConst*
dtype0*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙
q

Reshape_92Reshapevc/dense_1/bias/readReshape_92/shape*
Tshape0*
_output_shapes	
:*
T0
c
Reshape_93/shapeConst*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙*
_output_shapes
:
s

Reshape_93Reshapevc/dense_2/kernel/readReshape_93/shape*
Tshape0*
T0*
_output_shapes	
:
c
Reshape_94/shapeConst*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙*
_output_shapes
:
p

Reshape_94Reshapevc/dense_2/bias/readReshape_94/shape*
Tshape0*
T0*
_output_shapes
:
c
Reshape_95/shapeConst*
dtype0*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙
l

Reshape_95Reshapebeta1_power/readReshape_95/shape*
_output_shapes
:*
Tshape0*
T0
c
Reshape_96/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
_output_shapes
:*
dtype0
l

Reshape_96Reshapebeta2_power/readReshape_96/shape*
_output_shapes
:*
Tshape0*
T0
c
Reshape_97/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
v

Reshape_97Reshapepi/dense/kernel/Adam/readReshape_97/shape*
_output_shapes	
:x*
T0*
Tshape0
c
Reshape_98/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
x

Reshape_98Reshapepi/dense/kernel/Adam_1/readReshape_98/shape*
T0*
Tshape0*
_output_shapes	
:x
c
Reshape_99/shapeConst*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙*
_output_shapes
:
t

Reshape_99Reshapepi/dense/bias/Adam/readReshape_99/shape*
Tshape0*
T0*
_output_shapes	
:
d
Reshape_100/shapeConst*
dtype0*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙
x
Reshape_100Reshapepi/dense/bias/Adam_1/readReshape_100/shape*
Tshape0*
T0*
_output_shapes	
:
d
Reshape_101/shapeConst*
dtype0*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙
{
Reshape_101Reshapepi/dense_1/kernel/Adam/readReshape_101/shape*
Tshape0*
_output_shapes

:*
T0
d
Reshape_102/shapeConst*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0
}
Reshape_102Reshapepi/dense_1/kernel/Adam_1/readReshape_102/shape*
Tshape0*
_output_shapes

:*
T0
d
Reshape_103/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
x
Reshape_103Reshapepi/dense_1/bias/Adam/readReshape_103/shape*
_output_shapes	
:*
T0*
Tshape0
d
Reshape_104/shapeConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙
z
Reshape_104Reshapepi/dense_1/bias/Adam_1/readReshape_104/shape*
Tshape0*
T0*
_output_shapes	
:
d
Reshape_105/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
z
Reshape_105Reshapepi/dense_2/kernel/Adam/readReshape_105/shape*
T0*
_output_shapes	
:*
Tshape0
d
Reshape_106/shapeConst*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙*
_output_shapes
:
|
Reshape_106Reshapepi/dense_2/kernel/Adam_1/readReshape_106/shape*
T0*
Tshape0*
_output_shapes	
:
d
Reshape_107/shapeConst*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0
w
Reshape_107Reshapepi/dense_2/bias/Adam/readReshape_107/shape*
_output_shapes
:*
T0*
Tshape0
d
Reshape_108/shapeConst*
dtype0*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙
y
Reshape_108Reshapepi/dense_2/bias/Adam_1/readReshape_108/shape*
Tshape0*
T0*
_output_shapes
:
d
Reshape_109/shapeConst*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0
r
Reshape_109Reshapepi/log_std/Adam/readReshape_109/shape*
T0*
Tshape0*
_output_shapes
:
d
Reshape_110/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
t
Reshape_110Reshapepi/log_std/Adam_1/readReshape_110/shape*
_output_shapes
:*
T0*
Tshape0
d
Reshape_111/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
p
Reshape_111Reshapebeta1_power_1/readReshape_111/shape*
Tshape0*
_output_shapes
:*
T0
d
Reshape_112/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
_output_shapes
:*
dtype0
p
Reshape_112Reshapebeta2_power_1/readReshape_112/shape*
_output_shapes
:*
Tshape0*
T0
d
Reshape_113/shapeConst*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙*
_output_shapes
:
x
Reshape_113Reshapevf/dense/kernel/Adam/readReshape_113/shape*
T0*
Tshape0*
_output_shapes	
:x
d
Reshape_114/shapeConst*
dtype0*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙
z
Reshape_114Reshapevf/dense/kernel/Adam_1/readReshape_114/shape*
_output_shapes	
:x*
T0*
Tshape0
d
Reshape_115/shapeConst*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙*
_output_shapes
:
v
Reshape_115Reshapevf/dense/bias/Adam/readReshape_115/shape*
_output_shapes	
:*
Tshape0*
T0
d
Reshape_116/shapeConst*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙*
_output_shapes
:
x
Reshape_116Reshapevf/dense/bias/Adam_1/readReshape_116/shape*
_output_shapes	
:*
T0*
Tshape0
d
Reshape_117/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
{
Reshape_117Reshapevf/dense_1/kernel/Adam/readReshape_117/shape*
Tshape0*
_output_shapes

:*
T0
d
Reshape_118/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
}
Reshape_118Reshapevf/dense_1/kernel/Adam_1/readReshape_118/shape*
T0*
_output_shapes

:*
Tshape0
d
Reshape_119/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
x
Reshape_119Reshapevf/dense_1/bias/Adam/readReshape_119/shape*
Tshape0*
_output_shapes	
:*
T0
d
Reshape_120/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
_output_shapes
:*
dtype0
z
Reshape_120Reshapevf/dense_1/bias/Adam_1/readReshape_120/shape*
Tshape0*
T0*
_output_shapes	
:
d
Reshape_121/shapeConst*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0
z
Reshape_121Reshapevf/dense_2/kernel/Adam/readReshape_121/shape*
Tshape0*
T0*
_output_shapes	
:
d
Reshape_122/shapeConst*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙*
_output_shapes
:
|
Reshape_122Reshapevf/dense_2/kernel/Adam_1/readReshape_122/shape*
T0*
Tshape0*
_output_shapes	
:
d
Reshape_123/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
w
Reshape_123Reshapevf/dense_2/bias/Adam/readReshape_123/shape*
Tshape0*
T0*
_output_shapes
:
d
Reshape_124/shapeConst*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0
y
Reshape_124Reshapevf/dense_2/bias/Adam_1/readReshape_124/shape*
_output_shapes
:*
Tshape0*
T0
d
Reshape_125/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
x
Reshape_125Reshapevc/dense/kernel/Adam/readReshape_125/shape*
Tshape0*
_output_shapes	
:x*
T0
d
Reshape_126/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
z
Reshape_126Reshapevc/dense/kernel/Adam_1/readReshape_126/shape*
T0*
_output_shapes	
:x*
Tshape0
d
Reshape_127/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
_output_shapes
:*
dtype0
v
Reshape_127Reshapevc/dense/bias/Adam/readReshape_127/shape*
T0*
_output_shapes	
:*
Tshape0
d
Reshape_128/shapeConst*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙*
_output_shapes
:
x
Reshape_128Reshapevc/dense/bias/Adam_1/readReshape_128/shape*
Tshape0*
T0*
_output_shapes	
:
d
Reshape_129/shapeConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙
{
Reshape_129Reshapevc/dense_1/kernel/Adam/readReshape_129/shape*
_output_shapes

:*
T0*
Tshape0
d
Reshape_130/shapeConst*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙*
_output_shapes
:
}
Reshape_130Reshapevc/dense_1/kernel/Adam_1/readReshape_130/shape*
Tshape0*
T0*
_output_shapes

:
d
Reshape_131/shapeConst*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0
x
Reshape_131Reshapevc/dense_1/bias/Adam/readReshape_131/shape*
_output_shapes	
:*
T0*
Tshape0
d
Reshape_132/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
_output_shapes
:*
dtype0
z
Reshape_132Reshapevc/dense_1/bias/Adam_1/readReshape_132/shape*
_output_shapes	
:*
T0*
Tshape0
d
Reshape_133/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
_output_shapes
:*
dtype0
z
Reshape_133Reshapevc/dense_2/kernel/Adam/readReshape_133/shape*
_output_shapes	
:*
Tshape0*
T0
d
Reshape_134/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
_output_shapes
:*
dtype0
|
Reshape_134Reshapevc/dense_2/kernel/Adam_1/readReshape_134/shape*
T0*
_output_shapes	
:*
Tshape0
d
Reshape_135/shapeConst*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0
w
Reshape_135Reshapevc/dense_2/bias/Adam/readReshape_135/shape*
T0*
Tshape0*
_output_shapes
:
d
Reshape_136/shapeConst*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0
y
Reshape_136Reshapevc/dense_2/bias/Adam_1/readReshape_136/shape*
_output_shapes
:*
T0*
Tshape0
O
concat_4/axisConst*
_output_shapes
: *
dtype0*
value	B : 
ŕ
concat_4ConcatV2
Reshape_76
Reshape_77
Reshape_78
Reshape_79
Reshape_80
Reshape_81
Reshape_82
Reshape_83
Reshape_84
Reshape_85
Reshape_86
Reshape_87
Reshape_88
Reshape_89
Reshape_90
Reshape_91
Reshape_92
Reshape_93
Reshape_94
Reshape_95
Reshape_96
Reshape_97
Reshape_98
Reshape_99Reshape_100Reshape_101Reshape_102Reshape_103Reshape_104Reshape_105Reshape_106Reshape_107Reshape_108Reshape_109Reshape_110Reshape_111Reshape_112Reshape_113Reshape_114Reshape_115Reshape_116Reshape_117Reshape_118Reshape_119Reshape_120Reshape_121Reshape_122Reshape_123Reshape_124Reshape_125Reshape_126Reshape_127Reshape_128Reshape_129Reshape_130Reshape_131Reshape_132Reshape_133Reshape_134Reshape_135Reshape_136concat_4/axis*

Tidx0*
T0*
N=*
_output_shapes

:ô,
h
PyFunc_4PyFuncconcat_4*
Tin
2*
token
pyfunc_4*
Tout
2*
_output_shapes
:
Č
Const_9Const*
valueB˙="ô <                     <                  <                        <   <                                             <   <                                 <   <                                *
dtype0*
_output_shapes
:=
S
split_4/split_dimConst*
value	B : *
dtype0*
_output_shapes
: 
ę
split_4SplitVPyFunc_4Const_9split_4/split_dim*
	num_split=*

Tlen0*
T0*
_output_shapes÷
ô:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
b
Reshape_137/shapeConst*
dtype0*
valueB"<      *
_output_shapes
:
j
Reshape_137Reshapesplit_4Reshape_137/shape*
_output_shapes
:	<*
Tshape0*
T0
\
Reshape_138/shapeConst*
dtype0*
_output_shapes
:*
valueB:
h
Reshape_138Reshape	split_4:1Reshape_138/shape*
T0*
_output_shapes	
:*
Tshape0
b
Reshape_139/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
m
Reshape_139Reshape	split_4:2Reshape_139/shape*
Tshape0*
T0* 
_output_shapes
:

\
Reshape_140/shapeConst*
valueB:*
_output_shapes
:*
dtype0
h
Reshape_140Reshape	split_4:3Reshape_140/shape*
_output_shapes	
:*
Tshape0*
T0
b
Reshape_141/shapeConst*
dtype0*
valueB"      *
_output_shapes
:
l
Reshape_141Reshape	split_4:4Reshape_141/shape*
_output_shapes
:	*
Tshape0*
T0
[
Reshape_142/shapeConst*
_output_shapes
:*
dtype0*
valueB:
g
Reshape_142Reshape	split_4:5Reshape_142/shape*
T0*
Tshape0*
_output_shapes
:
[
Reshape_143/shapeConst*
_output_shapes
:*
dtype0*
valueB:
g
Reshape_143Reshape	split_4:6Reshape_143/shape*
_output_shapes
:*
T0*
Tshape0
b
Reshape_144/shapeConst*
valueB"<      *
_output_shapes
:*
dtype0
l
Reshape_144Reshape	split_4:7Reshape_144/shape*
Tshape0*
_output_shapes
:	<*
T0
\
Reshape_145/shapeConst*
valueB:*
_output_shapes
:*
dtype0
h
Reshape_145Reshape	split_4:8Reshape_145/shape*
_output_shapes	
:*
T0*
Tshape0
b
Reshape_146/shapeConst*
_output_shapes
:*
valueB"      *
dtype0
m
Reshape_146Reshape	split_4:9Reshape_146/shape*
T0* 
_output_shapes
:
*
Tshape0
\
Reshape_147/shapeConst*
dtype0*
_output_shapes
:*
valueB:
i
Reshape_147Reshape
split_4:10Reshape_147/shape*
T0*
Tshape0*
_output_shapes	
:
b
Reshape_148/shapeConst*
dtype0*
valueB"      *
_output_shapes
:
m
Reshape_148Reshape
split_4:11Reshape_148/shape*
T0*
Tshape0*
_output_shapes
:	
[
Reshape_149/shapeConst*
_output_shapes
:*
valueB:*
dtype0
h
Reshape_149Reshape
split_4:12Reshape_149/shape*
Tshape0*
_output_shapes
:*
T0
b
Reshape_150/shapeConst*
dtype0*
_output_shapes
:*
valueB"<      
m
Reshape_150Reshape
split_4:13Reshape_150/shape*
_output_shapes
:	<*
Tshape0*
T0
\
Reshape_151/shapeConst*
valueB:*
dtype0*
_output_shapes
:
i
Reshape_151Reshape
split_4:14Reshape_151/shape*
T0*
Tshape0*
_output_shapes	
:
b
Reshape_152/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
n
Reshape_152Reshape
split_4:15Reshape_152/shape*
T0*
Tshape0* 
_output_shapes
:

\
Reshape_153/shapeConst*
valueB:*
dtype0*
_output_shapes
:
i
Reshape_153Reshape
split_4:16Reshape_153/shape*
Tshape0*
_output_shapes	
:*
T0
b
Reshape_154/shapeConst*
dtype0*
valueB"      *
_output_shapes
:
m
Reshape_154Reshape
split_4:17Reshape_154/shape*
Tshape0*
T0*
_output_shapes
:	
[
Reshape_155/shapeConst*
valueB:*
dtype0*
_output_shapes
:
h
Reshape_155Reshape
split_4:18Reshape_155/shape*
_output_shapes
:*
T0*
Tshape0
T
Reshape_156/shapeConst*
_output_shapes
: *
dtype0*
valueB 
d
Reshape_156Reshape
split_4:19Reshape_156/shape*
T0*
Tshape0*
_output_shapes
: 
T
Reshape_157/shapeConst*
dtype0*
valueB *
_output_shapes
: 
d
Reshape_157Reshape
split_4:20Reshape_157/shape*
Tshape0*
T0*
_output_shapes
: 
b
Reshape_158/shapeConst*
_output_shapes
:*
dtype0*
valueB"<      
m
Reshape_158Reshape
split_4:21Reshape_158/shape*
_output_shapes
:	<*
T0*
Tshape0
b
Reshape_159/shapeConst*
_output_shapes
:*
dtype0*
valueB"<      
m
Reshape_159Reshape
split_4:22Reshape_159/shape*
_output_shapes
:	<*
T0*
Tshape0
\
Reshape_160/shapeConst*
dtype0*
_output_shapes
:*
valueB:
i
Reshape_160Reshape
split_4:23Reshape_160/shape*
_output_shapes	
:*
Tshape0*
T0
\
Reshape_161/shapeConst*
dtype0*
valueB:*
_output_shapes
:
i
Reshape_161Reshape
split_4:24Reshape_161/shape*
Tshape0*
_output_shapes	
:*
T0
b
Reshape_162/shapeConst*
_output_shapes
:*
valueB"      *
dtype0
n
Reshape_162Reshape
split_4:25Reshape_162/shape*
T0*
Tshape0* 
_output_shapes
:

b
Reshape_163/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
n
Reshape_163Reshape
split_4:26Reshape_163/shape*
T0*
Tshape0* 
_output_shapes
:

\
Reshape_164/shapeConst*
dtype0*
_output_shapes
:*
valueB:
i
Reshape_164Reshape
split_4:27Reshape_164/shape*
T0*
_output_shapes	
:*
Tshape0
\
Reshape_165/shapeConst*
valueB:*
dtype0*
_output_shapes
:
i
Reshape_165Reshape
split_4:28Reshape_165/shape*
_output_shapes	
:*
Tshape0*
T0
b
Reshape_166/shapeConst*
dtype0*
_output_shapes
:*
valueB"      
m
Reshape_166Reshape
split_4:29Reshape_166/shape*
Tshape0*
_output_shapes
:	*
T0
b
Reshape_167/shapeConst*
dtype0*
valueB"      *
_output_shapes
:
m
Reshape_167Reshape
split_4:30Reshape_167/shape*
T0*
_output_shapes
:	*
Tshape0
[
Reshape_168/shapeConst*
valueB:*
_output_shapes
:*
dtype0
h
Reshape_168Reshape
split_4:31Reshape_168/shape*
T0*
Tshape0*
_output_shapes
:
[
Reshape_169/shapeConst*
dtype0*
valueB:*
_output_shapes
:
h
Reshape_169Reshape
split_4:32Reshape_169/shape*
_output_shapes
:*
T0*
Tshape0
[
Reshape_170/shapeConst*
_output_shapes
:*
dtype0*
valueB:
h
Reshape_170Reshape
split_4:33Reshape_170/shape*
_output_shapes
:*
T0*
Tshape0
[
Reshape_171/shapeConst*
_output_shapes
:*
valueB:*
dtype0
h
Reshape_171Reshape
split_4:34Reshape_171/shape*
T0*
_output_shapes
:*
Tshape0
T
Reshape_172/shapeConst*
valueB *
dtype0*
_output_shapes
: 
d
Reshape_172Reshape
split_4:35Reshape_172/shape*
T0*
_output_shapes
: *
Tshape0
T
Reshape_173/shapeConst*
dtype0*
_output_shapes
: *
valueB 
d
Reshape_173Reshape
split_4:36Reshape_173/shape*
Tshape0*
T0*
_output_shapes
: 
b
Reshape_174/shapeConst*
_output_shapes
:*
dtype0*
valueB"<      
m
Reshape_174Reshape
split_4:37Reshape_174/shape*
_output_shapes
:	<*
T0*
Tshape0
b
Reshape_175/shapeConst*
dtype0*
valueB"<      *
_output_shapes
:
m
Reshape_175Reshape
split_4:38Reshape_175/shape*
Tshape0*
_output_shapes
:	<*
T0
\
Reshape_176/shapeConst*
valueB:*
dtype0*
_output_shapes
:
i
Reshape_176Reshape
split_4:39Reshape_176/shape*
Tshape0*
_output_shapes	
:*
T0
\
Reshape_177/shapeConst*
_output_shapes
:*
valueB:*
dtype0
i
Reshape_177Reshape
split_4:40Reshape_177/shape*
Tshape0*
T0*
_output_shapes	
:
b
Reshape_178/shapeConst*
dtype0*
_output_shapes
:*
valueB"      
n
Reshape_178Reshape
split_4:41Reshape_178/shape*
T0*
Tshape0* 
_output_shapes
:

b
Reshape_179/shapeConst*
dtype0*
valueB"      *
_output_shapes
:
n
Reshape_179Reshape
split_4:42Reshape_179/shape* 
_output_shapes
:
*
T0*
Tshape0
\
Reshape_180/shapeConst*
dtype0*
_output_shapes
:*
valueB:
i
Reshape_180Reshape
split_4:43Reshape_180/shape*
Tshape0*
T0*
_output_shapes	
:
\
Reshape_181/shapeConst*
_output_shapes
:*
valueB:*
dtype0
i
Reshape_181Reshape
split_4:44Reshape_181/shape*
_output_shapes	
:*
T0*
Tshape0
b
Reshape_182/shapeConst*
valueB"      *
_output_shapes
:*
dtype0
m
Reshape_182Reshape
split_4:45Reshape_182/shape*
Tshape0*
_output_shapes
:	*
T0
b
Reshape_183/shapeConst*
dtype0*
_output_shapes
:*
valueB"      
m
Reshape_183Reshape
split_4:46Reshape_183/shape*
T0*
Tshape0*
_output_shapes
:	
[
Reshape_184/shapeConst*
_output_shapes
:*
dtype0*
valueB:
h
Reshape_184Reshape
split_4:47Reshape_184/shape*
_output_shapes
:*
Tshape0*
T0
[
Reshape_185/shapeConst*
_output_shapes
:*
valueB:*
dtype0
h
Reshape_185Reshape
split_4:48Reshape_185/shape*
T0*
_output_shapes
:*
Tshape0
b
Reshape_186/shapeConst*
_output_shapes
:*
dtype0*
valueB"<      
m
Reshape_186Reshape
split_4:49Reshape_186/shape*
_output_shapes
:	<*
Tshape0*
T0
b
Reshape_187/shapeConst*
dtype0*
valueB"<      *
_output_shapes
:
m
Reshape_187Reshape
split_4:50Reshape_187/shape*
Tshape0*
_output_shapes
:	<*
T0
\
Reshape_188/shapeConst*
valueB:*
_output_shapes
:*
dtype0
i
Reshape_188Reshape
split_4:51Reshape_188/shape*
Tshape0*
T0*
_output_shapes	
:
\
Reshape_189/shapeConst*
valueB:*
dtype0*
_output_shapes
:
i
Reshape_189Reshape
split_4:52Reshape_189/shape*
Tshape0*
T0*
_output_shapes	
:
b
Reshape_190/shapeConst*
_output_shapes
:*
valueB"      *
dtype0
n
Reshape_190Reshape
split_4:53Reshape_190/shape*
T0*
Tshape0* 
_output_shapes
:

b
Reshape_191/shapeConst*
dtype0*
valueB"      *
_output_shapes
:
n
Reshape_191Reshape
split_4:54Reshape_191/shape*
T0* 
_output_shapes
:
*
Tshape0
\
Reshape_192/shapeConst*
dtype0*
_output_shapes
:*
valueB:
i
Reshape_192Reshape
split_4:55Reshape_192/shape*
Tshape0*
T0*
_output_shapes	
:
\
Reshape_193/shapeConst*
_output_shapes
:*
valueB:*
dtype0
i
Reshape_193Reshape
split_4:56Reshape_193/shape*
T0*
Tshape0*
_output_shapes	
:
b
Reshape_194/shapeConst*
valueB"      *
_output_shapes
:*
dtype0
m
Reshape_194Reshape
split_4:57Reshape_194/shape*
T0*
_output_shapes
:	*
Tshape0
b
Reshape_195/shapeConst*
dtype0*
valueB"      *
_output_shapes
:
m
Reshape_195Reshape
split_4:58Reshape_195/shape*
Tshape0*
T0*
_output_shapes
:	
[
Reshape_196/shapeConst*
valueB:*
_output_shapes
:*
dtype0
h
Reshape_196Reshape
split_4:59Reshape_196/shape*
T0*
Tshape0*
_output_shapes
:
[
Reshape_197/shapeConst*
dtype0*
_output_shapes
:*
valueB:
h
Reshape_197Reshape
split_4:60Reshape_197/shape*
T0*
Tshape0*
_output_shapes
:
¨
	Assign_19Assignpi/dense/kernelReshape_137*
_output_shapes
:	<*
use_locking(*"
_class
loc:@pi/dense/kernel*
validate_shape(*
T0
 
	Assign_20Assignpi/dense/biasReshape_138*
use_locking(*
_output_shapes	
:*
validate_shape(*
T0* 
_class
loc:@pi/dense/bias
­
	Assign_21Assignpi/dense_1/kernelReshape_139* 
_output_shapes
:
*$
_class
loc:@pi/dense_1/kernel*
use_locking(*
T0*
validate_shape(
¤
	Assign_22Assignpi/dense_1/biasReshape_140*
use_locking(*
T0*
_output_shapes	
:*"
_class
loc:@pi/dense_1/bias*
validate_shape(
Ź
	Assign_23Assignpi/dense_2/kernelReshape_141*$
_class
loc:@pi/dense_2/kernel*
use_locking(*
validate_shape(*
_output_shapes
:	*
T0
Ł
	Assign_24Assignpi/dense_2/biasReshape_142*
validate_shape(*
_output_shapes
:*
T0*
use_locking(*"
_class
loc:@pi/dense_2/bias

	Assign_25Assign
pi/log_stdReshape_143*
_class
loc:@pi/log_std*
_output_shapes
:*
validate_shape(*
T0*
use_locking(
¨
	Assign_26Assignvf/dense/kernelReshape_144*
T0*"
_class
loc:@vf/dense/kernel*
validate_shape(*
_output_shapes
:	<*
use_locking(
 
	Assign_27Assignvf/dense/biasReshape_145* 
_class
loc:@vf/dense/bias*
validate_shape(*
use_locking(*
T0*
_output_shapes	
:
­
	Assign_28Assignvf/dense_1/kernelReshape_146*
T0*
use_locking(*
validate_shape(*$
_class
loc:@vf/dense_1/kernel* 
_output_shapes
:

¤
	Assign_29Assignvf/dense_1/biasReshape_147*"
_class
loc:@vf/dense_1/bias*
use_locking(*
validate_shape(*
_output_shapes	
:*
T0
Ź
	Assign_30Assignvf/dense_2/kernelReshape_148*
validate_shape(*
use_locking(*
T0*$
_class
loc:@vf/dense_2/kernel*
_output_shapes
:	
Ł
	Assign_31Assignvf/dense_2/biasReshape_149*
T0*
_output_shapes
:*"
_class
loc:@vf/dense_2/bias*
use_locking(*
validate_shape(
¨
	Assign_32Assignvc/dense/kernelReshape_150*
validate_shape(*"
_class
loc:@vc/dense/kernel*
T0*
use_locking(*
_output_shapes
:	<
 
	Assign_33Assignvc/dense/biasReshape_151*
validate_shape(*
use_locking(*
T0* 
_class
loc:@vc/dense/bias*
_output_shapes	
:
­
	Assign_34Assignvc/dense_1/kernelReshape_152*
T0*
use_locking(*
validate_shape(* 
_output_shapes
:
*$
_class
loc:@vc/dense_1/kernel
¤
	Assign_35Assignvc/dense_1/biasReshape_153*
_output_shapes	
:*
T0*
use_locking(*"
_class
loc:@vc/dense_1/bias*
validate_shape(
Ź
	Assign_36Assignvc/dense_2/kernelReshape_154*$
_class
loc:@vc/dense_2/kernel*
use_locking(*
_output_shapes
:	*
T0*
validate_shape(
Ł
	Assign_37Assignvc/dense_2/biasReshape_155*
use_locking(*
_output_shapes
:*
T0*"
_class
loc:@vc/dense_2/bias*
validate_shape(

	Assign_38Assignbeta1_powerReshape_156* 
_class
loc:@pi/dense/bias*
validate_shape(*
T0*
use_locking(*
_output_shapes
: 

	Assign_39Assignbeta2_powerReshape_157*
use_locking(* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes
: *
T0
­
	Assign_40Assignpi/dense/kernel/AdamReshape_158*
validate_shape(*
use_locking(*
_output_shapes
:	<*"
_class
loc:@pi/dense/kernel*
T0
Ż
	Assign_41Assignpi/dense/kernel/Adam_1Reshape_159*"
_class
loc:@pi/dense/kernel*
_output_shapes
:	<*
validate_shape(*
use_locking(*
T0
Ľ
	Assign_42Assignpi/dense/bias/AdamReshape_160*
use_locking(*
T0*
_output_shapes	
:*
validate_shape(* 
_class
loc:@pi/dense/bias
§
	Assign_43Assignpi/dense/bias/Adam_1Reshape_161*
T0*
_output_shapes	
:*
use_locking(*
validate_shape(* 
_class
loc:@pi/dense/bias
˛
	Assign_44Assignpi/dense_1/kernel/AdamReshape_162* 
_output_shapes
:
*
validate_shape(*
use_locking(*
T0*$
_class
loc:@pi/dense_1/kernel
´
	Assign_45Assignpi/dense_1/kernel/Adam_1Reshape_163*
validate_shape(*
T0*$
_class
loc:@pi/dense_1/kernel* 
_output_shapes
:
*
use_locking(
Š
	Assign_46Assignpi/dense_1/bias/AdamReshape_164*
_output_shapes	
:*
use_locking(*
validate_shape(*
T0*"
_class
loc:@pi/dense_1/bias
Ť
	Assign_47Assignpi/dense_1/bias/Adam_1Reshape_165*"
_class
loc:@pi/dense_1/bias*
T0*
validate_shape(*
use_locking(*
_output_shapes	
:
ą
	Assign_48Assignpi/dense_2/kernel/AdamReshape_166*
use_locking(*
validate_shape(*
_output_shapes
:	*$
_class
loc:@pi/dense_2/kernel*
T0
ł
	Assign_49Assignpi/dense_2/kernel/Adam_1Reshape_167*
T0*
validate_shape(*
_output_shapes
:	*$
_class
loc:@pi/dense_2/kernel*
use_locking(
¨
	Assign_50Assignpi/dense_2/bias/AdamReshape_168*
T0*
validate_shape(*"
_class
loc:@pi/dense_2/bias*
_output_shapes
:*
use_locking(
Ş
	Assign_51Assignpi/dense_2/bias/Adam_1Reshape_169*
_output_shapes
:*
validate_shape(*
T0*"
_class
loc:@pi/dense_2/bias*
use_locking(

	Assign_52Assignpi/log_std/AdamReshape_170*
validate_shape(*
T0*
use_locking(*
_class
loc:@pi/log_std*
_output_shapes
:
 
	Assign_53Assignpi/log_std/Adam_1Reshape_171*
use_locking(*
_class
loc:@pi/log_std*
T0*
_output_shapes
:*
validate_shape(

	Assign_54Assignbeta1_power_1Reshape_172*
use_locking(* 
_class
loc:@vc/dense/bias*
T0*
validate_shape(*
_output_shapes
: 

	Assign_55Assignbeta2_power_1Reshape_173*
T0*
_output_shapes
: *
validate_shape(*
use_locking(* 
_class
loc:@vc/dense/bias
­
	Assign_56Assignvf/dense/kernel/AdamReshape_174*
validate_shape(*
use_locking(*"
_class
loc:@vf/dense/kernel*
_output_shapes
:	<*
T0
Ż
	Assign_57Assignvf/dense/kernel/Adam_1Reshape_175*
_output_shapes
:	<*"
_class
loc:@vf/dense/kernel*
use_locking(*
validate_shape(*
T0
Ľ
	Assign_58Assignvf/dense/bias/AdamReshape_176*
_output_shapes	
:*
T0* 
_class
loc:@vf/dense/bias*
use_locking(*
validate_shape(
§
	Assign_59Assignvf/dense/bias/Adam_1Reshape_177*
validate_shape(*
use_locking(* 
_class
loc:@vf/dense/bias*
_output_shapes	
:*
T0
˛
	Assign_60Assignvf/dense_1/kernel/AdamReshape_178*
validate_shape(*$
_class
loc:@vf/dense_1/kernel* 
_output_shapes
:
*
T0*
use_locking(
´
	Assign_61Assignvf/dense_1/kernel/Adam_1Reshape_179*
validate_shape(*$
_class
loc:@vf/dense_1/kernel* 
_output_shapes
:
*
use_locking(*
T0
Š
	Assign_62Assignvf/dense_1/bias/AdamReshape_180*
use_locking(*
validate_shape(*"
_class
loc:@vf/dense_1/bias*
T0*
_output_shapes	
:
Ť
	Assign_63Assignvf/dense_1/bias/Adam_1Reshape_181*
T0*
_output_shapes	
:*
validate_shape(*
use_locking(*"
_class
loc:@vf/dense_1/bias
ą
	Assign_64Assignvf/dense_2/kernel/AdamReshape_182*
use_locking(*
T0*
validate_shape(*$
_class
loc:@vf/dense_2/kernel*
_output_shapes
:	
ł
	Assign_65Assignvf/dense_2/kernel/Adam_1Reshape_183*
validate_shape(*
use_locking(*
_output_shapes
:	*$
_class
loc:@vf/dense_2/kernel*
T0
¨
	Assign_66Assignvf/dense_2/bias/AdamReshape_184*
T0*
use_locking(*
_output_shapes
:*"
_class
loc:@vf/dense_2/bias*
validate_shape(
Ş
	Assign_67Assignvf/dense_2/bias/Adam_1Reshape_185*
T0*"
_class
loc:@vf/dense_2/bias*
validate_shape(*
_output_shapes
:*
use_locking(
­
	Assign_68Assignvc/dense/kernel/AdamReshape_186*
validate_shape(*"
_class
loc:@vc/dense/kernel*
_output_shapes
:	<*
T0*
use_locking(
Ż
	Assign_69Assignvc/dense/kernel/Adam_1Reshape_187*
validate_shape(*
_output_shapes
:	<*
use_locking(*
T0*"
_class
loc:@vc/dense/kernel
Ľ
	Assign_70Assignvc/dense/bias/AdamReshape_188*
use_locking(*
validate_shape(*
_output_shapes	
:*
T0* 
_class
loc:@vc/dense/bias
§
	Assign_71Assignvc/dense/bias/Adam_1Reshape_189*
validate_shape(* 
_class
loc:@vc/dense/bias*
_output_shapes	
:*
use_locking(*
T0
˛
	Assign_72Assignvc/dense_1/kernel/AdamReshape_190* 
_output_shapes
:
*
validate_shape(*$
_class
loc:@vc/dense_1/kernel*
T0*
use_locking(
´
	Assign_73Assignvc/dense_1/kernel/Adam_1Reshape_191*
validate_shape(*
use_locking(*$
_class
loc:@vc/dense_1/kernel* 
_output_shapes
:
*
T0
Š
	Assign_74Assignvc/dense_1/bias/AdamReshape_192*
T0*
use_locking(*
validate_shape(*"
_class
loc:@vc/dense_1/bias*
_output_shapes	
:
Ť
	Assign_75Assignvc/dense_1/bias/Adam_1Reshape_193*
_output_shapes	
:*
validate_shape(*
use_locking(*"
_class
loc:@vc/dense_1/bias*
T0
ą
	Assign_76Assignvc/dense_2/kernel/AdamReshape_194*$
_class
loc:@vc/dense_2/kernel*
_output_shapes
:	*
use_locking(*
validate_shape(*
T0
ł
	Assign_77Assignvc/dense_2/kernel/Adam_1Reshape_195*
_output_shapes
:	*
T0*
use_locking(*
validate_shape(*$
_class
loc:@vc/dense_2/kernel
¨
	Assign_78Assignvc/dense_2/bias/AdamReshape_196*
validate_shape(*
use_locking(*
_output_shapes
:*
T0*"
_class
loc:@vc/dense_2/bias
Ş
	Assign_79Assignvc/dense_2/bias/Adam_1Reshape_197*
T0*"
_class
loc:@vc/dense_2/bias*
validate_shape(*
_output_shapes
:*
use_locking(
đ
group_deps_4NoOp
^Assign_19
^Assign_20
^Assign_21
^Assign_22
^Assign_23
^Assign_24
^Assign_25
^Assign_26
^Assign_27
^Assign_28
^Assign_29
^Assign_30
^Assign_31
^Assign_32
^Assign_33
^Assign_34
^Assign_35
^Assign_36
^Assign_37
^Assign_38
^Assign_39
^Assign_40
^Assign_41
^Assign_42
^Assign_43
^Assign_44
^Assign_45
^Assign_46
^Assign_47
^Assign_48
^Assign_49
^Assign_50
^Assign_51
^Assign_52
^Assign_53
^Assign_54
^Assign_55
^Assign_56
^Assign_57
^Assign_58
^Assign_59
^Assign_60
^Assign_61
^Assign_62
^Assign_63
^Assign_64
^Assign_65
^Assign_66
^Assign_67
^Assign_68
^Assign_69
^Assign_70
^Assign_71
^Assign_72
^Assign_73
^Assign_74
^Assign_75
^Assign_76
^Assign_77
^Assign_78
^Assign_79
Y
save/filename/inputConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
n
save/filenamePlaceholderWithDefaultsave/filename/input*
dtype0*
_output_shapes
: *
shape: 
e

save/ConstPlaceholderWithDefaultsave/filename*
_output_shapes
: *
dtype0*
shape: 

save/StringJoin/inputs_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+_temp_4bbdcd98e2754018978bc1090c85a3b8/part
u
save/StringJoin
StringJoin
save/Constsave/StringJoin/inputs_1*
_output_shapes
: *
	separator *
N
Q
save/num_shardsConst*
value	B :*
_output_shapes
: *
dtype0
\
save/ShardedFilename/shardConst*
_output_shapes
: *
dtype0*
value	B : 
}
save/ShardedFilenameShardedFilenamesave/StringJoinsave/ShardedFilename/shardsave/num_shards*
_output_shapes
: 
Ę

save/SaveV2/tensor_namesConst*ý	
valueó	Bđ	=Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1B
pi/log_stdBpi/log_std/AdamBpi/log_std/Adam_1Bvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1*
dtype0*
_output_shapes
:=
ŕ
save/SaveV2/shape_and_slicesConst*
valueB=B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
_output_shapes
:=*
dtype0

save/SaveV2SaveV2save/ShardedFilenamesave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta2_powerbeta2_power_1pi/dense/biaspi/dense/bias/Adampi/dense/bias/Adam_1pi/dense/kernelpi/dense/kernel/Adampi/dense/kernel/Adam_1pi/dense_1/biaspi/dense_1/bias/Adampi/dense_1/bias/Adam_1pi/dense_1/kernelpi/dense_1/kernel/Adampi/dense_1/kernel/Adam_1pi/dense_2/biaspi/dense_2/bias/Adampi/dense_2/bias/Adam_1pi/dense_2/kernelpi/dense_2/kernel/Adampi/dense_2/kernel/Adam_1
pi/log_stdpi/log_std/Adampi/log_std/Adam_1vc/dense/biasvc/dense/bias/Adamvc/dense/bias/Adam_1vc/dense/kernelvc/dense/kernel/Adamvc/dense/kernel/Adam_1vc/dense_1/biasvc/dense_1/bias/Adamvc/dense_1/bias/Adam_1vc/dense_1/kernelvc/dense_1/kernel/Adamvc/dense_1/kernel/Adam_1vc/dense_2/biasvc/dense_2/bias/Adamvc/dense_2/bias/Adam_1vc/dense_2/kernelvc/dense_2/kernel/Adamvc/dense_2/kernel/Adam_1vf/dense/biasvf/dense/bias/Adamvf/dense/bias/Adam_1vf/dense/kernelvf/dense/kernel/Adamvf/dense/kernel/Adam_1vf/dense_1/biasvf/dense_1/bias/Adamvf/dense_1/bias/Adam_1vf/dense_1/kernelvf/dense_1/kernel/Adamvf/dense_1/kernel/Adam_1vf/dense_2/biasvf/dense_2/bias/Adamvf/dense_2/bias/Adam_1vf/dense_2/kernelvf/dense_2/kernel/Adamvf/dense_2/kernel/Adam_1*K
dtypesA
?2=

save/control_dependencyIdentitysave/ShardedFilename^save/SaveV2*
T0*
_output_shapes
: *'
_class
loc:@save/ShardedFilename

+save/MergeV2Checkpoints/checkpoint_prefixesPacksave/ShardedFilename^save/control_dependency*

axis *
N*
T0*
_output_shapes
:
}
save/MergeV2CheckpointsMergeV2Checkpoints+save/MergeV2Checkpoints/checkpoint_prefixes
save/Const*
delete_old_dirs(
z
save/IdentityIdentity
save/Const^save/MergeV2Checkpoints^save/control_dependency*
T0*
_output_shapes
: 
Í

save/RestoreV2/tensor_namesConst*ý	
valueó	Bđ	=Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1B
pi/log_stdBpi/log_std/AdamBpi/log_std/Adam_1Bvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1*
_output_shapes
:=*
dtype0
ă
save/RestoreV2/shape_and_slicesConst*
_output_shapes
:=*
valueB=B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0
ż
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices*K
dtypesA
?2=*
_output_shapes÷
ô:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

save/AssignAssignbeta1_powersave/RestoreV2*
_output_shapes
: *
T0*
validate_shape(*
use_locking(* 
_class
loc:@pi/dense/bias
¤
save/Assign_1Assignbeta1_power_1save/RestoreV2:1*
T0*
use_locking(*
_output_shapes
: * 
_class
loc:@vc/dense/bias*
validate_shape(
˘
save/Assign_2Assignbeta2_powersave/RestoreV2:2*
_output_shapes
: *
validate_shape(*
use_locking(*
T0* 
_class
loc:@pi/dense/bias
¤
save/Assign_3Assignbeta2_power_1save/RestoreV2:3* 
_class
loc:@vc/dense/bias*
_output_shapes
: *
T0*
use_locking(*
validate_shape(
Š
save/Assign_4Assignpi/dense/biassave/RestoreV2:4*
use_locking(*
validate_shape(* 
_class
loc:@pi/dense/bias*
T0*
_output_shapes	
:
Ž
save/Assign_5Assignpi/dense/bias/Adamsave/RestoreV2:5* 
_class
loc:@pi/dense/bias*
_output_shapes	
:*
validate_shape(*
use_locking(*
T0
°
save/Assign_6Assignpi/dense/bias/Adam_1save/RestoreV2:6*
_output_shapes	
:*
T0*
validate_shape(* 
_class
loc:@pi/dense/bias*
use_locking(
ą
save/Assign_7Assignpi/dense/kernelsave/RestoreV2:7*
_output_shapes
:	<*
use_locking(*
validate_shape(*
T0*"
_class
loc:@pi/dense/kernel
ś
save/Assign_8Assignpi/dense/kernel/Adamsave/RestoreV2:8*"
_class
loc:@pi/dense/kernel*
_output_shapes
:	<*
T0*
use_locking(*
validate_shape(
¸
save/Assign_9Assignpi/dense/kernel/Adam_1save/RestoreV2:9*"
_class
loc:@pi/dense/kernel*
T0*
validate_shape(*
_output_shapes
:	<*
use_locking(
Ż
save/Assign_10Assignpi/dense_1/biassave/RestoreV2:10*"
_class
loc:@pi/dense_1/bias*
use_locking(*
_output_shapes	
:*
validate_shape(*
T0
´
save/Assign_11Assignpi/dense_1/bias/Adamsave/RestoreV2:11*
validate_shape(*
_output_shapes	
:*
T0*
use_locking(*"
_class
loc:@pi/dense_1/bias
ś
save/Assign_12Assignpi/dense_1/bias/Adam_1save/RestoreV2:12*"
_class
loc:@pi/dense_1/bias*
use_locking(*
T0*
_output_shapes	
:*
validate_shape(
¸
save/Assign_13Assignpi/dense_1/kernelsave/RestoreV2:13*
use_locking(*
T0*
validate_shape(*$
_class
loc:@pi/dense_1/kernel* 
_output_shapes
:

˝
save/Assign_14Assignpi/dense_1/kernel/Adamsave/RestoreV2:14*
T0*
use_locking(*$
_class
loc:@pi/dense_1/kernel*
validate_shape(* 
_output_shapes
:

ż
save/Assign_15Assignpi/dense_1/kernel/Adam_1save/RestoreV2:15*
T0*
use_locking(*
validate_shape(*$
_class
loc:@pi/dense_1/kernel* 
_output_shapes
:

Ž
save/Assign_16Assignpi/dense_2/biassave/RestoreV2:16*
validate_shape(*
T0*
use_locking(*
_output_shapes
:*"
_class
loc:@pi/dense_2/bias
ł
save/Assign_17Assignpi/dense_2/bias/Adamsave/RestoreV2:17*
validate_shape(*
use_locking(*
T0*
_output_shapes
:*"
_class
loc:@pi/dense_2/bias
ľ
save/Assign_18Assignpi/dense_2/bias/Adam_1save/RestoreV2:18*
use_locking(*
_output_shapes
:*
T0*"
_class
loc:@pi/dense_2/bias*
validate_shape(
ˇ
save/Assign_19Assignpi/dense_2/kernelsave/RestoreV2:19*$
_class
loc:@pi/dense_2/kernel*
T0*
validate_shape(*
use_locking(*
_output_shapes
:	
ź
save/Assign_20Assignpi/dense_2/kernel/Adamsave/RestoreV2:20*
T0*
_output_shapes
:	*
validate_shape(*$
_class
loc:@pi/dense_2/kernel*
use_locking(
ž
save/Assign_21Assignpi/dense_2/kernel/Adam_1save/RestoreV2:21*$
_class
loc:@pi/dense_2/kernel*
validate_shape(*
use_locking(*
_output_shapes
:	*
T0
¤
save/Assign_22Assign
pi/log_stdsave/RestoreV2:22*
_output_shapes
:*
_class
loc:@pi/log_std*
T0*
validate_shape(*
use_locking(
Š
save/Assign_23Assignpi/log_std/Adamsave/RestoreV2:23*
T0*
_class
loc:@pi/log_std*
_output_shapes
:*
validate_shape(*
use_locking(
Ť
save/Assign_24Assignpi/log_std/Adam_1save/RestoreV2:24*
validate_shape(*
_output_shapes
:*
_class
loc:@pi/log_std*
T0*
use_locking(
Ť
save/Assign_25Assignvc/dense/biassave/RestoreV2:25*
_output_shapes	
:*
use_locking(*
T0*
validate_shape(* 
_class
loc:@vc/dense/bias
°
save/Assign_26Assignvc/dense/bias/Adamsave/RestoreV2:26* 
_class
loc:@vc/dense/bias*
_output_shapes	
:*
T0*
validate_shape(*
use_locking(
˛
save/Assign_27Assignvc/dense/bias/Adam_1save/RestoreV2:27*
validate_shape(*
T0*
use_locking(*
_output_shapes	
:* 
_class
loc:@vc/dense/bias
ł
save/Assign_28Assignvc/dense/kernelsave/RestoreV2:28*
validate_shape(*
use_locking(*"
_class
loc:@vc/dense/kernel*
T0*
_output_shapes
:	<
¸
save/Assign_29Assignvc/dense/kernel/Adamsave/RestoreV2:29*
_output_shapes
:	<*
validate_shape(*"
_class
loc:@vc/dense/kernel*
T0*
use_locking(
ş
save/Assign_30Assignvc/dense/kernel/Adam_1save/RestoreV2:30*"
_class
loc:@vc/dense/kernel*
validate_shape(*
T0*
_output_shapes
:	<*
use_locking(
Ż
save/Assign_31Assignvc/dense_1/biassave/RestoreV2:31*
_output_shapes	
:*"
_class
loc:@vc/dense_1/bias*
validate_shape(*
T0*
use_locking(
´
save/Assign_32Assignvc/dense_1/bias/Adamsave/RestoreV2:32*
_output_shapes	
:*"
_class
loc:@vc/dense_1/bias*
T0*
use_locking(*
validate_shape(
ś
save/Assign_33Assignvc/dense_1/bias/Adam_1save/RestoreV2:33*
T0*
validate_shape(*"
_class
loc:@vc/dense_1/bias*
use_locking(*
_output_shapes	
:
¸
save/Assign_34Assignvc/dense_1/kernelsave/RestoreV2:34*$
_class
loc:@vc/dense_1/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0
˝
save/Assign_35Assignvc/dense_1/kernel/Adamsave/RestoreV2:35*
validate_shape(*
use_locking(*$
_class
loc:@vc/dense_1/kernel*
T0* 
_output_shapes
:

ż
save/Assign_36Assignvc/dense_1/kernel/Adam_1save/RestoreV2:36*
T0*
use_locking(*
validate_shape(*$
_class
loc:@vc/dense_1/kernel* 
_output_shapes
:

Ž
save/Assign_37Assignvc/dense_2/biassave/RestoreV2:37*
T0*
use_locking(*
_output_shapes
:*"
_class
loc:@vc/dense_2/bias*
validate_shape(
ł
save/Assign_38Assignvc/dense_2/bias/Adamsave/RestoreV2:38*"
_class
loc:@vc/dense_2/bias*
use_locking(*
_output_shapes
:*
T0*
validate_shape(
ľ
save/Assign_39Assignvc/dense_2/bias/Adam_1save/RestoreV2:39*
use_locking(*"
_class
loc:@vc/dense_2/bias*
T0*
_output_shapes
:*
validate_shape(
ˇ
save/Assign_40Assignvc/dense_2/kernelsave/RestoreV2:40*
_output_shapes
:	*
use_locking(*
T0*
validate_shape(*$
_class
loc:@vc/dense_2/kernel
ź
save/Assign_41Assignvc/dense_2/kernel/Adamsave/RestoreV2:41*
validate_shape(*
T0*
_output_shapes
:	*$
_class
loc:@vc/dense_2/kernel*
use_locking(
ž
save/Assign_42Assignvc/dense_2/kernel/Adam_1save/RestoreV2:42*
_output_shapes
:	*$
_class
loc:@vc/dense_2/kernel*
validate_shape(*
T0*
use_locking(
Ť
save/Assign_43Assignvf/dense/biassave/RestoreV2:43* 
_class
loc:@vf/dense/bias*
use_locking(*
validate_shape(*
T0*
_output_shapes	
:
°
save/Assign_44Assignvf/dense/bias/Adamsave/RestoreV2:44*
T0*
validate_shape(* 
_class
loc:@vf/dense/bias*
_output_shapes	
:*
use_locking(
˛
save/Assign_45Assignvf/dense/bias/Adam_1save/RestoreV2:45*
validate_shape(*
T0*
use_locking(* 
_class
loc:@vf/dense/bias*
_output_shapes	
:
ł
save/Assign_46Assignvf/dense/kernelsave/RestoreV2:46*
T0*
use_locking(*
_output_shapes
:	<*
validate_shape(*"
_class
loc:@vf/dense/kernel
¸
save/Assign_47Assignvf/dense/kernel/Adamsave/RestoreV2:47*
T0*
validate_shape(*"
_class
loc:@vf/dense/kernel*
_output_shapes
:	<*
use_locking(
ş
save/Assign_48Assignvf/dense/kernel/Adam_1save/RestoreV2:48*
validate_shape(*
_output_shapes
:	<*
use_locking(*
T0*"
_class
loc:@vf/dense/kernel
Ż
save/Assign_49Assignvf/dense_1/biassave/RestoreV2:49*
T0*
use_locking(*
_output_shapes	
:*"
_class
loc:@vf/dense_1/bias*
validate_shape(
´
save/Assign_50Assignvf/dense_1/bias/Adamsave/RestoreV2:50*
validate_shape(*"
_class
loc:@vf/dense_1/bias*
_output_shapes	
:*
use_locking(*
T0
ś
save/Assign_51Assignvf/dense_1/bias/Adam_1save/RestoreV2:51*
validate_shape(*"
_class
loc:@vf/dense_1/bias*
use_locking(*
T0*
_output_shapes	
:
¸
save/Assign_52Assignvf/dense_1/kernelsave/RestoreV2:52*
validate_shape(* 
_output_shapes
:
*$
_class
loc:@vf/dense_1/kernel*
use_locking(*
T0
˝
save/Assign_53Assignvf/dense_1/kernel/Adamsave/RestoreV2:53*
T0*
use_locking(* 
_output_shapes
:
*
validate_shape(*$
_class
loc:@vf/dense_1/kernel
ż
save/Assign_54Assignvf/dense_1/kernel/Adam_1save/RestoreV2:54*
T0*
validate_shape(* 
_output_shapes
:
*
use_locking(*$
_class
loc:@vf/dense_1/kernel
Ž
save/Assign_55Assignvf/dense_2/biassave/RestoreV2:55*
validate_shape(*
T0*
use_locking(*"
_class
loc:@vf/dense_2/bias*
_output_shapes
:
ł
save/Assign_56Assignvf/dense_2/bias/Adamsave/RestoreV2:56*
T0*"
_class
loc:@vf/dense_2/bias*
_output_shapes
:*
validate_shape(*
use_locking(
ľ
save/Assign_57Assignvf/dense_2/bias/Adam_1save/RestoreV2:57*
validate_shape(*
_output_shapes
:*
T0*
use_locking(*"
_class
loc:@vf/dense_2/bias
ˇ
save/Assign_58Assignvf/dense_2/kernelsave/RestoreV2:58*
_output_shapes
:	*
validate_shape(*
use_locking(*$
_class
loc:@vf/dense_2/kernel*
T0
ź
save/Assign_59Assignvf/dense_2/kernel/Adamsave/RestoreV2:59*
validate_shape(*
T0*
use_locking(*$
_class
loc:@vf/dense_2/kernel*
_output_shapes
:	
ž
save/Assign_60Assignvf/dense_2/kernel/Adam_1save/RestoreV2:60*$
_class
loc:@vf/dense_2/kernel*
validate_shape(*
_output_shapes
:	*
use_locking(*
T0

save/restore_shardNoOp^save/Assign^save/Assign_1^save/Assign_10^save/Assign_11^save/Assign_12^save/Assign_13^save/Assign_14^save/Assign_15^save/Assign_16^save/Assign_17^save/Assign_18^save/Assign_19^save/Assign_2^save/Assign_20^save/Assign_21^save/Assign_22^save/Assign_23^save/Assign_24^save/Assign_25^save/Assign_26^save/Assign_27^save/Assign_28^save/Assign_29^save/Assign_3^save/Assign_30^save/Assign_31^save/Assign_32^save/Assign_33^save/Assign_34^save/Assign_35^save/Assign_36^save/Assign_37^save/Assign_38^save/Assign_39^save/Assign_4^save/Assign_40^save/Assign_41^save/Assign_42^save/Assign_43^save/Assign_44^save/Assign_45^save/Assign_46^save/Assign_47^save/Assign_48^save/Assign_49^save/Assign_5^save/Assign_50^save/Assign_51^save/Assign_52^save/Assign_53^save/Assign_54^save/Assign_55^save/Assign_56^save/Assign_57^save/Assign_58^save/Assign_59^save/Assign_6^save/Assign_60^save/Assign_7^save/Assign_8^save/Assign_9
-
save/restore_allNoOp^save/restore_shard
[
save_1/filename/inputConst*
dtype0*
valueB Bmodel*
_output_shapes
: 
r
save_1/filenamePlaceholderWithDefaultsave_1/filename/input*
shape: *
_output_shapes
: *
dtype0
i
save_1/ConstPlaceholderWithDefaultsave_1/filename*
dtype0*
shape: *
_output_shapes
: 

save_1/StringJoin/inputs_1Const*
dtype0*
_output_shapes
: *<
value3B1 B+_temp_4c749bfeb46c4b0da72b853466c2c777/part
{
save_1/StringJoin
StringJoinsave_1/Constsave_1/StringJoin/inputs_1*
N*
_output_shapes
: *
	separator 
S
save_1/num_shardsConst*
value	B :*
_output_shapes
: *
dtype0
^
save_1/ShardedFilename/shardConst*
_output_shapes
: *
value	B : *
dtype0

save_1/ShardedFilenameShardedFilenamesave_1/StringJoinsave_1/ShardedFilename/shardsave_1/num_shards*
_output_shapes
: 
Ě

save_1/SaveV2/tensor_namesConst*
dtype0*ý	
valueó	Bđ	=Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1B
pi/log_stdBpi/log_std/AdamBpi/log_std/Adam_1Bvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1*
_output_shapes
:=
â
save_1/SaveV2/shape_and_slicesConst*
_output_shapes
:=*
valueB=B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0
 
save_1/SaveV2SaveV2save_1/ShardedFilenamesave_1/SaveV2/tensor_namessave_1/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta2_powerbeta2_power_1pi/dense/biaspi/dense/bias/Adampi/dense/bias/Adam_1pi/dense/kernelpi/dense/kernel/Adampi/dense/kernel/Adam_1pi/dense_1/biaspi/dense_1/bias/Adampi/dense_1/bias/Adam_1pi/dense_1/kernelpi/dense_1/kernel/Adampi/dense_1/kernel/Adam_1pi/dense_2/biaspi/dense_2/bias/Adampi/dense_2/bias/Adam_1pi/dense_2/kernelpi/dense_2/kernel/Adampi/dense_2/kernel/Adam_1
pi/log_stdpi/log_std/Adampi/log_std/Adam_1vc/dense/biasvc/dense/bias/Adamvc/dense/bias/Adam_1vc/dense/kernelvc/dense/kernel/Adamvc/dense/kernel/Adam_1vc/dense_1/biasvc/dense_1/bias/Adamvc/dense_1/bias/Adam_1vc/dense_1/kernelvc/dense_1/kernel/Adamvc/dense_1/kernel/Adam_1vc/dense_2/biasvc/dense_2/bias/Adamvc/dense_2/bias/Adam_1vc/dense_2/kernelvc/dense_2/kernel/Adamvc/dense_2/kernel/Adam_1vf/dense/biasvf/dense/bias/Adamvf/dense/bias/Adam_1vf/dense/kernelvf/dense/kernel/Adamvf/dense/kernel/Adam_1vf/dense_1/biasvf/dense_1/bias/Adamvf/dense_1/bias/Adam_1vf/dense_1/kernelvf/dense_1/kernel/Adamvf/dense_1/kernel/Adam_1vf/dense_2/biasvf/dense_2/bias/Adamvf/dense_2/bias/Adam_1vf/dense_2/kernelvf/dense_2/kernel/Adamvf/dense_2/kernel/Adam_1*K
dtypesA
?2=

save_1/control_dependencyIdentitysave_1/ShardedFilename^save_1/SaveV2*)
_class
loc:@save_1/ShardedFilename*
T0*
_output_shapes
: 
Ł
-save_1/MergeV2Checkpoints/checkpoint_prefixesPacksave_1/ShardedFilename^save_1/control_dependency*
_output_shapes
:*

axis *
N*
T0

save_1/MergeV2CheckpointsMergeV2Checkpoints-save_1/MergeV2Checkpoints/checkpoint_prefixessave_1/Const*
delete_old_dirs(

save_1/IdentityIdentitysave_1/Const^save_1/MergeV2Checkpoints^save_1/control_dependency*
_output_shapes
: *
T0
Ď

save_1/RestoreV2/tensor_namesConst*ý	
valueó	Bđ	=Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1B
pi/log_stdBpi/log_std/AdamBpi/log_std/Adam_1Bvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1*
dtype0*
_output_shapes
:=
ĺ
!save_1/RestoreV2/shape_and_slicesConst*
valueB=B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
_output_shapes
:=*
dtype0
Ç
save_1/RestoreV2	RestoreV2save_1/Constsave_1/RestoreV2/tensor_names!save_1/RestoreV2/shape_and_slices*
_output_shapes÷
ô:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*K
dtypesA
?2=
˘
save_1/AssignAssignbeta1_powersave_1/RestoreV2*
T0*
use_locking(*
_output_shapes
: *
validate_shape(* 
_class
loc:@pi/dense/bias
¨
save_1/Assign_1Assignbeta1_power_1save_1/RestoreV2:1*
_output_shapes
: * 
_class
loc:@vc/dense/bias*
T0*
use_locking(*
validate_shape(
Ś
save_1/Assign_2Assignbeta2_powersave_1/RestoreV2:2*
T0*
_output_shapes
: * 
_class
loc:@pi/dense/bias*
validate_shape(*
use_locking(
¨
save_1/Assign_3Assignbeta2_power_1save_1/RestoreV2:3* 
_class
loc:@vc/dense/bias*
T0*
validate_shape(*
use_locking(*
_output_shapes
: 
­
save_1/Assign_4Assignpi/dense/biassave_1/RestoreV2:4* 
_class
loc:@pi/dense/bias*
_output_shapes	
:*
T0*
use_locking(*
validate_shape(
˛
save_1/Assign_5Assignpi/dense/bias/Adamsave_1/RestoreV2:5* 
_class
loc:@pi/dense/bias*
use_locking(*
_output_shapes	
:*
T0*
validate_shape(
´
save_1/Assign_6Assignpi/dense/bias/Adam_1save_1/RestoreV2:6*
T0*
use_locking(*
validate_shape(* 
_class
loc:@pi/dense/bias*
_output_shapes	
:
ľ
save_1/Assign_7Assignpi/dense/kernelsave_1/RestoreV2:7*
T0*
validate_shape(*
use_locking(*"
_class
loc:@pi/dense/kernel*
_output_shapes
:	<
ş
save_1/Assign_8Assignpi/dense/kernel/Adamsave_1/RestoreV2:8*
validate_shape(*"
_class
loc:@pi/dense/kernel*
use_locking(*
_output_shapes
:	<*
T0
ź
save_1/Assign_9Assignpi/dense/kernel/Adam_1save_1/RestoreV2:9*
_output_shapes
:	<*
validate_shape(*
use_locking(*"
_class
loc:@pi/dense/kernel*
T0
ł
save_1/Assign_10Assignpi/dense_1/biassave_1/RestoreV2:10*
use_locking(*
validate_shape(*
T0*
_output_shapes	
:*"
_class
loc:@pi/dense_1/bias
¸
save_1/Assign_11Assignpi/dense_1/bias/Adamsave_1/RestoreV2:11*
validate_shape(*
_output_shapes	
:*
T0*
use_locking(*"
_class
loc:@pi/dense_1/bias
ş
save_1/Assign_12Assignpi/dense_1/bias/Adam_1save_1/RestoreV2:12*
T0*"
_class
loc:@pi/dense_1/bias*
_output_shapes	
:*
use_locking(*
validate_shape(
ź
save_1/Assign_13Assignpi/dense_1/kernelsave_1/RestoreV2:13*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0*$
_class
loc:@pi/dense_1/kernel
Á
save_1/Assign_14Assignpi/dense_1/kernel/Adamsave_1/RestoreV2:14*
use_locking(*
validate_shape(* 
_output_shapes
:
*$
_class
loc:@pi/dense_1/kernel*
T0
Ă
save_1/Assign_15Assignpi/dense_1/kernel/Adam_1save_1/RestoreV2:15* 
_output_shapes
:
*
validate_shape(*
use_locking(*$
_class
loc:@pi/dense_1/kernel*
T0
˛
save_1/Assign_16Assignpi/dense_2/biassave_1/RestoreV2:16*
validate_shape(*
T0*
use_locking(*"
_class
loc:@pi/dense_2/bias*
_output_shapes
:
ˇ
save_1/Assign_17Assignpi/dense_2/bias/Adamsave_1/RestoreV2:17*
validate_shape(*
use_locking(*
T0*
_output_shapes
:*"
_class
loc:@pi/dense_2/bias
š
save_1/Assign_18Assignpi/dense_2/bias/Adam_1save_1/RestoreV2:18*
use_locking(*
T0*
validate_shape(*
_output_shapes
:*"
_class
loc:@pi/dense_2/bias
ť
save_1/Assign_19Assignpi/dense_2/kernelsave_1/RestoreV2:19*
use_locking(*
_output_shapes
:	*
validate_shape(*
T0*$
_class
loc:@pi/dense_2/kernel
Ŕ
save_1/Assign_20Assignpi/dense_2/kernel/Adamsave_1/RestoreV2:20*
validate_shape(*
_output_shapes
:	*
T0*$
_class
loc:@pi/dense_2/kernel*
use_locking(
Â
save_1/Assign_21Assignpi/dense_2/kernel/Adam_1save_1/RestoreV2:21*
validate_shape(*
T0*
use_locking(*
_output_shapes
:	*$
_class
loc:@pi/dense_2/kernel
¨
save_1/Assign_22Assign
pi/log_stdsave_1/RestoreV2:22*
_output_shapes
:*
_class
loc:@pi/log_std*
use_locking(*
T0*
validate_shape(
­
save_1/Assign_23Assignpi/log_std/Adamsave_1/RestoreV2:23*
use_locking(*
_class
loc:@pi/log_std*
T0*
validate_shape(*
_output_shapes
:
Ż
save_1/Assign_24Assignpi/log_std/Adam_1save_1/RestoreV2:24*
_class
loc:@pi/log_std*
T0*
use_locking(*
_output_shapes
:*
validate_shape(
Ż
save_1/Assign_25Assignvc/dense/biassave_1/RestoreV2:25*
use_locking(*
T0*
validate_shape(*
_output_shapes	
:* 
_class
loc:@vc/dense/bias
´
save_1/Assign_26Assignvc/dense/bias/Adamsave_1/RestoreV2:26*
_output_shapes	
:*
validate_shape(*
T0* 
_class
loc:@vc/dense/bias*
use_locking(
ś
save_1/Assign_27Assignvc/dense/bias/Adam_1save_1/RestoreV2:27* 
_class
loc:@vc/dense/bias*
T0*
use_locking(*
validate_shape(*
_output_shapes	
:
ˇ
save_1/Assign_28Assignvc/dense/kernelsave_1/RestoreV2:28*
T0*"
_class
loc:@vc/dense/kernel*
_output_shapes
:	<*
validate_shape(*
use_locking(
ź
save_1/Assign_29Assignvc/dense/kernel/Adamsave_1/RestoreV2:29*
use_locking(*
T0*
validate_shape(*
_output_shapes
:	<*"
_class
loc:@vc/dense/kernel
ž
save_1/Assign_30Assignvc/dense/kernel/Adam_1save_1/RestoreV2:30*
validate_shape(*"
_class
loc:@vc/dense/kernel*
T0*
_output_shapes
:	<*
use_locking(
ł
save_1/Assign_31Assignvc/dense_1/biassave_1/RestoreV2:31*
use_locking(*
validate_shape(*"
_class
loc:@vc/dense_1/bias*
T0*
_output_shapes	
:
¸
save_1/Assign_32Assignvc/dense_1/bias/Adamsave_1/RestoreV2:32*
use_locking(*
_output_shapes	
:*"
_class
loc:@vc/dense_1/bias*
validate_shape(*
T0
ş
save_1/Assign_33Assignvc/dense_1/bias/Adam_1save_1/RestoreV2:33*
_output_shapes	
:*
validate_shape(*"
_class
loc:@vc/dense_1/bias*
use_locking(*
T0
ź
save_1/Assign_34Assignvc/dense_1/kernelsave_1/RestoreV2:34*$
_class
loc:@vc/dense_1/kernel*
T0* 
_output_shapes
:
*
validate_shape(*
use_locking(
Á
save_1/Assign_35Assignvc/dense_1/kernel/Adamsave_1/RestoreV2:35*
validate_shape(*
use_locking(*$
_class
loc:@vc/dense_1/kernel* 
_output_shapes
:
*
T0
Ă
save_1/Assign_36Assignvc/dense_1/kernel/Adam_1save_1/RestoreV2:36* 
_output_shapes
:
*
validate_shape(*
use_locking(*$
_class
loc:@vc/dense_1/kernel*
T0
˛
save_1/Assign_37Assignvc/dense_2/biassave_1/RestoreV2:37*
validate_shape(*
use_locking(*
T0*
_output_shapes
:*"
_class
loc:@vc/dense_2/bias
ˇ
save_1/Assign_38Assignvc/dense_2/bias/Adamsave_1/RestoreV2:38*"
_class
loc:@vc/dense_2/bias*
_output_shapes
:*
T0*
use_locking(*
validate_shape(
š
save_1/Assign_39Assignvc/dense_2/bias/Adam_1save_1/RestoreV2:39*"
_class
loc:@vc/dense_2/bias*
validate_shape(*
use_locking(*
T0*
_output_shapes
:
ť
save_1/Assign_40Assignvc/dense_2/kernelsave_1/RestoreV2:40*$
_class
loc:@vc/dense_2/kernel*
validate_shape(*
use_locking(*
_output_shapes
:	*
T0
Ŕ
save_1/Assign_41Assignvc/dense_2/kernel/Adamsave_1/RestoreV2:41*
use_locking(*
T0*
_output_shapes
:	*$
_class
loc:@vc/dense_2/kernel*
validate_shape(
Â
save_1/Assign_42Assignvc/dense_2/kernel/Adam_1save_1/RestoreV2:42*
use_locking(*$
_class
loc:@vc/dense_2/kernel*
T0*
_output_shapes
:	*
validate_shape(
Ż
save_1/Assign_43Assignvf/dense/biassave_1/RestoreV2:43*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0* 
_class
loc:@vf/dense/bias
´
save_1/Assign_44Assignvf/dense/bias/Adamsave_1/RestoreV2:44*
T0*
validate_shape(*
use_locking(*
_output_shapes	
:* 
_class
loc:@vf/dense/bias
ś
save_1/Assign_45Assignvf/dense/bias/Adam_1save_1/RestoreV2:45*
use_locking(*
T0*
validate_shape(* 
_class
loc:@vf/dense/bias*
_output_shapes	
:
ˇ
save_1/Assign_46Assignvf/dense/kernelsave_1/RestoreV2:46*
validate_shape(*"
_class
loc:@vf/dense/kernel*
T0*
_output_shapes
:	<*
use_locking(
ź
save_1/Assign_47Assignvf/dense/kernel/Adamsave_1/RestoreV2:47*"
_class
loc:@vf/dense/kernel*
_output_shapes
:	<*
validate_shape(*
T0*
use_locking(
ž
save_1/Assign_48Assignvf/dense/kernel/Adam_1save_1/RestoreV2:48*
use_locking(*
T0*
_output_shapes
:	<*"
_class
loc:@vf/dense/kernel*
validate_shape(
ł
save_1/Assign_49Assignvf/dense_1/biassave_1/RestoreV2:49*
use_locking(*
_output_shapes	
:*
T0*
validate_shape(*"
_class
loc:@vf/dense_1/bias
¸
save_1/Assign_50Assignvf/dense_1/bias/Adamsave_1/RestoreV2:50*
use_locking(*
_output_shapes	
:*
T0*
validate_shape(*"
_class
loc:@vf/dense_1/bias
ş
save_1/Assign_51Assignvf/dense_1/bias/Adam_1save_1/RestoreV2:51*
validate_shape(*
use_locking(*"
_class
loc:@vf/dense_1/bias*
_output_shapes	
:*
T0
ź
save_1/Assign_52Assignvf/dense_1/kernelsave_1/RestoreV2:52*
T0* 
_output_shapes
:
*
validate_shape(*$
_class
loc:@vf/dense_1/kernel*
use_locking(
Á
save_1/Assign_53Assignvf/dense_1/kernel/Adamsave_1/RestoreV2:53* 
_output_shapes
:
*
use_locking(*
validate_shape(*$
_class
loc:@vf/dense_1/kernel*
T0
Ă
save_1/Assign_54Assignvf/dense_1/kernel/Adam_1save_1/RestoreV2:54*$
_class
loc:@vf/dense_1/kernel*
validate_shape(* 
_output_shapes
:
*
T0*
use_locking(
˛
save_1/Assign_55Assignvf/dense_2/biassave_1/RestoreV2:55*
validate_shape(*
use_locking(*"
_class
loc:@vf/dense_2/bias*
T0*
_output_shapes
:
ˇ
save_1/Assign_56Assignvf/dense_2/bias/Adamsave_1/RestoreV2:56*
validate_shape(*
_output_shapes
:*"
_class
loc:@vf/dense_2/bias*
use_locking(*
T0
š
save_1/Assign_57Assignvf/dense_2/bias/Adam_1save_1/RestoreV2:57*
use_locking(*
T0*
validate_shape(*"
_class
loc:@vf/dense_2/bias*
_output_shapes
:
ť
save_1/Assign_58Assignvf/dense_2/kernelsave_1/RestoreV2:58*$
_class
loc:@vf/dense_2/kernel*
_output_shapes
:	*
use_locking(*
validate_shape(*
T0
Ŕ
save_1/Assign_59Assignvf/dense_2/kernel/Adamsave_1/RestoreV2:59*
use_locking(*
_output_shapes
:	*
validate_shape(*$
_class
loc:@vf/dense_2/kernel*
T0
Â
save_1/Assign_60Assignvf/dense_2/kernel/Adam_1save_1/RestoreV2:60*$
_class
loc:@vf/dense_2/kernel*
validate_shape(*
T0*
_output_shapes
:	*
use_locking(
	
save_1/restore_shardNoOp^save_1/Assign^save_1/Assign_1^save_1/Assign_10^save_1/Assign_11^save_1/Assign_12^save_1/Assign_13^save_1/Assign_14^save_1/Assign_15^save_1/Assign_16^save_1/Assign_17^save_1/Assign_18^save_1/Assign_19^save_1/Assign_2^save_1/Assign_20^save_1/Assign_21^save_1/Assign_22^save_1/Assign_23^save_1/Assign_24^save_1/Assign_25^save_1/Assign_26^save_1/Assign_27^save_1/Assign_28^save_1/Assign_29^save_1/Assign_3^save_1/Assign_30^save_1/Assign_31^save_1/Assign_32^save_1/Assign_33^save_1/Assign_34^save_1/Assign_35^save_1/Assign_36^save_1/Assign_37^save_1/Assign_38^save_1/Assign_39^save_1/Assign_4^save_1/Assign_40^save_1/Assign_41^save_1/Assign_42^save_1/Assign_43^save_1/Assign_44^save_1/Assign_45^save_1/Assign_46^save_1/Assign_47^save_1/Assign_48^save_1/Assign_49^save_1/Assign_5^save_1/Assign_50^save_1/Assign_51^save_1/Assign_52^save_1/Assign_53^save_1/Assign_54^save_1/Assign_55^save_1/Assign_56^save_1/Assign_57^save_1/Assign_58^save_1/Assign_59^save_1/Assign_6^save_1/Assign_60^save_1/Assign_7^save_1/Assign_8^save_1/Assign_9
1
save_1/restore_allNoOp^save_1/restore_shard "B
save_1/Const:0save_1/Identity:0save_1/restore_all (5 @F8"
train_op

Adam
Adam_1"ˇ:
	variablesŠ:Ś:
s
pi/dense/kernel:0pi/dense/kernel/Assignpi/dense/kernel/read:02,pi/dense/kernel/Initializer/random_uniform:08
b
pi/dense/bias:0pi/dense/bias/Assignpi/dense/bias/read:02!pi/dense/bias/Initializer/zeros:08
{
pi/dense_1/kernel:0pi/dense_1/kernel/Assignpi/dense_1/kernel/read:02.pi/dense_1/kernel/Initializer/random_uniform:08
j
pi/dense_1/bias:0pi/dense_1/bias/Assignpi/dense_1/bias/read:02#pi/dense_1/bias/Initializer/zeros:08
{
pi/dense_2/kernel:0pi/dense_2/kernel/Assignpi/dense_2/kernel/read:02.pi/dense_2/kernel/Initializer/random_uniform:08
j
pi/dense_2/bias:0pi/dense_2/bias/Assignpi/dense_2/bias/read:02#pi/dense_2/bias/Initializer/zeros:08
R
pi/log_std:0pi/log_std/Assignpi/log_std/read:02pi/log_std/initial_value:08
s
vf/dense/kernel:0vf/dense/kernel/Assignvf/dense/kernel/read:02,vf/dense/kernel/Initializer/random_uniform:08
b
vf/dense/bias:0vf/dense/bias/Assignvf/dense/bias/read:02!vf/dense/bias/Initializer/zeros:08
{
vf/dense_1/kernel:0vf/dense_1/kernel/Assignvf/dense_1/kernel/read:02.vf/dense_1/kernel/Initializer/random_uniform:08
j
vf/dense_1/bias:0vf/dense_1/bias/Assignvf/dense_1/bias/read:02#vf/dense_1/bias/Initializer/zeros:08
{
vf/dense_2/kernel:0vf/dense_2/kernel/Assignvf/dense_2/kernel/read:02.vf/dense_2/kernel/Initializer/random_uniform:08
j
vf/dense_2/bias:0vf/dense_2/bias/Assignvf/dense_2/bias/read:02#vf/dense_2/bias/Initializer/zeros:08
s
vc/dense/kernel:0vc/dense/kernel/Assignvc/dense/kernel/read:02,vc/dense/kernel/Initializer/random_uniform:08
b
vc/dense/bias:0vc/dense/bias/Assignvc/dense/bias/read:02!vc/dense/bias/Initializer/zeros:08
{
vc/dense_1/kernel:0vc/dense_1/kernel/Assignvc/dense_1/kernel/read:02.vc/dense_1/kernel/Initializer/random_uniform:08
j
vc/dense_1/bias:0vc/dense_1/bias/Assignvc/dense_1/bias/read:02#vc/dense_1/bias/Initializer/zeros:08
{
vc/dense_2/kernel:0vc/dense_2/kernel/Assignvc/dense_2/kernel/read:02.vc/dense_2/kernel/Initializer/random_uniform:08
j
vc/dense_2/bias:0vc/dense_2/bias/Assignvc/dense_2/bias/read:02#vc/dense_2/bias/Initializer/zeros:08
T
beta1_power:0beta1_power/Assignbeta1_power/read:02beta1_power/initial_value:0
T
beta2_power:0beta2_power/Assignbeta2_power/read:02beta2_power/initial_value:0
|
pi/dense/kernel/Adam:0pi/dense/kernel/Adam/Assignpi/dense/kernel/Adam/read:02(pi/dense/kernel/Adam/Initializer/zeros:0

pi/dense/kernel/Adam_1:0pi/dense/kernel/Adam_1/Assignpi/dense/kernel/Adam_1/read:02*pi/dense/kernel/Adam_1/Initializer/zeros:0
t
pi/dense/bias/Adam:0pi/dense/bias/Adam/Assignpi/dense/bias/Adam/read:02&pi/dense/bias/Adam/Initializer/zeros:0
|
pi/dense/bias/Adam_1:0pi/dense/bias/Adam_1/Assignpi/dense/bias/Adam_1/read:02(pi/dense/bias/Adam_1/Initializer/zeros:0

pi/dense_1/kernel/Adam:0pi/dense_1/kernel/Adam/Assignpi/dense_1/kernel/Adam/read:02*pi/dense_1/kernel/Adam/Initializer/zeros:0

pi/dense_1/kernel/Adam_1:0pi/dense_1/kernel/Adam_1/Assignpi/dense_1/kernel/Adam_1/read:02,pi/dense_1/kernel/Adam_1/Initializer/zeros:0
|
pi/dense_1/bias/Adam:0pi/dense_1/bias/Adam/Assignpi/dense_1/bias/Adam/read:02(pi/dense_1/bias/Adam/Initializer/zeros:0

pi/dense_1/bias/Adam_1:0pi/dense_1/bias/Adam_1/Assignpi/dense_1/bias/Adam_1/read:02*pi/dense_1/bias/Adam_1/Initializer/zeros:0

pi/dense_2/kernel/Adam:0pi/dense_2/kernel/Adam/Assignpi/dense_2/kernel/Adam/read:02*pi/dense_2/kernel/Adam/Initializer/zeros:0

pi/dense_2/kernel/Adam_1:0pi/dense_2/kernel/Adam_1/Assignpi/dense_2/kernel/Adam_1/read:02,pi/dense_2/kernel/Adam_1/Initializer/zeros:0
|
pi/dense_2/bias/Adam:0pi/dense_2/bias/Adam/Assignpi/dense_2/bias/Adam/read:02(pi/dense_2/bias/Adam/Initializer/zeros:0

pi/dense_2/bias/Adam_1:0pi/dense_2/bias/Adam_1/Assignpi/dense_2/bias/Adam_1/read:02*pi/dense_2/bias/Adam_1/Initializer/zeros:0
h
pi/log_std/Adam:0pi/log_std/Adam/Assignpi/log_std/Adam/read:02#pi/log_std/Adam/Initializer/zeros:0
p
pi/log_std/Adam_1:0pi/log_std/Adam_1/Assignpi/log_std/Adam_1/read:02%pi/log_std/Adam_1/Initializer/zeros:0
\
beta1_power_1:0beta1_power_1/Assignbeta1_power_1/read:02beta1_power_1/initial_value:0
\
beta2_power_1:0beta2_power_1/Assignbeta2_power_1/read:02beta2_power_1/initial_value:0
|
vf/dense/kernel/Adam:0vf/dense/kernel/Adam/Assignvf/dense/kernel/Adam/read:02(vf/dense/kernel/Adam/Initializer/zeros:0

vf/dense/kernel/Adam_1:0vf/dense/kernel/Adam_1/Assignvf/dense/kernel/Adam_1/read:02*vf/dense/kernel/Adam_1/Initializer/zeros:0
t
vf/dense/bias/Adam:0vf/dense/bias/Adam/Assignvf/dense/bias/Adam/read:02&vf/dense/bias/Adam/Initializer/zeros:0
|
vf/dense/bias/Adam_1:0vf/dense/bias/Adam_1/Assignvf/dense/bias/Adam_1/read:02(vf/dense/bias/Adam_1/Initializer/zeros:0

vf/dense_1/kernel/Adam:0vf/dense_1/kernel/Adam/Assignvf/dense_1/kernel/Adam/read:02*vf/dense_1/kernel/Adam/Initializer/zeros:0

vf/dense_1/kernel/Adam_1:0vf/dense_1/kernel/Adam_1/Assignvf/dense_1/kernel/Adam_1/read:02,vf/dense_1/kernel/Adam_1/Initializer/zeros:0
|
vf/dense_1/bias/Adam:0vf/dense_1/bias/Adam/Assignvf/dense_1/bias/Adam/read:02(vf/dense_1/bias/Adam/Initializer/zeros:0

vf/dense_1/bias/Adam_1:0vf/dense_1/bias/Adam_1/Assignvf/dense_1/bias/Adam_1/read:02*vf/dense_1/bias/Adam_1/Initializer/zeros:0

vf/dense_2/kernel/Adam:0vf/dense_2/kernel/Adam/Assignvf/dense_2/kernel/Adam/read:02*vf/dense_2/kernel/Adam/Initializer/zeros:0

vf/dense_2/kernel/Adam_1:0vf/dense_2/kernel/Adam_1/Assignvf/dense_2/kernel/Adam_1/read:02,vf/dense_2/kernel/Adam_1/Initializer/zeros:0
|
vf/dense_2/bias/Adam:0vf/dense_2/bias/Adam/Assignvf/dense_2/bias/Adam/read:02(vf/dense_2/bias/Adam/Initializer/zeros:0

vf/dense_2/bias/Adam_1:0vf/dense_2/bias/Adam_1/Assignvf/dense_2/bias/Adam_1/read:02*vf/dense_2/bias/Adam_1/Initializer/zeros:0
|
vc/dense/kernel/Adam:0vc/dense/kernel/Adam/Assignvc/dense/kernel/Adam/read:02(vc/dense/kernel/Adam/Initializer/zeros:0

vc/dense/kernel/Adam_1:0vc/dense/kernel/Adam_1/Assignvc/dense/kernel/Adam_1/read:02*vc/dense/kernel/Adam_1/Initializer/zeros:0
t
vc/dense/bias/Adam:0vc/dense/bias/Adam/Assignvc/dense/bias/Adam/read:02&vc/dense/bias/Adam/Initializer/zeros:0
|
vc/dense/bias/Adam_1:0vc/dense/bias/Adam_1/Assignvc/dense/bias/Adam_1/read:02(vc/dense/bias/Adam_1/Initializer/zeros:0

vc/dense_1/kernel/Adam:0vc/dense_1/kernel/Adam/Assignvc/dense_1/kernel/Adam/read:02*vc/dense_1/kernel/Adam/Initializer/zeros:0

vc/dense_1/kernel/Adam_1:0vc/dense_1/kernel/Adam_1/Assignvc/dense_1/kernel/Adam_1/read:02,vc/dense_1/kernel/Adam_1/Initializer/zeros:0
|
vc/dense_1/bias/Adam:0vc/dense_1/bias/Adam/Assignvc/dense_1/bias/Adam/read:02(vc/dense_1/bias/Adam/Initializer/zeros:0

vc/dense_1/bias/Adam_1:0vc/dense_1/bias/Adam_1/Assignvc/dense_1/bias/Adam_1/read:02*vc/dense_1/bias/Adam_1/Initializer/zeros:0

vc/dense_2/kernel/Adam:0vc/dense_2/kernel/Adam/Assignvc/dense_2/kernel/Adam/read:02*vc/dense_2/kernel/Adam/Initializer/zeros:0

vc/dense_2/kernel/Adam_1:0vc/dense_2/kernel/Adam_1/Assignvc/dense_2/kernel/Adam_1/read:02,vc/dense_2/kernel/Adam_1/Initializer/zeros:0
|
vc/dense_2/bias/Adam:0vc/dense_2/bias/Adam/Assignvc/dense_2/bias/Adam/read:02(vc/dense_2/bias/Adam/Initializer/zeros:0

vc/dense_2/bias/Adam_1:0vc/dense_2/bias/Adam_1/Assignvc/dense_2/bias/Adam_1/read:02*vc/dense_2/bias/Adam_1/Initializer/zeros:0"đ
trainable_variablesŘŐ
s
pi/dense/kernel:0pi/dense/kernel/Assignpi/dense/kernel/read:02,pi/dense/kernel/Initializer/random_uniform:08
b
pi/dense/bias:0pi/dense/bias/Assignpi/dense/bias/read:02!pi/dense/bias/Initializer/zeros:08
{
pi/dense_1/kernel:0pi/dense_1/kernel/Assignpi/dense_1/kernel/read:02.pi/dense_1/kernel/Initializer/random_uniform:08
j
pi/dense_1/bias:0pi/dense_1/bias/Assignpi/dense_1/bias/read:02#pi/dense_1/bias/Initializer/zeros:08
{
pi/dense_2/kernel:0pi/dense_2/kernel/Assignpi/dense_2/kernel/read:02.pi/dense_2/kernel/Initializer/random_uniform:08
j
pi/dense_2/bias:0pi/dense_2/bias/Assignpi/dense_2/bias/read:02#pi/dense_2/bias/Initializer/zeros:08
R
pi/log_std:0pi/log_std/Assignpi/log_std/read:02pi/log_std/initial_value:08
s
vf/dense/kernel:0vf/dense/kernel/Assignvf/dense/kernel/read:02,vf/dense/kernel/Initializer/random_uniform:08
b
vf/dense/bias:0vf/dense/bias/Assignvf/dense/bias/read:02!vf/dense/bias/Initializer/zeros:08
{
vf/dense_1/kernel:0vf/dense_1/kernel/Assignvf/dense_1/kernel/read:02.vf/dense_1/kernel/Initializer/random_uniform:08
j
vf/dense_1/bias:0vf/dense_1/bias/Assignvf/dense_1/bias/read:02#vf/dense_1/bias/Initializer/zeros:08
{
vf/dense_2/kernel:0vf/dense_2/kernel/Assignvf/dense_2/kernel/read:02.vf/dense_2/kernel/Initializer/random_uniform:08
j
vf/dense_2/bias:0vf/dense_2/bias/Assignvf/dense_2/bias/read:02#vf/dense_2/bias/Initializer/zeros:08
s
vc/dense/kernel:0vc/dense/kernel/Assignvc/dense/kernel/read:02,vc/dense/kernel/Initializer/random_uniform:08
b
vc/dense/bias:0vc/dense/bias/Assignvc/dense/bias/read:02!vc/dense/bias/Initializer/zeros:08
{
vc/dense_1/kernel:0vc/dense_1/kernel/Assignvc/dense_1/kernel/read:02.vc/dense_1/kernel/Initializer/random_uniform:08
j
vc/dense_1/bias:0vc/dense_1/bias/Assignvc/dense_1/bias/read:02#vc/dense_1/bias/Initializer/zeros:08
{
vc/dense_2/kernel:0vc/dense_2/kernel/Assignvc/dense_2/kernel/read:02.vc/dense_2/kernel/Initializer/random_uniform:08
j
vc/dense_2/bias:0vc/dense_2/bias/Assignvc/dense_2/bias/read:02#vc/dense_2/bias/Initializer/zeros:08*Ď
serving_defaultť
)
x$
Placeholder:0˙˙˙˙˙˙˙˙˙<%
vc
vc/Squeeze:0˙˙˙˙˙˙˙˙˙%
pi
pi/add:0˙˙˙˙˙˙˙˙˙$
v
vf/Squeeze:0˙˙˙˙˙˙˙˙˙tensorflow/serving/predict