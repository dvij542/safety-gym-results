ěý:
é%˝%
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
@
Softplus
features"T
activations"T"
Ttype:
2
R
SoftplusGrad
	gradients"T
features"T
	backprops"T"
Ttype:
2
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
Ttype"serve*1.13.12b'v1.13.1-0-g6612da8951'ŕ:
n
PlaceholderPlaceholder*'
_output_shapes
:˙˙˙˙˙˙˙˙˙L*
shape:˙˙˙˙˙˙˙˙˙L*
dtype0
p
Placeholder_1Placeholder*
shape:˙˙˙˙˙˙˙˙˙*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0
h
Placeholder_2Placeholder*
dtype0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
shape:˙˙˙˙˙˙˙˙˙
h
Placeholder_3Placeholder*
shape:˙˙˙˙˙˙˙˙˙*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0
h
Placeholder_4Placeholder*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*
shape:˙˙˙˙˙˙˙˙˙
h
Placeholder_5Placeholder*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
shape:˙˙˙˙˙˙˙˙˙*
dtype0
h
Placeholder_6Placeholder*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*
shape:˙˙˙˙˙˙˙˙˙
N
Placeholder_7Placeholder*
_output_shapes
: *
shape: *
dtype0
N
Placeholder_8Placeholder*
shape: *
dtype0*
_output_shapes
: 
Ľ
0pi/dense/kernel/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*
valueB"L      *"
_class
loc:@pi/dense/kernel

.pi/dense/kernel/Initializer/random_uniform/minConst*
dtype0*"
_class
loc:@pi/dense/kernel*
valueB
 *Ü¨	ž*
_output_shapes
: 

.pi/dense/kernel/Initializer/random_uniform/maxConst*"
_class
loc:@pi/dense/kernel*
dtype0*
_output_shapes
: *
valueB
 *Ü¨	>
ď
8pi/dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform0pi/dense/kernel/Initializer/random_uniform/shape*"
_class
loc:@pi/dense/kernel*
T0*
dtype0*
_output_shapes
:	L*

seed *
seed2
Ú
.pi/dense/kernel/Initializer/random_uniform/subSub.pi/dense/kernel/Initializer/random_uniform/max.pi/dense/kernel/Initializer/random_uniform/min*
T0*
_output_shapes
: *"
_class
loc:@pi/dense/kernel
í
.pi/dense/kernel/Initializer/random_uniform/mulMul8pi/dense/kernel/Initializer/random_uniform/RandomUniform.pi/dense/kernel/Initializer/random_uniform/sub*
_output_shapes
:	L*
T0*"
_class
loc:@pi/dense/kernel
ß
*pi/dense/kernel/Initializer/random_uniformAdd.pi/dense/kernel/Initializer/random_uniform/mul.pi/dense/kernel/Initializer/random_uniform/min*
_output_shapes
:	L*
T0*"
_class
loc:@pi/dense/kernel
Š
pi/dense/kernel
VariableV2*
shape:	L*
_output_shapes
:	L*
dtype0*
	container *"
_class
loc:@pi/dense/kernel*
shared_name 
Ô
pi/dense/kernel/AssignAssignpi/dense/kernel*pi/dense/kernel/Initializer/random_uniform*"
_class
loc:@pi/dense/kernel*
T0*
validate_shape(*
_output_shapes
:	L*
use_locking(

pi/dense/kernel/readIdentitypi/dense/kernel*
_output_shapes
:	L*
T0*"
_class
loc:@pi/dense/kernel

pi/dense/bias/Initializer/zerosConst*
valueB*    *
_output_shapes	
:* 
_class
loc:@pi/dense/bias*
dtype0

pi/dense/bias
VariableV2* 
_class
loc:@pi/dense/bias*
shape:*
shared_name *
dtype0*
_output_shapes	
:*
	container 
ż
pi/dense/bias/AssignAssignpi/dense/biaspi/dense/bias/Initializer/zeros*
T0*
validate_shape(* 
_class
loc:@pi/dense/bias*
_output_shapes	
:*
use_locking(
u
pi/dense/bias/readIdentitypi/dense/bias* 
_class
loc:@pi/dense/bias*
T0*
_output_shapes	
:

pi/dense/MatMulMatMulPlaceholderpi/dense/kernel/read*
transpose_b( *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
transpose_a( 

pi/dense/BiasAddBiasAddpi/dense/MatMulpi/dense/bias/read*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
data_formatNHWC*
T0
Z
pi/dense/TanhTanhpi/dense/BiasAdd*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Š
2pi/dense_1/kernel/Initializer/random_uniform/shapeConst*
valueB"      *
_output_shapes
:*
dtype0*$
_class
loc:@pi/dense_1/kernel

0pi/dense_1/kernel/Initializer/random_uniform/minConst*$
_class
loc:@pi/dense_1/kernel*
dtype0*
valueB
 *×łÝ˝*
_output_shapes
: 

0pi/dense_1/kernel/Initializer/random_uniform/maxConst*
valueB
 *×łÝ=*
_output_shapes
: *
dtype0*$
_class
loc:@pi/dense_1/kernel
ö
:pi/dense_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform2pi/dense_1/kernel/Initializer/random_uniform/shape*
T0*
dtype0*

seed *$
_class
loc:@pi/dense_1/kernel* 
_output_shapes
:
*
seed2
â
0pi/dense_1/kernel/Initializer/random_uniform/subSub0pi/dense_1/kernel/Initializer/random_uniform/max0pi/dense_1/kernel/Initializer/random_uniform/min*
T0*$
_class
loc:@pi/dense_1/kernel*
_output_shapes
: 
ö
0pi/dense_1/kernel/Initializer/random_uniform/mulMul:pi/dense_1/kernel/Initializer/random_uniform/RandomUniform0pi/dense_1/kernel/Initializer/random_uniform/sub* 
_output_shapes
:
*
T0*$
_class
loc:@pi/dense_1/kernel
č
,pi/dense_1/kernel/Initializer/random_uniformAdd0pi/dense_1/kernel/Initializer/random_uniform/mul0pi/dense_1/kernel/Initializer/random_uniform/min*
T0* 
_output_shapes
:
*$
_class
loc:@pi/dense_1/kernel
Ż
pi/dense_1/kernel
VariableV2* 
_output_shapes
:
*$
_class
loc:@pi/dense_1/kernel*
shared_name *
shape:
*
	container *
dtype0
Ý
pi/dense_1/kernel/AssignAssignpi/dense_1/kernel,pi/dense_1/kernel/Initializer/random_uniform* 
_output_shapes
:
*
T0*
validate_shape(*
use_locking(*$
_class
loc:@pi/dense_1/kernel

pi/dense_1/kernel/readIdentitypi/dense_1/kernel* 
_output_shapes
:
*
T0*$
_class
loc:@pi/dense_1/kernel

!pi/dense_1/bias/Initializer/zerosConst*
_output_shapes	
:*"
_class
loc:@pi/dense_1/bias*
valueB*    *
dtype0
Ą
pi/dense_1/bias
VariableV2*
	container *
dtype0*"
_class
loc:@pi/dense_1/bias*
shape:*
shared_name *
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
pi/dense_1/bias/readIdentitypi/dense_1/bias*"
_class
loc:@pi/dense_1/bias*
_output_shapes	
:*
T0

pi/dense_1/MatMulMatMulpi/dense/Tanhpi/dense_1/kernel/read*
T0*
transpose_b( *
transpose_a( *(
_output_shapes
:˙˙˙˙˙˙˙˙˙

pi/dense_1/BiasAddBiasAddpi/dense_1/MatMulpi/dense_1/bias/read*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
data_formatNHWC*
T0
^
pi/dense_1/TanhTanhpi/dense_1/BiasAdd*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Š
2pi/dense_2/kernel/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*$
_class
loc:@pi/dense_2/kernel*
valueB"      

0pi/dense_2/kernel/Initializer/random_uniform/minConst*
_output_shapes
: *$
_class
loc:@pi/dense_2/kernel*
valueB
 *(ž*
dtype0

0pi/dense_2/kernel/Initializer/random_uniform/maxConst*
valueB
 *(>*$
_class
loc:@pi/dense_2/kernel*
dtype0*
_output_shapes
: 
ő
:pi/dense_2/kernel/Initializer/random_uniform/RandomUniformRandomUniform2pi/dense_2/kernel/Initializer/random_uniform/shape*$
_class
loc:@pi/dense_2/kernel*

seed *
T0*
seed2.*
_output_shapes
:	*
dtype0
â
0pi/dense_2/kernel/Initializer/random_uniform/subSub0pi/dense_2/kernel/Initializer/random_uniform/max0pi/dense_2/kernel/Initializer/random_uniform/min*$
_class
loc:@pi/dense_2/kernel*
_output_shapes
: *
T0
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
:	*
T0*$
_class
loc:@pi/dense_2/kernel
­
pi/dense_2/kernel
VariableV2*
dtype0*
shape:	*
_output_shapes
:	*
	container *$
_class
loc:@pi/dense_2/kernel*
shared_name 
Ü
pi/dense_2/kernel/AssignAssignpi/dense_2/kernel,pi/dense_2/kernel/Initializer/random_uniform*
use_locking(*
_output_shapes
:	*
validate_shape(*
T0*$
_class
loc:@pi/dense_2/kernel

pi/dense_2/kernel/readIdentitypi/dense_2/kernel*
_output_shapes
:	*
T0*$
_class
loc:@pi/dense_2/kernel

!pi/dense_2/bias/Initializer/zerosConst*
_output_shapes
:*"
_class
loc:@pi/dense_2/bias*
valueB*    *
dtype0

pi/dense_2/bias
VariableV2*
shared_name *
_output_shapes
:*
	container *"
_class
loc:@pi/dense_2/bias*
shape:*
dtype0
Ć
pi/dense_2/bias/AssignAssignpi/dense_2/bias!pi/dense_2/bias/Initializer/zeros*
validate_shape(*
_output_shapes
:*
T0*"
_class
loc:@pi/dense_2/bias*
use_locking(
z
pi/dense_2/bias/readIdentitypi/dense_2/bias*"
_class
loc:@pi/dense_2/bias*
T0*
_output_shapes
:

pi/dense_2/MatMulMatMulpi/dense_1/Tanhpi/dense_2/kernel/read*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_b( *
transpose_a( 

pi/dense_2/BiasAddBiasAddpi/dense_2/MatMulpi/dense_2/bias/read*
data_formatNHWC*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
i
pi/log_std/initial_valueConst*
valueB"   ż   ż*
dtype0*
_output_shapes
:
v

pi/log_std
VariableV2*
shape:*
dtype0*
_output_shapes
:*
shared_name *
	container 
Ž
pi/log_std/AssignAssign
pi/log_stdpi/log_std/initial_value*
use_locking(*
T0*
_output_shapes
:*
_class
loc:@pi/log_std*
validate_shape(
k
pi/log_std/readIdentity
pi/log_std*
_class
loc:@pi/log_std*
_output_shapes
:*
T0
C
pi/ExpExppi/log_std/read*
T0*
_output_shapes
:
Z
pi/ShapeShapepi/dense_2/BiasAdd*
_output_shapes
:*
out_type0*
T0
Z
pi/random_normal/meanConst*
dtype0*
_output_shapes
: *
valueB
 *    
\
pi/random_normal/stddevConst*
valueB
 *  ?*
_output_shapes
: *
dtype0

%pi/random_normal/RandomStandardNormalRandomStandardNormalpi/Shape*

seed *
seed2C*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0

pi/random_normal/mulMul%pi/random_normal/RandomStandardNormalpi/random_normal/stddev*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
v
pi/random_normalAddpi/random_normal/mulpi/random_normal/mean*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Y
pi/mulMulpi/random_normalpi/Exp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
[
pi/addAddpi/dense_2/BiasAddpi/mul*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
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

pi/add_1/yConst*
_output_shapes
: *
valueB
 *wĚ+2*
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
dtype0*
valueB
 *   @*
_output_shapes
: 
U
pi/powPow
pi/truedivpi/pow/y*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
O

pi/mul_1/xConst*
valueB
 *   @*
dtype0*
_output_shapes
: 
Q
pi/mul_1Mul
pi/mul_1/xpi/log_std/read*
_output_shapes
:*
T0
S
pi/add_2Addpi/powpi/mul_1*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
O

pi/add_3/yConst*
dtype0*
valueB
 *?ë?*
_output_shapes
: 
W
pi/add_3Addpi/add_2
pi/add_3/y*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
O

pi/mul_2/xConst*
_output_shapes
: *
valueB
 *   ż*
dtype0
W
pi/mul_2Mul
pi/mul_2/xpi/add_3*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Z
pi/Sum/reduction_indicesConst*
_output_shapes
: *
value	B :*
dtype0
|
pi/SumSumpi/mul_2pi/Sum/reduction_indices*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
	keep_dims( *

Tidx0
]
pi/sub_1Subpi/addpi/dense_2/BiasAdd*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
E
pi/Exp_2Exppi/log_std/read*
_output_shapes
:*
T0
O

pi/add_4/yConst*
dtype0*
valueB
 *wĚ+2*
_output_shapes
: 
J
pi/add_4Addpi/Exp_2
pi/add_4/y*
T0*
_output_shapes
:
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

pi/mul_3/xConst*
dtype0*
_output_shapes
: *
valueB
 *   @
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

pi/add_6/yConst*
dtype0*
_output_shapes
: *
valueB
 *?ë?
W
pi/add_6Addpi/add_5
pi/add_6/y*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
O

pi/mul_4/xConst*
_output_shapes
: *
valueB
 *   ż*
dtype0
W
pi/mul_4Mul
pi/mul_4/xpi/add_6*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
\
pi/Sum_1/reduction_indicesConst*
dtype0*
value	B :*
_output_shapes
: 

pi/Sum_1Sumpi/mul_4pi/Sum_1/reduction_indices*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
	keep_dims( *

Tidx0*
T0
q
pi/PlaceholderPlaceholder*
dtype0*
shape:˙˙˙˙˙˙˙˙˙*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
s
pi/Placeholder_1Placeholder*
dtype0*
shape:˙˙˙˙˙˙˙˙˙*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
O

pi/mul_5/xConst*
dtype0*
_output_shapes
: *
valueB
 *   @
Q
pi/mul_5Mul
pi/mul_5/xpi/log_std/read*
_output_shapes
:*
T0
>
pi/Exp_3Exppi/mul_5*
_output_shapes
:*
T0
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

pi/pow_2/yConst*
_output_shapes
: *
valueB
 *   @*
dtype0
W
pi/pow_2Powpi/sub_2
pi/pow_2/y*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
U
pi/add_7Addpi/pow_2pi/Exp_3*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
O

pi/add_8/yConst*
dtype0*
valueB
 *wĚ+2*
_output_shapes
: 
W
pi/add_8Addpi/Exp_4
pi/add_8/y*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
]
pi/truediv_2RealDivpi/add_7pi/add_8*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
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

pi/mul_7/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
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
pi/sub_4Subpi/add_9pi/log_std/read*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
\
pi/Sum_2/reduction_indicesConst*
dtype0*
value	B :*
_output_shapes
: 

pi/Sum_2Sumpi/sub_4pi/Sum_2/reduction_indices*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
	keep_dims( *

Tidx0*
T0
R
pi/ConstConst*
valueB: *
_output_shapes
:*
dtype0
a
pi/MeanMeanpi/Sum_2pi/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
P
pi/add_10/yConst*
dtype0*
_output_shapes
: *
valueB
 *Çľ?
S
	pi/add_10Addpi/log_std/readpi/add_10/y*
T0*
_output_shapes
:
e
pi/Sum_3/reduction_indicesConst*
valueB :
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
: 
t
pi/Sum_3Sum	pi/add_10pi/Sum_3/reduction_indices*

Tidx0*
	keep_dims( *
_output_shapes
: *
T0
M

pi/Const_1Const*
_output_shapes
: *
dtype0*
valueB 
e
	pi/Mean_1Meanpi/Sum_3
pi/Const_1*
T0*
	keep_dims( *
_output_shapes
: *

Tidx0
Ľ
0vf/dense/kernel/Initializer/random_uniform/shapeConst*"
_class
loc:@vf/dense/kernel*
valueB"L      *
dtype0*
_output_shapes
:

.vf/dense/kernel/Initializer/random_uniform/minConst*
dtype0*
valueB
 *Ü¨	ž*
_output_shapes
: *"
_class
loc:@vf/dense/kernel

.vf/dense/kernel/Initializer/random_uniform/maxConst*
dtype0*
valueB
 *Ü¨	>*"
_class
loc:@vf/dense/kernel*
_output_shapes
: 
đ
8vf/dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform0vf/dense/kernel/Initializer/random_uniform/shape*

seed *
T0*
dtype0*
seed2*
_output_shapes
:	L*"
_class
loc:@vf/dense/kernel
Ú
.vf/dense/kernel/Initializer/random_uniform/subSub.vf/dense/kernel/Initializer/random_uniform/max.vf/dense/kernel/Initializer/random_uniform/min*"
_class
loc:@vf/dense/kernel*
T0*
_output_shapes
: 
í
.vf/dense/kernel/Initializer/random_uniform/mulMul8vf/dense/kernel/Initializer/random_uniform/RandomUniform.vf/dense/kernel/Initializer/random_uniform/sub*"
_class
loc:@vf/dense/kernel*
_output_shapes
:	L*
T0
ß
*vf/dense/kernel/Initializer/random_uniformAdd.vf/dense/kernel/Initializer/random_uniform/mul.vf/dense/kernel/Initializer/random_uniform/min*
_output_shapes
:	L*
T0*"
_class
loc:@vf/dense/kernel
Š
vf/dense/kernel
VariableV2*
_output_shapes
:	L*
shape:	L*
shared_name *"
_class
loc:@vf/dense/kernel*
dtype0*
	container 
Ô
vf/dense/kernel/AssignAssignvf/dense/kernel*vf/dense/kernel/Initializer/random_uniform*
use_locking(*"
_class
loc:@vf/dense/kernel*
validate_shape(*
_output_shapes
:	L*
T0

vf/dense/kernel/readIdentityvf/dense/kernel*
T0*"
_class
loc:@vf/dense/kernel*
_output_shapes
:	L

vf/dense/bias/Initializer/zerosConst* 
_class
loc:@vf/dense/bias*
valueB*    *
dtype0*
_output_shapes	
:

vf/dense/bias
VariableV2*
_output_shapes	
:*
shared_name *
shape:*
	container * 
_class
loc:@vf/dense/bias*
dtype0
ż
vf/dense/bias/AssignAssignvf/dense/biasvf/dense/bias/Initializer/zeros*
use_locking(*
_output_shapes	
:*
T0* 
_class
loc:@vf/dense/bias*
validate_shape(
u
vf/dense/bias/readIdentityvf/dense/bias*
_output_shapes	
:*
T0* 
_class
loc:@vf/dense/bias

vf/dense/MatMulMatMulPlaceholdervf/dense/kernel/read*
transpose_a( *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_b( *
T0

vf/dense/BiasAddBiasAddvf/dense/MatMulvf/dense/bias/read*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
data_formatNHWC
Z
vf/dense/TanhTanhvf/dense/BiasAdd*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Š
2vf/dense_1/kernel/Initializer/random_uniform/shapeConst*$
_class
loc:@vf/dense_1/kernel*
_output_shapes
:*
valueB"      *
dtype0

0vf/dense_1/kernel/Initializer/random_uniform/minConst*
dtype0*$
_class
loc:@vf/dense_1/kernel*
valueB
 *×łÝ˝*
_output_shapes
: 

0vf/dense_1/kernel/Initializer/random_uniform/maxConst*
_output_shapes
: *$
_class
loc:@vf/dense_1/kernel*
dtype0*
valueB
 *×łÝ=
÷
:vf/dense_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform2vf/dense_1/kernel/Initializer/random_uniform/shape*
T0*
dtype0*$
_class
loc:@vf/dense_1/kernel* 
_output_shapes
:
*

seed *
seed2
â
0vf/dense_1/kernel/Initializer/random_uniform/subSub0vf/dense_1/kernel/Initializer/random_uniform/max0vf/dense_1/kernel/Initializer/random_uniform/min*$
_class
loc:@vf/dense_1/kernel*
_output_shapes
: *
T0
ö
0vf/dense_1/kernel/Initializer/random_uniform/mulMul:vf/dense_1/kernel/Initializer/random_uniform/RandomUniform0vf/dense_1/kernel/Initializer/random_uniform/sub*
T0* 
_output_shapes
:
*$
_class
loc:@vf/dense_1/kernel
č
,vf/dense_1/kernel/Initializer/random_uniformAdd0vf/dense_1/kernel/Initializer/random_uniform/mul0vf/dense_1/kernel/Initializer/random_uniform/min* 
_output_shapes
:
*
T0*$
_class
loc:@vf/dense_1/kernel
Ż
vf/dense_1/kernel
VariableV2* 
_output_shapes
:
*
shape:
*
dtype0*
shared_name *
	container *$
_class
loc:@vf/dense_1/kernel
Ý
vf/dense_1/kernel/AssignAssignvf/dense_1/kernel,vf/dense_1/kernel/Initializer/random_uniform*
use_locking(*
validate_shape(*
T0* 
_output_shapes
:
*$
_class
loc:@vf/dense_1/kernel

vf/dense_1/kernel/readIdentityvf/dense_1/kernel*$
_class
loc:@vf/dense_1/kernel*
T0* 
_output_shapes
:


!vf/dense_1/bias/Initializer/zerosConst*
_output_shapes	
:*
valueB*    *"
_class
loc:@vf/dense_1/bias*
dtype0
Ą
vf/dense_1/bias
VariableV2*"
_class
loc:@vf/dense_1/bias*
shape:*
dtype0*
_output_shapes	
:*
shared_name *
	container 
Ç
vf/dense_1/bias/AssignAssignvf/dense_1/bias!vf/dense_1/bias/Initializer/zeros*
_output_shapes	
:*
validate_shape(*
use_locking(*
T0*"
_class
loc:@vf/dense_1/bias
{
vf/dense_1/bias/readIdentityvf/dense_1/bias*
T0*
_output_shapes	
:*"
_class
loc:@vf/dense_1/bias

vf/dense_1/MatMulMatMulvf/dense/Tanhvf/dense_1/kernel/read*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
T0*
transpose_b( 

vf/dense_1/BiasAddBiasAddvf/dense_1/MatMulvf/dense_1/bias/read*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
data_formatNHWC
^
vf/dense_1/TanhTanhvf/dense_1/BiasAdd*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Š
2vf/dense_2/kernel/Initializer/random_uniform/shapeConst*$
_class
loc:@vf/dense_2/kernel*
_output_shapes
:*
valueB"      *
dtype0

0vf/dense_2/kernel/Initializer/random_uniform/minConst*
valueB
 *Ivž*
dtype0*
_output_shapes
: *$
_class
loc:@vf/dense_2/kernel

0vf/dense_2/kernel/Initializer/random_uniform/maxConst*$
_class
loc:@vf/dense_2/kernel*
valueB
 *Iv>*
dtype0*
_output_shapes
: 
ö
:vf/dense_2/kernel/Initializer/random_uniform/RandomUniformRandomUniform2vf/dense_2/kernel/Initializer/random_uniform/shape*
dtype0*
seed2Ź*
T0*
_output_shapes
:	*

seed *$
_class
loc:@vf/dense_2/kernel
â
0vf/dense_2/kernel/Initializer/random_uniform/subSub0vf/dense_2/kernel/Initializer/random_uniform/max0vf/dense_2/kernel/Initializer/random_uniform/min*$
_class
loc:@vf/dense_2/kernel*
T0*
_output_shapes
: 
ő
0vf/dense_2/kernel/Initializer/random_uniform/mulMul:vf/dense_2/kernel/Initializer/random_uniform/RandomUniform0vf/dense_2/kernel/Initializer/random_uniform/sub*
_output_shapes
:	*$
_class
loc:@vf/dense_2/kernel*
T0
ç
,vf/dense_2/kernel/Initializer/random_uniformAdd0vf/dense_2/kernel/Initializer/random_uniform/mul0vf/dense_2/kernel/Initializer/random_uniform/min*
_output_shapes
:	*
T0*$
_class
loc:@vf/dense_2/kernel
­
vf/dense_2/kernel
VariableV2*
shared_name *
dtype0*
shape:	*
_output_shapes
:	*$
_class
loc:@vf/dense_2/kernel*
	container 
Ü
vf/dense_2/kernel/AssignAssignvf/dense_2/kernel,vf/dense_2/kernel/Initializer/random_uniform*
T0*
_output_shapes
:	*
use_locking(*
validate_shape(*$
_class
loc:@vf/dense_2/kernel

vf/dense_2/kernel/readIdentityvf/dense_2/kernel*$
_class
loc:@vf/dense_2/kernel*
T0*
_output_shapes
:	

!vf/dense_2/bias/Initializer/zerosConst*
valueB*    *"
_class
loc:@vf/dense_2/bias*
_output_shapes
:*
dtype0

vf/dense_2/bias
VariableV2*
_output_shapes
:*
shared_name *
dtype0*
shape:*
	container *"
_class
loc:@vf/dense_2/bias
Ć
vf/dense_2/bias/AssignAssignvf/dense_2/bias!vf/dense_2/bias/Initializer/zeros*
validate_shape(*"
_class
loc:@vf/dense_2/bias*
T0*
use_locking(*
_output_shapes
:
z
vf/dense_2/bias/readIdentityvf/dense_2/bias*
T0*
_output_shapes
:*"
_class
loc:@vf/dense_2/bias

vf/dense_2/MatMulMatMulvf/dense_1/Tanhvf/dense_2/kernel/read*
transpose_a( *
transpose_b( *'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

vf/dense_2/BiasAddBiasAddvf/dense_2/MatMulvf/dense_2/bias/read*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
data_formatNHWC
n

vf/SqueezeSqueezevf/dense_2/BiasAdd*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
squeeze_dims

Ľ
0vc/dense/kernel/Initializer/random_uniform/shapeConst*
valueB"L      *"
_class
loc:@vc/dense/kernel*
_output_shapes
:*
dtype0

.vc/dense/kernel/Initializer/random_uniform/minConst*"
_class
loc:@vc/dense/kernel*
valueB
 *Ü¨	ž*
_output_shapes
: *
dtype0

.vc/dense/kernel/Initializer/random_uniform/maxConst*
valueB
 *Ü¨	>*
_output_shapes
: *"
_class
loc:@vc/dense/kernel*
dtype0
đ
8vc/dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform0vc/dense/kernel/Initializer/random_uniform/shape*
_output_shapes
:	L*"
_class
loc:@vc/dense/kernel*
seed2˝*
dtype0*
T0*

seed 
Ú
.vc/dense/kernel/Initializer/random_uniform/subSub.vc/dense/kernel/Initializer/random_uniform/max.vc/dense/kernel/Initializer/random_uniform/min*"
_class
loc:@vc/dense/kernel*
_output_shapes
: *
T0
í
.vc/dense/kernel/Initializer/random_uniform/mulMul8vc/dense/kernel/Initializer/random_uniform/RandomUniform.vc/dense/kernel/Initializer/random_uniform/sub*
_output_shapes
:	L*
T0*"
_class
loc:@vc/dense/kernel
ß
*vc/dense/kernel/Initializer/random_uniformAdd.vc/dense/kernel/Initializer/random_uniform/mul.vc/dense/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@vc/dense/kernel*
_output_shapes
:	L
Š
vc/dense/kernel
VariableV2*
	container *
shape:	L*"
_class
loc:@vc/dense/kernel*
shared_name *
_output_shapes
:	L*
dtype0
Ô
vc/dense/kernel/AssignAssignvc/dense/kernel*vc/dense/kernel/Initializer/random_uniform*"
_class
loc:@vc/dense/kernel*
use_locking(*
T0*
validate_shape(*
_output_shapes
:	L

vc/dense/kernel/readIdentityvc/dense/kernel*
T0*"
_class
loc:@vc/dense/kernel*
_output_shapes
:	L

vc/dense/bias/Initializer/zerosConst* 
_class
loc:@vc/dense/bias*
dtype0*
_output_shapes	
:*
valueB*    

vc/dense/bias
VariableV2*
dtype0*
	container * 
_class
loc:@vc/dense/bias*
_output_shapes	
:*
shared_name *
shape:
ż
vc/dense/bias/AssignAssignvc/dense/biasvc/dense/bias/Initializer/zeros*
validate_shape(*
use_locking(* 
_class
loc:@vc/dense/bias*
_output_shapes	
:*
T0
u
vc/dense/bias/readIdentityvc/dense/bias* 
_class
loc:@vc/dense/bias*
T0*
_output_shapes	
:

vc/dense/MatMulMatMulPlaceholdervc/dense/kernel/read*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_b( *
T0*
transpose_a( 

vc/dense/BiasAddBiasAddvc/dense/MatMulvc/dense/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Z
vc/dense/TanhTanhvc/dense/BiasAdd*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Š
2vc/dense_1/kernel/Initializer/random_uniform/shapeConst*
_output_shapes
:*$
_class
loc:@vc/dense_1/kernel*
valueB"      *
dtype0

0vc/dense_1/kernel/Initializer/random_uniform/minConst*
_output_shapes
: *$
_class
loc:@vc/dense_1/kernel*
valueB
 *×łÝ˝*
dtype0

0vc/dense_1/kernel/Initializer/random_uniform/maxConst*
_output_shapes
: *
valueB
 *×łÝ=*
dtype0*$
_class
loc:@vc/dense_1/kernel
÷
:vc/dense_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform2vc/dense_1/kernel/Initializer/random_uniform/shape*
dtype0*$
_class
loc:@vc/dense_1/kernel*

seed *
seed2Î*
T0* 
_output_shapes
:

â
0vc/dense_1/kernel/Initializer/random_uniform/subSub0vc/dense_1/kernel/Initializer/random_uniform/max0vc/dense_1/kernel/Initializer/random_uniform/min*$
_class
loc:@vc/dense_1/kernel*
_output_shapes
: *
T0
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
*
T0*$
_class
loc:@vc/dense_1/kernel
Ż
vc/dense_1/kernel
VariableV2*
	container *
shape:
*$
_class
loc:@vc/dense_1/kernel*
shared_name *
dtype0* 
_output_shapes
:

Ý
vc/dense_1/kernel/AssignAssignvc/dense_1/kernel,vc/dense_1/kernel/Initializer/random_uniform*
validate_shape(*
use_locking(*$
_class
loc:@vc/dense_1/kernel* 
_output_shapes
:
*
T0

vc/dense_1/kernel/readIdentityvc/dense_1/kernel*
T0*$
_class
loc:@vc/dense_1/kernel* 
_output_shapes
:


!vc/dense_1/bias/Initializer/zerosConst*
dtype0*
_output_shapes	
:*
valueB*    *"
_class
loc:@vc/dense_1/bias
Ą
vc/dense_1/bias
VariableV2*"
_class
loc:@vc/dense_1/bias*
dtype0*
shape:*
_output_shapes	
:*
	container *
shared_name 
Ç
vc/dense_1/bias/AssignAssignvc/dense_1/bias!vc/dense_1/bias/Initializer/zeros*"
_class
loc:@vc/dense_1/bias*
use_locking(*
_output_shapes	
:*
validate_shape(*
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
transpose_b( *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
transpose_a( 

vc/dense_1/BiasAddBiasAddvc/dense_1/MatMulvc/dense_1/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
^
vc/dense_1/TanhTanhvc/dense_1/BiasAdd*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Š
2vc/dense_2/kernel/Initializer/random_uniform/shapeConst*
dtype0*$
_class
loc:@vc/dense_2/kernel*
valueB"      *
_output_shapes
:

0vc/dense_2/kernel/Initializer/random_uniform/minConst*$
_class
loc:@vc/dense_2/kernel*
_output_shapes
: *
valueB
 *Ivž*
dtype0

0vc/dense_2/kernel/Initializer/random_uniform/maxConst*$
_class
loc:@vc/dense_2/kernel*
dtype0*
_output_shapes
: *
valueB
 *Iv>
ö
:vc/dense_2/kernel/Initializer/random_uniform/RandomUniformRandomUniform2vc/dense_2/kernel/Initializer/random_uniform/shape*
_output_shapes
:	*
T0*
dtype0*
seed2ß*

seed *$
_class
loc:@vc/dense_2/kernel
â
0vc/dense_2/kernel/Initializer/random_uniform/subSub0vc/dense_2/kernel/Initializer/random_uniform/max0vc/dense_2/kernel/Initializer/random_uniform/min*
_output_shapes
: *$
_class
loc:@vc/dense_2/kernel*
T0
ő
0vc/dense_2/kernel/Initializer/random_uniform/mulMul:vc/dense_2/kernel/Initializer/random_uniform/RandomUniform0vc/dense_2/kernel/Initializer/random_uniform/sub*$
_class
loc:@vc/dense_2/kernel*
T0*
_output_shapes
:	
ç
,vc/dense_2/kernel/Initializer/random_uniformAdd0vc/dense_2/kernel/Initializer/random_uniform/mul0vc/dense_2/kernel/Initializer/random_uniform/min*$
_class
loc:@vc/dense_2/kernel*
T0*
_output_shapes
:	
­
vc/dense_2/kernel
VariableV2*
_output_shapes
:	*
dtype0*
	container *
shared_name *$
_class
loc:@vc/dense_2/kernel*
shape:	
Ü
vc/dense_2/kernel/AssignAssignvc/dense_2/kernel,vc/dense_2/kernel/Initializer/random_uniform*$
_class
loc:@vc/dense_2/kernel*
use_locking(*
validate_shape(*
T0*
_output_shapes
:	

vc/dense_2/kernel/readIdentityvc/dense_2/kernel*
T0*$
_class
loc:@vc/dense_2/kernel*
_output_shapes
:	

!vc/dense_2/bias/Initializer/zerosConst*
dtype0*
_output_shapes
:*
valueB*    *"
_class
loc:@vc/dense_2/bias

vc/dense_2/bias
VariableV2*
	container *"
_class
loc:@vc/dense_2/bias*
_output_shapes
:*
dtype0*
shape:*
shared_name 
Ć
vc/dense_2/bias/AssignAssignvc/dense_2/bias!vc/dense_2/bias/Initializer/zeros*
validate_shape(*
_output_shapes
:*
T0*"
_class
loc:@vc/dense_2/bias*
use_locking(
z
vc/dense_2/bias/readIdentityvc/dense_2/bias*
T0*"
_class
loc:@vc/dense_2/bias*
_output_shapes
:

vc/dense_2/MatMulMatMulvc/dense_1/Tanhvc/dense_2/kernel/read*
transpose_b( *
T0*
transpose_a( *'
_output_shapes
:˙˙˙˙˙˙˙˙˙

vc/dense_2/BiasAddBiasAddvc/dense_2/MatMulvc/dense_2/bias/read*
data_formatNHWC*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
n

vc/SqueezeSqueezevc/dense_2/BiasAdd*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
squeeze_dims

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
MeanMeanNegConst*
T0*

Tidx0*
	keep_dims( *
_output_shapes
: 
L
Const_1Const*
dtype0*
valueB
 *D
?*
_output_shapes
: 
h
#penalty/penalty_param/initial_valueConst*
dtype0*
valueB
 *D
?*
_output_shapes
: 
y
penalty/penalty_param
VariableV2*
shared_name *
dtype0*
_output_shapes
: *
	container *
shape: 
Ö
penalty/penalty_param/AssignAssignpenalty/penalty_param#penalty/penalty_param/initial_value*
_output_shapes
: *
T0*
use_locking(*
validate_shape(*(
_class
loc:@penalty/penalty_param

penalty/penalty_param/readIdentitypenalty/penalty_param*(
_class
loc:@penalty/penalty_param*
T0*
_output_shapes
: 
Q
SoftplusSoftpluspenalty/penalty_param/read*
T0*
_output_shapes
: 
I
Neg_1Negpenalty/penalty_param/read*
T0*
_output_shapes
: 
J
sub/yConst*
dtype0*
valueB
 *  ČA*
_output_shapes
: 
A
subSubPlaceholder_8sub/y*
_output_shapes
: *
T0
7
mulMulNeg_1sub*
_output_shapes
: *
T0
R
gradients/ShapeConst*
dtype0*
_output_shapes
: *
valueB 
X
gradients/grad_ys_0Const*
dtype0*
_output_shapes
: *
valueB
 *  ?
o
gradients/FillFillgradients/Shapegradients/grad_ys_0*

index_type0*
T0*
_output_shapes
: 
S
gradients/mul_grad/MulMulgradients/Fillsub*
T0*
_output_shapes
: 
W
gradients/mul_grad/Mul_1Mulgradients/FillNeg_1*
_output_shapes
: *
T0
_
#gradients/mul_grad/tuple/group_depsNoOp^gradients/mul_grad/Mul^gradients/mul_grad/Mul_1
Á
+gradients/mul_grad/tuple/control_dependencyIdentitygradients/mul_grad/Mul$^gradients/mul_grad/tuple/group_deps*)
_class
loc:@gradients/mul_grad/Mul*
T0*
_output_shapes
: 
Ç
-gradients/mul_grad/tuple/control_dependency_1Identitygradients/mul_grad/Mul_1$^gradients/mul_grad/tuple/group_deps*
_output_shapes
: *+
_class!
loc:@gradients/mul_grad/Mul_1*
T0
m
gradients/Neg_1_grad/NegNeg+gradients/mul_grad/tuple/control_dependency*
T0*
_output_shapes
: 
`
Reshape/shapeConst*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙*
_output_shapes
:
n
ReshapeReshapegradients/Neg_1_grad/NegReshape/shape*
T0*
Tshape0*
_output_shapes
:
S
concat/concat_dimConst*
dtype0*
_output_shapes
: *
value	B : 
@
concatIdentityReshape*
T0*
_output_shapes
:
f
PyFuncPyFuncconcat*
_output_shapes
:*
Tout
2*
token
pyfunc_0*
Tin
2
Q
Const_2Const*
_output_shapes
:*
valueB:*
dtype0
Q
split/split_dimConst*
dtype0*
_output_shapes
: *
value	B : 
s
splitSplitVPyFuncConst_2split/split_dim*
_output_shapes
:*
	num_split*

Tlen0*
T0
R
Reshape_1/shapeConst*
dtype0*
valueB *
_output_shapes
: 
[
	Reshape_1ReshapesplitReshape_1/shape*
Tshape0*
T0*
_output_shapes
: 

beta1_power/initial_valueConst*
valueB
 *fff?*
_output_shapes
: *
dtype0*(
_class
loc:@penalty/penalty_param

beta1_power
VariableV2*
	container *
shared_name *(
_class
loc:@penalty/penalty_param*
shape: *
_output_shapes
: *
dtype0
¸
beta1_power/AssignAssignbeta1_powerbeta1_power/initial_value*
_output_shapes
: *
use_locking(*
validate_shape(*
T0*(
_class
loc:@penalty/penalty_param
t
beta1_power/readIdentitybeta1_power*
_output_shapes
: *
T0*(
_class
loc:@penalty/penalty_param

beta2_power/initial_valueConst*
valueB
 *wž?*
_output_shapes
: *(
_class
loc:@penalty/penalty_param*
dtype0

beta2_power
VariableV2*
_output_shapes
: *
shared_name *
shape: *
dtype0*
	container *(
_class
loc:@penalty/penalty_param
¸
beta2_power/AssignAssignbeta2_powerbeta2_power/initial_value*
T0*
validate_shape(*
use_locking(*
_output_shapes
: *(
_class
loc:@penalty/penalty_param
t
beta2_power/readIdentitybeta2_power*
_output_shapes
: *
T0*(
_class
loc:@penalty/penalty_param

,penalty/penalty_param/Adam/Initializer/zerosConst*
dtype0*
_output_shapes
: *(
_class
loc:@penalty/penalty_param*
valueB
 *    
¨
penalty/penalty_param/Adam
VariableV2*(
_class
loc:@penalty/penalty_param*
shared_name *
_output_shapes
: *
dtype0*
	container *
shape: 
é
!penalty/penalty_param/Adam/AssignAssignpenalty/penalty_param/Adam,penalty/penalty_param/Adam/Initializer/zeros*
_output_shapes
: *(
_class
loc:@penalty/penalty_param*
use_locking(*
T0*
validate_shape(

penalty/penalty_param/Adam/readIdentitypenalty/penalty_param/Adam*(
_class
loc:@penalty/penalty_param*
T0*
_output_shapes
: 

.penalty/penalty_param/Adam_1/Initializer/zerosConst*
_output_shapes
: *
dtype0*(
_class
loc:@penalty/penalty_param*
valueB
 *    
Ş
penalty/penalty_param/Adam_1
VariableV2*
_output_shapes
: *
	container *(
_class
loc:@penalty/penalty_param*
shape: *
shared_name *
dtype0
ď
#penalty/penalty_param/Adam_1/AssignAssignpenalty/penalty_param/Adam_1.penalty/penalty_param/Adam_1/Initializer/zeros*
_output_shapes
: *
use_locking(*
validate_shape(*(
_class
loc:@penalty/penalty_param*
T0

!penalty/penalty_param/Adam_1/readIdentitypenalty/penalty_param/Adam_1*
T0*(
_class
loc:@penalty/penalty_param*
_output_shapes
: 
W
Adam/learning_rateConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL=
O

Adam/beta1Const*
dtype0*
_output_shapes
: *
valueB
 *fff?
O

Adam/beta2Const*
valueB
 *wž?*
_output_shapes
: *
dtype0
Q
Adam/epsilonConst*
valueB
 *wĚ+2*
dtype0*
_output_shapes
: 
ä
+Adam/update_penalty/penalty_param/ApplyAdam	ApplyAdampenalty/penalty_parampenalty/penalty_param/Adampenalty/penalty_param/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon	Reshape_1*(
_class
loc:@penalty/penalty_param*
T0*
use_nesterov( *
_output_shapes
: *
use_locking( 
Ś
Adam/mulMulbeta1_power/read
Adam/beta1,^Adam/update_penalty/penalty_param/ApplyAdam*
T0*(
_class
loc:@penalty/penalty_param*
_output_shapes
: 
 
Adam/AssignAssignbeta1_powerAdam/mul*
T0*(
_class
loc:@penalty/penalty_param*
use_locking( *
_output_shapes
: *
validate_shape(
¨

Adam/mul_1Mulbeta2_power/read
Adam/beta2,^Adam/update_penalty/penalty_param/ApplyAdam*
T0*
_output_shapes
: *(
_class
loc:@penalty/penalty_param
¤
Adam/Assign_1Assignbeta2_power
Adam/mul_1*
validate_shape(*(
_class
loc:@penalty/penalty_param*
_output_shapes
: *
use_locking( *
T0
X
AdamNoOp^Adam/Assign^Adam/Assign_1,^Adam/update_penalty/penalty_param/ApplyAdam
i
Reshape_2/shapeConst^Adam*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
t
	Reshape_2Reshapepenalty/penalty_param/readReshape_2/shape*
Tshape0*
_output_shapes
:*
T0
\
concat_1/concat_dimConst^Adam*
_output_shapes
: *
value	B : *
dtype0
D
concat_1Identity	Reshape_2*
T0*
_output_shapes
:
h
PyFunc_1PyFuncconcat_1*
Tout
2*
_output_shapes
:*
token
pyfunc_1*
Tin
2
X
Const_3Const^Adam*
_output_shapes
:*
dtype0*
valueB:
Z
split_1/split_dimConst^Adam*
dtype0*
_output_shapes
: *
value	B : 
w
split_1SplitVPyFunc_1Const_3split_1/split_dim*
	num_split*
T0*
_output_shapes
:*

Tlen0
Y
Reshape_3/shapeConst^Adam*
valueB *
_output_shapes
: *
dtype0
]
	Reshape_3Reshapesplit_1Reshape_3/shape*
Tshape0*
T0*
_output_shapes
: 
Ś
AssignAssignpenalty/penalty_param	Reshape_3*
validate_shape(*(
_class
loc:@penalty/penalty_param*
T0*
use_locking(*
_output_shapes
: 
"

group_depsNoOp^Adam^Assign
(
group_deps_1NoOp^Adam^group_deps
Q
sub_1Subpi/SumPlaceholder_6*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
?
ExpExpsub_1*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
N
	Greater/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Z
GreaterGreaterPlaceholder_2	Greater/y*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
L
mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *?
R
mul_1Mulmul_1/xPlaceholder_2*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
L
mul_2/xConst*
valueB
 *ÍĚL?*
_output_shapes
: *
dtype0
R
mul_2Mulmul_2/xPlaceholder_2*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
U
SelectSelectGreatermul_1mul_2*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
N
mul_3MulExpPlaceholder_2*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
O
MinimumMinimummul_3Select*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Q
Const_4Const*
_output_shapes
:*
dtype0*
valueB: 
^
Mean_1MeanMinimumConst_4*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
N
mul_4MulExpPlaceholder_3*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
Q
Const_5Const*
valueB: *
_output_shapes
:*
dtype0
\
Mean_2Meanmul_4Const_5*

Tidx0*
	keep_dims( *
_output_shapes
: *
T0
L
mul_5/xConst*
valueB
 *    *
dtype0*
_output_shapes
: 
A
mul_5Mulmul_5/x	pi/Mean_1*
_output_shapes
: *
T0
:
addAddMean_1mul_5*
_output_shapes
: *
T0
?
mul_6MulSoftplusMean_2*
T0*
_output_shapes
: 
9
sub_2Subaddmul_6*
T0*
_output_shapes
: 
L
add_1/xConst*
valueB
 *  ?*
_output_shapes
: *
dtype0
@
add_1Addadd_1/xSoftplus*
T0*
_output_shapes
: 
A
truedivRealDivsub_2add_1*
_output_shapes
: *
T0
6
Neg_2Negtruediv*
_output_shapes
: *
T0
T
gradients_1/ShapeConst*
valueB *
_output_shapes
: *
dtype0
Z
gradients_1/grad_ys_0Const*
valueB
 *  ?*
_output_shapes
: *
dtype0
u
gradients_1/FillFillgradients_1/Shapegradients_1/grad_ys_0*

index_type0*
_output_shapes
: *
T0
T
gradients_1/Neg_2_grad/NegNeggradients_1/Fill*
T0*
_output_shapes
: 
a
gradients_1/truediv_grad/ShapeConst*
_output_shapes
: *
dtype0*
valueB 
c
 gradients_1/truediv_grad/Shape_1Const*
dtype0*
valueB *
_output_shapes
: 
Ć
.gradients_1/truediv_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/truediv_grad/Shape gradients_1/truediv_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
o
 gradients_1/truediv_grad/RealDivRealDivgradients_1/Neg_2_grad/Negadd_1*
_output_shapes
: *
T0
ł
gradients_1/truediv_grad/SumSum gradients_1/truediv_grad/RealDiv.gradients_1/truediv_grad/BroadcastGradientArgs*

Tidx0*
T0*
	keep_dims( *
_output_shapes
: 

 gradients_1/truediv_grad/ReshapeReshapegradients_1/truediv_grad/Sumgradients_1/truediv_grad/Shape*
Tshape0*
_output_shapes
: *
T0
K
gradients_1/truediv_grad/NegNegsub_2*
_output_shapes
: *
T0
s
"gradients_1/truediv_grad/RealDiv_1RealDivgradients_1/truediv_grad/Negadd_1*
_output_shapes
: *
T0
y
"gradients_1/truediv_grad/RealDiv_2RealDiv"gradients_1/truediv_grad/RealDiv_1add_1*
_output_shapes
: *
T0

gradients_1/truediv_grad/mulMulgradients_1/Neg_2_grad/Neg"gradients_1/truediv_grad/RealDiv_2*
_output_shapes
: *
T0
ł
gradients_1/truediv_grad/Sum_1Sumgradients_1/truediv_grad/mul0gradients_1/truediv_grad/BroadcastGradientArgs:1*

Tidx0*
_output_shapes
: *
	keep_dims( *
T0

"gradients_1/truediv_grad/Reshape_1Reshapegradients_1/truediv_grad/Sum_1 gradients_1/truediv_grad/Shape_1*
T0*
_output_shapes
: *
Tshape0
y
)gradients_1/truediv_grad/tuple/group_depsNoOp!^gradients_1/truediv_grad/Reshape#^gradients_1/truediv_grad/Reshape_1
á
1gradients_1/truediv_grad/tuple/control_dependencyIdentity gradients_1/truediv_grad/Reshape*^gradients_1/truediv_grad/tuple/group_deps*3
_class)
'%loc:@gradients_1/truediv_grad/Reshape*
_output_shapes
: *
T0
ç
3gradients_1/truediv_grad/tuple/control_dependency_1Identity"gradients_1/truediv_grad/Reshape_1*^gradients_1/truediv_grad/tuple/group_deps*5
_class+
)'loc:@gradients_1/truediv_grad/Reshape_1*
T0*
_output_shapes
: 
u
gradients_1/sub_2_grad/NegNeg1gradients_1/truediv_grad/tuple/control_dependency*
T0*
_output_shapes
: 

'gradients_1/sub_2_grad/tuple/group_depsNoOp^gradients_1/sub_2_grad/Neg2^gradients_1/truediv_grad/tuple/control_dependency
î
/gradients_1/sub_2_grad/tuple/control_dependencyIdentity1gradients_1/truediv_grad/tuple/control_dependency(^gradients_1/sub_2_grad/tuple/group_deps*
_output_shapes
: *3
_class)
'%loc:@gradients_1/truediv_grad/Reshape*
T0
Ó
1gradients_1/sub_2_grad/tuple/control_dependency_1Identitygradients_1/sub_2_grad/Neg(^gradients_1/sub_2_grad/tuple/group_deps*-
_class#
!loc:@gradients_1/sub_2_grad/Neg*
_output_shapes
: *
T0
e
'gradients_1/add_1_grad/tuple/group_depsNoOp4^gradients_1/truediv_grad/tuple/control_dependency_1
ň
/gradients_1/add_1_grad/tuple/control_dependencyIdentity3gradients_1/truediv_grad/tuple/control_dependency_1(^gradients_1/add_1_grad/tuple/group_deps*5
_class+
)'loc:@gradients_1/truediv_grad/Reshape_1*
T0*
_output_shapes
: 
ô
1gradients_1/add_1_grad/tuple/control_dependency_1Identity3gradients_1/truediv_grad/tuple/control_dependency_1(^gradients_1/add_1_grad/tuple/group_deps*
T0*5
_class+
)'loc:@gradients_1/truediv_grad/Reshape_1*
_output_shapes
: 
_
%gradients_1/add_grad/tuple/group_depsNoOp0^gradients_1/sub_2_grad/tuple/control_dependency
č
-gradients_1/add_grad/tuple/control_dependencyIdentity/gradients_1/sub_2_grad/tuple/control_dependency&^gradients_1/add_grad/tuple/group_deps*
_output_shapes
: *3
_class)
'%loc:@gradients_1/truediv_grad/Reshape*
T0
ę
/gradients_1/add_grad/tuple/control_dependency_1Identity/gradients_1/sub_2_grad/tuple/control_dependency&^gradients_1/add_grad/tuple/group_deps*3
_class)
'%loc:@gradients_1/truediv_grad/Reshape*
T0*
_output_shapes
: 
}
gradients_1/mul_6_grad/MulMul1gradients_1/sub_2_grad/tuple/control_dependency_1Mean_2*
T0*
_output_shapes
: 

gradients_1/mul_6_grad/Mul_1Mul1gradients_1/sub_2_grad/tuple/control_dependency_1Softplus*
T0*
_output_shapes
: 
k
'gradients_1/mul_6_grad/tuple/group_depsNoOp^gradients_1/mul_6_grad/Mul^gradients_1/mul_6_grad/Mul_1
Ń
/gradients_1/mul_6_grad/tuple/control_dependencyIdentitygradients_1/mul_6_grad/Mul(^gradients_1/mul_6_grad/tuple/group_deps*
T0*-
_class#
!loc:@gradients_1/mul_6_grad/Mul*
_output_shapes
: 
×
1gradients_1/mul_6_grad/tuple/control_dependency_1Identitygradients_1/mul_6_grad/Mul_1(^gradients_1/mul_6_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients_1/mul_6_grad/Mul_1*
_output_shapes
: 
o
%gradients_1/Mean_1_grad/Reshape/shapeConst*
dtype0*
_output_shapes
:*
valueB:
ł
gradients_1/Mean_1_grad/ReshapeReshape-gradients_1/add_grad/tuple/control_dependency%gradients_1/Mean_1_grad/Reshape/shape*
T0*
_output_shapes
:*
Tshape0
d
gradients_1/Mean_1_grad/ShapeShapeMinimum*
_output_shapes
:*
out_type0*
T0
¤
gradients_1/Mean_1_grad/TileTilegradients_1/Mean_1_grad/Reshapegradients_1/Mean_1_grad/Shape*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*

Tmultiples0
f
gradients_1/Mean_1_grad/Shape_1ShapeMinimum*
T0*
_output_shapes
:*
out_type0
b
gradients_1/Mean_1_grad/Shape_2Const*
_output_shapes
: *
dtype0*
valueB 
g
gradients_1/Mean_1_grad/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
˘
gradients_1/Mean_1_grad/ProdProdgradients_1/Mean_1_grad/Shape_1gradients_1/Mean_1_grad/Const*
	keep_dims( *
_output_shapes
: *

Tidx0*
T0
i
gradients_1/Mean_1_grad/Const_1Const*
dtype0*
_output_shapes
:*
valueB: 
Ś
gradients_1/Mean_1_grad/Prod_1Prodgradients_1/Mean_1_grad/Shape_2gradients_1/Mean_1_grad/Const_1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
c
!gradients_1/Mean_1_grad/Maximum/yConst*
dtype0*
_output_shapes
: *
value	B :

gradients_1/Mean_1_grad/MaximumMaximumgradients_1/Mean_1_grad/Prod_1!gradients_1/Mean_1_grad/Maximum/y*
_output_shapes
: *
T0

 gradients_1/Mean_1_grad/floordivFloorDivgradients_1/Mean_1_grad/Prodgradients_1/Mean_1_grad/Maximum*
T0*
_output_shapes
: 

gradients_1/Mean_1_grad/CastCast gradients_1/Mean_1_grad/floordiv*
Truncate( *

SrcT0*

DstT0*
_output_shapes
: 

gradients_1/Mean_1_grad/truedivRealDivgradients_1/Mean_1_grad/Tilegradients_1/Mean_1_grad/Cast*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
~
gradients_1/mul_5_grad/MulMul/gradients_1/add_grad/tuple/control_dependency_1	pi/Mean_1*
_output_shapes
: *
T0
~
gradients_1/mul_5_grad/Mul_1Mul/gradients_1/add_grad/tuple/control_dependency_1mul_5/x*
_output_shapes
: *
T0
k
'gradients_1/mul_5_grad/tuple/group_depsNoOp^gradients_1/mul_5_grad/Mul^gradients_1/mul_5_grad/Mul_1
Ń
/gradients_1/mul_5_grad/tuple/control_dependencyIdentitygradients_1/mul_5_grad/Mul(^gradients_1/mul_5_grad/tuple/group_deps*-
_class#
!loc:@gradients_1/mul_5_grad/Mul*
T0*
_output_shapes
: 
×
1gradients_1/mul_5_grad/tuple/control_dependency_1Identitygradients_1/mul_5_grad/Mul_1(^gradients_1/mul_5_grad/tuple/group_deps*
_output_shapes
: *
T0*/
_class%
#!loc:@gradients_1/mul_5_grad/Mul_1
Ý
gradients_1/AddNAddN1gradients_1/add_1_grad/tuple/control_dependency_1/gradients_1/mul_6_grad/tuple/control_dependency*
N*
_output_shapes
: *5
_class+
)'loc:@gradients_1/truediv_grad/Reshape_1*
T0

&gradients_1/Softplus_grad/SoftplusGradSoftplusGradgradients_1/AddNpenalty/penalty_param/read*
_output_shapes
: *
T0
o
%gradients_1/Mean_2_grad/Reshape/shapeConst*
dtype0*
valueB:*
_output_shapes
:
ˇ
gradients_1/Mean_2_grad/ReshapeReshape1gradients_1/mul_6_grad/tuple/control_dependency_1%gradients_1/Mean_2_grad/Reshape/shape*
Tshape0*
_output_shapes
:*
T0
b
gradients_1/Mean_2_grad/ShapeShapemul_4*
_output_shapes
:*
out_type0*
T0
¤
gradients_1/Mean_2_grad/TileTilegradients_1/Mean_2_grad/Reshapegradients_1/Mean_2_grad/Shape*

Tmultiples0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
d
gradients_1/Mean_2_grad/Shape_1Shapemul_4*
_output_shapes
:*
out_type0*
T0
b
gradients_1/Mean_2_grad/Shape_2Const*
dtype0*
_output_shapes
: *
valueB 
g
gradients_1/Mean_2_grad/ConstConst*
dtype0*
valueB: *
_output_shapes
:
˘
gradients_1/Mean_2_grad/ProdProdgradients_1/Mean_2_grad/Shape_1gradients_1/Mean_2_grad/Const*

Tidx0*
_output_shapes
: *
T0*
	keep_dims( 
i
gradients_1/Mean_2_grad/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
Ś
gradients_1/Mean_2_grad/Prod_1Prodgradients_1/Mean_2_grad/Shape_2gradients_1/Mean_2_grad/Const_1*

Tidx0*
T0*
_output_shapes
: *
	keep_dims( 
c
!gradients_1/Mean_2_grad/Maximum/yConst*
_output_shapes
: *
value	B :*
dtype0

gradients_1/Mean_2_grad/MaximumMaximumgradients_1/Mean_2_grad/Prod_1!gradients_1/Mean_2_grad/Maximum/y*
_output_shapes
: *
T0

 gradients_1/Mean_2_grad/floordivFloorDivgradients_1/Mean_2_grad/Prodgradients_1/Mean_2_grad/Maximum*
T0*
_output_shapes
: 

gradients_1/Mean_2_grad/CastCast gradients_1/Mean_2_grad/floordiv*
_output_shapes
: *

DstT0*
Truncate( *

SrcT0

gradients_1/Mean_2_grad/truedivRealDivgradients_1/Mean_2_grad/Tilegradients_1/Mean_2_grad/Cast*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
c
gradients_1/Minimum_grad/ShapeShapemul_3*
_output_shapes
:*
out_type0*
T0
f
 gradients_1/Minimum_grad/Shape_1ShapeSelect*
out_type0*
_output_shapes
:*
T0

 gradients_1/Minimum_grad/Shape_2Shapegradients_1/Mean_1_grad/truediv*
out_type0*
T0*
_output_shapes
:
i
$gradients_1/Minimum_grad/zeros/ConstConst*
_output_shapes
: *
valueB
 *    *
dtype0
Ž
gradients_1/Minimum_grad/zerosFill gradients_1/Minimum_grad/Shape_2$gradients_1/Minimum_grad/zeros/Const*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*

index_type0
l
"gradients_1/Minimum_grad/LessEqual	LessEqualmul_3Select*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ć
.gradients_1/Minimum_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/Minimum_grad/Shape gradients_1/Minimum_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
ź
gradients_1/Minimum_grad/SelectSelect"gradients_1/Minimum_grad/LessEqualgradients_1/Mean_1_grad/truedivgradients_1/Minimum_grad/zeros*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
ž
!gradients_1/Minimum_grad/Select_1Select"gradients_1/Minimum_grad/LessEqualgradients_1/Minimum_grad/zerosgradients_1/Mean_1_grad/truediv*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
´
gradients_1/Minimum_grad/SumSumgradients_1/Minimum_grad/Select.gradients_1/Minimum_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
Ľ
 gradients_1/Minimum_grad/ReshapeReshapegradients_1/Minimum_grad/Sumgradients_1/Minimum_grad/Shape*
Tshape0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
ş
gradients_1/Minimum_grad/Sum_1Sum!gradients_1/Minimum_grad/Select_10gradients_1/Minimum_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*
_output_shapes
:*

Tidx0
Ť
"gradients_1/Minimum_grad/Reshape_1Reshapegradients_1/Minimum_grad/Sum_1 gradients_1/Minimum_grad/Shape_1*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
Tshape0*
T0
y
)gradients_1/Minimum_grad/tuple/group_depsNoOp!^gradients_1/Minimum_grad/Reshape#^gradients_1/Minimum_grad/Reshape_1
î
1gradients_1/Minimum_grad/tuple/control_dependencyIdentity gradients_1/Minimum_grad/Reshape*^gradients_1/Minimum_grad/tuple/group_deps*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*3
_class)
'%loc:@gradients_1/Minimum_grad/Reshape*
T0
ô
3gradients_1/Minimum_grad/tuple/control_dependency_1Identity"gradients_1/Minimum_grad/Reshape_1*^gradients_1/Minimum_grad/tuple/group_deps*
T0*5
_class+
)'loc:@gradients_1/Minimum_grad/Reshape_1*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
k
(gradients_1/pi/Mean_1_grad/Reshape/shapeConst*
valueB *
dtype0*
_output_shapes
: 
š
"gradients_1/pi/Mean_1_grad/ReshapeReshape1gradients_1/mul_5_grad/tuple/control_dependency_1(gradients_1/pi/Mean_1_grad/Reshape/shape*
_output_shapes
: *
T0*
Tshape0
c
 gradients_1/pi/Mean_1_grad/ConstConst*
valueB *
_output_shapes
: *
dtype0
 
gradients_1/pi/Mean_1_grad/TileTile"gradients_1/pi/Mean_1_grad/Reshape gradients_1/pi/Mean_1_grad/Const*

Tmultiples0*
T0*
_output_shapes
: 
g
"gradients_1/pi/Mean_1_grad/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *  ?

"gradients_1/pi/Mean_1_grad/truedivRealDivgradients_1/pi/Mean_1_grad/Tile"gradients_1/pi/Mean_1_grad/Const_1*
_output_shapes
: *
T0
_
gradients_1/mul_4_grad/ShapeShapeExp*
T0*
_output_shapes
:*
out_type0
k
gradients_1/mul_4_grad/Shape_1ShapePlaceholder_3*
_output_shapes
:*
T0*
out_type0
Ŕ
,gradients_1/mul_4_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/mul_4_grad/Shapegradients_1/mul_4_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0

gradients_1/mul_4_grad/MulMulgradients_1/Mean_2_grad/truedivPlaceholder_3*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ť
gradients_1/mul_4_grad/SumSumgradients_1/mul_4_grad/Mul,gradients_1/mul_4_grad/BroadcastGradientArgs*
	keep_dims( *
T0*
_output_shapes
:*

Tidx0

gradients_1/mul_4_grad/ReshapeReshapegradients_1/mul_4_grad/Sumgradients_1/mul_4_grad/Shape*
T0*
Tshape0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
w
gradients_1/mul_4_grad/Mul_1MulExpgradients_1/Mean_2_grad/truediv*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
ą
gradients_1/mul_4_grad/Sum_1Sumgradients_1/mul_4_grad/Mul_1.gradients_1/mul_4_grad/BroadcastGradientArgs:1*
	keep_dims( *
_output_shapes
:*

Tidx0*
T0
Ľ
 gradients_1/mul_4_grad/Reshape_1Reshapegradients_1/mul_4_grad/Sum_1gradients_1/mul_4_grad/Shape_1*
Tshape0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
s
'gradients_1/mul_4_grad/tuple/group_depsNoOp^gradients_1/mul_4_grad/Reshape!^gradients_1/mul_4_grad/Reshape_1
ć
/gradients_1/mul_4_grad/tuple/control_dependencyIdentitygradients_1/mul_4_grad/Reshape(^gradients_1/mul_4_grad/tuple/group_deps*1
_class'
%#loc:@gradients_1/mul_4_grad/Reshape*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
ě
1gradients_1/mul_4_grad/tuple/control_dependency_1Identity gradients_1/mul_4_grad/Reshape_1(^gradients_1/mul_4_grad/tuple/group_deps*3
_class)
'%loc:@gradients_1/mul_4_grad/Reshape_1*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
_
gradients_1/mul_3_grad/ShapeShapeExp*
out_type0*
_output_shapes
:*
T0
k
gradients_1/mul_3_grad/Shape_1ShapePlaceholder_2*
out_type0*
_output_shapes
:*
T0
Ŕ
,gradients_1/mul_3_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/mul_3_grad/Shapegradients_1/mul_3_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙

gradients_1/mul_3_grad/MulMul1gradients_1/Minimum_grad/tuple/control_dependencyPlaceholder_2*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ť
gradients_1/mul_3_grad/SumSumgradients_1/mul_3_grad/Mul,gradients_1/mul_3_grad/BroadcastGradientArgs*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:

gradients_1/mul_3_grad/ReshapeReshapegradients_1/mul_3_grad/Sumgradients_1/mul_3_grad/Shape*
Tshape0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

gradients_1/mul_3_grad/Mul_1MulExp1gradients_1/Minimum_grad/tuple/control_dependency*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
ą
gradients_1/mul_3_grad/Sum_1Sumgradients_1/mul_3_grad/Mul_1.gradients_1/mul_3_grad/BroadcastGradientArgs:1*
	keep_dims( *
_output_shapes
:*

Tidx0*
T0
Ľ
 gradients_1/mul_3_grad/Reshape_1Reshapegradients_1/mul_3_grad/Sum_1gradients_1/mul_3_grad/Shape_1*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
Tshape0
s
'gradients_1/mul_3_grad/tuple/group_depsNoOp^gradients_1/mul_3_grad/Reshape!^gradients_1/mul_3_grad/Reshape_1
ć
/gradients_1/mul_3_grad/tuple/control_dependencyIdentitygradients_1/mul_3_grad/Reshape(^gradients_1/mul_3_grad/tuple/group_deps*1
_class'
%#loc:@gradients_1/mul_3_grad/Reshape*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
ě
1gradients_1/mul_3_grad/tuple/control_dependency_1Identity gradients_1/mul_3_grad/Reshape_1(^gradients_1/mul_3_grad/tuple/group_deps*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*3
_class)
'%loc:@gradients_1/mul_3_grad/Reshape_1*
T0
i
gradients_1/pi/Sum_3_grad/ShapeConst*
dtype0*
_output_shapes
:*
valueB:

gradients_1/pi/Sum_3_grad/SizeConst*
value	B :*
dtype0*2
_class(
&$loc:@gradients_1/pi/Sum_3_grad/Shape*
_output_shapes
: 
ľ
gradients_1/pi/Sum_3_grad/addAddpi/Sum_3/reduction_indicesgradients_1/pi/Sum_3_grad/Size*
T0*2
_class(
&$loc:@gradients_1/pi/Sum_3_grad/Shape*
_output_shapes
: 
˝
gradients_1/pi/Sum_3_grad/modFloorModgradients_1/pi/Sum_3_grad/addgradients_1/pi/Sum_3_grad/Size*
T0*
_output_shapes
: *2
_class(
&$loc:@gradients_1/pi/Sum_3_grad/Shape

!gradients_1/pi/Sum_3_grad/Shape_1Const*
_output_shapes
: *
dtype0*
valueB *2
_class(
&$loc:@gradients_1/pi/Sum_3_grad/Shape

%gradients_1/pi/Sum_3_grad/range/startConst*
value	B : *2
_class(
&$loc:@gradients_1/pi/Sum_3_grad/Shape*
dtype0*
_output_shapes
: 

%gradients_1/pi/Sum_3_grad/range/deltaConst*2
_class(
&$loc:@gradients_1/pi/Sum_3_grad/Shape*
dtype0*
_output_shapes
: *
value	B :
ň
gradients_1/pi/Sum_3_grad/rangeRange%gradients_1/pi/Sum_3_grad/range/startgradients_1/pi/Sum_3_grad/Size%gradients_1/pi/Sum_3_grad/range/delta*

Tidx0*
_output_shapes
:*2
_class(
&$loc:@gradients_1/pi/Sum_3_grad/Shape

$gradients_1/pi/Sum_3_grad/Fill/valueConst*
dtype0*
_output_shapes
: *
value	B :*2
_class(
&$loc:@gradients_1/pi/Sum_3_grad/Shape
Ö
gradients_1/pi/Sum_3_grad/FillFill!gradients_1/pi/Sum_3_grad/Shape_1$gradients_1/pi/Sum_3_grad/Fill/value*
T0*

index_type0*
_output_shapes
: *2
_class(
&$loc:@gradients_1/pi/Sum_3_grad/Shape

'gradients_1/pi/Sum_3_grad/DynamicStitchDynamicStitchgradients_1/pi/Sum_3_grad/rangegradients_1/pi/Sum_3_grad/modgradients_1/pi/Sum_3_grad/Shapegradients_1/pi/Sum_3_grad/Fill*
T0*2
_class(
&$loc:@gradients_1/pi/Sum_3_grad/Shape*
N*
_output_shapes
:

#gradients_1/pi/Sum_3_grad/Maximum/yConst*2
_class(
&$loc:@gradients_1/pi/Sum_3_grad/Shape*
_output_shapes
: *
dtype0*
value	B :
Ó
!gradients_1/pi/Sum_3_grad/MaximumMaximum'gradients_1/pi/Sum_3_grad/DynamicStitch#gradients_1/pi/Sum_3_grad/Maximum/y*
T0*2
_class(
&$loc:@gradients_1/pi/Sum_3_grad/Shape*
_output_shapes
:
Ë
"gradients_1/pi/Sum_3_grad/floordivFloorDivgradients_1/pi/Sum_3_grad/Shape!gradients_1/pi/Sum_3_grad/Maximum*
T0*
_output_shapes
:*2
_class(
&$loc:@gradients_1/pi/Sum_3_grad/Shape
Ź
!gradients_1/pi/Sum_3_grad/ReshapeReshape"gradients_1/pi/Mean_1_grad/truediv'gradients_1/pi/Sum_3_grad/DynamicStitch*
Tshape0*
T0*
_output_shapes
:
¤
gradients_1/pi/Sum_3_grad/TileTile!gradients_1/pi/Sum_3_grad/Reshape"gradients_1/pi/Sum_3_grad/floordiv*
T0*

Tmultiples0*
_output_shapes
:
ć
gradients_1/AddN_1AddN/gradients_1/mul_4_grad/tuple/control_dependency/gradients_1/mul_3_grad/tuple/control_dependency*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*1
_class'
%#loc:@gradients_1/mul_4_grad/Reshape*
N*
T0
f
gradients_1/Exp_grad/mulMulgradients_1/AddN_1Exp*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
j
 gradients_1/pi/add_10_grad/ShapeConst*
dtype0*
_output_shapes
:*
valueB:
e
"gradients_1/pi/add_10_grad/Shape_1Const*
_output_shapes
: *
valueB *
dtype0
Ě
0gradients_1/pi/add_10_grad/BroadcastGradientArgsBroadcastGradientArgs gradients_1/pi/add_10_grad/Shape"gradients_1/pi/add_10_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0
š
gradients_1/pi/add_10_grad/SumSumgradients_1/pi/Sum_3_grad/Tile0gradients_1/pi/add_10_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
_output_shapes
:*
T0
˘
"gradients_1/pi/add_10_grad/ReshapeReshapegradients_1/pi/add_10_grad/Sum gradients_1/pi/add_10_grad/Shape*
T0*
_output_shapes
:*
Tshape0
š
 gradients_1/pi/add_10_grad/Sum_1Sumgradients_1/pi/Sum_3_grad/Tile2gradients_1/pi/add_10_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
_output_shapes
: *
T0
¤
$gradients_1/pi/add_10_grad/Reshape_1Reshape gradients_1/pi/add_10_grad/Sum_1"gradients_1/pi/add_10_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 

+gradients_1/pi/add_10_grad/tuple/group_depsNoOp#^gradients_1/pi/add_10_grad/Reshape%^gradients_1/pi/add_10_grad/Reshape_1
í
3gradients_1/pi/add_10_grad/tuple/control_dependencyIdentity"gradients_1/pi/add_10_grad/Reshape,^gradients_1/pi/add_10_grad/tuple/group_deps*
T0*5
_class+
)'loc:@gradients_1/pi/add_10_grad/Reshape*
_output_shapes
:
ď
5gradients_1/pi/add_10_grad/tuple/control_dependency_1Identity$gradients_1/pi/add_10_grad/Reshape_1,^gradients_1/pi/add_10_grad/tuple/group_deps*7
_class-
+)loc:@gradients_1/pi/add_10_grad/Reshape_1*
_output_shapes
: *
T0
b
gradients_1/sub_1_grad/ShapeShapepi/Sum*
out_type0*
_output_shapes
:*
T0
k
gradients_1/sub_1_grad/Shape_1ShapePlaceholder_6*
T0*
out_type0*
_output_shapes
:
Ŕ
,gradients_1/sub_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/sub_1_grad/Shapegradients_1/sub_1_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
Š
gradients_1/sub_1_grad/SumSumgradients_1/Exp_grad/mul,gradients_1/sub_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0

gradients_1/sub_1_grad/ReshapeReshapegradients_1/sub_1_grad/Sumgradients_1/sub_1_grad/Shape*
Tshape0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
­
gradients_1/sub_1_grad/Sum_1Sumgradients_1/Exp_grad/mul.gradients_1/sub_1_grad/BroadcastGradientArgs:1*

Tidx0*
T0*
_output_shapes
:*
	keep_dims( 
b
gradients_1/sub_1_grad/NegNeggradients_1/sub_1_grad/Sum_1*
_output_shapes
:*
T0
Ł
 gradients_1/sub_1_grad/Reshape_1Reshapegradients_1/sub_1_grad/Neggradients_1/sub_1_grad/Shape_1*
Tshape0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
s
'gradients_1/sub_1_grad/tuple/group_depsNoOp^gradients_1/sub_1_grad/Reshape!^gradients_1/sub_1_grad/Reshape_1
ć
/gradients_1/sub_1_grad/tuple/control_dependencyIdentitygradients_1/sub_1_grad/Reshape(^gradients_1/sub_1_grad/tuple/group_deps*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*1
_class'
%#loc:@gradients_1/sub_1_grad/Reshape
ě
1gradients_1/sub_1_grad/tuple/control_dependency_1Identity gradients_1/sub_1_grad/Reshape_1(^gradients_1/sub_1_grad/tuple/group_deps*3
_class)
'%loc:@gradients_1/sub_1_grad/Reshape_1*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
e
gradients_1/pi/Sum_grad/ShapeShapepi/mul_2*
T0*
_output_shapes
:*
out_type0

gradients_1/pi/Sum_grad/SizeConst*0
_class&
$"loc:@gradients_1/pi/Sum_grad/Shape*
value	B :*
_output_shapes
: *
dtype0
­
gradients_1/pi/Sum_grad/addAddpi/Sum/reduction_indicesgradients_1/pi/Sum_grad/Size*
_output_shapes
: *
T0*0
_class&
$"loc:@gradients_1/pi/Sum_grad/Shape
ľ
gradients_1/pi/Sum_grad/modFloorModgradients_1/pi/Sum_grad/addgradients_1/pi/Sum_grad/Size*
T0*
_output_shapes
: *0
_class&
$"loc:@gradients_1/pi/Sum_grad/Shape

gradients_1/pi/Sum_grad/Shape_1Const*
dtype0*0
_class&
$"loc:@gradients_1/pi/Sum_grad/Shape*
_output_shapes
: *
valueB 

#gradients_1/pi/Sum_grad/range/startConst*
value	B : *
_output_shapes
: *
dtype0*0
_class&
$"loc:@gradients_1/pi/Sum_grad/Shape

#gradients_1/pi/Sum_grad/range/deltaConst*
value	B :*
_output_shapes
: *
dtype0*0
_class&
$"loc:@gradients_1/pi/Sum_grad/Shape
č
gradients_1/pi/Sum_grad/rangeRange#gradients_1/pi/Sum_grad/range/startgradients_1/pi/Sum_grad/Size#gradients_1/pi/Sum_grad/range/delta*
_output_shapes
:*0
_class&
$"loc:@gradients_1/pi/Sum_grad/Shape*

Tidx0

"gradients_1/pi/Sum_grad/Fill/valueConst*
value	B :*
dtype0*0
_class&
$"loc:@gradients_1/pi/Sum_grad/Shape*
_output_shapes
: 
Î
gradients_1/pi/Sum_grad/FillFillgradients_1/pi/Sum_grad/Shape_1"gradients_1/pi/Sum_grad/Fill/value*0
_class&
$"loc:@gradients_1/pi/Sum_grad/Shape*
_output_shapes
: *

index_type0*
T0

%gradients_1/pi/Sum_grad/DynamicStitchDynamicStitchgradients_1/pi/Sum_grad/rangegradients_1/pi/Sum_grad/modgradients_1/pi/Sum_grad/Shapegradients_1/pi/Sum_grad/Fill*0
_class&
$"loc:@gradients_1/pi/Sum_grad/Shape*
_output_shapes
:*
N*
T0

!gradients_1/pi/Sum_grad/Maximum/yConst*
value	B :*
dtype0*0
_class&
$"loc:@gradients_1/pi/Sum_grad/Shape*
_output_shapes
: 
Ë
gradients_1/pi/Sum_grad/MaximumMaximum%gradients_1/pi/Sum_grad/DynamicStitch!gradients_1/pi/Sum_grad/Maximum/y*
T0*
_output_shapes
:*0
_class&
$"loc:@gradients_1/pi/Sum_grad/Shape
Ă
 gradients_1/pi/Sum_grad/floordivFloorDivgradients_1/pi/Sum_grad/Shapegradients_1/pi/Sum_grad/Maximum*
_output_shapes
:*
T0*0
_class&
$"loc:@gradients_1/pi/Sum_grad/Shape
Ë
gradients_1/pi/Sum_grad/ReshapeReshape/gradients_1/sub_1_grad/tuple/control_dependency%gradients_1/pi/Sum_grad/DynamicStitch*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
T0*
Tshape0
Ť
gradients_1/pi/Sum_grad/TileTilegradients_1/pi/Sum_grad/Reshape gradients_1/pi/Sum_grad/floordiv*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*

Tmultiples0*
T0
b
gradients_1/pi/mul_2_grad/ShapeConst*
_output_shapes
: *
dtype0*
valueB 
i
!gradients_1/pi/mul_2_grad/Shape_1Shapepi/add_3*
T0*
out_type0*
_output_shapes
:
É
/gradients_1/pi/mul_2_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/pi/mul_2_grad/Shape!gradients_1/pi/mul_2_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0
~
gradients_1/pi/mul_2_grad/MulMulgradients_1/pi/Sum_grad/Tilepi/add_3*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
´
gradients_1/pi/mul_2_grad/SumSumgradients_1/pi/mul_2_grad/Mul/gradients_1/pi/mul_2_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0

!gradients_1/pi/mul_2_grad/ReshapeReshapegradients_1/pi/mul_2_grad/Sumgradients_1/pi/mul_2_grad/Shape*
Tshape0*
_output_shapes
: *
T0

gradients_1/pi/mul_2_grad/Mul_1Mul
pi/mul_2/xgradients_1/pi/Sum_grad/Tile*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
ş
gradients_1/pi/mul_2_grad/Sum_1Sumgradients_1/pi/mul_2_grad/Mul_11gradients_1/pi/mul_2_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( *
_output_shapes
:
˛
#gradients_1/pi/mul_2_grad/Reshape_1Reshapegradients_1/pi/mul_2_grad/Sum_1!gradients_1/pi/mul_2_grad/Shape_1*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
Tshape0*
T0
|
*gradients_1/pi/mul_2_grad/tuple/group_depsNoOp"^gradients_1/pi/mul_2_grad/Reshape$^gradients_1/pi/mul_2_grad/Reshape_1
ĺ
2gradients_1/pi/mul_2_grad/tuple/control_dependencyIdentity!gradients_1/pi/mul_2_grad/Reshape+^gradients_1/pi/mul_2_grad/tuple/group_deps*4
_class*
(&loc:@gradients_1/pi/mul_2_grad/Reshape*
T0*
_output_shapes
: 
ü
4gradients_1/pi/mul_2_grad/tuple/control_dependency_1Identity#gradients_1/pi/mul_2_grad/Reshape_1+^gradients_1/pi/mul_2_grad/tuple/group_deps*6
_class,
*(loc:@gradients_1/pi/mul_2_grad/Reshape_1*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
g
gradients_1/pi/add_3_grad/ShapeShapepi/add_2*
out_type0*
T0*
_output_shapes
:
d
!gradients_1/pi/add_3_grad/Shape_1Const*
dtype0*
_output_shapes
: *
valueB 
É
/gradients_1/pi/add_3_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/pi/add_3_grad/Shape!gradients_1/pi/add_3_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
Ë
gradients_1/pi/add_3_grad/SumSum4gradients_1/pi/mul_2_grad/tuple/control_dependency_1/gradients_1/pi/add_3_grad/BroadcastGradientArgs*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
Ź
!gradients_1/pi/add_3_grad/ReshapeReshapegradients_1/pi/add_3_grad/Sumgradients_1/pi/add_3_grad/Shape*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
Tshape0
Ď
gradients_1/pi/add_3_grad/Sum_1Sum4gradients_1/pi/mul_2_grad/tuple/control_dependency_11gradients_1/pi/add_3_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*
_output_shapes
:*

Tidx0
Ą
#gradients_1/pi/add_3_grad/Reshape_1Reshapegradients_1/pi/add_3_grad/Sum_1!gradients_1/pi/add_3_grad/Shape_1*
_output_shapes
: *
Tshape0*
T0
|
*gradients_1/pi/add_3_grad/tuple/group_depsNoOp"^gradients_1/pi/add_3_grad/Reshape$^gradients_1/pi/add_3_grad/Reshape_1
ö
2gradients_1/pi/add_3_grad/tuple/control_dependencyIdentity!gradients_1/pi/add_3_grad/Reshape+^gradients_1/pi/add_3_grad/tuple/group_deps*
T0*4
_class*
(&loc:@gradients_1/pi/add_3_grad/Reshape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
ë
4gradients_1/pi/add_3_grad/tuple/control_dependency_1Identity#gradients_1/pi/add_3_grad/Reshape_1+^gradients_1/pi/add_3_grad/tuple/group_deps*6
_class,
*(loc:@gradients_1/pi/add_3_grad/Reshape_1*
T0*
_output_shapes
: 
e
gradients_1/pi/add_2_grad/ShapeShapepi/pow*
T0*
out_type0*
_output_shapes
:
k
!gradients_1/pi/add_2_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB:
É
/gradients_1/pi/add_2_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/pi/add_2_grad/Shape!gradients_1/pi/add_2_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
É
gradients_1/pi/add_2_grad/SumSum2gradients_1/pi/add_3_grad/tuple/control_dependency/gradients_1/pi/add_2_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
Ź
!gradients_1/pi/add_2_grad/ReshapeReshapegradients_1/pi/add_2_grad/Sumgradients_1/pi/add_2_grad/Shape*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
Tshape0
Í
gradients_1/pi/add_2_grad/Sum_1Sum2gradients_1/pi/add_3_grad/tuple/control_dependency1gradients_1/pi/add_2_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
Ľ
#gradients_1/pi/add_2_grad/Reshape_1Reshapegradients_1/pi/add_2_grad/Sum_1!gradients_1/pi/add_2_grad/Shape_1*
_output_shapes
:*
Tshape0*
T0
|
*gradients_1/pi/add_2_grad/tuple/group_depsNoOp"^gradients_1/pi/add_2_grad/Reshape$^gradients_1/pi/add_2_grad/Reshape_1
ö
2gradients_1/pi/add_2_grad/tuple/control_dependencyIdentity!gradients_1/pi/add_2_grad/Reshape+^gradients_1/pi/add_2_grad/tuple/group_deps*4
_class*
(&loc:@gradients_1/pi/add_2_grad/Reshape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
ď
4gradients_1/pi/add_2_grad/tuple/control_dependency_1Identity#gradients_1/pi/add_2_grad/Reshape_1+^gradients_1/pi/add_2_grad/tuple/group_deps*6
_class,
*(loc:@gradients_1/pi/add_2_grad/Reshape_1*
T0*
_output_shapes
:
g
gradients_1/pi/pow_grad/ShapeShape
pi/truediv*
out_type0*
_output_shapes
:*
T0
b
gradients_1/pi/pow_grad/Shape_1Const*
dtype0*
_output_shapes
: *
valueB 
Ă
-gradients_1/pi/pow_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/pi/pow_grad/Shapegradients_1/pi/pow_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0

gradients_1/pi/pow_grad/mulMul2gradients_1/pi/add_2_grad/tuple/control_dependencypi/pow/y*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
b
gradients_1/pi/pow_grad/sub/yConst*
valueB
 *  ?*
_output_shapes
: *
dtype0
l
gradients_1/pi/pow_grad/subSubpi/pow/ygradients_1/pi/pow_grad/sub/y*
_output_shapes
: *
T0
}
gradients_1/pi/pow_grad/PowPow
pi/truedivgradients_1/pi/pow_grad/sub*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

gradients_1/pi/pow_grad/mul_1Mulgradients_1/pi/pow_grad/mulgradients_1/pi/pow_grad/Pow*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
°
gradients_1/pi/pow_grad/SumSumgradients_1/pi/pow_grad/mul_1-gradients_1/pi/pow_grad/BroadcastGradientArgs*

Tidx0*
_output_shapes
:*
	keep_dims( *
T0
Ś
gradients_1/pi/pow_grad/ReshapeReshapegradients_1/pi/pow_grad/Sumgradients_1/pi/pow_grad/Shape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
Tshape0
f
!gradients_1/pi/pow_grad/Greater/yConst*
dtype0*
valueB
 *    *
_output_shapes
: 

gradients_1/pi/pow_grad/GreaterGreater
pi/truediv!gradients_1/pi/pow_grad/Greater/y*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
q
'gradients_1/pi/pow_grad/ones_like/ShapeShape
pi/truediv*
_output_shapes
:*
out_type0*
T0
l
'gradients_1/pi/pow_grad/ones_like/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *  ?
ż
!gradients_1/pi/pow_grad/ones_likeFill'gradients_1/pi/pow_grad/ones_like/Shape'gradients_1/pi/pow_grad/ones_like/Const*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*

index_type0
Ş
gradients_1/pi/pow_grad/SelectSelectgradients_1/pi/pow_grad/Greater
pi/truediv!gradients_1/pi/pow_grad/ones_like*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
t
gradients_1/pi/pow_grad/LogLoggradients_1/pi/pow_grad/Select*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
m
"gradients_1/pi/pow_grad/zeros_like	ZerosLike
pi/truediv*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
ž
 gradients_1/pi/pow_grad/Select_1Selectgradients_1/pi/pow_grad/Greatergradients_1/pi/pow_grad/Log"gradients_1/pi/pow_grad/zeros_like*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

gradients_1/pi/pow_grad/mul_2Mul2gradients_1/pi/add_2_grad/tuple/control_dependencypi/pow*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

gradients_1/pi/pow_grad/mul_3Mulgradients_1/pi/pow_grad/mul_2 gradients_1/pi/pow_grad/Select_1*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
´
gradients_1/pi/pow_grad/Sum_1Sumgradients_1/pi/pow_grad/mul_3/gradients_1/pi/pow_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 

!gradients_1/pi/pow_grad/Reshape_1Reshapegradients_1/pi/pow_grad/Sum_1gradients_1/pi/pow_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
v
(gradients_1/pi/pow_grad/tuple/group_depsNoOp ^gradients_1/pi/pow_grad/Reshape"^gradients_1/pi/pow_grad/Reshape_1
î
0gradients_1/pi/pow_grad/tuple/control_dependencyIdentitygradients_1/pi/pow_grad/Reshape)^gradients_1/pi/pow_grad/tuple/group_deps*2
_class(
&$loc:@gradients_1/pi/pow_grad/Reshape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
ă
2gradients_1/pi/pow_grad/tuple/control_dependency_1Identity!gradients_1/pi/pow_grad/Reshape_1)^gradients_1/pi/pow_grad/tuple/group_deps*
_output_shapes
: *
T0*4
_class*
(&loc:@gradients_1/pi/pow_grad/Reshape_1
b
gradients_1/pi/mul_1_grad/ShapeConst*
dtype0*
_output_shapes
: *
valueB 
k
!gradients_1/pi/mul_1_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:
É
/gradients_1/pi/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/pi/mul_1_grad/Shape!gradients_1/pi/mul_1_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0

gradients_1/pi/mul_1_grad/MulMul4gradients_1/pi/add_2_grad/tuple/control_dependency_1pi/log_std/read*
_output_shapes
:*
T0
˛
gradients_1/pi/mul_1_grad/SumSumgradients_1/pi/mul_1_grad/Mul/gradients_1/pi/mul_1_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( *
_output_shapes
: 

!gradients_1/pi/mul_1_grad/ReshapeReshapegradients_1/pi/mul_1_grad/Sumgradients_1/pi/mul_1_grad/Shape*
Tshape0*
T0*
_output_shapes
: 

gradients_1/pi/mul_1_grad/Mul_1Mul
pi/mul_1/x4gradients_1/pi/add_2_grad/tuple/control_dependency_1*
_output_shapes
:*
T0
ź
gradients_1/pi/mul_1_grad/Sum_1Sumgradients_1/pi/mul_1_grad/Mul_11gradients_1/pi/mul_1_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
Ľ
#gradients_1/pi/mul_1_grad/Reshape_1Reshapegradients_1/pi/mul_1_grad/Sum_1!gradients_1/pi/mul_1_grad/Shape_1*
Tshape0*
T0*
_output_shapes
:
|
*gradients_1/pi/mul_1_grad/tuple/group_depsNoOp"^gradients_1/pi/mul_1_grad/Reshape$^gradients_1/pi/mul_1_grad/Reshape_1
ĺ
2gradients_1/pi/mul_1_grad/tuple/control_dependencyIdentity!gradients_1/pi/mul_1_grad/Reshape+^gradients_1/pi/mul_1_grad/tuple/group_deps*
_output_shapes
: *
T0*4
_class*
(&loc:@gradients_1/pi/mul_1_grad/Reshape
ď
4gradients_1/pi/mul_1_grad/tuple/control_dependency_1Identity#gradients_1/pi/mul_1_grad/Reshape_1+^gradients_1/pi/mul_1_grad/tuple/group_deps*
T0*
_output_shapes
:*6
_class,
*(loc:@gradients_1/pi/mul_1_grad/Reshape_1
g
!gradients_1/pi/truediv_grad/ShapeShapepi/sub*
out_type0*
_output_shapes
:*
T0
m
#gradients_1/pi/truediv_grad/Shape_1Const*
valueB:*
_output_shapes
:*
dtype0
Ď
1gradients_1/pi/truediv_grad/BroadcastGradientArgsBroadcastGradientArgs!gradients_1/pi/truediv_grad/Shape#gradients_1/pi/truediv_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0

#gradients_1/pi/truediv_grad/RealDivRealDiv0gradients_1/pi/pow_grad/tuple/control_dependencypi/add_1*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
ž
gradients_1/pi/truediv_grad/SumSum#gradients_1/pi/truediv_grad/RealDiv1gradients_1/pi/truediv_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
_output_shapes
:*
T0
˛
#gradients_1/pi/truediv_grad/ReshapeReshapegradients_1/pi/truediv_grad/Sum!gradients_1/pi/truediv_grad/Shape*
T0*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
`
gradients_1/pi/truediv_grad/NegNegpi/sub*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

%gradients_1/pi/truediv_grad/RealDiv_1RealDivgradients_1/pi/truediv_grad/Negpi/add_1*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

%gradients_1/pi/truediv_grad/RealDiv_2RealDiv%gradients_1/pi/truediv_grad/RealDiv_1pi/add_1*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
ą
gradients_1/pi/truediv_grad/mulMul0gradients_1/pi/pow_grad/tuple/control_dependency%gradients_1/pi/truediv_grad/RealDiv_2*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
ž
!gradients_1/pi/truediv_grad/Sum_1Sumgradients_1/pi/truediv_grad/mul3gradients_1/pi/truediv_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
_output_shapes
:*
T0
Ť
%gradients_1/pi/truediv_grad/Reshape_1Reshape!gradients_1/pi/truediv_grad/Sum_1#gradients_1/pi/truediv_grad/Shape_1*
T0*
_output_shapes
:*
Tshape0

,gradients_1/pi/truediv_grad/tuple/group_depsNoOp$^gradients_1/pi/truediv_grad/Reshape&^gradients_1/pi/truediv_grad/Reshape_1
ţ
4gradients_1/pi/truediv_grad/tuple/control_dependencyIdentity#gradients_1/pi/truediv_grad/Reshape-^gradients_1/pi/truediv_grad/tuple/group_deps*6
_class,
*(loc:@gradients_1/pi/truediv_grad/Reshape*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
÷
6gradients_1/pi/truediv_grad/tuple/control_dependency_1Identity%gradients_1/pi/truediv_grad/Reshape_1-^gradients_1/pi/truediv_grad/tuple/group_deps*
T0*
_output_shapes
:*8
_class.
,*loc:@gradients_1/pi/truediv_grad/Reshape_1
j
gradients_1/pi/sub_grad/ShapeShapePlaceholder_1*
out_type0*
T0*
_output_shapes
:
q
gradients_1/pi/sub_grad/Shape_1Shapepi/dense_2/BiasAdd*
_output_shapes
:*
out_type0*
T0
Ă
-gradients_1/pi/sub_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/pi/sub_grad/Shapegradients_1/pi/sub_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
Ç
gradients_1/pi/sub_grad/SumSum4gradients_1/pi/truediv_grad/tuple/control_dependency-gradients_1/pi/sub_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
_output_shapes
:*
T0
Ś
gradients_1/pi/sub_grad/ReshapeReshapegradients_1/pi/sub_grad/Sumgradients_1/pi/sub_grad/Shape*
T0*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ë
gradients_1/pi/sub_grad/Sum_1Sum4gradients_1/pi/truediv_grad/tuple/control_dependency/gradients_1/pi/sub_grad/BroadcastGradientArgs:1*
T0*
	keep_dims( *

Tidx0*
_output_shapes
:
d
gradients_1/pi/sub_grad/NegNeggradients_1/pi/sub_grad/Sum_1*
T0*
_output_shapes
:
Ş
!gradients_1/pi/sub_grad/Reshape_1Reshapegradients_1/pi/sub_grad/Neggradients_1/pi/sub_grad/Shape_1*
Tshape0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
v
(gradients_1/pi/sub_grad/tuple/group_depsNoOp ^gradients_1/pi/sub_grad/Reshape"^gradients_1/pi/sub_grad/Reshape_1
î
0gradients_1/pi/sub_grad/tuple/control_dependencyIdentitygradients_1/pi/sub_grad/Reshape)^gradients_1/pi/sub_grad/tuple/group_deps*2
_class(
&$loc:@gradients_1/pi/sub_grad/Reshape*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
ô
2gradients_1/pi/sub_grad/tuple/control_dependency_1Identity!gradients_1/pi/sub_grad/Reshape_1)^gradients_1/pi/sub_grad/tuple/group_deps*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*4
_class*
(&loc:@gradients_1/pi/sub_grad/Reshape_1
i
gradients_1/pi/add_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:
d
!gradients_1/pi/add_1_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
É
/gradients_1/pi/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/pi/add_1_grad/Shape!gradients_1/pi/add_1_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0
Ď
gradients_1/pi/add_1_grad/SumSum6gradients_1/pi/truediv_grad/tuple/control_dependency_1/gradients_1/pi/add_1_grad/BroadcastGradientArgs*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:

!gradients_1/pi/add_1_grad/ReshapeReshapegradients_1/pi/add_1_grad/Sumgradients_1/pi/add_1_grad/Shape*
T0*
_output_shapes
:*
Tshape0
Ď
gradients_1/pi/add_1_grad/Sum_1Sum6gradients_1/pi/truediv_grad/tuple/control_dependency_11gradients_1/pi/add_1_grad/BroadcastGradientArgs:1*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
Ą
#gradients_1/pi/add_1_grad/Reshape_1Reshapegradients_1/pi/add_1_grad/Sum_1!gradients_1/pi/add_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
|
*gradients_1/pi/add_1_grad/tuple/group_depsNoOp"^gradients_1/pi/add_1_grad/Reshape$^gradients_1/pi/add_1_grad/Reshape_1
é
2gradients_1/pi/add_1_grad/tuple/control_dependencyIdentity!gradients_1/pi/add_1_grad/Reshape+^gradients_1/pi/add_1_grad/tuple/group_deps*4
_class*
(&loc:@gradients_1/pi/add_1_grad/Reshape*
T0*
_output_shapes
:
ë
4gradients_1/pi/add_1_grad/tuple/control_dependency_1Identity#gradients_1/pi/add_1_grad/Reshape_1+^gradients_1/pi/add_1_grad/tuple/group_deps*
_output_shapes
: *
T0*6
_class,
*(loc:@gradients_1/pi/add_1_grad/Reshape_1
Ž
/gradients_1/pi/dense_2/BiasAdd_grad/BiasAddGradBiasAddGrad2gradients_1/pi/sub_grad/tuple/control_dependency_1*
_output_shapes
:*
data_formatNHWC*
T0
Ł
4gradients_1/pi/dense_2/BiasAdd_grad/tuple/group_depsNoOp0^gradients_1/pi/dense_2/BiasAdd_grad/BiasAddGrad3^gradients_1/pi/sub_grad/tuple/control_dependency_1

<gradients_1/pi/dense_2/BiasAdd_grad/tuple/control_dependencyIdentity2gradients_1/pi/sub_grad/tuple/control_dependency_15^gradients_1/pi/dense_2/BiasAdd_grad/tuple/group_deps*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*4
_class*
(&loc:@gradients_1/pi/sub_grad/Reshape_1

>gradients_1/pi/dense_2/BiasAdd_grad/tuple/control_dependency_1Identity/gradients_1/pi/dense_2/BiasAdd_grad/BiasAddGrad5^gradients_1/pi/dense_2/BiasAdd_grad/tuple/group_deps*
T0*B
_class8
64loc:@gradients_1/pi/dense_2/BiasAdd_grad/BiasAddGrad*
_output_shapes
:

gradients_1/pi/Exp_1_grad/mulMul2gradients_1/pi/add_1_grad/tuple/control_dependencypi/Exp_1*
_output_shapes
:*
T0
â
)gradients_1/pi/dense_2/MatMul_grad/MatMulMatMul<gradients_1/pi/dense_2/BiasAdd_grad/tuple/control_dependencypi/dense_2/kernel/read*
transpose_b(*
T0*
transpose_a( *(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ô
+gradients_1/pi/dense_2/MatMul_grad/MatMul_1MatMulpi/dense_1/Tanh<gradients_1/pi/dense_2/BiasAdd_grad/tuple/control_dependency*
T0*
_output_shapes
:	*
transpose_b( *
transpose_a(

3gradients_1/pi/dense_2/MatMul_grad/tuple/group_depsNoOp*^gradients_1/pi/dense_2/MatMul_grad/MatMul,^gradients_1/pi/dense_2/MatMul_grad/MatMul_1

;gradients_1/pi/dense_2/MatMul_grad/tuple/control_dependencyIdentity)gradients_1/pi/dense_2/MatMul_grad/MatMul4^gradients_1/pi/dense_2/MatMul_grad/tuple/group_deps*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*<
_class2
0.loc:@gradients_1/pi/dense_2/MatMul_grad/MatMul

=gradients_1/pi/dense_2/MatMul_grad/tuple/control_dependency_1Identity+gradients_1/pi/dense_2/MatMul_grad/MatMul_14^gradients_1/pi/dense_2/MatMul_grad/tuple/group_deps*
_output_shapes
:	*
T0*>
_class4
20loc:@gradients_1/pi/dense_2/MatMul_grad/MatMul_1

gradients_1/AddN_2AddN3gradients_1/pi/add_10_grad/tuple/control_dependency4gradients_1/pi/mul_1_grad/tuple/control_dependency_1gradients_1/pi/Exp_1_grad/mul*
_output_shapes
:*5
_class+
)'loc:@gradients_1/pi/add_10_grad/Reshape*
T0*
N
ś
)gradients_1/pi/dense_1/Tanh_grad/TanhGradTanhGradpi/dense_1/Tanh;gradients_1/pi/dense_2/MatMul_grad/tuple/control_dependency*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ś
/gradients_1/pi/dense_1/BiasAdd_grad/BiasAddGradBiasAddGrad)gradients_1/pi/dense_1/Tanh_grad/TanhGrad*
_output_shapes	
:*
data_formatNHWC*
T0

4gradients_1/pi/dense_1/BiasAdd_grad/tuple/group_depsNoOp0^gradients_1/pi/dense_1/BiasAdd_grad/BiasAddGrad*^gradients_1/pi/dense_1/Tanh_grad/TanhGrad

<gradients_1/pi/dense_1/BiasAdd_grad/tuple/control_dependencyIdentity)gradients_1/pi/dense_1/Tanh_grad/TanhGrad5^gradients_1/pi/dense_1/BiasAdd_grad/tuple/group_deps*
T0*<
_class2
0.loc:@gradients_1/pi/dense_1/Tanh_grad/TanhGrad*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

>gradients_1/pi/dense_1/BiasAdd_grad/tuple/control_dependency_1Identity/gradients_1/pi/dense_1/BiasAdd_grad/BiasAddGrad5^gradients_1/pi/dense_1/BiasAdd_grad/tuple/group_deps*B
_class8
64loc:@gradients_1/pi/dense_1/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:*
T0
â
)gradients_1/pi/dense_1/MatMul_grad/MatMulMatMul<gradients_1/pi/dense_1/BiasAdd_grad/tuple/control_dependencypi/dense_1/kernel/read*
T0*
transpose_b(*
transpose_a( *(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ó
+gradients_1/pi/dense_1/MatMul_grad/MatMul_1MatMulpi/dense/Tanh<gradients_1/pi/dense_1/BiasAdd_grad/tuple/control_dependency*
transpose_a(*
T0*
transpose_b( * 
_output_shapes
:


3gradients_1/pi/dense_1/MatMul_grad/tuple/group_depsNoOp*^gradients_1/pi/dense_1/MatMul_grad/MatMul,^gradients_1/pi/dense_1/MatMul_grad/MatMul_1

;gradients_1/pi/dense_1/MatMul_grad/tuple/control_dependencyIdentity)gradients_1/pi/dense_1/MatMul_grad/MatMul4^gradients_1/pi/dense_1/MatMul_grad/tuple/group_deps*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*<
_class2
0.loc:@gradients_1/pi/dense_1/MatMul_grad/MatMul

=gradients_1/pi/dense_1/MatMul_grad/tuple/control_dependency_1Identity+gradients_1/pi/dense_1/MatMul_grad/MatMul_14^gradients_1/pi/dense_1/MatMul_grad/tuple/group_deps* 
_output_shapes
:
*
T0*>
_class4
20loc:@gradients_1/pi/dense_1/MatMul_grad/MatMul_1
˛
'gradients_1/pi/dense/Tanh_grad/TanhGradTanhGradpi/dense/Tanh;gradients_1/pi/dense_1/MatMul_grad/tuple/control_dependency*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
˘
-gradients_1/pi/dense/BiasAdd_grad/BiasAddGradBiasAddGrad'gradients_1/pi/dense/Tanh_grad/TanhGrad*
T0*
data_formatNHWC*
_output_shapes	
:

2gradients_1/pi/dense/BiasAdd_grad/tuple/group_depsNoOp.^gradients_1/pi/dense/BiasAdd_grad/BiasAddGrad(^gradients_1/pi/dense/Tanh_grad/TanhGrad

:gradients_1/pi/dense/BiasAdd_grad/tuple/control_dependencyIdentity'gradients_1/pi/dense/Tanh_grad/TanhGrad3^gradients_1/pi/dense/BiasAdd_grad/tuple/group_deps*:
_class0
.,loc:@gradients_1/pi/dense/Tanh_grad/TanhGrad*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

<gradients_1/pi/dense/BiasAdd_grad/tuple/control_dependency_1Identity-gradients_1/pi/dense/BiasAdd_grad/BiasAddGrad3^gradients_1/pi/dense/BiasAdd_grad/tuple/group_deps*
_output_shapes	
:*@
_class6
42loc:@gradients_1/pi/dense/BiasAdd_grad/BiasAddGrad*
T0
Ű
'gradients_1/pi/dense/MatMul_grad/MatMulMatMul:gradients_1/pi/dense/BiasAdd_grad/tuple/control_dependencypi/dense/kernel/read*
transpose_a( *
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙L*
transpose_b(
Ě
)gradients_1/pi/dense/MatMul_grad/MatMul_1MatMulPlaceholder:gradients_1/pi/dense/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
T0*
transpose_a(*
_output_shapes
:	L

1gradients_1/pi/dense/MatMul_grad/tuple/group_depsNoOp(^gradients_1/pi/dense/MatMul_grad/MatMul*^gradients_1/pi/dense/MatMul_grad/MatMul_1

9gradients_1/pi/dense/MatMul_grad/tuple/control_dependencyIdentity'gradients_1/pi/dense/MatMul_grad/MatMul2^gradients_1/pi/dense/MatMul_grad/tuple/group_deps*'
_output_shapes
:˙˙˙˙˙˙˙˙˙L*:
_class0
.,loc:@gradients_1/pi/dense/MatMul_grad/MatMul*
T0

;gradients_1/pi/dense/MatMul_grad/tuple/control_dependency_1Identity)gradients_1/pi/dense/MatMul_grad/MatMul_12^gradients_1/pi/dense/MatMul_grad/tuple/group_deps*
_output_shapes
:	L*
T0*<
_class2
0.loc:@gradients_1/pi/dense/MatMul_grad/MatMul_1
b
Reshape_4/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:

	Reshape_4Reshape;gradients_1/pi/dense/MatMul_grad/tuple/control_dependency_1Reshape_4/shape*
_output_shapes

:*
Tshape0*
T0
b
Reshape_5/shapeConst*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0

	Reshape_5Reshape<gradients_1/pi/dense/BiasAdd_grad/tuple/control_dependency_1Reshape_5/shape*
Tshape0*
T0*
_output_shapes	
:
b
Reshape_6/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:

	Reshape_6Reshape=gradients_1/pi/dense_1/MatMul_grad/tuple/control_dependency_1Reshape_6/shape*
_output_shapes

:*
Tshape0*
T0
b
Reshape_7/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
_output_shapes
:*
dtype0

	Reshape_7Reshape>gradients_1/pi/dense_1/BiasAdd_grad/tuple/control_dependency_1Reshape_7/shape*
Tshape0*
_output_shapes	
:*
T0
b
Reshape_8/shapeConst*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙*
_output_shapes
:

	Reshape_8Reshape=gradients_1/pi/dense_2/MatMul_grad/tuple/control_dependency_1Reshape_8/shape*
T0*
Tshape0*
_output_shapes	
:
b
Reshape_9/shapeConst*
dtype0*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙

	Reshape_9Reshape>gradients_1/pi/dense_2/BiasAdd_grad/tuple/control_dependency_1Reshape_9/shape*
T0*
Tshape0*
_output_shapes
:
c
Reshape_10/shapeConst*
dtype0*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙
n

Reshape_10Reshapegradients_1/AddN_2Reshape_10/shape*
_output_shapes
:*
Tshape0*
T0
c
Reshape_11/shapeConst*
dtype0*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙


Reshape_11Reshape&gradients_1/Softplus_grad/SoftplusGradReshape_11/shape*
_output_shapes
:*
T0*
Tshape0
O
concat_2/axisConst*
value	B : *
dtype0*
_output_shapes
: 
š
concat_2ConcatV2	Reshape_4	Reshape_5	Reshape_6	Reshape_7	Reshape_8	Reshape_9
Reshape_10
Reshape_11concat_2/axis*
T0*
_output_shapes

: *
N*

Tidx0
l
PyFunc_2PyFuncconcat_2*
Tout
2*
token
pyfunc_2*
Tin
2*
_output_shapes

: 
p
Const_6Const*5
value,B*"  L                       *
_output_shapes
:*
dtype0
S
split_2/split_dimConst*
dtype0*
_output_shapes
: *
value	B : 
Ş
split_2SplitVPyFunc_2Const_6split_2/split_dim*K
_output_shapes9
7::::::::*
	num_split*
T0*

Tlen0
a
Reshape_12/shapeConst*
_output_shapes
:*
valueB"L      *
dtype0
h

Reshape_12Reshapesplit_2Reshape_12/shape*
T0*
_output_shapes
:	L*
Tshape0
[
Reshape_13/shapeConst*
_output_shapes
:*
valueB:*
dtype0
f

Reshape_13Reshape	split_2:1Reshape_13/shape*
Tshape0*
_output_shapes	
:*
T0
a
Reshape_14/shapeConst*
_output_shapes
:*
valueB"      *
dtype0
k

Reshape_14Reshape	split_2:2Reshape_14/shape*
Tshape0* 
_output_shapes
:
*
T0
[
Reshape_15/shapeConst*
dtype0*
valueB:*
_output_shapes
:
f

Reshape_15Reshape	split_2:3Reshape_15/shape*
_output_shapes	
:*
T0*
Tshape0
a
Reshape_16/shapeConst*
dtype0*
valueB"      *
_output_shapes
:
j

Reshape_16Reshape	split_2:4Reshape_16/shape*
T0*
Tshape0*
_output_shapes
:	
Z
Reshape_17/shapeConst*
_output_shapes
:*
dtype0*
valueB:
e

Reshape_17Reshape	split_2:5Reshape_17/shape*
Tshape0*
T0*
_output_shapes
:
Z
Reshape_18/shapeConst*
valueB:*
dtype0*
_output_shapes
:
e

Reshape_18Reshape	split_2:6Reshape_18/shape*
Tshape0*
_output_shapes
:*
T0
S
Reshape_19/shapeConst*
valueB *
dtype0*
_output_shapes
: 
a

Reshape_19Reshape	split_2:7Reshape_19/shape*
_output_shapes
: *
T0*
Tshape0

beta1_power_1/initial_valueConst*
valueB
 *fff?*
dtype0*
_output_shapes
: *(
_class
loc:@penalty/penalty_param

beta1_power_1
VariableV2*
_output_shapes
: *
shape: *
shared_name *
	container *
dtype0*(
_class
loc:@penalty/penalty_param
ž
beta1_power_1/AssignAssignbeta1_power_1beta1_power_1/initial_value*
T0*
use_locking(*
validate_shape(*(
_class
loc:@penalty/penalty_param*
_output_shapes
: 
x
beta1_power_1/readIdentitybeta1_power_1*
_output_shapes
: *
T0*(
_class
loc:@penalty/penalty_param

beta2_power_1/initial_valueConst*
_output_shapes
: *
valueB
 *wž?*
dtype0*(
_class
loc:@penalty/penalty_param

beta2_power_1
VariableV2*
dtype0*
shared_name *(
_class
loc:@penalty/penalty_param*
shape: *
_output_shapes
: *
	container 
ž
beta2_power_1/AssignAssignbeta2_power_1beta2_power_1/initial_value*
use_locking(*
validate_shape(*
T0*(
_class
loc:@penalty/penalty_param*
_output_shapes
: 
x
beta2_power_1/readIdentitybeta2_power_1*(
_class
loc:@penalty/penalty_param*
T0*
_output_shapes
: 
Ť
6pi/dense/kernel/Adam/Initializer/zeros/shape_as_tensorConst*
valueB"L      *
_output_shapes
:*
dtype0*"
_class
loc:@pi/dense/kernel

,pi/dense/kernel/Adam/Initializer/zeros/ConstConst*
dtype0*"
_class
loc:@pi/dense/kernel*
_output_shapes
: *
valueB
 *    
ô
&pi/dense/kernel/Adam/Initializer/zerosFill6pi/dense/kernel/Adam/Initializer/zeros/shape_as_tensor,pi/dense/kernel/Adam/Initializer/zeros/Const*
_output_shapes
:	L*

index_type0*"
_class
loc:@pi/dense/kernel*
T0
Ž
pi/dense/kernel/Adam
VariableV2*
shared_name *"
_class
loc:@pi/dense/kernel*
	container *
shape:	L*
_output_shapes
:	L*
dtype0
Ú
pi/dense/kernel/Adam/AssignAssignpi/dense/kernel/Adam&pi/dense/kernel/Adam/Initializer/zeros*
validate_shape(*
T0*
use_locking(*"
_class
loc:@pi/dense/kernel*
_output_shapes
:	L

pi/dense/kernel/Adam/readIdentitypi/dense/kernel/Adam*
T0*"
_class
loc:@pi/dense/kernel*
_output_shapes
:	L
­
8pi/dense/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*
valueB"L      *"
_class
loc:@pi/dense/kernel

.pi/dense/kernel/Adam_1/Initializer/zeros/ConstConst*"
_class
loc:@pi/dense/kernel*
_output_shapes
: *
valueB
 *    *
dtype0
ú
(pi/dense/kernel/Adam_1/Initializer/zerosFill8pi/dense/kernel/Adam_1/Initializer/zeros/shape_as_tensor.pi/dense/kernel/Adam_1/Initializer/zeros/Const*
_output_shapes
:	L*
T0*

index_type0*"
_class
loc:@pi/dense/kernel
°
pi/dense/kernel/Adam_1
VariableV2*
dtype0*
shared_name *"
_class
loc:@pi/dense/kernel*
_output_shapes
:	L*
	container *
shape:	L
ŕ
pi/dense/kernel/Adam_1/AssignAssignpi/dense/kernel/Adam_1(pi/dense/kernel/Adam_1/Initializer/zeros*
use_locking(*
validate_shape(*
_output_shapes
:	L*
T0*"
_class
loc:@pi/dense/kernel

pi/dense/kernel/Adam_1/readIdentitypi/dense/kernel/Adam_1*
T0*"
_class
loc:@pi/dense/kernel*
_output_shapes
:	L

$pi/dense/bias/Adam/Initializer/zerosConst*
valueB*    * 
_class
loc:@pi/dense/bias*
dtype0*
_output_shapes	
:
˘
pi/dense/bias/Adam
VariableV2*
	container *
shared_name *
dtype0*
_output_shapes	
:* 
_class
loc:@pi/dense/bias*
shape:
Î
pi/dense/bias/Adam/AssignAssignpi/dense/bias/Adam$pi/dense/bias/Adam/Initializer/zeros*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0* 
_class
loc:@pi/dense/bias

pi/dense/bias/Adam/readIdentitypi/dense/bias/Adam* 
_class
loc:@pi/dense/bias*
_output_shapes	
:*
T0

&pi/dense/bias/Adam_1/Initializer/zerosConst*
_output_shapes	
:*
dtype0* 
_class
loc:@pi/dense/bias*
valueB*    
¤
pi/dense/bias/Adam_1
VariableV2*
dtype0*
	container *
shared_name *
_output_shapes	
:* 
_class
loc:@pi/dense/bias*
shape:
Ô
pi/dense/bias/Adam_1/AssignAssignpi/dense/bias/Adam_1&pi/dense/bias/Adam_1/Initializer/zeros*
use_locking(* 
_class
loc:@pi/dense/bias*
validate_shape(*
T0*
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
dtype0*
valueB"      *$
_class
loc:@pi/dense_1/kernel

.pi/dense_1/kernel/Adam/Initializer/zeros/ConstConst*
_output_shapes
: *
valueB
 *    *
dtype0*$
_class
loc:@pi/dense_1/kernel
ý
(pi/dense_1/kernel/Adam/Initializer/zerosFill8pi/dense_1/kernel/Adam/Initializer/zeros/shape_as_tensor.pi/dense_1/kernel/Adam/Initializer/zeros/Const*

index_type0*
T0* 
_output_shapes
:
*$
_class
loc:@pi/dense_1/kernel
´
pi/dense_1/kernel/Adam
VariableV2*
shape:
*
dtype0*
	container *
shared_name *$
_class
loc:@pi/dense_1/kernel* 
_output_shapes
:

ă
pi/dense_1/kernel/Adam/AssignAssignpi/dense_1/kernel/Adam(pi/dense_1/kernel/Adam/Initializer/zeros*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0*$
_class
loc:@pi/dense_1/kernel

pi/dense_1/kernel/Adam/readIdentitypi/dense_1/kernel/Adam*
T0* 
_output_shapes
:
*$
_class
loc:@pi/dense_1/kernel
ą
:pi/dense_1/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
valueB"      *
_output_shapes
:*$
_class
loc:@pi/dense_1/kernel*
dtype0

0pi/dense_1/kernel/Adam_1/Initializer/zeros/ConstConst*
dtype0*$
_class
loc:@pi/dense_1/kernel*
valueB
 *    *
_output_shapes
: 

*pi/dense_1/kernel/Adam_1/Initializer/zerosFill:pi/dense_1/kernel/Adam_1/Initializer/zeros/shape_as_tensor0pi/dense_1/kernel/Adam_1/Initializer/zeros/Const* 
_output_shapes
:
*
T0*$
_class
loc:@pi/dense_1/kernel*

index_type0
ś
pi/dense_1/kernel/Adam_1
VariableV2*
dtype0* 
_output_shapes
:
*
	container *
shape:
*
shared_name *$
_class
loc:@pi/dense_1/kernel
é
pi/dense_1/kernel/Adam_1/AssignAssignpi/dense_1/kernel/Adam_1*pi/dense_1/kernel/Adam_1/Initializer/zeros*
use_locking(*
validate_shape(*$
_class
loc:@pi/dense_1/kernel*
T0* 
_output_shapes
:


pi/dense_1/kernel/Adam_1/readIdentitypi/dense_1/kernel/Adam_1* 
_output_shapes
:
*
T0*$
_class
loc:@pi/dense_1/kernel

&pi/dense_1/bias/Adam/Initializer/zerosConst*
_output_shapes	
:*"
_class
loc:@pi/dense_1/bias*
valueB*    *
dtype0
Ś
pi/dense_1/bias/Adam
VariableV2*
	container *"
_class
loc:@pi/dense_1/bias*
shape:*
dtype0*
shared_name *
_output_shapes	
:
Ö
pi/dense_1/bias/Adam/AssignAssignpi/dense_1/bias/Adam&pi/dense_1/bias/Adam/Initializer/zeros*
T0*
validate_shape(*"
_class
loc:@pi/dense_1/bias*
use_locking(*
_output_shapes	
:
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
:*
valueB*    *"
_class
loc:@pi/dense_1/bias*
dtype0
¨
pi/dense_1/bias/Adam_1
VariableV2*
shared_name *"
_class
loc:@pi/dense_1/bias*
	container *
dtype0*
_output_shapes	
:*
shape:
Ü
pi/dense_1/bias/Adam_1/AssignAssignpi/dense_1/bias/Adam_1(pi/dense_1/bias/Adam_1/Initializer/zeros*
use_locking(*"
_class
loc:@pi/dense_1/bias*
validate_shape(*
T0*
_output_shapes	
:

pi/dense_1/bias/Adam_1/readIdentitypi/dense_1/bias/Adam_1*
_output_shapes	
:*
T0*"
_class
loc:@pi/dense_1/bias
Ľ
(pi/dense_2/kernel/Adam/Initializer/zerosConst*$
_class
loc:@pi/dense_2/kernel*
dtype0*
valueB	*    *
_output_shapes
:	
˛
pi/dense_2/kernel/Adam
VariableV2*
	container *
dtype0*
shape:	*
_output_shapes
:	*
shared_name *$
_class
loc:@pi/dense_2/kernel
â
pi/dense_2/kernel/Adam/AssignAssignpi/dense_2/kernel/Adam(pi/dense_2/kernel/Adam/Initializer/zeros*
validate_shape(*
_output_shapes
:	*
use_locking(*
T0*$
_class
loc:@pi/dense_2/kernel

pi/dense_2/kernel/Adam/readIdentitypi/dense_2/kernel/Adam*
_output_shapes
:	*$
_class
loc:@pi/dense_2/kernel*
T0
§
*pi/dense_2/kernel/Adam_1/Initializer/zerosConst*$
_class
loc:@pi/dense_2/kernel*
_output_shapes
:	*
dtype0*
valueB	*    
´
pi/dense_2/kernel/Adam_1
VariableV2*$
_class
loc:@pi/dense_2/kernel*
shared_name *
_output_shapes
:	*
shape:	*
	container *
dtype0
č
pi/dense_2/kernel/Adam_1/AssignAssignpi/dense_2/kernel/Adam_1*pi/dense_2/kernel/Adam_1/Initializer/zeros*
validate_shape(*
T0*$
_class
loc:@pi/dense_2/kernel*
_output_shapes
:	*
use_locking(

pi/dense_2/kernel/Adam_1/readIdentitypi/dense_2/kernel/Adam_1*
T0*$
_class
loc:@pi/dense_2/kernel*
_output_shapes
:	

&pi/dense_2/bias/Adam/Initializer/zerosConst*"
_class
loc:@pi/dense_2/bias*
dtype0*
valueB*    *
_output_shapes
:
¤
pi/dense_2/bias/Adam
VariableV2*
	container *
dtype0*"
_class
loc:@pi/dense_2/bias*
shared_name *
shape:*
_output_shapes
:
Ő
pi/dense_2/bias/Adam/AssignAssignpi/dense_2/bias/Adam&pi/dense_2/bias/Adam/Initializer/zeros*
_output_shapes
:*
T0*
use_locking(*"
_class
loc:@pi/dense_2/bias*
validate_shape(

pi/dense_2/bias/Adam/readIdentitypi/dense_2/bias/Adam*
T0*"
_class
loc:@pi/dense_2/bias*
_output_shapes
:

(pi/dense_2/bias/Adam_1/Initializer/zerosConst*
_output_shapes
:*
valueB*    *"
_class
loc:@pi/dense_2/bias*
dtype0
Ś
pi/dense_2/bias/Adam_1
VariableV2*
dtype0*"
_class
loc:@pi/dense_2/bias*
shared_name *
	container *
shape:*
_output_shapes
:
Ű
pi/dense_2/bias/Adam_1/AssignAssignpi/dense_2/bias/Adam_1(pi/dense_2/bias/Adam_1/Initializer/zeros*
use_locking(*
_output_shapes
:*
T0*
validate_shape(*"
_class
loc:@pi/dense_2/bias

pi/dense_2/bias/Adam_1/readIdentitypi/dense_2/bias/Adam_1*
T0*"
_class
loc:@pi/dense_2/bias*
_output_shapes
:

!pi/log_std/Adam/Initializer/zerosConst*
_output_shapes
:*
dtype0*
_class
loc:@pi/log_std*
valueB*    

pi/log_std/Adam
VariableV2*
_output_shapes
:*
shape:*
_class
loc:@pi/log_std*
shared_name *
dtype0*
	container 
Á
pi/log_std/Adam/AssignAssignpi/log_std/Adam!pi/log_std/Adam/Initializer/zeros*
_class
loc:@pi/log_std*
T0*
validate_shape(*
_output_shapes
:*
use_locking(
u
pi/log_std/Adam/readIdentitypi/log_std/Adam*
_output_shapes
:*
_class
loc:@pi/log_std*
T0

#pi/log_std/Adam_1/Initializer/zerosConst*
valueB*    *
_class
loc:@pi/log_std*
_output_shapes
:*
dtype0

pi/log_std/Adam_1
VariableV2*
dtype0*
_class
loc:@pi/log_std*
shared_name *
_output_shapes
:*
shape:*
	container 
Ç
pi/log_std/Adam_1/AssignAssignpi/log_std/Adam_1#pi/log_std/Adam_1/Initializer/zeros*
T0*
_class
loc:@pi/log_std*
validate_shape(*
use_locking(*
_output_shapes
:
y
pi/log_std/Adam_1/readIdentitypi/log_std/Adam_1*
T0*
_class
loc:@pi/log_std*
_output_shapes
:

.penalty/penalty_param/Adam_2/Initializer/zerosConst*
valueB
 *    *
dtype0*(
_class
loc:@penalty/penalty_param*
_output_shapes
: 
Ş
penalty/penalty_param/Adam_2
VariableV2*
	container *
dtype0*(
_class
loc:@penalty/penalty_param*
shape: *
_output_shapes
: *
shared_name 
ď
#penalty/penalty_param/Adam_2/AssignAssignpenalty/penalty_param/Adam_2.penalty/penalty_param/Adam_2/Initializer/zeros*
validate_shape(*
_output_shapes
: *(
_class
loc:@penalty/penalty_param*
use_locking(*
T0

!penalty/penalty_param/Adam_2/readIdentitypenalty/penalty_param/Adam_2*(
_class
loc:@penalty/penalty_param*
_output_shapes
: *
T0

.penalty/penalty_param/Adam_3/Initializer/zerosConst*
_output_shapes
: *
valueB
 *    *(
_class
loc:@penalty/penalty_param*
dtype0
Ş
penalty/penalty_param/Adam_3
VariableV2*
shape: *
dtype0*
shared_name *
_output_shapes
: *(
_class
loc:@penalty/penalty_param*
	container 
ď
#penalty/penalty_param/Adam_3/AssignAssignpenalty/penalty_param/Adam_3.penalty/penalty_param/Adam_3/Initializer/zeros*
_output_shapes
: *
validate_shape(*
T0*(
_class
loc:@penalty/penalty_param*
use_locking(

!penalty/penalty_param/Adam_3/readIdentitypenalty/penalty_param/Adam_3*
T0*(
_class
loc:@penalty/penalty_param*
_output_shapes
: 
Y
Adam_1/learning_rateConst*
_output_shapes
: *
dtype0*
valueB
 *RI9
Q
Adam_1/beta1Const*
dtype0*
valueB
 *fff?*
_output_shapes
: 
Q
Adam_1/beta2Const*
dtype0*
_output_shapes
: *
valueB
 *wž?
S
Adam_1/epsilonConst*
_output_shapes
: *
valueB
 *wĚ+2*
dtype0
Ţ
'Adam_1/update_pi/dense/kernel/ApplyAdam	ApplyAdampi/dense/kernelpi/dense/kernel/Adampi/dense/kernel/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilon
Reshape_12*
use_nesterov( *
use_locking( *
_output_shapes
:	L*
T0*"
_class
loc:@pi/dense/kernel
Đ
%Adam_1/update_pi/dense/bias/ApplyAdam	ApplyAdampi/dense/biaspi/dense/bias/Adampi/dense/bias/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilon
Reshape_13*
_output_shapes	
:*
use_nesterov( *
T0*
use_locking( * 
_class
loc:@pi/dense/bias
é
)Adam_1/update_pi/dense_1/kernel/ApplyAdam	ApplyAdampi/dense_1/kernelpi/dense_1/kernel/Adampi/dense_1/kernel/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilon
Reshape_14* 
_output_shapes
:
*$
_class
loc:@pi/dense_1/kernel*
use_locking( *
T0*
use_nesterov( 
Ú
'Adam_1/update_pi/dense_1/bias/ApplyAdam	ApplyAdampi/dense_1/biaspi/dense_1/bias/Adampi/dense_1/bias/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilon
Reshape_15*
T0*
use_nesterov( *
use_locking( *
_output_shapes	
:*"
_class
loc:@pi/dense_1/bias
č
)Adam_1/update_pi/dense_2/kernel/ApplyAdam	ApplyAdampi/dense_2/kernelpi/dense_2/kernel/Adampi/dense_2/kernel/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilon
Reshape_16*$
_class
loc:@pi/dense_2/kernel*
use_nesterov( *
_output_shapes
:	*
use_locking( *
T0
Ů
'Adam_1/update_pi/dense_2/bias/ApplyAdam	ApplyAdampi/dense_2/biaspi/dense_2/bias/Adampi/dense_2/bias/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilon
Reshape_17*
T0*
use_locking( *"
_class
loc:@pi/dense_2/bias*
use_nesterov( *
_output_shapes
:
Ŕ
"Adam_1/update_pi/log_std/ApplyAdam	ApplyAdam
pi/log_stdpi/log_std/Adampi/log_std/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilon
Reshape_18*
use_locking( *
T0*
use_nesterov( *
_output_shapes
:*
_class
loc:@pi/log_std
ő
-Adam_1/update_penalty/penalty_param/ApplyAdam	ApplyAdampenalty/penalty_parampenalty/penalty_param/Adam_2penalty/penalty_param/Adam_3beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilon
Reshape_19*
T0*
use_locking( *
use_nesterov( *(
_class
loc:@penalty/penalty_param*
_output_shapes
: 
Ń

Adam_1/mulMulbeta1_power_1/readAdam_1/beta1.^Adam_1/update_penalty/penalty_param/ApplyAdam&^Adam_1/update_pi/dense/bias/ApplyAdam(^Adam_1/update_pi/dense/kernel/ApplyAdam(^Adam_1/update_pi/dense_1/bias/ApplyAdam*^Adam_1/update_pi/dense_1/kernel/ApplyAdam(^Adam_1/update_pi/dense_2/bias/ApplyAdam*^Adam_1/update_pi/dense_2/kernel/ApplyAdam#^Adam_1/update_pi/log_std/ApplyAdam*
_output_shapes
: *(
_class
loc:@penalty/penalty_param*
T0
Ś
Adam_1/AssignAssignbeta1_power_1
Adam_1/mul*
_output_shapes
: *
use_locking( *(
_class
loc:@penalty/penalty_param*
validate_shape(*
T0
Ó
Adam_1/mul_1Mulbeta2_power_1/readAdam_1/beta2.^Adam_1/update_penalty/penalty_param/ApplyAdam&^Adam_1/update_pi/dense/bias/ApplyAdam(^Adam_1/update_pi/dense/kernel/ApplyAdam(^Adam_1/update_pi/dense_1/bias/ApplyAdam*^Adam_1/update_pi/dense_1/kernel/ApplyAdam(^Adam_1/update_pi/dense_2/bias/ApplyAdam*^Adam_1/update_pi/dense_2/kernel/ApplyAdam#^Adam_1/update_pi/log_std/ApplyAdam*(
_class
loc:@penalty/penalty_param*
T0*
_output_shapes
: 
Ş
Adam_1/Assign_1Assignbeta2_power_1Adam_1/mul_1*
use_locking( *
validate_shape(*
T0*(
_class
loc:@penalty/penalty_param*
_output_shapes
: 

Adam_1NoOp^Adam_1/Assign^Adam_1/Assign_1.^Adam_1/update_penalty/penalty_param/ApplyAdam&^Adam_1/update_pi/dense/bias/ApplyAdam(^Adam_1/update_pi/dense/kernel/ApplyAdam(^Adam_1/update_pi/dense_1/bias/ApplyAdam*^Adam_1/update_pi/dense_1/kernel/ApplyAdam(^Adam_1/update_pi/dense_2/bias/ApplyAdam*^Adam_1/update_pi/dense_2/kernel/ApplyAdam#^Adam_1/update_pi/log_std/ApplyAdam
l
Reshape_20/shapeConst^Adam_1*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0
r

Reshape_20Reshapepi/dense/kernel/readReshape_20/shape*
T0*
_output_shapes

:*
Tshape0
l
Reshape_21/shapeConst^Adam_1*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0
o

Reshape_21Reshapepi/dense/bias/readReshape_21/shape*
Tshape0*
_output_shapes	
:*
T0
l
Reshape_22/shapeConst^Adam_1*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0
t

Reshape_22Reshapepi/dense_1/kernel/readReshape_22/shape*
_output_shapes

:*
Tshape0*
T0
l
Reshape_23/shapeConst^Adam_1*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙*
_output_shapes
:
q

Reshape_23Reshapepi/dense_1/bias/readReshape_23/shape*
_output_shapes	
:*
T0*
Tshape0
l
Reshape_24/shapeConst^Adam_1*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙*
_output_shapes
:
s

Reshape_24Reshapepi/dense_2/kernel/readReshape_24/shape*
_output_shapes	
:*
T0*
Tshape0
l
Reshape_25/shapeConst^Adam_1*
valueB:
˙˙˙˙˙˙˙˙˙*
_output_shapes
:*
dtype0
p

Reshape_25Reshapepi/dense_2/bias/readReshape_25/shape*
_output_shapes
:*
Tshape0*
T0
l
Reshape_26/shapeConst^Adam_1*
dtype0*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙
k

Reshape_26Reshapepi/log_std/readReshape_26/shape*
Tshape0*
_output_shapes
:*
T0
l
Reshape_27/shapeConst^Adam_1*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
v

Reshape_27Reshapepenalty/penalty_param/readReshape_27/shape*
Tshape0*
_output_shapes
:*
T0
X
concat_3/axisConst^Adam_1*
_output_shapes
: *
dtype0*
value	B : 
ż
concat_3ConcatV2
Reshape_20
Reshape_21
Reshape_22
Reshape_23
Reshape_24
Reshape_25
Reshape_26
Reshape_27concat_3/axis*

Tidx0*
N*
_output_shapes

: *
T0
h
PyFunc_3PyFuncconcat_3*
token
pyfunc_3*
_output_shapes
:*
Tout
2*
Tin
2
y
Const_7Const^Adam_1*5
value,B*"  L                       *
_output_shapes
:*
dtype0
\
split_3/split_dimConst^Adam_1*
dtype0*
_output_shapes
: *
value	B : 

split_3SplitVPyFunc_3Const_7split_3/split_dim*

Tlen0*
T0*
	num_split*4
_output_shapes"
 ::::::::
j
Reshape_28/shapeConst^Adam_1*
_output_shapes
:*
valueB"L      *
dtype0
h

Reshape_28Reshapesplit_3Reshape_28/shape*
Tshape0*
T0*
_output_shapes
:	L
d
Reshape_29/shapeConst^Adam_1*
valueB:*
_output_shapes
:*
dtype0
f

Reshape_29Reshape	split_3:1Reshape_29/shape*
Tshape0*
_output_shapes	
:*
T0
j
Reshape_30/shapeConst^Adam_1*
_output_shapes
:*
valueB"      *
dtype0
k

Reshape_30Reshape	split_3:2Reshape_30/shape*
Tshape0* 
_output_shapes
:
*
T0
d
Reshape_31/shapeConst^Adam_1*
_output_shapes
:*
valueB:*
dtype0
f

Reshape_31Reshape	split_3:3Reshape_31/shape*
T0*
Tshape0*
_output_shapes	
:
j
Reshape_32/shapeConst^Adam_1*
_output_shapes
:*
dtype0*
valueB"      
j

Reshape_32Reshape	split_3:4Reshape_32/shape*
_output_shapes
:	*
T0*
Tshape0
c
Reshape_33/shapeConst^Adam_1*
valueB:*
dtype0*
_output_shapes
:
e

Reshape_33Reshape	split_3:5Reshape_33/shape*
_output_shapes
:*
Tshape0*
T0
c
Reshape_34/shapeConst^Adam_1*
valueB:*
dtype0*
_output_shapes
:
e

Reshape_34Reshape	split_3:6Reshape_34/shape*
Tshape0*
_output_shapes
:*
T0
\
Reshape_35/shapeConst^Adam_1*
valueB *
dtype0*
_output_shapes
: 
a

Reshape_35Reshape	split_3:7Reshape_35/shape*
Tshape0*
T0*
_output_shapes
: 
Ś
Assign_1Assignpi/dense/kernel
Reshape_28*"
_class
loc:@pi/dense/kernel*
use_locking(*
_output_shapes
:	L*
validate_shape(*
T0

Assign_2Assignpi/dense/bias
Reshape_29*
_output_shapes	
:*
use_locking(*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(
Ť
Assign_3Assignpi/dense_1/kernel
Reshape_30* 
_output_shapes
:
*
validate_shape(*
T0*$
_class
loc:@pi/dense_1/kernel*
use_locking(
˘
Assign_4Assignpi/dense_1/bias
Reshape_31*
validate_shape(*"
_class
loc:@pi/dense_1/bias*
_output_shapes	
:*
T0*
use_locking(
Ş
Assign_5Assignpi/dense_2/kernel
Reshape_32*
use_locking(*
validate_shape(*
T0*$
_class
loc:@pi/dense_2/kernel*
_output_shapes
:	
Ą
Assign_6Assignpi/dense_2/bias
Reshape_33*
_output_shapes
:*
use_locking(*
T0*"
_class
loc:@pi/dense_2/bias*
validate_shape(

Assign_7Assign
pi/log_std
Reshape_34*
use_locking(*
T0*
_class
loc:@pi/log_std*
_output_shapes
:*
validate_shape(
Š
Assign_8Assignpenalty/penalty_param
Reshape_35*
_output_shapes
: *
use_locking(*
validate_shape(*(
_class
loc:@penalty/penalty_param*
T0
u
group_deps_2NoOp^Adam_1	^Assign_1	^Assign_2	^Assign_3	^Assign_4	^Assign_5	^Assign_6	^Assign_7	^Assign_8
,
group_deps_3NoOp^Adam_1^group_deps_2
U
sub_3SubPlaceholder_4
vf/Squeeze*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
J
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @
F
powPowsub_3pow/y*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Q
Const_8Const*
dtype0*
valueB: *
_output_shapes
:
Z
Mean_3MeanpowConst_8*
T0*
	keep_dims( *
_output_shapes
: *

Tidx0
U
sub_4SubPlaceholder_5
vc/Squeeze*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
L
pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @
J
pow_1Powsub_4pow_1/y*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Q
Const_9Const*
dtype0*
valueB: *
_output_shapes
:
\
Mean_4Meanpow_1Const_9*

Tidx0*
_output_shapes
: *
T0*
	keep_dims( 
=
add_2AddMean_3Mean_4*
_output_shapes
: *
T0
T
gradients_2/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
Z
gradients_2/grad_ys_0Const*
dtype0*
_output_shapes
: *
valueB
 *  ?
u
gradients_2/FillFillgradients_2/Shapegradients_2/grad_ys_0*
T0*
_output_shapes
: *

index_type0
B
'gradients_2/add_2_grad/tuple/group_depsNoOp^gradients_2/Fill
˝
/gradients_2/add_2_grad/tuple/control_dependencyIdentitygradients_2/Fill(^gradients_2/add_2_grad/tuple/group_deps*
_output_shapes
: *
T0*#
_class
loc:@gradients_2/Fill
ż
1gradients_2/add_2_grad/tuple/control_dependency_1Identitygradients_2/Fill(^gradients_2/add_2_grad/tuple/group_deps*#
_class
loc:@gradients_2/Fill*
_output_shapes
: *
T0
o
%gradients_2/Mean_3_grad/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
ľ
gradients_2/Mean_3_grad/ReshapeReshape/gradients_2/add_2_grad/tuple/control_dependency%gradients_2/Mean_3_grad/Reshape/shape*
_output_shapes
:*
Tshape0*
T0
`
gradients_2/Mean_3_grad/ShapeShapepow*
T0*
_output_shapes
:*
out_type0
¤
gradients_2/Mean_3_grad/TileTilegradients_2/Mean_3_grad/Reshapegradients_2/Mean_3_grad/Shape*

Tmultiples0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
b
gradients_2/Mean_3_grad/Shape_1Shapepow*
out_type0*
_output_shapes
:*
T0
b
gradients_2/Mean_3_grad/Shape_2Const*
dtype0*
_output_shapes
: *
valueB 
g
gradients_2/Mean_3_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:
˘
gradients_2/Mean_3_grad/ProdProdgradients_2/Mean_3_grad/Shape_1gradients_2/Mean_3_grad/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
i
gradients_2/Mean_3_grad/Const_1Const*
dtype0*
_output_shapes
:*
valueB: 
Ś
gradients_2/Mean_3_grad/Prod_1Prodgradients_2/Mean_3_grad/Shape_2gradients_2/Mean_3_grad/Const_1*
	keep_dims( *
T0*

Tidx0*
_output_shapes
: 
c
!gradients_2/Mean_3_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 

gradients_2/Mean_3_grad/MaximumMaximumgradients_2/Mean_3_grad/Prod_1!gradients_2/Mean_3_grad/Maximum/y*
_output_shapes
: *
T0

 gradients_2/Mean_3_grad/floordivFloorDivgradients_2/Mean_3_grad/Prodgradients_2/Mean_3_grad/Maximum*
_output_shapes
: *
T0

gradients_2/Mean_3_grad/CastCast gradients_2/Mean_3_grad/floordiv*

SrcT0*
Truncate( *

DstT0*
_output_shapes
: 

gradients_2/Mean_3_grad/truedivRealDivgradients_2/Mean_3_grad/Tilegradients_2/Mean_3_grad/Cast*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
o
%gradients_2/Mean_4_grad/Reshape/shapeConst*
valueB:*
_output_shapes
:*
dtype0
ˇ
gradients_2/Mean_4_grad/ReshapeReshape1gradients_2/add_2_grad/tuple/control_dependency_1%gradients_2/Mean_4_grad/Reshape/shape*
Tshape0*
_output_shapes
:*
T0
b
gradients_2/Mean_4_grad/ShapeShapepow_1*
_output_shapes
:*
out_type0*
T0
¤
gradients_2/Mean_4_grad/TileTilegradients_2/Mean_4_grad/Reshapegradients_2/Mean_4_grad/Shape*
T0*

Tmultiples0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
d
gradients_2/Mean_4_grad/Shape_1Shapepow_1*
T0*
out_type0*
_output_shapes
:
b
gradients_2/Mean_4_grad/Shape_2Const*
dtype0*
valueB *
_output_shapes
: 
g
gradients_2/Mean_4_grad/ConstConst*
valueB: *
_output_shapes
:*
dtype0
˘
gradients_2/Mean_4_grad/ProdProdgradients_2/Mean_4_grad/Shape_1gradients_2/Mean_4_grad/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
i
gradients_2/Mean_4_grad/Const_1Const*
valueB: *
_output_shapes
:*
dtype0
Ś
gradients_2/Mean_4_grad/Prod_1Prodgradients_2/Mean_4_grad/Shape_2gradients_2/Mean_4_grad/Const_1*

Tidx0*
T0*
_output_shapes
: *
	keep_dims( 
c
!gradients_2/Mean_4_grad/Maximum/yConst*
_output_shapes
: *
value	B :*
dtype0

gradients_2/Mean_4_grad/MaximumMaximumgradients_2/Mean_4_grad/Prod_1!gradients_2/Mean_4_grad/Maximum/y*
_output_shapes
: *
T0

 gradients_2/Mean_4_grad/floordivFloorDivgradients_2/Mean_4_grad/Prodgradients_2/Mean_4_grad/Maximum*
T0*
_output_shapes
: 

gradients_2/Mean_4_grad/CastCast gradients_2/Mean_4_grad/floordiv*
_output_shapes
: *

SrcT0*

DstT0*
Truncate( 

gradients_2/Mean_4_grad/truedivRealDivgradients_2/Mean_4_grad/Tilegradients_2/Mean_4_grad/Cast*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
_
gradients_2/pow_grad/ShapeShapesub_3*
T0*
out_type0*
_output_shapes
:
_
gradients_2/pow_grad/Shape_1Const*
valueB *
_output_shapes
: *
dtype0
ş
*gradients_2/pow_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_2/pow_grad/Shapegradients_2/pow_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0
u
gradients_2/pow_grad/mulMulgradients_2/Mean_3_grad/truedivpow/y*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
_
gradients_2/pow_grad/sub/yConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
c
gradients_2/pow_grad/subSubpow/ygradients_2/pow_grad/sub/y*
_output_shapes
: *
T0
n
gradients_2/pow_grad/PowPowsub_3gradients_2/pow_grad/sub*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙

gradients_2/pow_grad/mul_1Mulgradients_2/pow_grad/mulgradients_2/pow_grad/Pow*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
§
gradients_2/pow_grad/SumSumgradients_2/pow_grad/mul_1*gradients_2/pow_grad/BroadcastGradientArgs*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:

gradients_2/pow_grad/ReshapeReshapegradients_2/pow_grad/Sumgradients_2/pow_grad/Shape*
Tshape0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
c
gradients_2/pow_grad/Greater/yConst*
_output_shapes
: *
valueB
 *    *
dtype0
|
gradients_2/pow_grad/GreaterGreatersub_3gradients_2/pow_grad/Greater/y*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
i
$gradients_2/pow_grad/ones_like/ShapeShapesub_3*
out_type0*
T0*
_output_shapes
:
i
$gradients_2/pow_grad/ones_like/ConstConst*
_output_shapes
: *
valueB
 *  ?*
dtype0
˛
gradients_2/pow_grad/ones_likeFill$gradients_2/pow_grad/ones_like/Shape$gradients_2/pow_grad/ones_like/Const*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*

index_type0

gradients_2/pow_grad/SelectSelectgradients_2/pow_grad/Greatersub_3gradients_2/pow_grad/ones_like*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
j
gradients_2/pow_grad/LogLoggradients_2/pow_grad/Select*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
a
gradients_2/pow_grad/zeros_like	ZerosLikesub_3*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ž
gradients_2/pow_grad/Select_1Selectgradients_2/pow_grad/Greatergradients_2/pow_grad/Loggradients_2/pow_grad/zeros_like*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
u
gradients_2/pow_grad/mul_2Mulgradients_2/Mean_3_grad/truedivpow*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

gradients_2/pow_grad/mul_3Mulgradients_2/pow_grad/mul_2gradients_2/pow_grad/Select_1*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ť
gradients_2/pow_grad/Sum_1Sumgradients_2/pow_grad/mul_3,gradients_2/pow_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *
T0*

Tidx0

gradients_2/pow_grad/Reshape_1Reshapegradients_2/pow_grad/Sum_1gradients_2/pow_grad/Shape_1*
_output_shapes
: *
T0*
Tshape0
m
%gradients_2/pow_grad/tuple/group_depsNoOp^gradients_2/pow_grad/Reshape^gradients_2/pow_grad/Reshape_1
Ţ
-gradients_2/pow_grad/tuple/control_dependencyIdentitygradients_2/pow_grad/Reshape&^gradients_2/pow_grad/tuple/group_deps*/
_class%
#!loc:@gradients_2/pow_grad/Reshape*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
×
/gradients_2/pow_grad/tuple/control_dependency_1Identitygradients_2/pow_grad/Reshape_1&^gradients_2/pow_grad/tuple/group_deps*
_output_shapes
: *
T0*1
_class'
%#loc:@gradients_2/pow_grad/Reshape_1
a
gradients_2/pow_1_grad/ShapeShapesub_4*
_output_shapes
:*
T0*
out_type0
a
gradients_2/pow_1_grad/Shape_1Const*
dtype0*
_output_shapes
: *
valueB 
Ŕ
,gradients_2/pow_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_2/pow_1_grad/Shapegradients_2/pow_1_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0
y
gradients_2/pow_1_grad/mulMulgradients_2/Mean_4_grad/truedivpow_1/y*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
a
gradients_2/pow_1_grad/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
i
gradients_2/pow_1_grad/subSubpow_1/ygradients_2/pow_1_grad/sub/y*
_output_shapes
: *
T0
r
gradients_2/pow_1_grad/PowPowsub_4gradients_2/pow_1_grad/sub*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

gradients_2/pow_1_grad/mul_1Mulgradients_2/pow_1_grad/mulgradients_2/pow_1_grad/Pow*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
­
gradients_2/pow_1_grad/SumSumgradients_2/pow_1_grad/mul_1,gradients_2/pow_1_grad/BroadcastGradientArgs*

Tidx0*
_output_shapes
:*
T0*
	keep_dims( 

gradients_2/pow_1_grad/ReshapeReshapegradients_2/pow_1_grad/Sumgradients_2/pow_1_grad/Shape*
T0*
Tshape0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
e
 gradients_2/pow_1_grad/Greater/yConst*
_output_shapes
: *
valueB
 *    *
dtype0

gradients_2/pow_1_grad/GreaterGreatersub_4 gradients_2/pow_1_grad/Greater/y*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
k
&gradients_2/pow_1_grad/ones_like/ShapeShapesub_4*
_output_shapes
:*
out_type0*
T0
k
&gradients_2/pow_1_grad/ones_like/ConstConst*
valueB
 *  ?*
_output_shapes
: *
dtype0
¸
 gradients_2/pow_1_grad/ones_likeFill&gradients_2/pow_1_grad/ones_like/Shape&gradients_2/pow_1_grad/ones_like/Const*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*

index_type0*
T0

gradients_2/pow_1_grad/SelectSelectgradients_2/pow_1_grad/Greatersub_4 gradients_2/pow_1_grad/ones_like*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
n
gradients_2/pow_1_grad/LogLoggradients_2/pow_1_grad/Select*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
c
!gradients_2/pow_1_grad/zeros_like	ZerosLikesub_4*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
ś
gradients_2/pow_1_grad/Select_1Selectgradients_2/pow_1_grad/Greatergradients_2/pow_1_grad/Log!gradients_2/pow_1_grad/zeros_like*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
y
gradients_2/pow_1_grad/mul_2Mulgradients_2/Mean_4_grad/truedivpow_1*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙

gradients_2/pow_1_grad/mul_3Mulgradients_2/pow_1_grad/mul_2gradients_2/pow_1_grad/Select_1*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
ą
gradients_2/pow_1_grad/Sum_1Sumgradients_2/pow_1_grad/mul_3.gradients_2/pow_1_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( *
_output_shapes
:

 gradients_2/pow_1_grad/Reshape_1Reshapegradients_2/pow_1_grad/Sum_1gradients_2/pow_1_grad/Shape_1*
Tshape0*
T0*
_output_shapes
: 
s
'gradients_2/pow_1_grad/tuple/group_depsNoOp^gradients_2/pow_1_grad/Reshape!^gradients_2/pow_1_grad/Reshape_1
ć
/gradients_2/pow_1_grad/tuple/control_dependencyIdentitygradients_2/pow_1_grad/Reshape(^gradients_2/pow_1_grad/tuple/group_deps*1
_class'
%#loc:@gradients_2/pow_1_grad/Reshape*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
ß
1gradients_2/pow_1_grad/tuple/control_dependency_1Identity gradients_2/pow_1_grad/Reshape_1(^gradients_2/pow_1_grad/tuple/group_deps*
T0*
_output_shapes
: *3
_class)
'%loc:@gradients_2/pow_1_grad/Reshape_1
i
gradients_2/sub_3_grad/ShapeShapePlaceholder_4*
out_type0*
_output_shapes
:*
T0
h
gradients_2/sub_3_grad/Shape_1Shape
vf/Squeeze*
out_type0*
T0*
_output_shapes
:
Ŕ
,gradients_2/sub_3_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_2/sub_3_grad/Shapegradients_2/sub_3_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
ž
gradients_2/sub_3_grad/SumSum-gradients_2/pow_grad/tuple/control_dependency,gradients_2/sub_3_grad/BroadcastGradientArgs*

Tidx0*
_output_shapes
:*
	keep_dims( *
T0

gradients_2/sub_3_grad/ReshapeReshapegradients_2/sub_3_grad/Sumgradients_2/sub_3_grad/Shape*
Tshape0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Â
gradients_2/sub_3_grad/Sum_1Sum-gradients_2/pow_grad/tuple/control_dependency.gradients_2/sub_3_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
b
gradients_2/sub_3_grad/NegNeggradients_2/sub_3_grad/Sum_1*
_output_shapes
:*
T0
Ł
 gradients_2/sub_3_grad/Reshape_1Reshapegradients_2/sub_3_grad/Neggradients_2/sub_3_grad/Shape_1*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
Tshape0
s
'gradients_2/sub_3_grad/tuple/group_depsNoOp^gradients_2/sub_3_grad/Reshape!^gradients_2/sub_3_grad/Reshape_1
ć
/gradients_2/sub_3_grad/tuple/control_dependencyIdentitygradients_2/sub_3_grad/Reshape(^gradients_2/sub_3_grad/tuple/group_deps*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*1
_class'
%#loc:@gradients_2/sub_3_grad/Reshape*
T0
ě
1gradients_2/sub_3_grad/tuple/control_dependency_1Identity gradients_2/sub_3_grad/Reshape_1(^gradients_2/sub_3_grad/tuple/group_deps*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*3
_class)
'%loc:@gradients_2/sub_3_grad/Reshape_1
i
gradients_2/sub_4_grad/ShapeShapePlaceholder_5*
T0*
_output_shapes
:*
out_type0
h
gradients_2/sub_4_grad/Shape_1Shape
vc/Squeeze*
out_type0*
_output_shapes
:*
T0
Ŕ
,gradients_2/sub_4_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_2/sub_4_grad/Shapegradients_2/sub_4_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
Ŕ
gradients_2/sub_4_grad/SumSum/gradients_2/pow_1_grad/tuple/control_dependency,gradients_2/sub_4_grad/BroadcastGradientArgs*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:

gradients_2/sub_4_grad/ReshapeReshapegradients_2/sub_4_grad/Sumgradients_2/sub_4_grad/Shape*
T0*
Tshape0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ä
gradients_2/sub_4_grad/Sum_1Sum/gradients_2/pow_1_grad/tuple/control_dependency.gradients_2/sub_4_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *
T0*

Tidx0
b
gradients_2/sub_4_grad/NegNeggradients_2/sub_4_grad/Sum_1*
T0*
_output_shapes
:
Ł
 gradients_2/sub_4_grad/Reshape_1Reshapegradients_2/sub_4_grad/Neggradients_2/sub_4_grad/Shape_1*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
Tshape0
s
'gradients_2/sub_4_grad/tuple/group_depsNoOp^gradients_2/sub_4_grad/Reshape!^gradients_2/sub_4_grad/Reshape_1
ć
/gradients_2/sub_4_grad/tuple/control_dependencyIdentitygradients_2/sub_4_grad/Reshape(^gradients_2/sub_4_grad/tuple/group_deps*1
_class'
%#loc:@gradients_2/sub_4_grad/Reshape*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
ě
1gradients_2/sub_4_grad/tuple/control_dependency_1Identity gradients_2/sub_4_grad/Reshape_1(^gradients_2/sub_4_grad/tuple/group_deps*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*3
_class)
'%loc:@gradients_2/sub_4_grad/Reshape_1
s
!gradients_2/vf/Squeeze_grad/ShapeShapevf/dense_2/BiasAdd*
_output_shapes
:*
T0*
out_type0
Ä
#gradients_2/vf/Squeeze_grad/ReshapeReshape1gradients_2/sub_3_grad/tuple/control_dependency_1!gradients_2/vf/Squeeze_grad/Shape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
Tshape0*
T0
s
!gradients_2/vc/Squeeze_grad/ShapeShapevc/dense_2/BiasAdd*
T0*
_output_shapes
:*
out_type0
Ä
#gradients_2/vc/Squeeze_grad/ReshapeReshape1gradients_2/sub_4_grad/tuple/control_dependency_1!gradients_2/vc/Squeeze_grad/Shape*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
Tshape0

/gradients_2/vf/dense_2/BiasAdd_grad/BiasAddGradBiasAddGrad#gradients_2/vf/Squeeze_grad/Reshape*
_output_shapes
:*
T0*
data_formatNHWC

4gradients_2/vf/dense_2/BiasAdd_grad/tuple/group_depsNoOp$^gradients_2/vf/Squeeze_grad/Reshape0^gradients_2/vf/dense_2/BiasAdd_grad/BiasAddGrad

<gradients_2/vf/dense_2/BiasAdd_grad/tuple/control_dependencyIdentity#gradients_2/vf/Squeeze_grad/Reshape5^gradients_2/vf/dense_2/BiasAdd_grad/tuple/group_deps*
T0*6
_class,
*(loc:@gradients_2/vf/Squeeze_grad/Reshape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

>gradients_2/vf/dense_2/BiasAdd_grad/tuple/control_dependency_1Identity/gradients_2/vf/dense_2/BiasAdd_grad/BiasAddGrad5^gradients_2/vf/dense_2/BiasAdd_grad/tuple/group_deps*
T0*
_output_shapes
:*B
_class8
64loc:@gradients_2/vf/dense_2/BiasAdd_grad/BiasAddGrad

/gradients_2/vc/dense_2/BiasAdd_grad/BiasAddGradBiasAddGrad#gradients_2/vc/Squeeze_grad/Reshape*
T0*
_output_shapes
:*
data_formatNHWC

4gradients_2/vc/dense_2/BiasAdd_grad/tuple/group_depsNoOp$^gradients_2/vc/Squeeze_grad/Reshape0^gradients_2/vc/dense_2/BiasAdd_grad/BiasAddGrad

<gradients_2/vc/dense_2/BiasAdd_grad/tuple/control_dependencyIdentity#gradients_2/vc/Squeeze_grad/Reshape5^gradients_2/vc/dense_2/BiasAdd_grad/tuple/group_deps*6
_class,
*(loc:@gradients_2/vc/Squeeze_grad/Reshape*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

>gradients_2/vc/dense_2/BiasAdd_grad/tuple/control_dependency_1Identity/gradients_2/vc/dense_2/BiasAdd_grad/BiasAddGrad5^gradients_2/vc/dense_2/BiasAdd_grad/tuple/group_deps*
_output_shapes
:*
T0*B
_class8
64loc:@gradients_2/vc/dense_2/BiasAdd_grad/BiasAddGrad
â
)gradients_2/vf/dense_2/MatMul_grad/MatMulMatMul<gradients_2/vf/dense_2/BiasAdd_grad/tuple/control_dependencyvf/dense_2/kernel/read*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
T0*
transpose_b(
Ô
+gradients_2/vf/dense_2/MatMul_grad/MatMul_1MatMulvf/dense_1/Tanh<gradients_2/vf/dense_2/BiasAdd_grad/tuple/control_dependency*
T0*
transpose_a(*
_output_shapes
:	*
transpose_b( 

3gradients_2/vf/dense_2/MatMul_grad/tuple/group_depsNoOp*^gradients_2/vf/dense_2/MatMul_grad/MatMul,^gradients_2/vf/dense_2/MatMul_grad/MatMul_1

;gradients_2/vf/dense_2/MatMul_grad/tuple/control_dependencyIdentity)gradients_2/vf/dense_2/MatMul_grad/MatMul4^gradients_2/vf/dense_2/MatMul_grad/tuple/group_deps*<
_class2
0.loc:@gradients_2/vf/dense_2/MatMul_grad/MatMul*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

=gradients_2/vf/dense_2/MatMul_grad/tuple/control_dependency_1Identity+gradients_2/vf/dense_2/MatMul_grad/MatMul_14^gradients_2/vf/dense_2/MatMul_grad/tuple/group_deps*
_output_shapes
:	*
T0*>
_class4
20loc:@gradients_2/vf/dense_2/MatMul_grad/MatMul_1
â
)gradients_2/vc/dense_2/MatMul_grad/MatMulMatMul<gradients_2/vc/dense_2/BiasAdd_grad/tuple/control_dependencyvc/dense_2/kernel/read*
T0*
transpose_b(*
transpose_a( *(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ô
+gradients_2/vc/dense_2/MatMul_grad/MatMul_1MatMulvc/dense_1/Tanh<gradients_2/vc/dense_2/BiasAdd_grad/tuple/control_dependency*
T0*
transpose_b( *
transpose_a(*
_output_shapes
:	

3gradients_2/vc/dense_2/MatMul_grad/tuple/group_depsNoOp*^gradients_2/vc/dense_2/MatMul_grad/MatMul,^gradients_2/vc/dense_2/MatMul_grad/MatMul_1

;gradients_2/vc/dense_2/MatMul_grad/tuple/control_dependencyIdentity)gradients_2/vc/dense_2/MatMul_grad/MatMul4^gradients_2/vc/dense_2/MatMul_grad/tuple/group_deps*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*<
_class2
0.loc:@gradients_2/vc/dense_2/MatMul_grad/MatMul

=gradients_2/vc/dense_2/MatMul_grad/tuple/control_dependency_1Identity+gradients_2/vc/dense_2/MatMul_grad/MatMul_14^gradients_2/vc/dense_2/MatMul_grad/tuple/group_deps*
_output_shapes
:	*
T0*>
_class4
20loc:@gradients_2/vc/dense_2/MatMul_grad/MatMul_1
ś
)gradients_2/vf/dense_1/Tanh_grad/TanhGradTanhGradvf/dense_1/Tanh;gradients_2/vf/dense_2/MatMul_grad/tuple/control_dependency*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ś
)gradients_2/vc/dense_1/Tanh_grad/TanhGradTanhGradvc/dense_1/Tanh;gradients_2/vc/dense_2/MatMul_grad/tuple/control_dependency*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ś
/gradients_2/vf/dense_1/BiasAdd_grad/BiasAddGradBiasAddGrad)gradients_2/vf/dense_1/Tanh_grad/TanhGrad*
T0*
_output_shapes	
:*
data_formatNHWC

4gradients_2/vf/dense_1/BiasAdd_grad/tuple/group_depsNoOp0^gradients_2/vf/dense_1/BiasAdd_grad/BiasAddGrad*^gradients_2/vf/dense_1/Tanh_grad/TanhGrad

<gradients_2/vf/dense_1/BiasAdd_grad/tuple/control_dependencyIdentity)gradients_2/vf/dense_1/Tanh_grad/TanhGrad5^gradients_2/vf/dense_1/BiasAdd_grad/tuple/group_deps*<
_class2
0.loc:@gradients_2/vf/dense_1/Tanh_grad/TanhGrad*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

>gradients_2/vf/dense_1/BiasAdd_grad/tuple/control_dependency_1Identity/gradients_2/vf/dense_1/BiasAdd_grad/BiasAddGrad5^gradients_2/vf/dense_1/BiasAdd_grad/tuple/group_deps*
_output_shapes	
:*B
_class8
64loc:@gradients_2/vf/dense_1/BiasAdd_grad/BiasAddGrad*
T0
Ś
/gradients_2/vc/dense_1/BiasAdd_grad/BiasAddGradBiasAddGrad)gradients_2/vc/dense_1/Tanh_grad/TanhGrad*
T0*
data_formatNHWC*
_output_shapes	
:

4gradients_2/vc/dense_1/BiasAdd_grad/tuple/group_depsNoOp0^gradients_2/vc/dense_1/BiasAdd_grad/BiasAddGrad*^gradients_2/vc/dense_1/Tanh_grad/TanhGrad

<gradients_2/vc/dense_1/BiasAdd_grad/tuple/control_dependencyIdentity)gradients_2/vc/dense_1/Tanh_grad/TanhGrad5^gradients_2/vc/dense_1/BiasAdd_grad/tuple/group_deps*<
_class2
0.loc:@gradients_2/vc/dense_1/Tanh_grad/TanhGrad*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

>gradients_2/vc/dense_1/BiasAdd_grad/tuple/control_dependency_1Identity/gradients_2/vc/dense_1/BiasAdd_grad/BiasAddGrad5^gradients_2/vc/dense_1/BiasAdd_grad/tuple/group_deps*
_output_shapes	
:*B
_class8
64loc:@gradients_2/vc/dense_1/BiasAdd_grad/BiasAddGrad*
T0
â
)gradients_2/vf/dense_1/MatMul_grad/MatMulMatMul<gradients_2/vf/dense_1/BiasAdd_grad/tuple/control_dependencyvf/dense_1/kernel/read*
transpose_b(*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( 
Ó
+gradients_2/vf/dense_1/MatMul_grad/MatMul_1MatMulvf/dense/Tanh<gradients_2/vf/dense_1/BiasAdd_grad/tuple/control_dependency*
transpose_b( * 
_output_shapes
:
*
T0*
transpose_a(

3gradients_2/vf/dense_1/MatMul_grad/tuple/group_depsNoOp*^gradients_2/vf/dense_1/MatMul_grad/MatMul,^gradients_2/vf/dense_1/MatMul_grad/MatMul_1

;gradients_2/vf/dense_1/MatMul_grad/tuple/control_dependencyIdentity)gradients_2/vf/dense_1/MatMul_grad/MatMul4^gradients_2/vf/dense_1/MatMul_grad/tuple/group_deps*
T0*<
_class2
0.loc:@gradients_2/vf/dense_1/MatMul_grad/MatMul*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

=gradients_2/vf/dense_1/MatMul_grad/tuple/control_dependency_1Identity+gradients_2/vf/dense_1/MatMul_grad/MatMul_14^gradients_2/vf/dense_1/MatMul_grad/tuple/group_deps* 
_output_shapes
:
*
T0*>
_class4
20loc:@gradients_2/vf/dense_1/MatMul_grad/MatMul_1
â
)gradients_2/vc/dense_1/MatMul_grad/MatMulMatMul<gradients_2/vc/dense_1/BiasAdd_grad/tuple/control_dependencyvc/dense_1/kernel/read*
transpose_b(*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
transpose_a( 
Ó
+gradients_2/vc/dense_1/MatMul_grad/MatMul_1MatMulvc/dense/Tanh<gradients_2/vc/dense_1/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
T0*
transpose_a(* 
_output_shapes
:


3gradients_2/vc/dense_1/MatMul_grad/tuple/group_depsNoOp*^gradients_2/vc/dense_1/MatMul_grad/MatMul,^gradients_2/vc/dense_1/MatMul_grad/MatMul_1

;gradients_2/vc/dense_1/MatMul_grad/tuple/control_dependencyIdentity)gradients_2/vc/dense_1/MatMul_grad/MatMul4^gradients_2/vc/dense_1/MatMul_grad/tuple/group_deps*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*<
_class2
0.loc:@gradients_2/vc/dense_1/MatMul_grad/MatMul*
T0

=gradients_2/vc/dense_1/MatMul_grad/tuple/control_dependency_1Identity+gradients_2/vc/dense_1/MatMul_grad/MatMul_14^gradients_2/vc/dense_1/MatMul_grad/tuple/group_deps* 
_output_shapes
:
*
T0*>
_class4
20loc:@gradients_2/vc/dense_1/MatMul_grad/MatMul_1
˛
'gradients_2/vf/dense/Tanh_grad/TanhGradTanhGradvf/dense/Tanh;gradients_2/vf/dense_1/MatMul_grad/tuple/control_dependency*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
˛
'gradients_2/vc/dense/Tanh_grad/TanhGradTanhGradvc/dense/Tanh;gradients_2/vc/dense_1/MatMul_grad/tuple/control_dependency*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
˘
-gradients_2/vf/dense/BiasAdd_grad/BiasAddGradBiasAddGrad'gradients_2/vf/dense/Tanh_grad/TanhGrad*
_output_shapes	
:*
T0*
data_formatNHWC

2gradients_2/vf/dense/BiasAdd_grad/tuple/group_depsNoOp.^gradients_2/vf/dense/BiasAdd_grad/BiasAddGrad(^gradients_2/vf/dense/Tanh_grad/TanhGrad

:gradients_2/vf/dense/BiasAdd_grad/tuple/control_dependencyIdentity'gradients_2/vf/dense/Tanh_grad/TanhGrad3^gradients_2/vf/dense/BiasAdd_grad/tuple/group_deps*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*:
_class0
.,loc:@gradients_2/vf/dense/Tanh_grad/TanhGrad

<gradients_2/vf/dense/BiasAdd_grad/tuple/control_dependency_1Identity-gradients_2/vf/dense/BiasAdd_grad/BiasAddGrad3^gradients_2/vf/dense/BiasAdd_grad/tuple/group_deps*@
_class6
42loc:@gradients_2/vf/dense/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes	
:
˘
-gradients_2/vc/dense/BiasAdd_grad/BiasAddGradBiasAddGrad'gradients_2/vc/dense/Tanh_grad/TanhGrad*
T0*
data_formatNHWC*
_output_shapes	
:

2gradients_2/vc/dense/BiasAdd_grad/tuple/group_depsNoOp.^gradients_2/vc/dense/BiasAdd_grad/BiasAddGrad(^gradients_2/vc/dense/Tanh_grad/TanhGrad

:gradients_2/vc/dense/BiasAdd_grad/tuple/control_dependencyIdentity'gradients_2/vc/dense/Tanh_grad/TanhGrad3^gradients_2/vc/dense/BiasAdd_grad/tuple/group_deps*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*:
_class0
.,loc:@gradients_2/vc/dense/Tanh_grad/TanhGrad

<gradients_2/vc/dense/BiasAdd_grad/tuple/control_dependency_1Identity-gradients_2/vc/dense/BiasAdd_grad/BiasAddGrad3^gradients_2/vc/dense/BiasAdd_grad/tuple/group_deps*
_output_shapes	
:*
T0*@
_class6
42loc:@gradients_2/vc/dense/BiasAdd_grad/BiasAddGrad
Ű
'gradients_2/vf/dense/MatMul_grad/MatMulMatMul:gradients_2/vf/dense/BiasAdd_grad/tuple/control_dependencyvf/dense/kernel/read*
transpose_a( *
transpose_b(*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙L
Ě
)gradients_2/vf/dense/MatMul_grad/MatMul_1MatMulPlaceholder:gradients_2/vf/dense/BiasAdd_grad/tuple/control_dependency*
transpose_a(*
T0*
_output_shapes
:	L*
transpose_b( 

1gradients_2/vf/dense/MatMul_grad/tuple/group_depsNoOp(^gradients_2/vf/dense/MatMul_grad/MatMul*^gradients_2/vf/dense/MatMul_grad/MatMul_1

9gradients_2/vf/dense/MatMul_grad/tuple/control_dependencyIdentity'gradients_2/vf/dense/MatMul_grad/MatMul2^gradients_2/vf/dense/MatMul_grad/tuple/group_deps*:
_class0
.,loc:@gradients_2/vf/dense/MatMul_grad/MatMul*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙L

;gradients_2/vf/dense/MatMul_grad/tuple/control_dependency_1Identity)gradients_2/vf/dense/MatMul_grad/MatMul_12^gradients_2/vf/dense/MatMul_grad/tuple/group_deps*<
_class2
0.loc:@gradients_2/vf/dense/MatMul_grad/MatMul_1*
T0*
_output_shapes
:	L
Ű
'gradients_2/vc/dense/MatMul_grad/MatMulMatMul:gradients_2/vc/dense/BiasAdd_grad/tuple/control_dependencyvc/dense/kernel/read*
transpose_b(*'
_output_shapes
:˙˙˙˙˙˙˙˙˙L*
T0*
transpose_a( 
Ě
)gradients_2/vc/dense/MatMul_grad/MatMul_1MatMulPlaceholder:gradients_2/vc/dense/BiasAdd_grad/tuple/control_dependency*
_output_shapes
:	L*
transpose_b( *
T0*
transpose_a(

1gradients_2/vc/dense/MatMul_grad/tuple/group_depsNoOp(^gradients_2/vc/dense/MatMul_grad/MatMul*^gradients_2/vc/dense/MatMul_grad/MatMul_1

9gradients_2/vc/dense/MatMul_grad/tuple/control_dependencyIdentity'gradients_2/vc/dense/MatMul_grad/MatMul2^gradients_2/vc/dense/MatMul_grad/tuple/group_deps*:
_class0
.,loc:@gradients_2/vc/dense/MatMul_grad/MatMul*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙L

;gradients_2/vc/dense/MatMul_grad/tuple/control_dependency_1Identity)gradients_2/vc/dense/MatMul_grad/MatMul_12^gradients_2/vc/dense/MatMul_grad/tuple/group_deps*<
_class2
0.loc:@gradients_2/vc/dense/MatMul_grad/MatMul_1*
_output_shapes
:	L*
T0
c
Reshape_36/shapeConst*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0


Reshape_36Reshape;gradients_2/vf/dense/MatMul_grad/tuple/control_dependency_1Reshape_36/shape*
Tshape0*
T0*
_output_shapes

:
c
Reshape_37/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
_output_shapes
:*
dtype0


Reshape_37Reshape<gradients_2/vf/dense/BiasAdd_grad/tuple/control_dependency_1Reshape_37/shape*
_output_shapes	
:*
Tshape0*
T0
c
Reshape_38/shapeConst*
dtype0*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙


Reshape_38Reshape=gradients_2/vf/dense_1/MatMul_grad/tuple/control_dependency_1Reshape_38/shape*
T0*
_output_shapes

:*
Tshape0
c
Reshape_39/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
_output_shapes
:*
dtype0


Reshape_39Reshape>gradients_2/vf/dense_1/BiasAdd_grad/tuple/control_dependency_1Reshape_39/shape*
_output_shapes	
:*
Tshape0*
T0
c
Reshape_40/shapeConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙


Reshape_40Reshape=gradients_2/vf/dense_2/MatMul_grad/tuple/control_dependency_1Reshape_40/shape*
T0*
Tshape0*
_output_shapes	
:
c
Reshape_41/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:


Reshape_41Reshape>gradients_2/vf/dense_2/BiasAdd_grad/tuple/control_dependency_1Reshape_41/shape*
_output_shapes
:*
T0*
Tshape0
c
Reshape_42/shapeConst*
dtype0*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙


Reshape_42Reshape;gradients_2/vc/dense/MatMul_grad/tuple/control_dependency_1Reshape_42/shape*
T0*
Tshape0*
_output_shapes

:
c
Reshape_43/shapeConst*
dtype0*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙


Reshape_43Reshape<gradients_2/vc/dense/BiasAdd_grad/tuple/control_dependency_1Reshape_43/shape*
Tshape0*
_output_shapes	
:*
T0
c
Reshape_44/shapeConst*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙*
_output_shapes
:


Reshape_44Reshape=gradients_2/vc/dense_1/MatMul_grad/tuple/control_dependency_1Reshape_44/shape*
Tshape0*
T0*
_output_shapes

:
c
Reshape_45/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:


Reshape_45Reshape>gradients_2/vc/dense_1/BiasAdd_grad/tuple/control_dependency_1Reshape_45/shape*
Tshape0*
T0*
_output_shapes	
:
c
Reshape_46/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
_output_shapes
:*
dtype0


Reshape_46Reshape=gradients_2/vc/dense_2/MatMul_grad/tuple/control_dependency_1Reshape_46/shape*
Tshape0*
T0*
_output_shapes	
:
c
Reshape_47/shapeConst*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0


Reshape_47Reshape>gradients_2/vc/dense_2/BiasAdd_grad/tuple/control_dependency_1Reshape_47/shape*
Tshape0*
_output_shapes
:*
T0
O
concat_4/axisConst*
value	B : *
dtype0*
_output_shapes
: 
ď
concat_4ConcatV2
Reshape_36
Reshape_37
Reshape_38
Reshape_39
Reshape_40
Reshape_41
Reshape_42
Reshape_43
Reshape_44
Reshape_45
Reshape_46
Reshape_47concat_4/axis*
_output_shapes

:ź
*
N*

Tidx0*
T0
l
PyFunc_4PyFuncconcat_4*
Tout
2*
_output_shapes

:ź
*
token
pyfunc_4*
Tin
2

Const_10Const*
_output_shapes
:*
dtype0*E
value<B:"0 L                  L                 
S
split_4/split_dimConst*
dtype0*
_output_shapes
: *
value	B : 
Ę
split_4SplitVPyFunc_4Const_10split_4/split_dim*
	num_split*

Tlen0*j
_output_shapesX
V::::::::::::*
T0
a
Reshape_48/shapeConst*
valueB"L      *
dtype0*
_output_shapes
:
h

Reshape_48Reshapesplit_4Reshape_48/shape*
_output_shapes
:	L*
Tshape0*
T0
[
Reshape_49/shapeConst*
valueB:*
_output_shapes
:*
dtype0
f

Reshape_49Reshape	split_4:1Reshape_49/shape*
Tshape0*
T0*
_output_shapes	
:
a
Reshape_50/shapeConst*
dtype0*
valueB"      *
_output_shapes
:
k

Reshape_50Reshape	split_4:2Reshape_50/shape* 
_output_shapes
:
*
T0*
Tshape0
[
Reshape_51/shapeConst*
_output_shapes
:*
valueB:*
dtype0
f

Reshape_51Reshape	split_4:3Reshape_51/shape*
Tshape0*
_output_shapes	
:*
T0
a
Reshape_52/shapeConst*
dtype0*
_output_shapes
:*
valueB"      
j

Reshape_52Reshape	split_4:4Reshape_52/shape*
Tshape0*
T0*
_output_shapes
:	
Z
Reshape_53/shapeConst*
dtype0*
valueB:*
_output_shapes
:
e

Reshape_53Reshape	split_4:5Reshape_53/shape*
T0*
_output_shapes
:*
Tshape0
a
Reshape_54/shapeConst*
dtype0*
_output_shapes
:*
valueB"L      
j

Reshape_54Reshape	split_4:6Reshape_54/shape*
Tshape0*
T0*
_output_shapes
:	L
[
Reshape_55/shapeConst*
_output_shapes
:*
valueB:*
dtype0
f

Reshape_55Reshape	split_4:7Reshape_55/shape*
T0*
_output_shapes	
:*
Tshape0
a
Reshape_56/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
k

Reshape_56Reshape	split_4:8Reshape_56/shape* 
_output_shapes
:
*
Tshape0*
T0
[
Reshape_57/shapeConst*
_output_shapes
:*
dtype0*
valueB:
f

Reshape_57Reshape	split_4:9Reshape_57/shape*
Tshape0*
T0*
_output_shapes	
:
a
Reshape_58/shapeConst*
valueB"      *
_output_shapes
:*
dtype0
k

Reshape_58Reshape
split_4:10Reshape_58/shape*
_output_shapes
:	*
Tshape0*
T0
Z
Reshape_59/shapeConst*
_output_shapes
:*
valueB:*
dtype0
f

Reshape_59Reshape
split_4:11Reshape_59/shape*
_output_shapes
:*
T0*
Tshape0

beta1_power_2/initial_valueConst*
_output_shapes
: * 
_class
loc:@vc/dense/bias*
valueB
 *fff?*
dtype0

beta1_power_2
VariableV2*
shared_name *
dtype0*
_output_shapes
: * 
_class
loc:@vc/dense/bias*
	container *
shape: 
ś
beta1_power_2/AssignAssignbeta1_power_2beta1_power_2/initial_value*
T0* 
_class
loc:@vc/dense/bias*
use_locking(*
_output_shapes
: *
validate_shape(
p
beta1_power_2/readIdentitybeta1_power_2* 
_class
loc:@vc/dense/bias*
_output_shapes
: *
T0

beta2_power_2/initial_valueConst* 
_class
loc:@vc/dense/bias*
dtype0*
valueB
 *wž?*
_output_shapes
: 

beta2_power_2
VariableV2*
_output_shapes
: *
shape: * 
_class
loc:@vc/dense/bias*
dtype0*
	container *
shared_name 
ś
beta2_power_2/AssignAssignbeta2_power_2beta2_power_2/initial_value*
use_locking(*
T0* 
_class
loc:@vc/dense/bias*
_output_shapes
: *
validate_shape(
p
beta2_power_2/readIdentitybeta2_power_2* 
_class
loc:@vc/dense/bias*
_output_shapes
: *
T0
Ť
6vf/dense/kernel/Adam/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*
valueB"L      *"
_class
loc:@vf/dense/kernel*
dtype0

,vf/dense/kernel/Adam/Initializer/zeros/ConstConst*"
_class
loc:@vf/dense/kernel*
dtype0*
_output_shapes
: *
valueB
 *    
ô
&vf/dense/kernel/Adam/Initializer/zerosFill6vf/dense/kernel/Adam/Initializer/zeros/shape_as_tensor,vf/dense/kernel/Adam/Initializer/zeros/Const*

index_type0*
T0*"
_class
loc:@vf/dense/kernel*
_output_shapes
:	L
Ž
vf/dense/kernel/Adam
VariableV2*
_output_shapes
:	L*
dtype0*
shape:	L*"
_class
loc:@vf/dense/kernel*
shared_name *
	container 
Ú
vf/dense/kernel/Adam/AssignAssignvf/dense/kernel/Adam&vf/dense/kernel/Adam/Initializer/zeros*
T0*
_output_shapes
:	L*
validate_shape(*
use_locking(*"
_class
loc:@vf/dense/kernel

vf/dense/kernel/Adam/readIdentityvf/dense/kernel/Adam*"
_class
loc:@vf/dense/kernel*
T0*
_output_shapes
:	L
­
8vf/dense/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*
valueB"L      *"
_class
loc:@vf/dense/kernel*
dtype0

.vf/dense/kernel/Adam_1/Initializer/zeros/ConstConst*
dtype0*
valueB
 *    *"
_class
loc:@vf/dense/kernel*
_output_shapes
: 
ú
(vf/dense/kernel/Adam_1/Initializer/zerosFill8vf/dense/kernel/Adam_1/Initializer/zeros/shape_as_tensor.vf/dense/kernel/Adam_1/Initializer/zeros/Const*"
_class
loc:@vf/dense/kernel*

index_type0*
_output_shapes
:	L*
T0
°
vf/dense/kernel/Adam_1
VariableV2*
shape:	L*
shared_name *
_output_shapes
:	L*"
_class
loc:@vf/dense/kernel*
dtype0*
	container 
ŕ
vf/dense/kernel/Adam_1/AssignAssignvf/dense/kernel/Adam_1(vf/dense/kernel/Adam_1/Initializer/zeros*
use_locking(*"
_class
loc:@vf/dense/kernel*
T0*
_output_shapes
:	L*
validate_shape(

vf/dense/kernel/Adam_1/readIdentityvf/dense/kernel/Adam_1*"
_class
loc:@vf/dense/kernel*
_output_shapes
:	L*
T0

$vf/dense/bias/Adam/Initializer/zerosConst*
dtype0* 
_class
loc:@vf/dense/bias*
_output_shapes	
:*
valueB*    
˘
vf/dense/bias/Adam
VariableV2*
_output_shapes	
:* 
_class
loc:@vf/dense/bias*
shared_name *
shape:*
	container *
dtype0
Î
vf/dense/bias/Adam/AssignAssignvf/dense/bias/Adam$vf/dense/bias/Adam/Initializer/zeros*
validate_shape(* 
_class
loc:@vf/dense/bias*
use_locking(*
T0*
_output_shapes	
:

vf/dense/bias/Adam/readIdentityvf/dense/bias/Adam*
T0*
_output_shapes	
:* 
_class
loc:@vf/dense/bias

&vf/dense/bias/Adam_1/Initializer/zerosConst*
valueB*    *
_output_shapes	
:* 
_class
loc:@vf/dense/bias*
dtype0
¤
vf/dense/bias/Adam_1
VariableV2*
shape:*
	container *
dtype0*
_output_shapes	
:* 
_class
loc:@vf/dense/bias*
shared_name 
Ô
vf/dense/bias/Adam_1/AssignAssignvf/dense/bias/Adam_1&vf/dense/bias/Adam_1/Initializer/zeros*
use_locking(*
validate_shape(* 
_class
loc:@vf/dense/bias*
T0*
_output_shapes	
:

vf/dense/bias/Adam_1/readIdentityvf/dense/bias/Adam_1*
T0* 
_class
loc:@vf/dense/bias*
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
.vf/dense_1/kernel/Adam/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    *$
_class
loc:@vf/dense_1/kernel
ý
(vf/dense_1/kernel/Adam/Initializer/zerosFill8vf/dense_1/kernel/Adam/Initializer/zeros/shape_as_tensor.vf/dense_1/kernel/Adam/Initializer/zeros/Const*

index_type0*
T0* 
_output_shapes
:
*$
_class
loc:@vf/dense_1/kernel
´
vf/dense_1/kernel/Adam
VariableV2*
shared_name * 
_output_shapes
:
*$
_class
loc:@vf/dense_1/kernel*
	container *
dtype0*
shape:

ă
vf/dense_1/kernel/Adam/AssignAssignvf/dense_1/kernel/Adam(vf/dense_1/kernel/Adam/Initializer/zeros*
T0*
use_locking(*$
_class
loc:@vf/dense_1/kernel*
validate_shape(* 
_output_shapes
:


vf/dense_1/kernel/Adam/readIdentityvf/dense_1/kernel/Adam*
T0*$
_class
loc:@vf/dense_1/kernel* 
_output_shapes
:

ą
:vf/dense_1/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
dtype0*
valueB"      *$
_class
loc:@vf/dense_1/kernel*
_output_shapes
:

0vf/dense_1/kernel/Adam_1/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    *$
_class
loc:@vf/dense_1/kernel

*vf/dense_1/kernel/Adam_1/Initializer/zerosFill:vf/dense_1/kernel/Adam_1/Initializer/zeros/shape_as_tensor0vf/dense_1/kernel/Adam_1/Initializer/zeros/Const* 
_output_shapes
:
*$
_class
loc:@vf/dense_1/kernel*

index_type0*
T0
ś
vf/dense_1/kernel/Adam_1
VariableV2*
	container *$
_class
loc:@vf/dense_1/kernel*
shared_name *
shape:
*
dtype0* 
_output_shapes
:

é
vf/dense_1/kernel/Adam_1/AssignAssignvf/dense_1/kernel/Adam_1*vf/dense_1/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*
validate_shape(* 
_output_shapes
:
*$
_class
loc:@vf/dense_1/kernel

vf/dense_1/kernel/Adam_1/readIdentityvf/dense_1/kernel/Adam_1*$
_class
loc:@vf/dense_1/kernel*
T0* 
_output_shapes
:


&vf/dense_1/bias/Adam/Initializer/zerosConst*"
_class
loc:@vf/dense_1/bias*
valueB*    *
dtype0*
_output_shapes	
:
Ś
vf/dense_1/bias/Adam
VariableV2*
_output_shapes	
:*
shape:*
	container *
dtype0*
shared_name *"
_class
loc:@vf/dense_1/bias
Ö
vf/dense_1/bias/Adam/AssignAssignvf/dense_1/bias/Adam&vf/dense_1/bias/Adam/Initializer/zeros*
use_locking(*
validate_shape(*
T0*
_output_shapes	
:*"
_class
loc:@vf/dense_1/bias

vf/dense_1/bias/Adam/readIdentityvf/dense_1/bias/Adam*
T0*"
_class
loc:@vf/dense_1/bias*
_output_shapes	
:

(vf/dense_1/bias/Adam_1/Initializer/zerosConst*"
_class
loc:@vf/dense_1/bias*
dtype0*
_output_shapes	
:*
valueB*    
¨
vf/dense_1/bias/Adam_1
VariableV2*
dtype0*
	container *
_output_shapes	
:*
shape:*
shared_name *"
_class
loc:@vf/dense_1/bias
Ü
vf/dense_1/bias/Adam_1/AssignAssignvf/dense_1/bias/Adam_1(vf/dense_1/bias/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*"
_class
loc:@vf/dense_1/bias

vf/dense_1/bias/Adam_1/readIdentityvf/dense_1/bias/Adam_1*
T0*
_output_shapes	
:*"
_class
loc:@vf/dense_1/bias
Ľ
(vf/dense_2/kernel/Adam/Initializer/zerosConst*
valueB	*    *
dtype0*
_output_shapes
:	*$
_class
loc:@vf/dense_2/kernel
˛
vf/dense_2/kernel/Adam
VariableV2*
shape:	*
dtype0*$
_class
loc:@vf/dense_2/kernel*
shared_name *
_output_shapes
:	*
	container 
â
vf/dense_2/kernel/Adam/AssignAssignvf/dense_2/kernel/Adam(vf/dense_2/kernel/Adam/Initializer/zeros*
_output_shapes
:	*$
_class
loc:@vf/dense_2/kernel*
use_locking(*
T0*
validate_shape(

vf/dense_2/kernel/Adam/readIdentityvf/dense_2/kernel/Adam*
_output_shapes
:	*
T0*$
_class
loc:@vf/dense_2/kernel
§
*vf/dense_2/kernel/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes
:	*$
_class
loc:@vf/dense_2/kernel*
valueB	*    
´
vf/dense_2/kernel/Adam_1
VariableV2*
	container *
shape:	*
_output_shapes
:	*$
_class
loc:@vf/dense_2/kernel*
shared_name *
dtype0
č
vf/dense_2/kernel/Adam_1/AssignAssignvf/dense_2/kernel/Adam_1*vf/dense_2/kernel/Adam_1/Initializer/zeros*$
_class
loc:@vf/dense_2/kernel*
use_locking(*
validate_shape(*
_output_shapes
:	*
T0

vf/dense_2/kernel/Adam_1/readIdentityvf/dense_2/kernel/Adam_1*$
_class
loc:@vf/dense_2/kernel*
T0*
_output_shapes
:	

&vf/dense_2/bias/Adam/Initializer/zerosConst*"
_class
loc:@vf/dense_2/bias*
_output_shapes
:*
valueB*    *
dtype0
¤
vf/dense_2/bias/Adam
VariableV2*
dtype0*
	container *"
_class
loc:@vf/dense_2/bias*
_output_shapes
:*
shared_name *
shape:
Ő
vf/dense_2/bias/Adam/AssignAssignvf/dense_2/bias/Adam&vf/dense_2/bias/Adam/Initializer/zeros*
_output_shapes
:*
T0*
validate_shape(*
use_locking(*"
_class
loc:@vf/dense_2/bias

vf/dense_2/bias/Adam/readIdentityvf/dense_2/bias/Adam*
_output_shapes
:*
T0*"
_class
loc:@vf/dense_2/bias

(vf/dense_2/bias/Adam_1/Initializer/zerosConst*"
_class
loc:@vf/dense_2/bias*
valueB*    *
_output_shapes
:*
dtype0
Ś
vf/dense_2/bias/Adam_1
VariableV2*
dtype0*
shared_name *
shape:*
_output_shapes
:*
	container *"
_class
loc:@vf/dense_2/bias
Ű
vf/dense_2/bias/Adam_1/AssignAssignvf/dense_2/bias/Adam_1(vf/dense_2/bias/Adam_1/Initializer/zeros*
_output_shapes
:*"
_class
loc:@vf/dense_2/bias*
T0*
use_locking(*
validate_shape(

vf/dense_2/bias/Adam_1/readIdentityvf/dense_2/bias/Adam_1*"
_class
loc:@vf/dense_2/bias*
_output_shapes
:*
T0
Ť
6vc/dense/kernel/Adam/Initializer/zeros/shape_as_tensorConst*"
_class
loc:@vc/dense/kernel*
dtype0*
valueB"L      *
_output_shapes
:

,vc/dense/kernel/Adam/Initializer/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    *"
_class
loc:@vc/dense/kernel
ô
&vc/dense/kernel/Adam/Initializer/zerosFill6vc/dense/kernel/Adam/Initializer/zeros/shape_as_tensor,vc/dense/kernel/Adam/Initializer/zeros/Const*
_output_shapes
:	L*
T0*"
_class
loc:@vc/dense/kernel*

index_type0
Ž
vc/dense/kernel/Adam
VariableV2*
dtype0*
shared_name *
shape:	L*
_output_shapes
:	L*
	container *"
_class
loc:@vc/dense/kernel
Ú
vc/dense/kernel/Adam/AssignAssignvc/dense/kernel/Adam&vc/dense/kernel/Adam/Initializer/zeros*
validate_shape(*"
_class
loc:@vc/dense/kernel*
_output_shapes
:	L*
use_locking(*
T0

vc/dense/kernel/Adam/readIdentityvc/dense/kernel/Adam*"
_class
loc:@vc/dense/kernel*
_output_shapes
:	L*
T0
­
8vc/dense/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*"
_class
loc:@vc/dense/kernel*
dtype0*
valueB"L      

.vc/dense/kernel/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    *
dtype0*"
_class
loc:@vc/dense/kernel*
_output_shapes
: 
ú
(vc/dense/kernel/Adam_1/Initializer/zerosFill8vc/dense/kernel/Adam_1/Initializer/zeros/shape_as_tensor.vc/dense/kernel/Adam_1/Initializer/zeros/Const*
T0*"
_class
loc:@vc/dense/kernel*
_output_shapes
:	L*

index_type0
°
vc/dense/kernel/Adam_1
VariableV2*
shared_name *"
_class
loc:@vc/dense/kernel*
dtype0*
shape:	L*
	container *
_output_shapes
:	L
ŕ
vc/dense/kernel/Adam_1/AssignAssignvc/dense/kernel/Adam_1(vc/dense/kernel/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes
:	L*
T0*
use_locking(*"
_class
loc:@vc/dense/kernel

vc/dense/kernel/Adam_1/readIdentityvc/dense/kernel/Adam_1*
_output_shapes
:	L*"
_class
loc:@vc/dense/kernel*
T0

$vc/dense/bias/Adam/Initializer/zerosConst*
dtype0*
valueB*    *
_output_shapes	
:* 
_class
loc:@vc/dense/bias
˘
vc/dense/bias/Adam
VariableV2*
shared_name *
shape:*
	container *
_output_shapes	
:*
dtype0* 
_class
loc:@vc/dense/bias
Î
vc/dense/bias/Adam/AssignAssignvc/dense/bias/Adam$vc/dense/bias/Adam/Initializer/zeros* 
_class
loc:@vc/dense/bias*
T0*
use_locking(*
_output_shapes	
:*
validate_shape(

vc/dense/bias/Adam/readIdentityvc/dense/bias/Adam* 
_class
loc:@vc/dense/bias*
_output_shapes	
:*
T0

&vc/dense/bias/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes	
:*
valueB*    * 
_class
loc:@vc/dense/bias
¤
vc/dense/bias/Adam_1
VariableV2*
	container * 
_class
loc:@vc/dense/bias*
shape:*
dtype0*
_output_shapes	
:*
shared_name 
Ô
vc/dense/bias/Adam_1/AssignAssignvc/dense/bias/Adam_1&vc/dense/bias/Adam_1/Initializer/zeros* 
_class
loc:@vc/dense/bias*
use_locking(*
validate_shape(*
_output_shapes	
:*
T0

vc/dense/bias/Adam_1/readIdentityvc/dense/bias/Adam_1*
_output_shapes	
:* 
_class
loc:@vc/dense/bias*
T0
Ż
8vc/dense_1/kernel/Adam/Initializer/zeros/shape_as_tensorConst*$
_class
loc:@vc/dense_1/kernel*
valueB"      *
dtype0*
_output_shapes
:

.vc/dense_1/kernel/Adam/Initializer/zeros/ConstConst*
_output_shapes
: *
valueB
 *    *
dtype0*$
_class
loc:@vc/dense_1/kernel
ý
(vc/dense_1/kernel/Adam/Initializer/zerosFill8vc/dense_1/kernel/Adam/Initializer/zeros/shape_as_tensor.vc/dense_1/kernel/Adam/Initializer/zeros/Const* 
_output_shapes
:
*
T0*$
_class
loc:@vc/dense_1/kernel*

index_type0
´
vc/dense_1/kernel/Adam
VariableV2*$
_class
loc:@vc/dense_1/kernel*
	container * 
_output_shapes
:
*
shared_name *
dtype0*
shape:

ă
vc/dense_1/kernel/Adam/AssignAssignvc/dense_1/kernel/Adam(vc/dense_1/kernel/Adam/Initializer/zeros*
T0*
use_locking(* 
_output_shapes
:
*$
_class
loc:@vc/dense_1/kernel*
validate_shape(

vc/dense_1/kernel/Adam/readIdentityvc/dense_1/kernel/Adam* 
_output_shapes
:
*
T0*$
_class
loc:@vc/dense_1/kernel
ą
:vc/dense_1/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*$
_class
loc:@vc/dense_1/kernel*
_output_shapes
:*
valueB"      *
dtype0

0vc/dense_1/kernel/Adam_1/Initializer/zeros/ConstConst*$
_class
loc:@vc/dense_1/kernel*
_output_shapes
: *
valueB
 *    *
dtype0

*vc/dense_1/kernel/Adam_1/Initializer/zerosFill:vc/dense_1/kernel/Adam_1/Initializer/zeros/shape_as_tensor0vc/dense_1/kernel/Adam_1/Initializer/zeros/Const*
T0*

index_type0* 
_output_shapes
:
*$
_class
loc:@vc/dense_1/kernel
ś
vc/dense_1/kernel/Adam_1
VariableV2*
shared_name *
dtype0*
	container *$
_class
loc:@vc/dense_1/kernel*
shape:
* 
_output_shapes
:

é
vc/dense_1/kernel/Adam_1/AssignAssignvc/dense_1/kernel/Adam_1*vc/dense_1/kernel/Adam_1/Initializer/zeros*
validate_shape(*$
_class
loc:@vc/dense_1/kernel* 
_output_shapes
:
*
use_locking(*
T0

vc/dense_1/kernel/Adam_1/readIdentityvc/dense_1/kernel/Adam_1*
T0*$
_class
loc:@vc/dense_1/kernel* 
_output_shapes
:


&vc/dense_1/bias/Adam/Initializer/zerosConst*"
_class
loc:@vc/dense_1/bias*
valueB*    *
dtype0*
_output_shapes	
:
Ś
vc/dense_1/bias/Adam
VariableV2*
	container *"
_class
loc:@vc/dense_1/bias*
shape:*
shared_name *
_output_shapes	
:*
dtype0
Ö
vc/dense_1/bias/Adam/AssignAssignvc/dense_1/bias/Adam&vc/dense_1/bias/Adam/Initializer/zeros*
_output_shapes	
:*
T0*
use_locking(*
validate_shape(*"
_class
loc:@vc/dense_1/bias

vc/dense_1/bias/Adam/readIdentityvc/dense_1/bias/Adam*"
_class
loc:@vc/dense_1/bias*
_output_shapes	
:*
T0

(vc/dense_1/bias/Adam_1/Initializer/zerosConst*
valueB*    *
dtype0*
_output_shapes	
:*"
_class
loc:@vc/dense_1/bias
¨
vc/dense_1/bias/Adam_1
VariableV2*
dtype0*"
_class
loc:@vc/dense_1/bias*
_output_shapes	
:*
shape:*
shared_name *
	container 
Ü
vc/dense_1/bias/Adam_1/AssignAssignvc/dense_1/bias/Adam_1(vc/dense_1/bias/Adam_1/Initializer/zeros*
T0*
_output_shapes	
:*
validate_shape(*"
_class
loc:@vc/dense_1/bias*
use_locking(

vc/dense_1/bias/Adam_1/readIdentityvc/dense_1/bias/Adam_1*
_output_shapes	
:*"
_class
loc:@vc/dense_1/bias*
T0
Ľ
(vc/dense_2/kernel/Adam/Initializer/zerosConst*$
_class
loc:@vc/dense_2/kernel*
valueB	*    *
dtype0*
_output_shapes
:	
˛
vc/dense_2/kernel/Adam
VariableV2*
shared_name *
	container *
shape:	*
dtype0*$
_class
loc:@vc/dense_2/kernel*
_output_shapes
:	
â
vc/dense_2/kernel/Adam/AssignAssignvc/dense_2/kernel/Adam(vc/dense_2/kernel/Adam/Initializer/zeros*$
_class
loc:@vc/dense_2/kernel*
use_locking(*
_output_shapes
:	*
validate_shape(*
T0

vc/dense_2/kernel/Adam/readIdentityvc/dense_2/kernel/Adam*$
_class
loc:@vc/dense_2/kernel*
T0*
_output_shapes
:	
§
*vc/dense_2/kernel/Adam_1/Initializer/zerosConst*
_output_shapes
:	*
dtype0*$
_class
loc:@vc/dense_2/kernel*
valueB	*    
´
vc/dense_2/kernel/Adam_1
VariableV2*
	container *
shape:	*
dtype0*
_output_shapes
:	*
shared_name *$
_class
loc:@vc/dense_2/kernel
č
vc/dense_2/kernel/Adam_1/AssignAssignvc/dense_2/kernel/Adam_1*vc/dense_2/kernel/Adam_1/Initializer/zeros*
_output_shapes
:	*
use_locking(*
validate_shape(*$
_class
loc:@vc/dense_2/kernel*
T0

vc/dense_2/kernel/Adam_1/readIdentityvc/dense_2/kernel/Adam_1*
T0*
_output_shapes
:	*$
_class
loc:@vc/dense_2/kernel

&vc/dense_2/bias/Adam/Initializer/zerosConst*
dtype0*
_output_shapes
:*
valueB*    *"
_class
loc:@vc/dense_2/bias
¤
vc/dense_2/bias/Adam
VariableV2*"
_class
loc:@vc/dense_2/bias*
_output_shapes
:*
	container *
shared_name *
shape:*
dtype0
Ő
vc/dense_2/bias/Adam/AssignAssignvc/dense_2/bias/Adam&vc/dense_2/bias/Adam/Initializer/zeros*
T0*
validate_shape(*
use_locking(*"
_class
loc:@vc/dense_2/bias*
_output_shapes
:

vc/dense_2/bias/Adam/readIdentityvc/dense_2/bias/Adam*"
_class
loc:@vc/dense_2/bias*
T0*
_output_shapes
:

(vc/dense_2/bias/Adam_1/Initializer/zerosConst*"
_class
loc:@vc/dense_2/bias*
dtype0*
_output_shapes
:*
valueB*    
Ś
vc/dense_2/bias/Adam_1
VariableV2*
shape:*
dtype0*
	container *
_output_shapes
:*"
_class
loc:@vc/dense_2/bias*
shared_name 
Ű
vc/dense_2/bias/Adam_1/AssignAssignvc/dense_2/bias/Adam_1(vc/dense_2/bias/Adam_1/Initializer/zeros*
_output_shapes
:*
use_locking(*"
_class
loc:@vc/dense_2/bias*
T0*
validate_shape(

vc/dense_2/bias/Adam_1/readIdentityvc/dense_2/bias/Adam_1*
_output_shapes
:*"
_class
loc:@vc/dense_2/bias*
T0
Y
Adam_2/learning_rateConst*
dtype0*
_output_shapes
: *
valueB
 *o:
Q
Adam_2/beta1Const*
_output_shapes
: *
valueB
 *fff?*
dtype0
Q
Adam_2/beta2Const*
valueB
 *wž?*
_output_shapes
: *
dtype0
S
Adam_2/epsilonConst*
_output_shapes
: *
valueB
 *wĚ+2*
dtype0
Ţ
'Adam_2/update_vf/dense/kernel/ApplyAdam	ApplyAdamvf/dense/kernelvf/dense/kernel/Adamvf/dense/kernel/Adam_1beta1_power_2/readbeta2_power_2/readAdam_2/learning_rateAdam_2/beta1Adam_2/beta2Adam_2/epsilon
Reshape_48*"
_class
loc:@vf/dense/kernel*
use_locking( *
_output_shapes
:	L*
T0*
use_nesterov( 
Đ
%Adam_2/update_vf/dense/bias/ApplyAdam	ApplyAdamvf/dense/biasvf/dense/bias/Adamvf/dense/bias/Adam_1beta1_power_2/readbeta2_power_2/readAdam_2/learning_rateAdam_2/beta1Adam_2/beta2Adam_2/epsilon
Reshape_49* 
_class
loc:@vf/dense/bias*
T0*
use_nesterov( *
_output_shapes	
:*
use_locking( 
é
)Adam_2/update_vf/dense_1/kernel/ApplyAdam	ApplyAdamvf/dense_1/kernelvf/dense_1/kernel/Adamvf/dense_1/kernel/Adam_1beta1_power_2/readbeta2_power_2/readAdam_2/learning_rateAdam_2/beta1Adam_2/beta2Adam_2/epsilon
Reshape_50*
use_locking( *
use_nesterov( *$
_class
loc:@vf/dense_1/kernel*
T0* 
_output_shapes
:

Ú
'Adam_2/update_vf/dense_1/bias/ApplyAdam	ApplyAdamvf/dense_1/biasvf/dense_1/bias/Adamvf/dense_1/bias/Adam_1beta1_power_2/readbeta2_power_2/readAdam_2/learning_rateAdam_2/beta1Adam_2/beta2Adam_2/epsilon
Reshape_51*
_output_shapes	
:*
use_nesterov( *
T0*
use_locking( *"
_class
loc:@vf/dense_1/bias
č
)Adam_2/update_vf/dense_2/kernel/ApplyAdam	ApplyAdamvf/dense_2/kernelvf/dense_2/kernel/Adamvf/dense_2/kernel/Adam_1beta1_power_2/readbeta2_power_2/readAdam_2/learning_rateAdam_2/beta1Adam_2/beta2Adam_2/epsilon
Reshape_52*
_output_shapes
:	*
use_locking( *
T0*$
_class
loc:@vf/dense_2/kernel*
use_nesterov( 
Ů
'Adam_2/update_vf/dense_2/bias/ApplyAdam	ApplyAdamvf/dense_2/biasvf/dense_2/bias/Adamvf/dense_2/bias/Adam_1beta1_power_2/readbeta2_power_2/readAdam_2/learning_rateAdam_2/beta1Adam_2/beta2Adam_2/epsilon
Reshape_53*
use_nesterov( *
T0*"
_class
loc:@vf/dense_2/bias*
_output_shapes
:*
use_locking( 
Ţ
'Adam_2/update_vc/dense/kernel/ApplyAdam	ApplyAdamvc/dense/kernelvc/dense/kernel/Adamvc/dense/kernel/Adam_1beta1_power_2/readbeta2_power_2/readAdam_2/learning_rateAdam_2/beta1Adam_2/beta2Adam_2/epsilon
Reshape_54*
use_locking( *
use_nesterov( *
T0*"
_class
loc:@vc/dense/kernel*
_output_shapes
:	L
Đ
%Adam_2/update_vc/dense/bias/ApplyAdam	ApplyAdamvc/dense/biasvc/dense/bias/Adamvc/dense/bias/Adam_1beta1_power_2/readbeta2_power_2/readAdam_2/learning_rateAdam_2/beta1Adam_2/beta2Adam_2/epsilon
Reshape_55*
use_locking( *
use_nesterov( *
T0*
_output_shapes	
:* 
_class
loc:@vc/dense/bias
é
)Adam_2/update_vc/dense_1/kernel/ApplyAdam	ApplyAdamvc/dense_1/kernelvc/dense_1/kernel/Adamvc/dense_1/kernel/Adam_1beta1_power_2/readbeta2_power_2/readAdam_2/learning_rateAdam_2/beta1Adam_2/beta2Adam_2/epsilon
Reshape_56*
use_nesterov( *
T0* 
_output_shapes
:
*
use_locking( *$
_class
loc:@vc/dense_1/kernel
Ú
'Adam_2/update_vc/dense_1/bias/ApplyAdam	ApplyAdamvc/dense_1/biasvc/dense_1/bias/Adamvc/dense_1/bias/Adam_1beta1_power_2/readbeta2_power_2/readAdam_2/learning_rateAdam_2/beta1Adam_2/beta2Adam_2/epsilon
Reshape_57*"
_class
loc:@vc/dense_1/bias*
use_locking( *
use_nesterov( *
T0*
_output_shapes	
:
č
)Adam_2/update_vc/dense_2/kernel/ApplyAdam	ApplyAdamvc/dense_2/kernelvc/dense_2/kernel/Adamvc/dense_2/kernel/Adam_1beta1_power_2/readbeta2_power_2/readAdam_2/learning_rateAdam_2/beta1Adam_2/beta2Adam_2/epsilon
Reshape_58*
use_locking( *
use_nesterov( *
T0*$
_class
loc:@vc/dense_2/kernel*
_output_shapes
:	
Ů
'Adam_2/update_vc/dense_2/bias/ApplyAdam	ApplyAdamvc/dense_2/biasvc/dense_2/bias/Adamvc/dense_2/bias/Adam_1beta1_power_2/readbeta2_power_2/readAdam_2/learning_rateAdam_2/beta1Adam_2/beta2Adam_2/epsilon
Reshape_59*
T0*"
_class
loc:@vc/dense_2/bias*
_output_shapes
:*
use_nesterov( *
use_locking( 
ň

Adam_2/mulMulbeta1_power_2/readAdam_2/beta1&^Adam_2/update_vc/dense/bias/ApplyAdam(^Adam_2/update_vc/dense/kernel/ApplyAdam(^Adam_2/update_vc/dense_1/bias/ApplyAdam*^Adam_2/update_vc/dense_1/kernel/ApplyAdam(^Adam_2/update_vc/dense_2/bias/ApplyAdam*^Adam_2/update_vc/dense_2/kernel/ApplyAdam&^Adam_2/update_vf/dense/bias/ApplyAdam(^Adam_2/update_vf/dense/kernel/ApplyAdam(^Adam_2/update_vf/dense_1/bias/ApplyAdam*^Adam_2/update_vf/dense_1/kernel/ApplyAdam(^Adam_2/update_vf/dense_2/bias/ApplyAdam*^Adam_2/update_vf/dense_2/kernel/ApplyAdam*
T0*
_output_shapes
: * 
_class
loc:@vc/dense/bias

Adam_2/AssignAssignbeta1_power_2
Adam_2/mul* 
_class
loc:@vc/dense/bias*
T0*
_output_shapes
: *
validate_shape(*
use_locking( 
ô
Adam_2/mul_1Mulbeta2_power_2/readAdam_2/beta2&^Adam_2/update_vc/dense/bias/ApplyAdam(^Adam_2/update_vc/dense/kernel/ApplyAdam(^Adam_2/update_vc/dense_1/bias/ApplyAdam*^Adam_2/update_vc/dense_1/kernel/ApplyAdam(^Adam_2/update_vc/dense_2/bias/ApplyAdam*^Adam_2/update_vc/dense_2/kernel/ApplyAdam&^Adam_2/update_vf/dense/bias/ApplyAdam(^Adam_2/update_vf/dense/kernel/ApplyAdam(^Adam_2/update_vf/dense_1/bias/ApplyAdam*^Adam_2/update_vf/dense_1/kernel/ApplyAdam(^Adam_2/update_vf/dense_2/bias/ApplyAdam*^Adam_2/update_vf/dense_2/kernel/ApplyAdam*
T0* 
_class
loc:@vc/dense/bias*
_output_shapes
: 
˘
Adam_2/Assign_1Assignbeta2_power_2Adam_2/mul_1*
T0*
_output_shapes
: * 
_class
loc:@vc/dense/bias*
use_locking( *
validate_shape(
Ź
Adam_2NoOp^Adam_2/Assign^Adam_2/Assign_1&^Adam_2/update_vc/dense/bias/ApplyAdam(^Adam_2/update_vc/dense/kernel/ApplyAdam(^Adam_2/update_vc/dense_1/bias/ApplyAdam*^Adam_2/update_vc/dense_1/kernel/ApplyAdam(^Adam_2/update_vc/dense_2/bias/ApplyAdam*^Adam_2/update_vc/dense_2/kernel/ApplyAdam&^Adam_2/update_vf/dense/bias/ApplyAdam(^Adam_2/update_vf/dense/kernel/ApplyAdam(^Adam_2/update_vf/dense_1/bias/ApplyAdam*^Adam_2/update_vf/dense_1/kernel/ApplyAdam(^Adam_2/update_vf/dense_2/bias/ApplyAdam*^Adam_2/update_vf/dense_2/kernel/ApplyAdam
l
Reshape_60/shapeConst^Adam_2*
dtype0*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙
r

Reshape_60Reshapevf/dense/kernel/readReshape_60/shape*
Tshape0*
_output_shapes

:*
T0
l
Reshape_61/shapeConst^Adam_2*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
o

Reshape_61Reshapevf/dense/bias/readReshape_61/shape*
_output_shapes	
:*
T0*
Tshape0
l
Reshape_62/shapeConst^Adam_2*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0
t

Reshape_62Reshapevf/dense_1/kernel/readReshape_62/shape*
Tshape0*
T0*
_output_shapes

:
l
Reshape_63/shapeConst^Adam_2*
valueB:
˙˙˙˙˙˙˙˙˙*
_output_shapes
:*
dtype0
q

Reshape_63Reshapevf/dense_1/bias/readReshape_63/shape*
T0*
Tshape0*
_output_shapes	
:
l
Reshape_64/shapeConst^Adam_2*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0
s

Reshape_64Reshapevf/dense_2/kernel/readReshape_64/shape*
_output_shapes	
:*
Tshape0*
T0
l
Reshape_65/shapeConst^Adam_2*
dtype0*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙
p

Reshape_65Reshapevf/dense_2/bias/readReshape_65/shape*
Tshape0*
_output_shapes
:*
T0
l
Reshape_66/shapeConst^Adam_2*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙*
_output_shapes
:
r

Reshape_66Reshapevc/dense/kernel/readReshape_66/shape*
T0*
Tshape0*
_output_shapes

:
l
Reshape_67/shapeConst^Adam_2*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙
o

Reshape_67Reshapevc/dense/bias/readReshape_67/shape*
_output_shapes	
:*
T0*
Tshape0
l
Reshape_68/shapeConst^Adam_2*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
t

Reshape_68Reshapevc/dense_1/kernel/readReshape_68/shape*
Tshape0*
_output_shapes

:*
T0
l
Reshape_69/shapeConst^Adam_2*
valueB:
˙˙˙˙˙˙˙˙˙*
_output_shapes
:*
dtype0
q

Reshape_69Reshapevc/dense_1/bias/readReshape_69/shape*
Tshape0*
T0*
_output_shapes	
:
l
Reshape_70/shapeConst^Adam_2*
valueB:
˙˙˙˙˙˙˙˙˙*
_output_shapes
:*
dtype0
s

Reshape_70Reshapevc/dense_2/kernel/readReshape_70/shape*
Tshape0*
T0*
_output_shapes	
:
l
Reshape_71/shapeConst^Adam_2*
dtype0*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙
p

Reshape_71Reshapevc/dense_2/bias/readReshape_71/shape*
T0*
Tshape0*
_output_shapes
:
X
concat_5/axisConst^Adam_2*
value	B : *
_output_shapes
: *
dtype0
ď
concat_5ConcatV2
Reshape_60
Reshape_61
Reshape_62
Reshape_63
Reshape_64
Reshape_65
Reshape_66
Reshape_67
Reshape_68
Reshape_69
Reshape_70
Reshape_71concat_5/axis*
T0*
_output_shapes

:ź
*
N*

Tidx0
h
PyFunc_5PyFuncconcat_5*
Tout
2*
token
pyfunc_5*
Tin
2*
_output_shapes
:

Const_11Const^Adam_2*
_output_shapes
:*
dtype0*E
value<B:"0 L                  L                 
\
split_5/split_dimConst^Adam_2*
_output_shapes
: *
dtype0*
value	B : 
¤
split_5SplitVPyFunc_5Const_11split_5/split_dim*
	num_split*D
_output_shapes2
0::::::::::::*

Tlen0*
T0
j
Reshape_72/shapeConst^Adam_2*
valueB"L      *
dtype0*
_output_shapes
:
h

Reshape_72Reshapesplit_5Reshape_72/shape*
_output_shapes
:	L*
Tshape0*
T0
d
Reshape_73/shapeConst^Adam_2*
dtype0*
_output_shapes
:*
valueB:
f

Reshape_73Reshape	split_5:1Reshape_73/shape*
_output_shapes	
:*
T0*
Tshape0
j
Reshape_74/shapeConst^Adam_2*
_output_shapes
:*
dtype0*
valueB"      
k

Reshape_74Reshape	split_5:2Reshape_74/shape* 
_output_shapes
:
*
Tshape0*
T0
d
Reshape_75/shapeConst^Adam_2*
valueB:*
dtype0*
_output_shapes
:
f

Reshape_75Reshape	split_5:3Reshape_75/shape*
Tshape0*
T0*
_output_shapes	
:
j
Reshape_76/shapeConst^Adam_2*
valueB"      *
dtype0*
_output_shapes
:
j

Reshape_76Reshape	split_5:4Reshape_76/shape*
Tshape0*
_output_shapes
:	*
T0
c
Reshape_77/shapeConst^Adam_2*
_output_shapes
:*
dtype0*
valueB:
e

Reshape_77Reshape	split_5:5Reshape_77/shape*
_output_shapes
:*
Tshape0*
T0
j
Reshape_78/shapeConst^Adam_2*
valueB"L      *
dtype0*
_output_shapes
:
j

Reshape_78Reshape	split_5:6Reshape_78/shape*
Tshape0*
_output_shapes
:	L*
T0
d
Reshape_79/shapeConst^Adam_2*
_output_shapes
:*
dtype0*
valueB:
f

Reshape_79Reshape	split_5:7Reshape_79/shape*
Tshape0*
T0*
_output_shapes	
:
j
Reshape_80/shapeConst^Adam_2*
valueB"      *
dtype0*
_output_shapes
:
k

Reshape_80Reshape	split_5:8Reshape_80/shape*
T0*
Tshape0* 
_output_shapes
:

d
Reshape_81/shapeConst^Adam_2*
valueB:*
dtype0*
_output_shapes
:
f

Reshape_81Reshape	split_5:9Reshape_81/shape*
_output_shapes	
:*
Tshape0*
T0
j
Reshape_82/shapeConst^Adam_2*
dtype0*
_output_shapes
:*
valueB"      
k

Reshape_82Reshape
split_5:10Reshape_82/shape*
Tshape0*
_output_shapes
:	*
T0
c
Reshape_83/shapeConst^Adam_2*
valueB:*
dtype0*
_output_shapes
:
f

Reshape_83Reshape
split_5:11Reshape_83/shape*
_output_shapes
:*
T0*
Tshape0
Ś
Assign_9Assignvf/dense/kernel
Reshape_72*
T0*
validate_shape(*
use_locking(*"
_class
loc:@vf/dense/kernel*
_output_shapes
:	L

	Assign_10Assignvf/dense/bias
Reshape_73*
use_locking(*
_output_shapes	
:* 
_class
loc:@vf/dense/bias*
T0*
validate_shape(
Ź
	Assign_11Assignvf/dense_1/kernel
Reshape_74*
use_locking(*$
_class
loc:@vf/dense_1/kernel* 
_output_shapes
:
*
T0*
validate_shape(
Ł
	Assign_12Assignvf/dense_1/bias
Reshape_75*"
_class
loc:@vf/dense_1/bias*
use_locking(*
_output_shapes	
:*
validate_shape(*
T0
Ť
	Assign_13Assignvf/dense_2/kernel
Reshape_76*
use_locking(*$
_class
loc:@vf/dense_2/kernel*
_output_shapes
:	*
validate_shape(*
T0
˘
	Assign_14Assignvf/dense_2/bias
Reshape_77*
validate_shape(*
T0*"
_class
loc:@vf/dense_2/bias*
_output_shapes
:*
use_locking(
§
	Assign_15Assignvc/dense/kernel
Reshape_78*
_output_shapes
:	L*"
_class
loc:@vc/dense/kernel*
T0*
validate_shape(*
use_locking(

	Assign_16Assignvc/dense/bias
Reshape_79* 
_class
loc:@vc/dense/bias*
T0*
validate_shape(*
_output_shapes	
:*
use_locking(
Ź
	Assign_17Assignvc/dense_1/kernel
Reshape_80*
T0*$
_class
loc:@vc/dense_1/kernel*
use_locking(*
validate_shape(* 
_output_shapes
:

Ł
	Assign_18Assignvc/dense_1/bias
Reshape_81*
T0*
_output_shapes	
:*
validate_shape(*"
_class
loc:@vc/dense_1/bias*
use_locking(
Ť
	Assign_19Assignvc/dense_2/kernel
Reshape_82*$
_class
loc:@vc/dense_2/kernel*
_output_shapes
:	*
use_locking(*
T0*
validate_shape(
˘
	Assign_20Assignvc/dense_2/bias
Reshape_83*
validate_shape(*
T0*"
_class
loc:@vc/dense_2/bias*
_output_shapes
:*
use_locking(
Ź
group_deps_4NoOp^Adam_2
^Assign_10
^Assign_11
^Assign_12
^Assign_13
^Assign_14
^Assign_15
^Assign_16
^Assign_17
^Assign_18
^Assign_19
^Assign_20	^Assign_9
,
group_deps_5NoOp^Adam_2^group_deps_4
ż
initNoOp^beta1_power/Assign^beta1_power_1/Assign^beta1_power_2/Assign^beta2_power/Assign^beta2_power_1/Assign^beta2_power_2/Assign"^penalty/penalty_param/Adam/Assign$^penalty/penalty_param/Adam_1/Assign$^penalty/penalty_param/Adam_2/Assign$^penalty/penalty_param/Adam_3/Assign^penalty/penalty_param/Assign^pi/dense/bias/Adam/Assign^pi/dense/bias/Adam_1/Assign^pi/dense/bias/Assign^pi/dense/kernel/Adam/Assign^pi/dense/kernel/Adam_1/Assign^pi/dense/kernel/Assign^pi/dense_1/bias/Adam/Assign^pi/dense_1/bias/Adam_1/Assign^pi/dense_1/bias/Assign^pi/dense_1/kernel/Adam/Assign ^pi/dense_1/kernel/Adam_1/Assign^pi/dense_1/kernel/Assign^pi/dense_2/bias/Adam/Assign^pi/dense_2/bias/Adam_1/Assign^pi/dense_2/bias/Assign^pi/dense_2/kernel/Adam/Assign ^pi/dense_2/kernel/Adam_1/Assign^pi/dense_2/kernel/Assign^pi/log_std/Adam/Assign^pi/log_std/Adam_1/Assign^pi/log_std/Assign^vc/dense/bias/Adam/Assign^vc/dense/bias/Adam_1/Assign^vc/dense/bias/Assign^vc/dense/kernel/Adam/Assign^vc/dense/kernel/Adam_1/Assign^vc/dense/kernel/Assign^vc/dense_1/bias/Adam/Assign^vc/dense_1/bias/Adam_1/Assign^vc/dense_1/bias/Assign^vc/dense_1/kernel/Adam/Assign ^vc/dense_1/kernel/Adam_1/Assign^vc/dense_1/kernel/Assign^vc/dense_2/bias/Adam/Assign^vc/dense_2/bias/Adam_1/Assign^vc/dense_2/bias/Assign^vc/dense_2/kernel/Adam/Assign ^vc/dense_2/kernel/Adam_1/Assign^vc/dense_2/kernel/Assign^vf/dense/bias/Adam/Assign^vf/dense/bias/Adam_1/Assign^vf/dense/bias/Assign^vf/dense/kernel/Adam/Assign^vf/dense/kernel/Adam_1/Assign^vf/dense/kernel/Assign^vf/dense_1/bias/Adam/Assign^vf/dense_1/bias/Adam_1/Assign^vf/dense_1/bias/Assign^vf/dense_1/kernel/Adam/Assign ^vf/dense_1/kernel/Adam_1/Assign^vf/dense_1/kernel/Assign^vf/dense_2/bias/Adam/Assign^vf/dense_2/bias/Adam_1/Assign^vf/dense_2/bias/Assign^vf/dense_2/kernel/Adam/Assign ^vf/dense_2/kernel/Adam_1/Assign^vf/dense_2/kernel/Assign
c
Reshape_84/shapeConst*
dtype0*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙
r

Reshape_84Reshapepi/dense/kernel/readReshape_84/shape*
T0*
Tshape0*
_output_shapes

:
c
Reshape_85/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
o

Reshape_85Reshapepi/dense/bias/readReshape_85/shape*
T0*
Tshape0*
_output_shapes	
:
c
Reshape_86/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
_output_shapes
:*
dtype0
t

Reshape_86Reshapepi/dense_1/kernel/readReshape_86/shape*
T0*
Tshape0*
_output_shapes

:
c
Reshape_87/shapeConst*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0
q

Reshape_87Reshapepi/dense_1/bias/readReshape_87/shape*
Tshape0*
_output_shapes	
:*
T0
c
Reshape_88/shapeConst*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙*
_output_shapes
:
s

Reshape_88Reshapepi/dense_2/kernel/readReshape_88/shape*
T0*
Tshape0*
_output_shapes	
:
c
Reshape_89/shapeConst*
dtype0*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙
p

Reshape_89Reshapepi/dense_2/bias/readReshape_89/shape*
_output_shapes
:*
Tshape0*
T0
c
Reshape_90/shapeConst*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙*
_output_shapes
:
k

Reshape_90Reshapepi/log_std/readReshape_90/shape*
T0*
_output_shapes
:*
Tshape0
c
Reshape_91/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
_output_shapes
:*
dtype0
r

Reshape_91Reshapevf/dense/kernel/readReshape_91/shape*
_output_shapes

:*
T0*
Tshape0
c
Reshape_92/shapeConst*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙*
_output_shapes
:
o

Reshape_92Reshapevf/dense/bias/readReshape_92/shape*
_output_shapes	
:*
Tshape0*
T0
c
Reshape_93/shapeConst*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙*
_output_shapes
:
t

Reshape_93Reshapevf/dense_1/kernel/readReshape_93/shape*
_output_shapes

:*
T0*
Tshape0
c
Reshape_94/shapeConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙
q

Reshape_94Reshapevf/dense_1/bias/readReshape_94/shape*
_output_shapes	
:*
T0*
Tshape0
c
Reshape_95/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
s

Reshape_95Reshapevf/dense_2/kernel/readReshape_95/shape*
_output_shapes	
:*
Tshape0*
T0
c
Reshape_96/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
p

Reshape_96Reshapevf/dense_2/bias/readReshape_96/shape*
T0*
Tshape0*
_output_shapes
:
c
Reshape_97/shapeConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙
r

Reshape_97Reshapevc/dense/kernel/readReshape_97/shape*
T0*
_output_shapes

:*
Tshape0
c
Reshape_98/shapeConst*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙*
_output_shapes
:
o

Reshape_98Reshapevc/dense/bias/readReshape_98/shape*
T0*
_output_shapes	
:*
Tshape0
c
Reshape_99/shapeConst*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙*
_output_shapes
:
t

Reshape_99Reshapevc/dense_1/kernel/readReshape_99/shape*
Tshape0*
_output_shapes

:*
T0
d
Reshape_100/shapeConst*
dtype0*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙
s
Reshape_100Reshapevc/dense_1/bias/readReshape_100/shape*
_output_shapes	
:*
Tshape0*
T0
d
Reshape_101/shapeConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙
u
Reshape_101Reshapevc/dense_2/kernel/readReshape_101/shape*
T0*
_output_shapes	
:*
Tshape0
d
Reshape_102/shapeConst*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0
r
Reshape_102Reshapevc/dense_2/bias/readReshape_102/shape*
Tshape0*
T0*
_output_shapes
:
d
Reshape_103/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
_output_shapes
:*
dtype0
x
Reshape_103Reshapepenalty/penalty_param/readReshape_103/shape*
_output_shapes
:*
Tshape0*
T0
d
Reshape_104/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
_output_shapes
:*
dtype0
n
Reshape_104Reshapebeta1_power/readReshape_104/shape*
Tshape0*
T0*
_output_shapes
:
d
Reshape_105/shapeConst*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙*
_output_shapes
:
n
Reshape_105Reshapebeta2_power/readReshape_105/shape*
T0*
_output_shapes
:*
Tshape0
d
Reshape_106/shapeConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙
}
Reshape_106Reshapepenalty/penalty_param/Adam/readReshape_106/shape*
T0*
_output_shapes
:*
Tshape0
d
Reshape_107/shapeConst*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0

Reshape_107Reshape!penalty/penalty_param/Adam_1/readReshape_107/shape*
T0*
Tshape0*
_output_shapes
:
d
Reshape_108/shapeConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙
p
Reshape_108Reshapebeta1_power_1/readReshape_108/shape*
_output_shapes
:*
T0*
Tshape0
d
Reshape_109/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
p
Reshape_109Reshapebeta2_power_1/readReshape_109/shape*
Tshape0*
T0*
_output_shapes
:
d
Reshape_110/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
_output_shapes
:*
dtype0
y
Reshape_110Reshapepi/dense/kernel/Adam/readReshape_110/shape*
T0*
Tshape0*
_output_shapes

:
d
Reshape_111/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
{
Reshape_111Reshapepi/dense/kernel/Adam_1/readReshape_111/shape*
T0*
Tshape0*
_output_shapes

:
d
Reshape_112/shapeConst*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0
v
Reshape_112Reshapepi/dense/bias/Adam/readReshape_112/shape*
T0*
Tshape0*
_output_shapes	
:
d
Reshape_113/shapeConst*
dtype0*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙
x
Reshape_113Reshapepi/dense/bias/Adam_1/readReshape_113/shape*
_output_shapes	
:*
T0*
Tshape0
d
Reshape_114/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
{
Reshape_114Reshapepi/dense_1/kernel/Adam/readReshape_114/shape*
T0*
_output_shapes

:*
Tshape0
d
Reshape_115/shapeConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙
}
Reshape_115Reshapepi/dense_1/kernel/Adam_1/readReshape_115/shape*
Tshape0*
_output_shapes

:*
T0
d
Reshape_116/shapeConst*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙*
_output_shapes
:
x
Reshape_116Reshapepi/dense_1/bias/Adam/readReshape_116/shape*
_output_shapes	
:*
T0*
Tshape0
d
Reshape_117/shapeConst*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙*
_output_shapes
:
z
Reshape_117Reshapepi/dense_1/bias/Adam_1/readReshape_117/shape*
T0*
Tshape0*
_output_shapes	
:
d
Reshape_118/shapeConst*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0
z
Reshape_118Reshapepi/dense_2/kernel/Adam/readReshape_118/shape*
Tshape0*
_output_shapes	
:*
T0
d
Reshape_119/shapeConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙
|
Reshape_119Reshapepi/dense_2/kernel/Adam_1/readReshape_119/shape*
T0*
Tshape0*
_output_shapes	
:
d
Reshape_120/shapeConst*
dtype0*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙
w
Reshape_120Reshapepi/dense_2/bias/Adam/readReshape_120/shape*
_output_shapes
:*
Tshape0*
T0
d
Reshape_121/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
y
Reshape_121Reshapepi/dense_2/bias/Adam_1/readReshape_121/shape*
Tshape0*
T0*
_output_shapes
:
d
Reshape_122/shapeConst*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙*
_output_shapes
:
r
Reshape_122Reshapepi/log_std/Adam/readReshape_122/shape*
_output_shapes
:*
Tshape0*
T0
d
Reshape_123/shapeConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙
t
Reshape_123Reshapepi/log_std/Adam_1/readReshape_123/shape*
_output_shapes
:*
T0*
Tshape0
d
Reshape_124/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:

Reshape_124Reshape!penalty/penalty_param/Adam_2/readReshape_124/shape*
Tshape0*
_output_shapes
:*
T0
d
Reshape_125/shapeConst*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0

Reshape_125Reshape!penalty/penalty_param/Adam_3/readReshape_125/shape*
T0*
Tshape0*
_output_shapes
:
d
Reshape_126/shapeConst*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0
p
Reshape_126Reshapebeta1_power_2/readReshape_126/shape*
T0*
_output_shapes
:*
Tshape0
d
Reshape_127/shapeConst*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙*
_output_shapes
:
p
Reshape_127Reshapebeta2_power_2/readReshape_127/shape*
_output_shapes
:*
Tshape0*
T0
d
Reshape_128/shapeConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙
y
Reshape_128Reshapevf/dense/kernel/Adam/readReshape_128/shape*
Tshape0*
_output_shapes

:*
T0
d
Reshape_129/shapeConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙
{
Reshape_129Reshapevf/dense/kernel/Adam_1/readReshape_129/shape*
T0*
Tshape0*
_output_shapes

:
d
Reshape_130/shapeConst*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0
v
Reshape_130Reshapevf/dense/bias/Adam/readReshape_130/shape*
T0*
_output_shapes	
:*
Tshape0
d
Reshape_131/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
_output_shapes
:*
dtype0
x
Reshape_131Reshapevf/dense/bias/Adam_1/readReshape_131/shape*
Tshape0*
_output_shapes	
:*
T0
d
Reshape_132/shapeConst*
dtype0*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙
{
Reshape_132Reshapevf/dense_1/kernel/Adam/readReshape_132/shape*
Tshape0*
_output_shapes

:*
T0
d
Reshape_133/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
_output_shapes
:*
dtype0
}
Reshape_133Reshapevf/dense_1/kernel/Adam_1/readReshape_133/shape*
Tshape0*
T0*
_output_shapes

:
d
Reshape_134/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
x
Reshape_134Reshapevf/dense_1/bias/Adam/readReshape_134/shape*
Tshape0*
_output_shapes	
:*
T0
d
Reshape_135/shapeConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙
z
Reshape_135Reshapevf/dense_1/bias/Adam_1/readReshape_135/shape*
T0*
_output_shapes	
:*
Tshape0
d
Reshape_136/shapeConst*
dtype0*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙
z
Reshape_136Reshapevf/dense_2/kernel/Adam/readReshape_136/shape*
Tshape0*
_output_shapes	
:*
T0
d
Reshape_137/shapeConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙
|
Reshape_137Reshapevf/dense_2/kernel/Adam_1/readReshape_137/shape*
T0*
_output_shapes	
:*
Tshape0
d
Reshape_138/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
w
Reshape_138Reshapevf/dense_2/bias/Adam/readReshape_138/shape*
_output_shapes
:*
T0*
Tshape0
d
Reshape_139/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
y
Reshape_139Reshapevf/dense_2/bias/Adam_1/readReshape_139/shape*
_output_shapes
:*
Tshape0*
T0
d
Reshape_140/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
y
Reshape_140Reshapevc/dense/kernel/Adam/readReshape_140/shape*
T0*
Tshape0*
_output_shapes

:
d
Reshape_141/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
{
Reshape_141Reshapevc/dense/kernel/Adam_1/readReshape_141/shape*
Tshape0*
T0*
_output_shapes

:
d
Reshape_142/shapeConst*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0
v
Reshape_142Reshapevc/dense/bias/Adam/readReshape_142/shape*
T0*
_output_shapes	
:*
Tshape0
d
Reshape_143/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
_output_shapes
:*
dtype0
x
Reshape_143Reshapevc/dense/bias/Adam_1/readReshape_143/shape*
Tshape0*
T0*
_output_shapes	
:
d
Reshape_144/shapeConst*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0
{
Reshape_144Reshapevc/dense_1/kernel/Adam/readReshape_144/shape*
_output_shapes

:*
T0*
Tshape0
d
Reshape_145/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
_output_shapes
:*
dtype0
}
Reshape_145Reshapevc/dense_1/kernel/Adam_1/readReshape_145/shape*
T0*
_output_shapes

:*
Tshape0
d
Reshape_146/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
x
Reshape_146Reshapevc/dense_1/bias/Adam/readReshape_146/shape*
T0*
Tshape0*
_output_shapes	
:
d
Reshape_147/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
_output_shapes
:*
dtype0
z
Reshape_147Reshapevc/dense_1/bias/Adam_1/readReshape_147/shape*
T0*
Tshape0*
_output_shapes	
:
d
Reshape_148/shapeConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙
z
Reshape_148Reshapevc/dense_2/kernel/Adam/readReshape_148/shape*
_output_shapes	
:*
T0*
Tshape0
d
Reshape_149/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
_output_shapes
:*
dtype0
|
Reshape_149Reshapevc/dense_2/kernel/Adam_1/readReshape_149/shape*
_output_shapes	
:*
Tshape0*
T0
d
Reshape_150/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
w
Reshape_150Reshapevc/dense_2/bias/Adam/readReshape_150/shape*
T0*
Tshape0*
_output_shapes
:
d
Reshape_151/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
_output_shapes
:*
dtype0
y
Reshape_151Reshapevc/dense_2/bias/Adam_1/readReshape_151/shape*
_output_shapes
:*
Tshape0*
T0
O
concat_6/axisConst*
value	B : *
dtype0*
_output_shapes
: 
Ă
concat_6ConcatV2
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
Reshape_99Reshape_100Reshape_101Reshape_102Reshape_103Reshape_104Reshape_105Reshape_106Reshape_107Reshape_108Reshape_109Reshape_110Reshape_111Reshape_112Reshape_113Reshape_114Reshape_115Reshape_116Reshape_117Reshape_118Reshape_119Reshape_120Reshape_121Reshape_122Reshape_123Reshape_124Reshape_125Reshape_126Reshape_127Reshape_128Reshape_129Reshape_130Reshape_131Reshape_132Reshape_133Reshape_134Reshape_135Reshape_136Reshape_137Reshape_138Reshape_139Reshape_140Reshape_141Reshape_142Reshape_143Reshape_144Reshape_145Reshape_146Reshape_147Reshape_148Reshape_149Reshape_150Reshape_151concat_6/axis*
T0*
_output_shapes

:/*
ND*

Tidx0
h
PyFunc_6PyFuncconcat_6*
_output_shapes
:*
Tin
2*
Tout
2*
token
pyfunc_6
ĺ
Const_12Const*
dtype0*
_output_shapes
:D*¨
valueBD" L                     L                  L                                       L   L                                                   L   L                                 L   L                                
S
split_6/split_dimConst*
value	B : *
_output_shapes
: *
dtype0

split_6SplitVPyFunc_6Const_12split_6/split_dim*
	num_splitD*
T0*

Tlen0*Ś
_output_shapes
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
b
Reshape_152/shapeConst*
_output_shapes
:*
dtype0*
valueB"L      
j
Reshape_152Reshapesplit_6Reshape_152/shape*
Tshape0*
T0*
_output_shapes
:	L
\
Reshape_153/shapeConst*
_output_shapes
:*
valueB:*
dtype0
h
Reshape_153Reshape	split_6:1Reshape_153/shape*
_output_shapes	
:*
Tshape0*
T0
b
Reshape_154/shapeConst*
valueB"      *
_output_shapes
:*
dtype0
m
Reshape_154Reshape	split_6:2Reshape_154/shape* 
_output_shapes
:
*
T0*
Tshape0
\
Reshape_155/shapeConst*
_output_shapes
:*
dtype0*
valueB:
h
Reshape_155Reshape	split_6:3Reshape_155/shape*
_output_shapes	
:*
Tshape0*
T0
b
Reshape_156/shapeConst*
valueB"      *
_output_shapes
:*
dtype0
l
Reshape_156Reshape	split_6:4Reshape_156/shape*
_output_shapes
:	*
Tshape0*
T0
[
Reshape_157/shapeConst*
dtype0*
valueB:*
_output_shapes
:
g
Reshape_157Reshape	split_6:5Reshape_157/shape*
_output_shapes
:*
Tshape0*
T0
[
Reshape_158/shapeConst*
dtype0*
valueB:*
_output_shapes
:
g
Reshape_158Reshape	split_6:6Reshape_158/shape*
Tshape0*
_output_shapes
:*
T0
b
Reshape_159/shapeConst*
valueB"L      *
_output_shapes
:*
dtype0
l
Reshape_159Reshape	split_6:7Reshape_159/shape*
T0*
Tshape0*
_output_shapes
:	L
\
Reshape_160/shapeConst*
_output_shapes
:*
dtype0*
valueB:
h
Reshape_160Reshape	split_6:8Reshape_160/shape*
Tshape0*
T0*
_output_shapes	
:
b
Reshape_161/shapeConst*
dtype0*
valueB"      *
_output_shapes
:
m
Reshape_161Reshape	split_6:9Reshape_161/shape*
Tshape0*
T0* 
_output_shapes
:

\
Reshape_162/shapeConst*
_output_shapes
:*
valueB:*
dtype0
i
Reshape_162Reshape
split_6:10Reshape_162/shape*
_output_shapes	
:*
T0*
Tshape0
b
Reshape_163/shapeConst*
dtype0*
valueB"      *
_output_shapes
:
m
Reshape_163Reshape
split_6:11Reshape_163/shape*
Tshape0*
T0*
_output_shapes
:	
[
Reshape_164/shapeConst*
dtype0*
_output_shapes
:*
valueB:
h
Reshape_164Reshape
split_6:12Reshape_164/shape*
T0*
Tshape0*
_output_shapes
:
b
Reshape_165/shapeConst*
valueB"L      *
dtype0*
_output_shapes
:
m
Reshape_165Reshape
split_6:13Reshape_165/shape*
_output_shapes
:	L*
Tshape0*
T0
\
Reshape_166/shapeConst*
dtype0*
valueB:*
_output_shapes
:
i
Reshape_166Reshape
split_6:14Reshape_166/shape*
T0*
Tshape0*
_output_shapes	
:
b
Reshape_167/shapeConst*
_output_shapes
:*
dtype0*
valueB"      
n
Reshape_167Reshape
split_6:15Reshape_167/shape*
Tshape0* 
_output_shapes
:
*
T0
\
Reshape_168/shapeConst*
_output_shapes
:*
valueB:*
dtype0
i
Reshape_168Reshape
split_6:16Reshape_168/shape*
T0*
Tshape0*
_output_shapes	
:
b
Reshape_169/shapeConst*
dtype0*
_output_shapes
:*
valueB"      
m
Reshape_169Reshape
split_6:17Reshape_169/shape*
_output_shapes
:	*
T0*
Tshape0
[
Reshape_170/shapeConst*
dtype0*
_output_shapes
:*
valueB:
h
Reshape_170Reshape
split_6:18Reshape_170/shape*
Tshape0*
T0*
_output_shapes
:
T
Reshape_171/shapeConst*
_output_shapes
: *
dtype0*
valueB 
d
Reshape_171Reshape
split_6:19Reshape_171/shape*
T0*
_output_shapes
: *
Tshape0
T
Reshape_172/shapeConst*
valueB *
_output_shapes
: *
dtype0
d
Reshape_172Reshape
split_6:20Reshape_172/shape*
Tshape0*
T0*
_output_shapes
: 
T
Reshape_173/shapeConst*
valueB *
_output_shapes
: *
dtype0
d
Reshape_173Reshape
split_6:21Reshape_173/shape*
_output_shapes
: *
Tshape0*
T0
T
Reshape_174/shapeConst*
dtype0*
valueB *
_output_shapes
: 
d
Reshape_174Reshape
split_6:22Reshape_174/shape*
_output_shapes
: *
Tshape0*
T0
T
Reshape_175/shapeConst*
_output_shapes
: *
valueB *
dtype0
d
Reshape_175Reshape
split_6:23Reshape_175/shape*
_output_shapes
: *
T0*
Tshape0
T
Reshape_176/shapeConst*
dtype0*
valueB *
_output_shapes
: 
d
Reshape_176Reshape
split_6:24Reshape_176/shape*
T0*
Tshape0*
_output_shapes
: 
T
Reshape_177/shapeConst*
_output_shapes
: *
valueB *
dtype0
d
Reshape_177Reshape
split_6:25Reshape_177/shape*
_output_shapes
: *
T0*
Tshape0
b
Reshape_178/shapeConst*
_output_shapes
:*
dtype0*
valueB"L      
m
Reshape_178Reshape
split_6:26Reshape_178/shape*
T0*
Tshape0*
_output_shapes
:	L
b
Reshape_179/shapeConst*
valueB"L      *
_output_shapes
:*
dtype0
m
Reshape_179Reshape
split_6:27Reshape_179/shape*
T0*
Tshape0*
_output_shapes
:	L
\
Reshape_180/shapeConst*
valueB:*
dtype0*
_output_shapes
:
i
Reshape_180Reshape
split_6:28Reshape_180/shape*
Tshape0*
T0*
_output_shapes	
:
\
Reshape_181/shapeConst*
_output_shapes
:*
dtype0*
valueB:
i
Reshape_181Reshape
split_6:29Reshape_181/shape*
Tshape0*
_output_shapes	
:*
T0
b
Reshape_182/shapeConst*
_output_shapes
:*
valueB"      *
dtype0
n
Reshape_182Reshape
split_6:30Reshape_182/shape*
Tshape0* 
_output_shapes
:
*
T0
b
Reshape_183/shapeConst*
_output_shapes
:*
valueB"      *
dtype0
n
Reshape_183Reshape
split_6:31Reshape_183/shape*
Tshape0*
T0* 
_output_shapes
:

\
Reshape_184/shapeConst*
valueB:*
dtype0*
_output_shapes
:
i
Reshape_184Reshape
split_6:32Reshape_184/shape*
_output_shapes	
:*
T0*
Tshape0
\
Reshape_185/shapeConst*
dtype0*
valueB:*
_output_shapes
:
i
Reshape_185Reshape
split_6:33Reshape_185/shape*
T0*
Tshape0*
_output_shapes	
:
b
Reshape_186/shapeConst*
_output_shapes
:*
valueB"      *
dtype0
m
Reshape_186Reshape
split_6:34Reshape_186/shape*
Tshape0*
_output_shapes
:	*
T0
b
Reshape_187/shapeConst*
_output_shapes
:*
dtype0*
valueB"      
m
Reshape_187Reshape
split_6:35Reshape_187/shape*
Tshape0*
T0*
_output_shapes
:	
[
Reshape_188/shapeConst*
valueB:*
dtype0*
_output_shapes
:
h
Reshape_188Reshape
split_6:36Reshape_188/shape*
T0*
_output_shapes
:*
Tshape0
[
Reshape_189/shapeConst*
valueB:*
_output_shapes
:*
dtype0
h
Reshape_189Reshape
split_6:37Reshape_189/shape*
Tshape0*
_output_shapes
:*
T0
[
Reshape_190/shapeConst*
_output_shapes
:*
valueB:*
dtype0
h
Reshape_190Reshape
split_6:38Reshape_190/shape*
T0*
_output_shapes
:*
Tshape0
[
Reshape_191/shapeConst*
dtype0*
valueB:*
_output_shapes
:
h
Reshape_191Reshape
split_6:39Reshape_191/shape*
T0*
_output_shapes
:*
Tshape0
T
Reshape_192/shapeConst*
valueB *
dtype0*
_output_shapes
: 
d
Reshape_192Reshape
split_6:40Reshape_192/shape*
_output_shapes
: *
T0*
Tshape0
T
Reshape_193/shapeConst*
valueB *
dtype0*
_output_shapes
: 
d
Reshape_193Reshape
split_6:41Reshape_193/shape*
T0*
Tshape0*
_output_shapes
: 
T
Reshape_194/shapeConst*
valueB *
_output_shapes
: *
dtype0
d
Reshape_194Reshape
split_6:42Reshape_194/shape*
T0*
Tshape0*
_output_shapes
: 
T
Reshape_195/shapeConst*
dtype0*
_output_shapes
: *
valueB 
d
Reshape_195Reshape
split_6:43Reshape_195/shape*
Tshape0*
T0*
_output_shapes
: 
b
Reshape_196/shapeConst*
valueB"L      *
_output_shapes
:*
dtype0
m
Reshape_196Reshape
split_6:44Reshape_196/shape*
_output_shapes
:	L*
Tshape0*
T0
b
Reshape_197/shapeConst*
dtype0*
_output_shapes
:*
valueB"L      
m
Reshape_197Reshape
split_6:45Reshape_197/shape*
T0*
Tshape0*
_output_shapes
:	L
\
Reshape_198/shapeConst*
dtype0*
valueB:*
_output_shapes
:
i
Reshape_198Reshape
split_6:46Reshape_198/shape*
_output_shapes	
:*
Tshape0*
T0
\
Reshape_199/shapeConst*
dtype0*
_output_shapes
:*
valueB:
i
Reshape_199Reshape
split_6:47Reshape_199/shape*
_output_shapes	
:*
T0*
Tshape0
b
Reshape_200/shapeConst*
dtype0*
valueB"      *
_output_shapes
:
n
Reshape_200Reshape
split_6:48Reshape_200/shape*
Tshape0*
T0* 
_output_shapes
:

b
Reshape_201/shapeConst*
valueB"      *
_output_shapes
:*
dtype0
n
Reshape_201Reshape
split_6:49Reshape_201/shape* 
_output_shapes
:
*
Tshape0*
T0
\
Reshape_202/shapeConst*
_output_shapes
:*
valueB:*
dtype0
i
Reshape_202Reshape
split_6:50Reshape_202/shape*
Tshape0*
_output_shapes	
:*
T0
\
Reshape_203/shapeConst*
valueB:*
dtype0*
_output_shapes
:
i
Reshape_203Reshape
split_6:51Reshape_203/shape*
T0*
Tshape0*
_output_shapes	
:
b
Reshape_204/shapeConst*
dtype0*
valueB"      *
_output_shapes
:
m
Reshape_204Reshape
split_6:52Reshape_204/shape*
T0*
_output_shapes
:	*
Tshape0
b
Reshape_205/shapeConst*
_output_shapes
:*
dtype0*
valueB"      
m
Reshape_205Reshape
split_6:53Reshape_205/shape*
_output_shapes
:	*
T0*
Tshape0
[
Reshape_206/shapeConst*
valueB:*
_output_shapes
:*
dtype0
h
Reshape_206Reshape
split_6:54Reshape_206/shape*
_output_shapes
:*
Tshape0*
T0
[
Reshape_207/shapeConst*
_output_shapes
:*
valueB:*
dtype0
h
Reshape_207Reshape
split_6:55Reshape_207/shape*
T0*
_output_shapes
:*
Tshape0
b
Reshape_208/shapeConst*
dtype0*
valueB"L      *
_output_shapes
:
m
Reshape_208Reshape
split_6:56Reshape_208/shape*
Tshape0*
T0*
_output_shapes
:	L
b
Reshape_209/shapeConst*
_output_shapes
:*
valueB"L      *
dtype0
m
Reshape_209Reshape
split_6:57Reshape_209/shape*
T0*
_output_shapes
:	L*
Tshape0
\
Reshape_210/shapeConst*
dtype0*
_output_shapes
:*
valueB:
i
Reshape_210Reshape
split_6:58Reshape_210/shape*
T0*
Tshape0*
_output_shapes	
:
\
Reshape_211/shapeConst*
_output_shapes
:*
dtype0*
valueB:
i
Reshape_211Reshape
split_6:59Reshape_211/shape*
_output_shapes	
:*
T0*
Tshape0
b
Reshape_212/shapeConst*
_output_shapes
:*
valueB"      *
dtype0
n
Reshape_212Reshape
split_6:60Reshape_212/shape* 
_output_shapes
:
*
Tshape0*
T0
b
Reshape_213/shapeConst*
valueB"      *
_output_shapes
:*
dtype0
n
Reshape_213Reshape
split_6:61Reshape_213/shape*
Tshape0*
T0* 
_output_shapes
:

\
Reshape_214/shapeConst*
valueB:*
_output_shapes
:*
dtype0
i
Reshape_214Reshape
split_6:62Reshape_214/shape*
_output_shapes	
:*
Tshape0*
T0
\
Reshape_215/shapeConst*
valueB:*
_output_shapes
:*
dtype0
i
Reshape_215Reshape
split_6:63Reshape_215/shape*
T0*
_output_shapes	
:*
Tshape0
b
Reshape_216/shapeConst*
valueB"      *
_output_shapes
:*
dtype0
m
Reshape_216Reshape
split_6:64Reshape_216/shape*
T0*
Tshape0*
_output_shapes
:	
b
Reshape_217/shapeConst*
_output_shapes
:*
dtype0*
valueB"      
m
Reshape_217Reshape
split_6:65Reshape_217/shape*
T0*
_output_shapes
:	*
Tshape0
[
Reshape_218/shapeConst*
_output_shapes
:*
valueB:*
dtype0
h
Reshape_218Reshape
split_6:66Reshape_218/shape*
Tshape0*
T0*
_output_shapes
:
[
Reshape_219/shapeConst*
valueB:*
dtype0*
_output_shapes
:
h
Reshape_219Reshape
split_6:67Reshape_219/shape*
T0*
_output_shapes
:*
Tshape0
¨
	Assign_21Assignpi/dense/kernelReshape_152*"
_class
loc:@pi/dense/kernel*
_output_shapes
:	L*
T0*
use_locking(*
validate_shape(
 
	Assign_22Assignpi/dense/biasReshape_153*
use_locking(*
validate_shape(* 
_class
loc:@pi/dense/bias*
T0*
_output_shapes	
:
­
	Assign_23Assignpi/dense_1/kernelReshape_154*
validate_shape(*$
_class
loc:@pi/dense_1/kernel*
use_locking(* 
_output_shapes
:
*
T0
¤
	Assign_24Assignpi/dense_1/biasReshape_155*
validate_shape(*
_output_shapes	
:*
T0*
use_locking(*"
_class
loc:@pi/dense_1/bias
Ź
	Assign_25Assignpi/dense_2/kernelReshape_156*
validate_shape(*
_output_shapes
:	*
use_locking(*
T0*$
_class
loc:@pi/dense_2/kernel
Ł
	Assign_26Assignpi/dense_2/biasReshape_157*
use_locking(*
T0*
_output_shapes
:*
validate_shape(*"
_class
loc:@pi/dense_2/bias

	Assign_27Assign
pi/log_stdReshape_158*
_output_shapes
:*
_class
loc:@pi/log_std*
validate_shape(*
use_locking(*
T0
¨
	Assign_28Assignvf/dense/kernelReshape_159*
T0*"
_class
loc:@vf/dense/kernel*
validate_shape(*
use_locking(*
_output_shapes
:	L
 
	Assign_29Assignvf/dense/biasReshape_160*
use_locking(*
validate_shape(* 
_class
loc:@vf/dense/bias*
_output_shapes	
:*
T0
­
	Assign_30Assignvf/dense_1/kernelReshape_161*
T0*
use_locking(*
validate_shape(*$
_class
loc:@vf/dense_1/kernel* 
_output_shapes
:

¤
	Assign_31Assignvf/dense_1/biasReshape_162*
use_locking(*
T0*"
_class
loc:@vf/dense_1/bias*
validate_shape(*
_output_shapes	
:
Ź
	Assign_32Assignvf/dense_2/kernelReshape_163*
_output_shapes
:	*
use_locking(*
T0*$
_class
loc:@vf/dense_2/kernel*
validate_shape(
Ł
	Assign_33Assignvf/dense_2/biasReshape_164*
use_locking(*
validate_shape(*
T0*
_output_shapes
:*"
_class
loc:@vf/dense_2/bias
¨
	Assign_34Assignvc/dense/kernelReshape_165*"
_class
loc:@vc/dense/kernel*
T0*
validate_shape(*
_output_shapes
:	L*
use_locking(
 
	Assign_35Assignvc/dense/biasReshape_166*
use_locking(*
_output_shapes	
:* 
_class
loc:@vc/dense/bias*
validate_shape(*
T0
­
	Assign_36Assignvc/dense_1/kernelReshape_167*
use_locking(* 
_output_shapes
:
*
T0*
validate_shape(*$
_class
loc:@vc/dense_1/kernel
¤
	Assign_37Assignvc/dense_1/biasReshape_168*
validate_shape(*
use_locking(*
_output_shapes	
:*"
_class
loc:@vc/dense_1/bias*
T0
Ź
	Assign_38Assignvc/dense_2/kernelReshape_169*
validate_shape(*$
_class
loc:@vc/dense_2/kernel*
use_locking(*
T0*
_output_shapes
:	
Ł
	Assign_39Assignvc/dense_2/biasReshape_170*"
_class
loc:@vc/dense_2/bias*
_output_shapes
:*
T0*
validate_shape(*
use_locking(
Ť
	Assign_40Assignpenalty/penalty_paramReshape_171*
T0*
use_locking(*
_output_shapes
: *(
_class
loc:@penalty/penalty_param*
validate_shape(
Ą
	Assign_41Assignbeta1_powerReshape_172*
use_locking(*
_output_shapes
: *(
_class
loc:@penalty/penalty_param*
validate_shape(*
T0
Ą
	Assign_42Assignbeta2_powerReshape_173*
T0*
validate_shape(*(
_class
loc:@penalty/penalty_param*
use_locking(*
_output_shapes
: 
°
	Assign_43Assignpenalty/penalty_param/AdamReshape_174*
_output_shapes
: *(
_class
loc:@penalty/penalty_param*
validate_shape(*
T0*
use_locking(
˛
	Assign_44Assignpenalty/penalty_param/Adam_1Reshape_175*
validate_shape(*
T0*
_output_shapes
: *(
_class
loc:@penalty/penalty_param*
use_locking(
Ł
	Assign_45Assignbeta1_power_1Reshape_176*(
_class
loc:@penalty/penalty_param*
_output_shapes
: *
use_locking(*
validate_shape(*
T0
Ł
	Assign_46Assignbeta2_power_1Reshape_177*
_output_shapes
: *
validate_shape(*
use_locking(*(
_class
loc:@penalty/penalty_param*
T0
­
	Assign_47Assignpi/dense/kernel/AdamReshape_178*
_output_shapes
:	L*
validate_shape(*
use_locking(*
T0*"
_class
loc:@pi/dense/kernel
Ż
	Assign_48Assignpi/dense/kernel/Adam_1Reshape_179*
use_locking(*"
_class
loc:@pi/dense/kernel*
T0*
validate_shape(*
_output_shapes
:	L
Ľ
	Assign_49Assignpi/dense/bias/AdamReshape_180*
T0*
_output_shapes	
:* 
_class
loc:@pi/dense/bias*
use_locking(*
validate_shape(
§
	Assign_50Assignpi/dense/bias/Adam_1Reshape_181*
T0*
_output_shapes	
:*
use_locking(*
validate_shape(* 
_class
loc:@pi/dense/bias
˛
	Assign_51Assignpi/dense_1/kernel/AdamReshape_182*
T0*
use_locking(*$
_class
loc:@pi/dense_1/kernel*
validate_shape(* 
_output_shapes
:

´
	Assign_52Assignpi/dense_1/kernel/Adam_1Reshape_183*$
_class
loc:@pi/dense_1/kernel* 
_output_shapes
:
*
use_locking(*
T0*
validate_shape(
Š
	Assign_53Assignpi/dense_1/bias/AdamReshape_184*"
_class
loc:@pi/dense_1/bias*
use_locking(*
_output_shapes	
:*
T0*
validate_shape(
Ť
	Assign_54Assignpi/dense_1/bias/Adam_1Reshape_185*"
_class
loc:@pi/dense_1/bias*
T0*
use_locking(*
validate_shape(*
_output_shapes	
:
ą
	Assign_55Assignpi/dense_2/kernel/AdamReshape_186*
T0*
validate_shape(*
use_locking(*$
_class
loc:@pi/dense_2/kernel*
_output_shapes
:	
ł
	Assign_56Assignpi/dense_2/kernel/Adam_1Reshape_187*
T0*
_output_shapes
:	*
use_locking(*
validate_shape(*$
_class
loc:@pi/dense_2/kernel
¨
	Assign_57Assignpi/dense_2/bias/AdamReshape_188*
validate_shape(*
use_locking(*
_output_shapes
:*"
_class
loc:@pi/dense_2/bias*
T0
Ş
	Assign_58Assignpi/dense_2/bias/Adam_1Reshape_189*
T0*
validate_shape(*
_output_shapes
:*
use_locking(*"
_class
loc:@pi/dense_2/bias

	Assign_59Assignpi/log_std/AdamReshape_190*
_output_shapes
:*
T0*
use_locking(*
validate_shape(*
_class
loc:@pi/log_std
 
	Assign_60Assignpi/log_std/Adam_1Reshape_191*
validate_shape(*
_class
loc:@pi/log_std*
_output_shapes
:*
use_locking(*
T0
˛
	Assign_61Assignpenalty/penalty_param/Adam_2Reshape_192*
T0*
validate_shape(*(
_class
loc:@penalty/penalty_param*
use_locking(*
_output_shapes
: 
˛
	Assign_62Assignpenalty/penalty_param/Adam_3Reshape_193*
_output_shapes
: *
T0*
use_locking(*
validate_shape(*(
_class
loc:@penalty/penalty_param

	Assign_63Assignbeta1_power_2Reshape_194*
validate_shape(*
T0*
_output_shapes
: *
use_locking(* 
_class
loc:@vc/dense/bias

	Assign_64Assignbeta2_power_2Reshape_195*
T0*
_output_shapes
: *
validate_shape(* 
_class
loc:@vc/dense/bias*
use_locking(
­
	Assign_65Assignvf/dense/kernel/AdamReshape_196*"
_class
loc:@vf/dense/kernel*
_output_shapes
:	L*
T0*
use_locking(*
validate_shape(
Ż
	Assign_66Assignvf/dense/kernel/Adam_1Reshape_197*
use_locking(*
T0*
_output_shapes
:	L*
validate_shape(*"
_class
loc:@vf/dense/kernel
Ľ
	Assign_67Assignvf/dense/bias/AdamReshape_198*
validate_shape(* 
_class
loc:@vf/dense/bias*
_output_shapes	
:*
use_locking(*
T0
§
	Assign_68Assignvf/dense/bias/Adam_1Reshape_199* 
_class
loc:@vf/dense/bias*
_output_shapes	
:*
T0*
use_locking(*
validate_shape(
˛
	Assign_69Assignvf/dense_1/kernel/AdamReshape_200*
use_locking(*
validate_shape(*$
_class
loc:@vf/dense_1/kernel* 
_output_shapes
:
*
T0
´
	Assign_70Assignvf/dense_1/kernel/Adam_1Reshape_201*
use_locking(*
validate_shape(* 
_output_shapes
:
*$
_class
loc:@vf/dense_1/kernel*
T0
Š
	Assign_71Assignvf/dense_1/bias/AdamReshape_202*
validate_shape(*
use_locking(*
T0*"
_class
loc:@vf/dense_1/bias*
_output_shapes	
:
Ť
	Assign_72Assignvf/dense_1/bias/Adam_1Reshape_203*
_output_shapes	
:*
use_locking(*
validate_shape(*
T0*"
_class
loc:@vf/dense_1/bias
ą
	Assign_73Assignvf/dense_2/kernel/AdamReshape_204*
T0*
_output_shapes
:	*
validate_shape(*$
_class
loc:@vf/dense_2/kernel*
use_locking(
ł
	Assign_74Assignvf/dense_2/kernel/Adam_1Reshape_205*
validate_shape(*
T0*
use_locking(*$
_class
loc:@vf/dense_2/kernel*
_output_shapes
:	
¨
	Assign_75Assignvf/dense_2/bias/AdamReshape_206*
use_locking(*"
_class
loc:@vf/dense_2/bias*
T0*
validate_shape(*
_output_shapes
:
Ş
	Assign_76Assignvf/dense_2/bias/Adam_1Reshape_207*
_output_shapes
:*"
_class
loc:@vf/dense_2/bias*
T0*
use_locking(*
validate_shape(
­
	Assign_77Assignvc/dense/kernel/AdamReshape_208*
use_locking(*
_output_shapes
:	L*
validate_shape(*
T0*"
_class
loc:@vc/dense/kernel
Ż
	Assign_78Assignvc/dense/kernel/Adam_1Reshape_209*
validate_shape(*
use_locking(*"
_class
loc:@vc/dense/kernel*
T0*
_output_shapes
:	L
Ľ
	Assign_79Assignvc/dense/bias/AdamReshape_210* 
_class
loc:@vc/dense/bias*
T0*
_output_shapes	
:*
validate_shape(*
use_locking(
§
	Assign_80Assignvc/dense/bias/Adam_1Reshape_211*
validate_shape(*
use_locking(*
T0* 
_class
loc:@vc/dense/bias*
_output_shapes	
:
˛
	Assign_81Assignvc/dense_1/kernel/AdamReshape_212*$
_class
loc:@vc/dense_1/kernel*
T0*
use_locking(*
validate_shape(* 
_output_shapes
:

´
	Assign_82Assignvc/dense_1/kernel/Adam_1Reshape_213*
use_locking(*$
_class
loc:@vc/dense_1/kernel* 
_output_shapes
:
*
T0*
validate_shape(
Š
	Assign_83Assignvc/dense_1/bias/AdamReshape_214*"
_class
loc:@vc/dense_1/bias*
validate_shape(*
T0*
_output_shapes	
:*
use_locking(
Ť
	Assign_84Assignvc/dense_1/bias/Adam_1Reshape_215*"
_class
loc:@vc/dense_1/bias*
_output_shapes	
:*
validate_shape(*
T0*
use_locking(
ą
	Assign_85Assignvc/dense_2/kernel/AdamReshape_216*
validate_shape(*
use_locking(*$
_class
loc:@vc/dense_2/kernel*
T0*
_output_shapes
:	
ł
	Assign_86Assignvc/dense_2/kernel/Adam_1Reshape_217*
use_locking(*
T0*$
_class
loc:@vc/dense_2/kernel*
validate_shape(*
_output_shapes
:	
¨
	Assign_87Assignvc/dense_2/bias/AdamReshape_218*"
_class
loc:@vc/dense_2/bias*
use_locking(*
T0*
validate_shape(*
_output_shapes
:
Ş
	Assign_88Assignvc/dense_2/bias/Adam_1Reshape_219*
T0*
_output_shapes
:*
use_locking(*
validate_shape(*"
_class
loc:@vc/dense_2/bias
Ä
group_deps_6NoOp
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
^Assign_79
^Assign_80
^Assign_81
^Assign_82
^Assign_83
^Assign_84
^Assign_85
^Assign_86
^Assign_87
^Assign_88
Y
save/filename/inputConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
n
save/filenamePlaceholderWithDefaultsave/filename/input*
_output_shapes
: *
dtype0*
shape: 
e

save/ConstPlaceholderWithDefaultsave/filename*
_output_shapes
: *
shape: *
dtype0

save/StringJoin/inputs_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+_temp_184755524bdb4d44af0e8c5752e39036/part
u
save/StringJoin
StringJoin
save/Constsave/StringJoin/inputs_1*
_output_shapes
: *
	separator *
N
Q
save/num_shardsConst*
_output_shapes
: *
dtype0*
value	B :
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
ő
save/SaveV2/tensor_namesConst*¨
valueBDBbeta1_powerBbeta1_power_1Bbeta1_power_2Bbeta2_powerBbeta2_power_1Bbeta2_power_2Bpenalty/penalty_paramBpenalty/penalty_param/AdamBpenalty/penalty_param/Adam_1Bpenalty/penalty_param/Adam_2Bpenalty/penalty_param/Adam_3Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1B
pi/log_stdBpi/log_std/AdamBpi/log_std/Adam_1Bvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1*
_output_shapes
:D*
dtype0
î
save/SaveV2/shape_and_slicesConst*
dtype0*
valueBDB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
_output_shapes
:D
Ę
save/SaveV2SaveV2save/ShardedFilenamesave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta1_power_2beta2_powerbeta2_power_1beta2_power_2penalty/penalty_parampenalty/penalty_param/Adampenalty/penalty_param/Adam_1penalty/penalty_param/Adam_2penalty/penalty_param/Adam_3pi/dense/biaspi/dense/bias/Adampi/dense/bias/Adam_1pi/dense/kernelpi/dense/kernel/Adampi/dense/kernel/Adam_1pi/dense_1/biaspi/dense_1/bias/Adampi/dense_1/bias/Adam_1pi/dense_1/kernelpi/dense_1/kernel/Adampi/dense_1/kernel/Adam_1pi/dense_2/biaspi/dense_2/bias/Adampi/dense_2/bias/Adam_1pi/dense_2/kernelpi/dense_2/kernel/Adampi/dense_2/kernel/Adam_1
pi/log_stdpi/log_std/Adampi/log_std/Adam_1vc/dense/biasvc/dense/bias/Adamvc/dense/bias/Adam_1vc/dense/kernelvc/dense/kernel/Adamvc/dense/kernel/Adam_1vc/dense_1/biasvc/dense_1/bias/Adamvc/dense_1/bias/Adam_1vc/dense_1/kernelvc/dense_1/kernel/Adamvc/dense_1/kernel/Adam_1vc/dense_2/biasvc/dense_2/bias/Adamvc/dense_2/bias/Adam_1vc/dense_2/kernelvc/dense_2/kernel/Adamvc/dense_2/kernel/Adam_1vf/dense/biasvf/dense/bias/Adamvf/dense/bias/Adam_1vf/dense/kernelvf/dense/kernel/Adamvf/dense/kernel/Adam_1vf/dense_1/biasvf/dense_1/bias/Adamvf/dense_1/bias/Adam_1vf/dense_1/kernelvf/dense_1/kernel/Adamvf/dense_1/kernel/Adam_1vf/dense_2/biasvf/dense_2/bias/Adamvf/dense_2/bias/Adam_1vf/dense_2/kernelvf/dense_2/kernel/Adamvf/dense_2/kernel/Adam_1*R
dtypesH
F2D

save/control_dependencyIdentitysave/ShardedFilename^save/SaveV2*
T0*'
_class
loc:@save/ShardedFilename*
_output_shapes
: 

+save/MergeV2Checkpoints/checkpoint_prefixesPacksave/ShardedFilename^save/control_dependency*
N*
_output_shapes
:*

axis *
T0
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
ř
save/RestoreV2/tensor_namesConst*
dtype0*¨
valueBDBbeta1_powerBbeta1_power_1Bbeta1_power_2Bbeta2_powerBbeta2_power_1Bbeta2_power_2Bpenalty/penalty_paramBpenalty/penalty_param/AdamBpenalty/penalty_param/Adam_1Bpenalty/penalty_param/Adam_2Bpenalty/penalty_param/Adam_3Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1B
pi/log_stdBpi/log_std/AdamBpi/log_std/Adam_1Bvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1*
_output_shapes
:D
ń
save/RestoreV2/shape_and_slicesConst*
_output_shapes
:D*
dtype0*
valueBDB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
â
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices*Ś
_output_shapes
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*R
dtypesH
F2D
Ś
save/AssignAssignbeta1_powersave/RestoreV2*
use_locking(*
validate_shape(*(
_class
loc:@penalty/penalty_param*
_output_shapes
: *
T0
Ź
save/Assign_1Assignbeta1_power_1save/RestoreV2:1*
T0*
use_locking(*
validate_shape(*
_output_shapes
: *(
_class
loc:@penalty/penalty_param
¤
save/Assign_2Assignbeta1_power_2save/RestoreV2:2* 
_class
loc:@vc/dense/bias*
validate_shape(*
_output_shapes
: *
T0*
use_locking(
Ş
save/Assign_3Assignbeta2_powersave/RestoreV2:3*
T0*
use_locking(*
_output_shapes
: *
validate_shape(*(
_class
loc:@penalty/penalty_param
Ź
save/Assign_4Assignbeta2_power_1save/RestoreV2:4*
_output_shapes
: *(
_class
loc:@penalty/penalty_param*
validate_shape(*
use_locking(*
T0
¤
save/Assign_5Assignbeta2_power_2save/RestoreV2:5*
use_locking(*
validate_shape(*
_output_shapes
: *
T0* 
_class
loc:@vc/dense/bias
´
save/Assign_6Assignpenalty/penalty_paramsave/RestoreV2:6*(
_class
loc:@penalty/penalty_param*
use_locking(*
_output_shapes
: *
T0*
validate_shape(
š
save/Assign_7Assignpenalty/penalty_param/Adamsave/RestoreV2:7*
_output_shapes
: *
T0*
use_locking(*
validate_shape(*(
_class
loc:@penalty/penalty_param
ť
save/Assign_8Assignpenalty/penalty_param/Adam_1save/RestoreV2:8*
_output_shapes
: *
validate_shape(*
use_locking(*(
_class
loc:@penalty/penalty_param*
T0
ť
save/Assign_9Assignpenalty/penalty_param/Adam_2save/RestoreV2:9*
_output_shapes
: *
use_locking(*(
_class
loc:@penalty/penalty_param*
validate_shape(*
T0
˝
save/Assign_10Assignpenalty/penalty_param/Adam_3save/RestoreV2:10*
_output_shapes
: *(
_class
loc:@penalty/penalty_param*
use_locking(*
T0*
validate_shape(
Ť
save/Assign_11Assignpi/dense/biassave/RestoreV2:11*
validate_shape(* 
_class
loc:@pi/dense/bias*
_output_shapes	
:*
use_locking(*
T0
°
save/Assign_12Assignpi/dense/bias/Adamsave/RestoreV2:12* 
_class
loc:@pi/dense/bias*
_output_shapes	
:*
use_locking(*
T0*
validate_shape(
˛
save/Assign_13Assignpi/dense/bias/Adam_1save/RestoreV2:13* 
_class
loc:@pi/dense/bias*
T0*
validate_shape(*
_output_shapes	
:*
use_locking(
ł
save/Assign_14Assignpi/dense/kernelsave/RestoreV2:14*
_output_shapes
:	L*
T0*"
_class
loc:@pi/dense/kernel*
validate_shape(*
use_locking(
¸
save/Assign_15Assignpi/dense/kernel/Adamsave/RestoreV2:15*
use_locking(*
T0*
_output_shapes
:	L*"
_class
loc:@pi/dense/kernel*
validate_shape(
ş
save/Assign_16Assignpi/dense/kernel/Adam_1save/RestoreV2:16*
use_locking(*
_output_shapes
:	L*
T0*"
_class
loc:@pi/dense/kernel*
validate_shape(
Ż
save/Assign_17Assignpi/dense_1/biassave/RestoreV2:17*"
_class
loc:@pi/dense_1/bias*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0
´
save/Assign_18Assignpi/dense_1/bias/Adamsave/RestoreV2:18*
validate_shape(*
use_locking(*"
_class
loc:@pi/dense_1/bias*
_output_shapes	
:*
T0
ś
save/Assign_19Assignpi/dense_1/bias/Adam_1save/RestoreV2:19*"
_class
loc:@pi/dense_1/bias*
_output_shapes	
:*
use_locking(*
validate_shape(*
T0
¸
save/Assign_20Assignpi/dense_1/kernelsave/RestoreV2:20*
use_locking(*
T0*$
_class
loc:@pi/dense_1/kernel* 
_output_shapes
:
*
validate_shape(
˝
save/Assign_21Assignpi/dense_1/kernel/Adamsave/RestoreV2:21*
use_locking(* 
_output_shapes
:
*$
_class
loc:@pi/dense_1/kernel*
T0*
validate_shape(
ż
save/Assign_22Assignpi/dense_1/kernel/Adam_1save/RestoreV2:22*
T0*
use_locking(*$
_class
loc:@pi/dense_1/kernel* 
_output_shapes
:
*
validate_shape(
Ž
save/Assign_23Assignpi/dense_2/biassave/RestoreV2:23*
T0*
validate_shape(*
use_locking(*
_output_shapes
:*"
_class
loc:@pi/dense_2/bias
ł
save/Assign_24Assignpi/dense_2/bias/Adamsave/RestoreV2:24*
validate_shape(*"
_class
loc:@pi/dense_2/bias*
_output_shapes
:*
T0*
use_locking(
ľ
save/Assign_25Assignpi/dense_2/bias/Adam_1save/RestoreV2:25*"
_class
loc:@pi/dense_2/bias*
validate_shape(*
T0*
use_locking(*
_output_shapes
:
ˇ
save/Assign_26Assignpi/dense_2/kernelsave/RestoreV2:26*$
_class
loc:@pi/dense_2/kernel*
validate_shape(*
T0*
use_locking(*
_output_shapes
:	
ź
save/Assign_27Assignpi/dense_2/kernel/Adamsave/RestoreV2:27*
validate_shape(*
_output_shapes
:	*
use_locking(*$
_class
loc:@pi/dense_2/kernel*
T0
ž
save/Assign_28Assignpi/dense_2/kernel/Adam_1save/RestoreV2:28*
_output_shapes
:	*
T0*
use_locking(*$
_class
loc:@pi/dense_2/kernel*
validate_shape(
¤
save/Assign_29Assign
pi/log_stdsave/RestoreV2:29*
_class
loc:@pi/log_std*
validate_shape(*
_output_shapes
:*
T0*
use_locking(
Š
save/Assign_30Assignpi/log_std/Adamsave/RestoreV2:30*
_output_shapes
:*
_class
loc:@pi/log_std*
T0*
use_locking(*
validate_shape(
Ť
save/Assign_31Assignpi/log_std/Adam_1save/RestoreV2:31*
_class
loc:@pi/log_std*
T0*
_output_shapes
:*
use_locking(*
validate_shape(
Ť
save/Assign_32Assignvc/dense/biassave/RestoreV2:32*
_output_shapes	
:*
validate_shape(*
use_locking(*
T0* 
_class
loc:@vc/dense/bias
°
save/Assign_33Assignvc/dense/bias/Adamsave/RestoreV2:33*
T0*
validate_shape(*
use_locking(* 
_class
loc:@vc/dense/bias*
_output_shapes	
:
˛
save/Assign_34Assignvc/dense/bias/Adam_1save/RestoreV2:34* 
_class
loc:@vc/dense/bias*
T0*
validate_shape(*
_output_shapes	
:*
use_locking(
ł
save/Assign_35Assignvc/dense/kernelsave/RestoreV2:35*
_output_shapes
:	L*
T0*
use_locking(*
validate_shape(*"
_class
loc:@vc/dense/kernel
¸
save/Assign_36Assignvc/dense/kernel/Adamsave/RestoreV2:36*
use_locking(*"
_class
loc:@vc/dense/kernel*
_output_shapes
:	L*
validate_shape(*
T0
ş
save/Assign_37Assignvc/dense/kernel/Adam_1save/RestoreV2:37*"
_class
loc:@vc/dense/kernel*
T0*
use_locking(*
_output_shapes
:	L*
validate_shape(
Ż
save/Assign_38Assignvc/dense_1/biassave/RestoreV2:38*"
_class
loc:@vc/dense_1/bias*
_output_shapes	
:*
validate_shape(*
use_locking(*
T0
´
save/Assign_39Assignvc/dense_1/bias/Adamsave/RestoreV2:39*
T0*"
_class
loc:@vc/dense_1/bias*
use_locking(*
validate_shape(*
_output_shapes	
:
ś
save/Assign_40Assignvc/dense_1/bias/Adam_1save/RestoreV2:40*
validate_shape(*
use_locking(*
T0*"
_class
loc:@vc/dense_1/bias*
_output_shapes	
:
¸
save/Assign_41Assignvc/dense_1/kernelsave/RestoreV2:41*$
_class
loc:@vc/dense_1/kernel*
validate_shape(*
use_locking(* 
_output_shapes
:
*
T0
˝
save/Assign_42Assignvc/dense_1/kernel/Adamsave/RestoreV2:42*
T0* 
_output_shapes
:
*
validate_shape(*$
_class
loc:@vc/dense_1/kernel*
use_locking(
ż
save/Assign_43Assignvc/dense_1/kernel/Adam_1save/RestoreV2:43*
validate_shape(*$
_class
loc:@vc/dense_1/kernel*
T0*
use_locking(* 
_output_shapes
:

Ž
save/Assign_44Assignvc/dense_2/biassave/RestoreV2:44*
use_locking(*"
_class
loc:@vc/dense_2/bias*
T0*
_output_shapes
:*
validate_shape(
ł
save/Assign_45Assignvc/dense_2/bias/Adamsave/RestoreV2:45*"
_class
loc:@vc/dense_2/bias*
_output_shapes
:*
validate_shape(*
T0*
use_locking(
ľ
save/Assign_46Assignvc/dense_2/bias/Adam_1save/RestoreV2:46*
_output_shapes
:*"
_class
loc:@vc/dense_2/bias*
validate_shape(*
use_locking(*
T0
ˇ
save/Assign_47Assignvc/dense_2/kernelsave/RestoreV2:47*$
_class
loc:@vc/dense_2/kernel*
use_locking(*
_output_shapes
:	*
T0*
validate_shape(
ź
save/Assign_48Assignvc/dense_2/kernel/Adamsave/RestoreV2:48*$
_class
loc:@vc/dense_2/kernel*
use_locking(*
validate_shape(*
T0*
_output_shapes
:	
ž
save/Assign_49Assignvc/dense_2/kernel/Adam_1save/RestoreV2:49*$
_class
loc:@vc/dense_2/kernel*
validate_shape(*
use_locking(*
_output_shapes
:	*
T0
Ť
save/Assign_50Assignvf/dense/biassave/RestoreV2:50*
T0*
validate_shape(*
_output_shapes	
:*
use_locking(* 
_class
loc:@vf/dense/bias
°
save/Assign_51Assignvf/dense/bias/Adamsave/RestoreV2:51*
validate_shape(*
use_locking(*
T0* 
_class
loc:@vf/dense/bias*
_output_shapes	
:
˛
save/Assign_52Assignvf/dense/bias/Adam_1save/RestoreV2:52*
T0* 
_class
loc:@vf/dense/bias*
validate_shape(*
_output_shapes	
:*
use_locking(
ł
save/Assign_53Assignvf/dense/kernelsave/RestoreV2:53*"
_class
loc:@vf/dense/kernel*
validate_shape(*
_output_shapes
:	L*
use_locking(*
T0
¸
save/Assign_54Assignvf/dense/kernel/Adamsave/RestoreV2:54*
_output_shapes
:	L*
T0*
use_locking(*"
_class
loc:@vf/dense/kernel*
validate_shape(
ş
save/Assign_55Assignvf/dense/kernel/Adam_1save/RestoreV2:55*
validate_shape(*
use_locking(*
_output_shapes
:	L*
T0*"
_class
loc:@vf/dense/kernel
Ż
save/Assign_56Assignvf/dense_1/biassave/RestoreV2:56*
_output_shapes	
:*
T0*
validate_shape(*"
_class
loc:@vf/dense_1/bias*
use_locking(
´
save/Assign_57Assignvf/dense_1/bias/Adamsave/RestoreV2:57*"
_class
loc:@vf/dense_1/bias*
validate_shape(*
T0*
use_locking(*
_output_shapes	
:
ś
save/Assign_58Assignvf/dense_1/bias/Adam_1save/RestoreV2:58*
_output_shapes	
:*
T0*
validate_shape(*
use_locking(*"
_class
loc:@vf/dense_1/bias
¸
save/Assign_59Assignvf/dense_1/kernelsave/RestoreV2:59*
use_locking(* 
_output_shapes
:
*$
_class
loc:@vf/dense_1/kernel*
validate_shape(*
T0
˝
save/Assign_60Assignvf/dense_1/kernel/Adamsave/RestoreV2:60*
use_locking(*$
_class
loc:@vf/dense_1/kernel* 
_output_shapes
:
*
validate_shape(*
T0
ż
save/Assign_61Assignvf/dense_1/kernel/Adam_1save/RestoreV2:61*$
_class
loc:@vf/dense_1/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0
Ž
save/Assign_62Assignvf/dense_2/biassave/RestoreV2:62*
validate_shape(*"
_class
loc:@vf/dense_2/bias*
_output_shapes
:*
use_locking(*
T0
ł
save/Assign_63Assignvf/dense_2/bias/Adamsave/RestoreV2:63*"
_class
loc:@vf/dense_2/bias*
T0*
validate_shape(*
_output_shapes
:*
use_locking(
ľ
save/Assign_64Assignvf/dense_2/bias/Adam_1save/RestoreV2:64*"
_class
loc:@vf/dense_2/bias*
T0*
_output_shapes
:*
validate_shape(*
use_locking(
ˇ
save/Assign_65Assignvf/dense_2/kernelsave/RestoreV2:65*
T0*
_output_shapes
:	*
validate_shape(*
use_locking(*$
_class
loc:@vf/dense_2/kernel
ź
save/Assign_66Assignvf/dense_2/kernel/Adamsave/RestoreV2:66*
_output_shapes
:	*
T0*
validate_shape(*$
_class
loc:@vf/dense_2/kernel*
use_locking(
ž
save/Assign_67Assignvf/dense_2/kernel/Adam_1save/RestoreV2:67*
use_locking(*$
_class
loc:@vf/dense_2/kernel*
T0*
validate_shape(*
_output_shapes
:	
	
save/restore_shardNoOp^save/Assign^save/Assign_1^save/Assign_10^save/Assign_11^save/Assign_12^save/Assign_13^save/Assign_14^save/Assign_15^save/Assign_16^save/Assign_17^save/Assign_18^save/Assign_19^save/Assign_2^save/Assign_20^save/Assign_21^save/Assign_22^save/Assign_23^save/Assign_24^save/Assign_25^save/Assign_26^save/Assign_27^save/Assign_28^save/Assign_29^save/Assign_3^save/Assign_30^save/Assign_31^save/Assign_32^save/Assign_33^save/Assign_34^save/Assign_35^save/Assign_36^save/Assign_37^save/Assign_38^save/Assign_39^save/Assign_4^save/Assign_40^save/Assign_41^save/Assign_42^save/Assign_43^save/Assign_44^save/Assign_45^save/Assign_46^save/Assign_47^save/Assign_48^save/Assign_49^save/Assign_5^save/Assign_50^save/Assign_51^save/Assign_52^save/Assign_53^save/Assign_54^save/Assign_55^save/Assign_56^save/Assign_57^save/Assign_58^save/Assign_59^save/Assign_6^save/Assign_60^save/Assign_61^save/Assign_62^save/Assign_63^save/Assign_64^save/Assign_65^save/Assign_66^save/Assign_67^save/Assign_7^save/Assign_8^save/Assign_9
-
save/restore_allNoOp^save/restore_shard
[
save_1/filename/inputConst*
_output_shapes
: *
valueB Bmodel*
dtype0
r
save_1/filenamePlaceholderWithDefaultsave_1/filename/input*
shape: *
_output_shapes
: *
dtype0
i
save_1/ConstPlaceholderWithDefaultsave_1/filename*
_output_shapes
: *
shape: *
dtype0

save_1/StringJoin/inputs_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+_temp_6ee89a47cf9c405d92fc9ed91c1ff751/part
{
save_1/StringJoin
StringJoinsave_1/Constsave_1/StringJoin/inputs_1*
	separator *
N*
_output_shapes
: 
S
save_1/num_shardsConst*
dtype0*
_output_shapes
: *
value	B :
^
save_1/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 

save_1/ShardedFilenameShardedFilenamesave_1/StringJoinsave_1/ShardedFilename/shardsave_1/num_shards*
_output_shapes
: 
÷
save_1/SaveV2/tensor_namesConst*
_output_shapes
:D*
dtype0*¨
valueBDBbeta1_powerBbeta1_power_1Bbeta1_power_2Bbeta2_powerBbeta2_power_1Bbeta2_power_2Bpenalty/penalty_paramBpenalty/penalty_param/AdamBpenalty/penalty_param/Adam_1Bpenalty/penalty_param/Adam_2Bpenalty/penalty_param/Adam_3Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1B
pi/log_stdBpi/log_std/AdamBpi/log_std/Adam_1Bvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1
đ
save_1/SaveV2/shape_and_slicesConst*
_output_shapes
:D*
valueBDB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0
Ň
save_1/SaveV2SaveV2save_1/ShardedFilenamesave_1/SaveV2/tensor_namessave_1/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta1_power_2beta2_powerbeta2_power_1beta2_power_2penalty/penalty_parampenalty/penalty_param/Adampenalty/penalty_param/Adam_1penalty/penalty_param/Adam_2penalty/penalty_param/Adam_3pi/dense/biaspi/dense/bias/Adampi/dense/bias/Adam_1pi/dense/kernelpi/dense/kernel/Adampi/dense/kernel/Adam_1pi/dense_1/biaspi/dense_1/bias/Adampi/dense_1/bias/Adam_1pi/dense_1/kernelpi/dense_1/kernel/Adampi/dense_1/kernel/Adam_1pi/dense_2/biaspi/dense_2/bias/Adampi/dense_2/bias/Adam_1pi/dense_2/kernelpi/dense_2/kernel/Adampi/dense_2/kernel/Adam_1
pi/log_stdpi/log_std/Adampi/log_std/Adam_1vc/dense/biasvc/dense/bias/Adamvc/dense/bias/Adam_1vc/dense/kernelvc/dense/kernel/Adamvc/dense/kernel/Adam_1vc/dense_1/biasvc/dense_1/bias/Adamvc/dense_1/bias/Adam_1vc/dense_1/kernelvc/dense_1/kernel/Adamvc/dense_1/kernel/Adam_1vc/dense_2/biasvc/dense_2/bias/Adamvc/dense_2/bias/Adam_1vc/dense_2/kernelvc/dense_2/kernel/Adamvc/dense_2/kernel/Adam_1vf/dense/biasvf/dense/bias/Adamvf/dense/bias/Adam_1vf/dense/kernelvf/dense/kernel/Adamvf/dense/kernel/Adam_1vf/dense_1/biasvf/dense_1/bias/Adamvf/dense_1/bias/Adam_1vf/dense_1/kernelvf/dense_1/kernel/Adamvf/dense_1/kernel/Adam_1vf/dense_2/biasvf/dense_2/bias/Adamvf/dense_2/bias/Adam_1vf/dense_2/kernelvf/dense_2/kernel/Adamvf/dense_2/kernel/Adam_1*R
dtypesH
F2D

save_1/control_dependencyIdentitysave_1/ShardedFilename^save_1/SaveV2*
_output_shapes
: *)
_class
loc:@save_1/ShardedFilename*
T0
Ł
-save_1/MergeV2Checkpoints/checkpoint_prefixesPacksave_1/ShardedFilename^save_1/control_dependency*
N*
_output_shapes
:*
T0*

axis 

save_1/MergeV2CheckpointsMergeV2Checkpoints-save_1/MergeV2Checkpoints/checkpoint_prefixessave_1/Const*
delete_old_dirs(

save_1/IdentityIdentitysave_1/Const^save_1/MergeV2Checkpoints^save_1/control_dependency*
_output_shapes
: *
T0
ú
save_1/RestoreV2/tensor_namesConst*¨
valueBDBbeta1_powerBbeta1_power_1Bbeta1_power_2Bbeta2_powerBbeta2_power_1Bbeta2_power_2Bpenalty/penalty_paramBpenalty/penalty_param/AdamBpenalty/penalty_param/Adam_1Bpenalty/penalty_param/Adam_2Bpenalty/penalty_param/Adam_3Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1B
pi/log_stdBpi/log_std/AdamBpi/log_std/Adam_1Bvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1*
_output_shapes
:D*
dtype0
ó
!save_1/RestoreV2/shape_and_slicesConst*
_output_shapes
:D*
dtype0*
valueBDB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
ę
save_1/RestoreV2	RestoreV2save_1/Constsave_1/RestoreV2/tensor_names!save_1/RestoreV2/shape_and_slices*Ś
_output_shapes
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*R
dtypesH
F2D
Ş
save_1/AssignAssignbeta1_powersave_1/RestoreV2*(
_class
loc:@penalty/penalty_param*
T0*
use_locking(*
_output_shapes
: *
validate_shape(
°
save_1/Assign_1Assignbeta1_power_1save_1/RestoreV2:1*
validate_shape(*
use_locking(*(
_class
loc:@penalty/penalty_param*
_output_shapes
: *
T0
¨
save_1/Assign_2Assignbeta1_power_2save_1/RestoreV2:2*
use_locking(* 
_class
loc:@vc/dense/bias*
validate_shape(*
_output_shapes
: *
T0
Ž
save_1/Assign_3Assignbeta2_powersave_1/RestoreV2:3*
use_locking(*
T0*(
_class
loc:@penalty/penalty_param*
validate_shape(*
_output_shapes
: 
°
save_1/Assign_4Assignbeta2_power_1save_1/RestoreV2:4*
T0*
validate_shape(*
use_locking(*
_output_shapes
: *(
_class
loc:@penalty/penalty_param
¨
save_1/Assign_5Assignbeta2_power_2save_1/RestoreV2:5*
validate_shape(*
use_locking(* 
_class
loc:@vc/dense/bias*
_output_shapes
: *
T0
¸
save_1/Assign_6Assignpenalty/penalty_paramsave_1/RestoreV2:6*
T0*(
_class
loc:@penalty/penalty_param*
use_locking(*
validate_shape(*
_output_shapes
: 
˝
save_1/Assign_7Assignpenalty/penalty_param/Adamsave_1/RestoreV2:7*(
_class
loc:@penalty/penalty_param*
T0*
_output_shapes
: *
use_locking(*
validate_shape(
ż
save_1/Assign_8Assignpenalty/penalty_param/Adam_1save_1/RestoreV2:8*
T0*
validate_shape(*
_output_shapes
: *
use_locking(*(
_class
loc:@penalty/penalty_param
ż
save_1/Assign_9Assignpenalty/penalty_param/Adam_2save_1/RestoreV2:9*
_output_shapes
: *
use_locking(*
T0*(
_class
loc:@penalty/penalty_param*
validate_shape(
Á
save_1/Assign_10Assignpenalty/penalty_param/Adam_3save_1/RestoreV2:10*
_output_shapes
: *
use_locking(*(
_class
loc:@penalty/penalty_param*
validate_shape(*
T0
Ż
save_1/Assign_11Assignpi/dense/biassave_1/RestoreV2:11*
T0*
validate_shape(*
_output_shapes	
:* 
_class
loc:@pi/dense/bias*
use_locking(
´
save_1/Assign_12Assignpi/dense/bias/Adamsave_1/RestoreV2:12* 
_class
loc:@pi/dense/bias*
use_locking(*
validate_shape(*
_output_shapes	
:*
T0
ś
save_1/Assign_13Assignpi/dense/bias/Adam_1save_1/RestoreV2:13*
T0*
use_locking(*
validate_shape(* 
_class
loc:@pi/dense/bias*
_output_shapes	
:
ˇ
save_1/Assign_14Assignpi/dense/kernelsave_1/RestoreV2:14*"
_class
loc:@pi/dense/kernel*
use_locking(*
_output_shapes
:	L*
validate_shape(*
T0
ź
save_1/Assign_15Assignpi/dense/kernel/Adamsave_1/RestoreV2:15*
_output_shapes
:	L*
T0*"
_class
loc:@pi/dense/kernel*
use_locking(*
validate_shape(
ž
save_1/Assign_16Assignpi/dense/kernel/Adam_1save_1/RestoreV2:16*"
_class
loc:@pi/dense/kernel*
validate_shape(*
_output_shapes
:	L*
T0*
use_locking(
ł
save_1/Assign_17Assignpi/dense_1/biassave_1/RestoreV2:17*
validate_shape(*"
_class
loc:@pi/dense_1/bias*
use_locking(*
_output_shapes	
:*
T0
¸
save_1/Assign_18Assignpi/dense_1/bias/Adamsave_1/RestoreV2:18*
_output_shapes	
:*"
_class
loc:@pi/dense_1/bias*
T0*
use_locking(*
validate_shape(
ş
save_1/Assign_19Assignpi/dense_1/bias/Adam_1save_1/RestoreV2:19*
_output_shapes	
:*
T0*"
_class
loc:@pi/dense_1/bias*
validate_shape(*
use_locking(
ź
save_1/Assign_20Assignpi/dense_1/kernelsave_1/RestoreV2:20* 
_output_shapes
:
*
use_locking(*
T0*
validate_shape(*$
_class
loc:@pi/dense_1/kernel
Á
save_1/Assign_21Assignpi/dense_1/kernel/Adamsave_1/RestoreV2:21*
use_locking(* 
_output_shapes
:
*
validate_shape(*$
_class
loc:@pi/dense_1/kernel*
T0
Ă
save_1/Assign_22Assignpi/dense_1/kernel/Adam_1save_1/RestoreV2:22*$
_class
loc:@pi/dense_1/kernel*
validate_shape(* 
_output_shapes
:
*
T0*
use_locking(
˛
save_1/Assign_23Assignpi/dense_2/biassave_1/RestoreV2:23*"
_class
loc:@pi/dense_2/bias*
validate_shape(*
_output_shapes
:*
T0*
use_locking(
ˇ
save_1/Assign_24Assignpi/dense_2/bias/Adamsave_1/RestoreV2:24*"
_class
loc:@pi/dense_2/bias*
validate_shape(*
use_locking(*
T0*
_output_shapes
:
š
save_1/Assign_25Assignpi/dense_2/bias/Adam_1save_1/RestoreV2:25*
validate_shape(*
use_locking(*
_output_shapes
:*"
_class
loc:@pi/dense_2/bias*
T0
ť
save_1/Assign_26Assignpi/dense_2/kernelsave_1/RestoreV2:26*
_output_shapes
:	*
T0*$
_class
loc:@pi/dense_2/kernel*
use_locking(*
validate_shape(
Ŕ
save_1/Assign_27Assignpi/dense_2/kernel/Adamsave_1/RestoreV2:27*
_output_shapes
:	*
validate_shape(*
use_locking(*$
_class
loc:@pi/dense_2/kernel*
T0
Â
save_1/Assign_28Assignpi/dense_2/kernel/Adam_1save_1/RestoreV2:28*
T0*
validate_shape(*$
_class
loc:@pi/dense_2/kernel*
_output_shapes
:	*
use_locking(
¨
save_1/Assign_29Assign
pi/log_stdsave_1/RestoreV2:29*
_class
loc:@pi/log_std*
validate_shape(*
use_locking(*
_output_shapes
:*
T0
­
save_1/Assign_30Assignpi/log_std/Adamsave_1/RestoreV2:30*
validate_shape(*
use_locking(*
_output_shapes
:*
_class
loc:@pi/log_std*
T0
Ż
save_1/Assign_31Assignpi/log_std/Adam_1save_1/RestoreV2:31*
_class
loc:@pi/log_std*
use_locking(*
_output_shapes
:*
T0*
validate_shape(
Ż
save_1/Assign_32Assignvc/dense/biassave_1/RestoreV2:32*
validate_shape(*
use_locking(*
_output_shapes	
:* 
_class
loc:@vc/dense/bias*
T0
´
save_1/Assign_33Assignvc/dense/bias/Adamsave_1/RestoreV2:33* 
_class
loc:@vc/dense/bias*
validate_shape(*
use_locking(*
_output_shapes	
:*
T0
ś
save_1/Assign_34Assignvc/dense/bias/Adam_1save_1/RestoreV2:34*
validate_shape(*
use_locking(*
T0*
_output_shapes	
:* 
_class
loc:@vc/dense/bias
ˇ
save_1/Assign_35Assignvc/dense/kernelsave_1/RestoreV2:35*
T0*
use_locking(*"
_class
loc:@vc/dense/kernel*
_output_shapes
:	L*
validate_shape(
ź
save_1/Assign_36Assignvc/dense/kernel/Adamsave_1/RestoreV2:36*
T0*
validate_shape(*
use_locking(*"
_class
loc:@vc/dense/kernel*
_output_shapes
:	L
ž
save_1/Assign_37Assignvc/dense/kernel/Adam_1save_1/RestoreV2:37*
validate_shape(*
_output_shapes
:	L*"
_class
loc:@vc/dense/kernel*
T0*
use_locking(
ł
save_1/Assign_38Assignvc/dense_1/biassave_1/RestoreV2:38*
use_locking(*
T0*
_output_shapes	
:*
validate_shape(*"
_class
loc:@vc/dense_1/bias
¸
save_1/Assign_39Assignvc/dense_1/bias/Adamsave_1/RestoreV2:39*
_output_shapes	
:*
validate_shape(*"
_class
loc:@vc/dense_1/bias*
T0*
use_locking(
ş
save_1/Assign_40Assignvc/dense_1/bias/Adam_1save_1/RestoreV2:40*
T0*
_output_shapes	
:*
use_locking(*"
_class
loc:@vc/dense_1/bias*
validate_shape(
ź
save_1/Assign_41Assignvc/dense_1/kernelsave_1/RestoreV2:41*
use_locking(* 
_output_shapes
:
*
T0*$
_class
loc:@vc/dense_1/kernel*
validate_shape(
Á
save_1/Assign_42Assignvc/dense_1/kernel/Adamsave_1/RestoreV2:42*$
_class
loc:@vc/dense_1/kernel*
T0*
validate_shape(*
use_locking(* 
_output_shapes
:

Ă
save_1/Assign_43Assignvc/dense_1/kernel/Adam_1save_1/RestoreV2:43* 
_output_shapes
:
*
T0*
validate_shape(*
use_locking(*$
_class
loc:@vc/dense_1/kernel
˛
save_1/Assign_44Assignvc/dense_2/biassave_1/RestoreV2:44*
_output_shapes
:*"
_class
loc:@vc/dense_2/bias*
validate_shape(*
T0*
use_locking(
ˇ
save_1/Assign_45Assignvc/dense_2/bias/Adamsave_1/RestoreV2:45*
use_locking(*
validate_shape(*"
_class
loc:@vc/dense_2/bias*
_output_shapes
:*
T0
š
save_1/Assign_46Assignvc/dense_2/bias/Adam_1save_1/RestoreV2:46*
T0*
use_locking(*
validate_shape(*
_output_shapes
:*"
_class
loc:@vc/dense_2/bias
ť
save_1/Assign_47Assignvc/dense_2/kernelsave_1/RestoreV2:47*
T0*
use_locking(*
_output_shapes
:	*$
_class
loc:@vc/dense_2/kernel*
validate_shape(
Ŕ
save_1/Assign_48Assignvc/dense_2/kernel/Adamsave_1/RestoreV2:48*
T0*
_output_shapes
:	*
validate_shape(*
use_locking(*$
_class
loc:@vc/dense_2/kernel
Â
save_1/Assign_49Assignvc/dense_2/kernel/Adam_1save_1/RestoreV2:49*
use_locking(*
_output_shapes
:	*$
_class
loc:@vc/dense_2/kernel*
T0*
validate_shape(
Ż
save_1/Assign_50Assignvf/dense/biassave_1/RestoreV2:50*
use_locking(*
_output_shapes	
:*
T0* 
_class
loc:@vf/dense/bias*
validate_shape(
´
save_1/Assign_51Assignvf/dense/bias/Adamsave_1/RestoreV2:51*
T0*
validate_shape(* 
_class
loc:@vf/dense/bias*
_output_shapes	
:*
use_locking(
ś
save_1/Assign_52Assignvf/dense/bias/Adam_1save_1/RestoreV2:52*
use_locking(*
_output_shapes	
:* 
_class
loc:@vf/dense/bias*
T0*
validate_shape(
ˇ
save_1/Assign_53Assignvf/dense/kernelsave_1/RestoreV2:53*
_output_shapes
:	L*"
_class
loc:@vf/dense/kernel*
T0*
use_locking(*
validate_shape(
ź
save_1/Assign_54Assignvf/dense/kernel/Adamsave_1/RestoreV2:54*
validate_shape(*
use_locking(*
_output_shapes
:	L*
T0*"
_class
loc:@vf/dense/kernel
ž
save_1/Assign_55Assignvf/dense/kernel/Adam_1save_1/RestoreV2:55*
validate_shape(*"
_class
loc:@vf/dense/kernel*
_output_shapes
:	L*
use_locking(*
T0
ł
save_1/Assign_56Assignvf/dense_1/biassave_1/RestoreV2:56*
T0*"
_class
loc:@vf/dense_1/bias*
validate_shape(*
_output_shapes	
:*
use_locking(
¸
save_1/Assign_57Assignvf/dense_1/bias/Adamsave_1/RestoreV2:57*
T0*
use_locking(*
validate_shape(*"
_class
loc:@vf/dense_1/bias*
_output_shapes	
:
ş
save_1/Assign_58Assignvf/dense_1/bias/Adam_1save_1/RestoreV2:58*
T0*
_output_shapes	
:*"
_class
loc:@vf/dense_1/bias*
validate_shape(*
use_locking(
ź
save_1/Assign_59Assignvf/dense_1/kernelsave_1/RestoreV2:59*$
_class
loc:@vf/dense_1/kernel*
use_locking(*
T0*
validate_shape(* 
_output_shapes
:

Á
save_1/Assign_60Assignvf/dense_1/kernel/Adamsave_1/RestoreV2:60*$
_class
loc:@vf/dense_1/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0
Ă
save_1/Assign_61Assignvf/dense_1/kernel/Adam_1save_1/RestoreV2:61*
validate_shape(*$
_class
loc:@vf/dense_1/kernel*
T0*
use_locking(* 
_output_shapes
:

˛
save_1/Assign_62Assignvf/dense_2/biassave_1/RestoreV2:62*
_output_shapes
:*
T0*"
_class
loc:@vf/dense_2/bias*
use_locking(*
validate_shape(
ˇ
save_1/Assign_63Assignvf/dense_2/bias/Adamsave_1/RestoreV2:63*
T0*
use_locking(*
validate_shape(*
_output_shapes
:*"
_class
loc:@vf/dense_2/bias
š
save_1/Assign_64Assignvf/dense_2/bias/Adam_1save_1/RestoreV2:64*"
_class
loc:@vf/dense_2/bias*
T0*
validate_shape(*
_output_shapes
:*
use_locking(
ť
save_1/Assign_65Assignvf/dense_2/kernelsave_1/RestoreV2:65*
validate_shape(*$
_class
loc:@vf/dense_2/kernel*
_output_shapes
:	*
use_locking(*
T0
Ŕ
save_1/Assign_66Assignvf/dense_2/kernel/Adamsave_1/RestoreV2:66*
use_locking(*
validate_shape(*
T0*
_output_shapes
:	*$
_class
loc:@vf/dense_2/kernel
Â
save_1/Assign_67Assignvf/dense_2/kernel/Adam_1save_1/RestoreV2:67*
T0*$
_class
loc:@vf/dense_2/kernel*
_output_shapes
:	*
validate_shape(*
use_locking(


save_1/restore_shardNoOp^save_1/Assign^save_1/Assign_1^save_1/Assign_10^save_1/Assign_11^save_1/Assign_12^save_1/Assign_13^save_1/Assign_14^save_1/Assign_15^save_1/Assign_16^save_1/Assign_17^save_1/Assign_18^save_1/Assign_19^save_1/Assign_2^save_1/Assign_20^save_1/Assign_21^save_1/Assign_22^save_1/Assign_23^save_1/Assign_24^save_1/Assign_25^save_1/Assign_26^save_1/Assign_27^save_1/Assign_28^save_1/Assign_29^save_1/Assign_3^save_1/Assign_30^save_1/Assign_31^save_1/Assign_32^save_1/Assign_33^save_1/Assign_34^save_1/Assign_35^save_1/Assign_36^save_1/Assign_37^save_1/Assign_38^save_1/Assign_39^save_1/Assign_4^save_1/Assign_40^save_1/Assign_41^save_1/Assign_42^save_1/Assign_43^save_1/Assign_44^save_1/Assign_45^save_1/Assign_46^save_1/Assign_47^save_1/Assign_48^save_1/Assign_49^save_1/Assign_5^save_1/Assign_50^save_1/Assign_51^save_1/Assign_52^save_1/Assign_53^save_1/Assign_54^save_1/Assign_55^save_1/Assign_56^save_1/Assign_57^save_1/Assign_58^save_1/Assign_59^save_1/Assign_6^save_1/Assign_60^save_1/Assign_61^save_1/Assign_62^save_1/Assign_63^save_1/Assign_64^save_1/Assign_65^save_1/Assign_66^save_1/Assign_67^save_1/Assign_7^save_1/Assign_8^save_1/Assign_9
1
save_1/restore_allNoOp^save_1/restore_shard
[
save_2/filename/inputConst*
_output_shapes
: *
valueB Bmodel*
dtype0
r
save_2/filenamePlaceholderWithDefaultsave_2/filename/input*
dtype0*
_output_shapes
: *
shape: 
i
save_2/ConstPlaceholderWithDefaultsave_2/filename*
dtype0*
_output_shapes
: *
shape: 

save_2/StringJoin/inputs_1Const*<
value3B1 B+_temp_e5ce19e91f9e4848b67a7506697908c4/part*
dtype0*
_output_shapes
: 
{
save_2/StringJoin
StringJoinsave_2/Constsave_2/StringJoin/inputs_1*
	separator *
N*
_output_shapes
: 
S
save_2/num_shardsConst*
dtype0*
_output_shapes
: *
value	B :
^
save_2/ShardedFilename/shardConst*
_output_shapes
: *
value	B : *
dtype0

save_2/ShardedFilenameShardedFilenamesave_2/StringJoinsave_2/ShardedFilename/shardsave_2/num_shards*
_output_shapes
: 
÷
save_2/SaveV2/tensor_namesConst*
_output_shapes
:D*¨
valueBDBbeta1_powerBbeta1_power_1Bbeta1_power_2Bbeta2_powerBbeta2_power_1Bbeta2_power_2Bpenalty/penalty_paramBpenalty/penalty_param/AdamBpenalty/penalty_param/Adam_1Bpenalty/penalty_param/Adam_2Bpenalty/penalty_param/Adam_3Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1B
pi/log_stdBpi/log_std/AdamBpi/log_std/Adam_1Bvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1*
dtype0
đ
save_2/SaveV2/shape_and_slicesConst*
valueBDB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:D
Ň
save_2/SaveV2SaveV2save_2/ShardedFilenamesave_2/SaveV2/tensor_namessave_2/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta1_power_2beta2_powerbeta2_power_1beta2_power_2penalty/penalty_parampenalty/penalty_param/Adampenalty/penalty_param/Adam_1penalty/penalty_param/Adam_2penalty/penalty_param/Adam_3pi/dense/biaspi/dense/bias/Adampi/dense/bias/Adam_1pi/dense/kernelpi/dense/kernel/Adampi/dense/kernel/Adam_1pi/dense_1/biaspi/dense_1/bias/Adampi/dense_1/bias/Adam_1pi/dense_1/kernelpi/dense_1/kernel/Adampi/dense_1/kernel/Adam_1pi/dense_2/biaspi/dense_2/bias/Adampi/dense_2/bias/Adam_1pi/dense_2/kernelpi/dense_2/kernel/Adampi/dense_2/kernel/Adam_1
pi/log_stdpi/log_std/Adampi/log_std/Adam_1vc/dense/biasvc/dense/bias/Adamvc/dense/bias/Adam_1vc/dense/kernelvc/dense/kernel/Adamvc/dense/kernel/Adam_1vc/dense_1/biasvc/dense_1/bias/Adamvc/dense_1/bias/Adam_1vc/dense_1/kernelvc/dense_1/kernel/Adamvc/dense_1/kernel/Adam_1vc/dense_2/biasvc/dense_2/bias/Adamvc/dense_2/bias/Adam_1vc/dense_2/kernelvc/dense_2/kernel/Adamvc/dense_2/kernel/Adam_1vf/dense/biasvf/dense/bias/Adamvf/dense/bias/Adam_1vf/dense/kernelvf/dense/kernel/Adamvf/dense/kernel/Adam_1vf/dense_1/biasvf/dense_1/bias/Adamvf/dense_1/bias/Adam_1vf/dense_1/kernelvf/dense_1/kernel/Adamvf/dense_1/kernel/Adam_1vf/dense_2/biasvf/dense_2/bias/Adamvf/dense_2/bias/Adam_1vf/dense_2/kernelvf/dense_2/kernel/Adamvf/dense_2/kernel/Adam_1*R
dtypesH
F2D

save_2/control_dependencyIdentitysave_2/ShardedFilename^save_2/SaveV2*)
_class
loc:@save_2/ShardedFilename*
T0*
_output_shapes
: 
Ł
-save_2/MergeV2Checkpoints/checkpoint_prefixesPacksave_2/ShardedFilename^save_2/control_dependency*
_output_shapes
:*

axis *
T0*
N

save_2/MergeV2CheckpointsMergeV2Checkpoints-save_2/MergeV2Checkpoints/checkpoint_prefixessave_2/Const*
delete_old_dirs(

save_2/IdentityIdentitysave_2/Const^save_2/MergeV2Checkpoints^save_2/control_dependency*
T0*
_output_shapes
: 
ú
save_2/RestoreV2/tensor_namesConst*
dtype0*¨
valueBDBbeta1_powerBbeta1_power_1Bbeta1_power_2Bbeta2_powerBbeta2_power_1Bbeta2_power_2Bpenalty/penalty_paramBpenalty/penalty_param/AdamBpenalty/penalty_param/Adam_1Bpenalty/penalty_param/Adam_2Bpenalty/penalty_param/Adam_3Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1B
pi/log_stdBpi/log_std/AdamBpi/log_std/Adam_1Bvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1*
_output_shapes
:D
ó
!save_2/RestoreV2/shape_and_slicesConst*
_output_shapes
:D*
valueBDB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0
ę
save_2/RestoreV2	RestoreV2save_2/Constsave_2/RestoreV2/tensor_names!save_2/RestoreV2/shape_and_slices*Ś
_output_shapes
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*R
dtypesH
F2D
Ş
save_2/AssignAssignbeta1_powersave_2/RestoreV2*
validate_shape(*
use_locking(*(
_class
loc:@penalty/penalty_param*
_output_shapes
: *
T0
°
save_2/Assign_1Assignbeta1_power_1save_2/RestoreV2:1*
validate_shape(*
_output_shapes
: *
T0*(
_class
loc:@penalty/penalty_param*
use_locking(
¨
save_2/Assign_2Assignbeta1_power_2save_2/RestoreV2:2* 
_class
loc:@vc/dense/bias*
T0*
validate_shape(*
use_locking(*
_output_shapes
: 
Ž
save_2/Assign_3Assignbeta2_powersave_2/RestoreV2:3*
T0*
use_locking(*
validate_shape(*
_output_shapes
: *(
_class
loc:@penalty/penalty_param
°
save_2/Assign_4Assignbeta2_power_1save_2/RestoreV2:4*
use_locking(*
_output_shapes
: *(
_class
loc:@penalty/penalty_param*
validate_shape(*
T0
¨
save_2/Assign_5Assignbeta2_power_2save_2/RestoreV2:5*
_output_shapes
: *
validate_shape(*
use_locking(* 
_class
loc:@vc/dense/bias*
T0
¸
save_2/Assign_6Assignpenalty/penalty_paramsave_2/RestoreV2:6*
validate_shape(*
use_locking(*
_output_shapes
: *
T0*(
_class
loc:@penalty/penalty_param
˝
save_2/Assign_7Assignpenalty/penalty_param/Adamsave_2/RestoreV2:7*
validate_shape(*
use_locking(*(
_class
loc:@penalty/penalty_param*
T0*
_output_shapes
: 
ż
save_2/Assign_8Assignpenalty/penalty_param/Adam_1save_2/RestoreV2:8*
_output_shapes
: *
T0*(
_class
loc:@penalty/penalty_param*
use_locking(*
validate_shape(
ż
save_2/Assign_9Assignpenalty/penalty_param/Adam_2save_2/RestoreV2:9*
T0*
use_locking(*
_output_shapes
: *(
_class
loc:@penalty/penalty_param*
validate_shape(
Á
save_2/Assign_10Assignpenalty/penalty_param/Adam_3save_2/RestoreV2:10*
T0*
_output_shapes
: *(
_class
loc:@penalty/penalty_param*
use_locking(*
validate_shape(
Ż
save_2/Assign_11Assignpi/dense/biassave_2/RestoreV2:11*
use_locking(* 
_class
loc:@pi/dense/bias*
_output_shapes	
:*
T0*
validate_shape(
´
save_2/Assign_12Assignpi/dense/bias/Adamsave_2/RestoreV2:12*
use_locking(*
_output_shapes	
:*
validate_shape(*
T0* 
_class
loc:@pi/dense/bias
ś
save_2/Assign_13Assignpi/dense/bias/Adam_1save_2/RestoreV2:13*
T0*
use_locking(* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes	
:
ˇ
save_2/Assign_14Assignpi/dense/kernelsave_2/RestoreV2:14*"
_class
loc:@pi/dense/kernel*
_output_shapes
:	L*
T0*
use_locking(*
validate_shape(
ź
save_2/Assign_15Assignpi/dense/kernel/Adamsave_2/RestoreV2:15*"
_class
loc:@pi/dense/kernel*
T0*
validate_shape(*
_output_shapes
:	L*
use_locking(
ž
save_2/Assign_16Assignpi/dense/kernel/Adam_1save_2/RestoreV2:16*
validate_shape(*
T0*"
_class
loc:@pi/dense/kernel*
use_locking(*
_output_shapes
:	L
ł
save_2/Assign_17Assignpi/dense_1/biassave_2/RestoreV2:17*
use_locking(*"
_class
loc:@pi/dense_1/bias*
validate_shape(*
_output_shapes	
:*
T0
¸
save_2/Assign_18Assignpi/dense_1/bias/Adamsave_2/RestoreV2:18*
T0*
_output_shapes	
:*
use_locking(*
validate_shape(*"
_class
loc:@pi/dense_1/bias
ş
save_2/Assign_19Assignpi/dense_1/bias/Adam_1save_2/RestoreV2:19*
T0*
_output_shapes	
:*
validate_shape(*"
_class
loc:@pi/dense_1/bias*
use_locking(
ź
save_2/Assign_20Assignpi/dense_1/kernelsave_2/RestoreV2:20* 
_output_shapes
:
*$
_class
loc:@pi/dense_1/kernel*
validate_shape(*
use_locking(*
T0
Á
save_2/Assign_21Assignpi/dense_1/kernel/Adamsave_2/RestoreV2:21*
use_locking(* 
_output_shapes
:
*
validate_shape(*
T0*$
_class
loc:@pi/dense_1/kernel
Ă
save_2/Assign_22Assignpi/dense_1/kernel/Adam_1save_2/RestoreV2:22*
validate_shape(*$
_class
loc:@pi/dense_1/kernel*
use_locking(* 
_output_shapes
:
*
T0
˛
save_2/Assign_23Assignpi/dense_2/biassave_2/RestoreV2:23*
validate_shape(*
use_locking(*
T0*"
_class
loc:@pi/dense_2/bias*
_output_shapes
:
ˇ
save_2/Assign_24Assignpi/dense_2/bias/Adamsave_2/RestoreV2:24*
T0*
validate_shape(*"
_class
loc:@pi/dense_2/bias*
use_locking(*
_output_shapes
:
š
save_2/Assign_25Assignpi/dense_2/bias/Adam_1save_2/RestoreV2:25*
_output_shapes
:*
use_locking(*
validate_shape(*"
_class
loc:@pi/dense_2/bias*
T0
ť
save_2/Assign_26Assignpi/dense_2/kernelsave_2/RestoreV2:26*
use_locking(*$
_class
loc:@pi/dense_2/kernel*
_output_shapes
:	*
validate_shape(*
T0
Ŕ
save_2/Assign_27Assignpi/dense_2/kernel/Adamsave_2/RestoreV2:27*
T0*
_output_shapes
:	*
validate_shape(*$
_class
loc:@pi/dense_2/kernel*
use_locking(
Â
save_2/Assign_28Assignpi/dense_2/kernel/Adam_1save_2/RestoreV2:28*
validate_shape(*$
_class
loc:@pi/dense_2/kernel*
use_locking(*
_output_shapes
:	*
T0
¨
save_2/Assign_29Assign
pi/log_stdsave_2/RestoreV2:29*
_class
loc:@pi/log_std*
_output_shapes
:*
T0*
validate_shape(*
use_locking(
­
save_2/Assign_30Assignpi/log_std/Adamsave_2/RestoreV2:30*
T0*
use_locking(*
validate_shape(*
_class
loc:@pi/log_std*
_output_shapes
:
Ż
save_2/Assign_31Assignpi/log_std/Adam_1save_2/RestoreV2:31*
use_locking(*
validate_shape(*
T0*
_output_shapes
:*
_class
loc:@pi/log_std
Ż
save_2/Assign_32Assignvc/dense/biassave_2/RestoreV2:32* 
_class
loc:@vc/dense/bias*
T0*
_output_shapes	
:*
use_locking(*
validate_shape(
´
save_2/Assign_33Assignvc/dense/bias/Adamsave_2/RestoreV2:33*
validate_shape(*
_output_shapes	
:*
T0* 
_class
loc:@vc/dense/bias*
use_locking(
ś
save_2/Assign_34Assignvc/dense/bias/Adam_1save_2/RestoreV2:34* 
_class
loc:@vc/dense/bias*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0
ˇ
save_2/Assign_35Assignvc/dense/kernelsave_2/RestoreV2:35*
T0*
use_locking(*
validate_shape(*"
_class
loc:@vc/dense/kernel*
_output_shapes
:	L
ź
save_2/Assign_36Assignvc/dense/kernel/Adamsave_2/RestoreV2:36*"
_class
loc:@vc/dense/kernel*
_output_shapes
:	L*
T0*
validate_shape(*
use_locking(
ž
save_2/Assign_37Assignvc/dense/kernel/Adam_1save_2/RestoreV2:37*
_output_shapes
:	L*
T0*
validate_shape(*
use_locking(*"
_class
loc:@vc/dense/kernel
ł
save_2/Assign_38Assignvc/dense_1/biassave_2/RestoreV2:38*"
_class
loc:@vc/dense_1/bias*
validate_shape(*
T0*
use_locking(*
_output_shapes	
:
¸
save_2/Assign_39Assignvc/dense_1/bias/Adamsave_2/RestoreV2:39*
T0*
_output_shapes	
:*"
_class
loc:@vc/dense_1/bias*
use_locking(*
validate_shape(
ş
save_2/Assign_40Assignvc/dense_1/bias/Adam_1save_2/RestoreV2:40*
T0*
validate_shape(*
use_locking(*"
_class
loc:@vc/dense_1/bias*
_output_shapes	
:
ź
save_2/Assign_41Assignvc/dense_1/kernelsave_2/RestoreV2:41*
T0*
use_locking(*
validate_shape(* 
_output_shapes
:
*$
_class
loc:@vc/dense_1/kernel
Á
save_2/Assign_42Assignvc/dense_1/kernel/Adamsave_2/RestoreV2:42* 
_output_shapes
:
*
T0*$
_class
loc:@vc/dense_1/kernel*
validate_shape(*
use_locking(
Ă
save_2/Assign_43Assignvc/dense_1/kernel/Adam_1save_2/RestoreV2:43* 
_output_shapes
:
*$
_class
loc:@vc/dense_1/kernel*
validate_shape(*
T0*
use_locking(
˛
save_2/Assign_44Assignvc/dense_2/biassave_2/RestoreV2:44*
T0*
validate_shape(*
use_locking(*
_output_shapes
:*"
_class
loc:@vc/dense_2/bias
ˇ
save_2/Assign_45Assignvc/dense_2/bias/Adamsave_2/RestoreV2:45*
T0*
_output_shapes
:*
validate_shape(*
use_locking(*"
_class
loc:@vc/dense_2/bias
š
save_2/Assign_46Assignvc/dense_2/bias/Adam_1save_2/RestoreV2:46*"
_class
loc:@vc/dense_2/bias*
_output_shapes
:*
T0*
validate_shape(*
use_locking(
ť
save_2/Assign_47Assignvc/dense_2/kernelsave_2/RestoreV2:47*
use_locking(*
_output_shapes
:	*
validate_shape(*$
_class
loc:@vc/dense_2/kernel*
T0
Ŕ
save_2/Assign_48Assignvc/dense_2/kernel/Adamsave_2/RestoreV2:48*$
_class
loc:@vc/dense_2/kernel*
use_locking(*
validate_shape(*
T0*
_output_shapes
:	
Â
save_2/Assign_49Assignvc/dense_2/kernel/Adam_1save_2/RestoreV2:49*$
_class
loc:@vc/dense_2/kernel*
validate_shape(*
T0*
_output_shapes
:	*
use_locking(
Ż
save_2/Assign_50Assignvf/dense/biassave_2/RestoreV2:50*
T0*
use_locking(*
_output_shapes	
:* 
_class
loc:@vf/dense/bias*
validate_shape(
´
save_2/Assign_51Assignvf/dense/bias/Adamsave_2/RestoreV2:51*
use_locking(*
validate_shape(*
_output_shapes	
:* 
_class
loc:@vf/dense/bias*
T0
ś
save_2/Assign_52Assignvf/dense/bias/Adam_1save_2/RestoreV2:52*
T0* 
_class
loc:@vf/dense/bias*
validate_shape(*
_output_shapes	
:*
use_locking(
ˇ
save_2/Assign_53Assignvf/dense/kernelsave_2/RestoreV2:53*
_output_shapes
:	L*
validate_shape(*
T0*
use_locking(*"
_class
loc:@vf/dense/kernel
ź
save_2/Assign_54Assignvf/dense/kernel/Adamsave_2/RestoreV2:54*
T0*
_output_shapes
:	L*
use_locking(*
validate_shape(*"
_class
loc:@vf/dense/kernel
ž
save_2/Assign_55Assignvf/dense/kernel/Adam_1save_2/RestoreV2:55*
_output_shapes
:	L*"
_class
loc:@vf/dense/kernel*
use_locking(*
validate_shape(*
T0
ł
save_2/Assign_56Assignvf/dense_1/biassave_2/RestoreV2:56*
use_locking(*
validate_shape(*
_output_shapes	
:*"
_class
loc:@vf/dense_1/bias*
T0
¸
save_2/Assign_57Assignvf/dense_1/bias/Adamsave_2/RestoreV2:57*
_output_shapes	
:*"
_class
loc:@vf/dense_1/bias*
T0*
validate_shape(*
use_locking(
ş
save_2/Assign_58Assignvf/dense_1/bias/Adam_1save_2/RestoreV2:58*
T0*
_output_shapes	
:*
use_locking(*
validate_shape(*"
_class
loc:@vf/dense_1/bias
ź
save_2/Assign_59Assignvf/dense_1/kernelsave_2/RestoreV2:59* 
_output_shapes
:
*
T0*
use_locking(*
validate_shape(*$
_class
loc:@vf/dense_1/kernel
Á
save_2/Assign_60Assignvf/dense_1/kernel/Adamsave_2/RestoreV2:60*$
_class
loc:@vf/dense_1/kernel* 
_output_shapes
:
*
use_locking(*
T0*
validate_shape(
Ă
save_2/Assign_61Assignvf/dense_1/kernel/Adam_1save_2/RestoreV2:61*$
_class
loc:@vf/dense_1/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0
˛
save_2/Assign_62Assignvf/dense_2/biassave_2/RestoreV2:62*
use_locking(*"
_class
loc:@vf/dense_2/bias*
T0*
validate_shape(*
_output_shapes
:
ˇ
save_2/Assign_63Assignvf/dense_2/bias/Adamsave_2/RestoreV2:63*
use_locking(*
T0*"
_class
loc:@vf/dense_2/bias*
_output_shapes
:*
validate_shape(
š
save_2/Assign_64Assignvf/dense_2/bias/Adam_1save_2/RestoreV2:64*
validate_shape(*
_output_shapes
:*
use_locking(*"
_class
loc:@vf/dense_2/bias*
T0
ť
save_2/Assign_65Assignvf/dense_2/kernelsave_2/RestoreV2:65*
use_locking(*$
_class
loc:@vf/dense_2/kernel*
T0*
_output_shapes
:	*
validate_shape(
Ŕ
save_2/Assign_66Assignvf/dense_2/kernel/Adamsave_2/RestoreV2:66*
_output_shapes
:	*
use_locking(*
T0*
validate_shape(*$
_class
loc:@vf/dense_2/kernel
Â
save_2/Assign_67Assignvf/dense_2/kernel/Adam_1save_2/RestoreV2:67*
use_locking(*
_output_shapes
:	*
validate_shape(*
T0*$
_class
loc:@vf/dense_2/kernel


save_2/restore_shardNoOp^save_2/Assign^save_2/Assign_1^save_2/Assign_10^save_2/Assign_11^save_2/Assign_12^save_2/Assign_13^save_2/Assign_14^save_2/Assign_15^save_2/Assign_16^save_2/Assign_17^save_2/Assign_18^save_2/Assign_19^save_2/Assign_2^save_2/Assign_20^save_2/Assign_21^save_2/Assign_22^save_2/Assign_23^save_2/Assign_24^save_2/Assign_25^save_2/Assign_26^save_2/Assign_27^save_2/Assign_28^save_2/Assign_29^save_2/Assign_3^save_2/Assign_30^save_2/Assign_31^save_2/Assign_32^save_2/Assign_33^save_2/Assign_34^save_2/Assign_35^save_2/Assign_36^save_2/Assign_37^save_2/Assign_38^save_2/Assign_39^save_2/Assign_4^save_2/Assign_40^save_2/Assign_41^save_2/Assign_42^save_2/Assign_43^save_2/Assign_44^save_2/Assign_45^save_2/Assign_46^save_2/Assign_47^save_2/Assign_48^save_2/Assign_49^save_2/Assign_5^save_2/Assign_50^save_2/Assign_51^save_2/Assign_52^save_2/Assign_53^save_2/Assign_54^save_2/Assign_55^save_2/Assign_56^save_2/Assign_57^save_2/Assign_58^save_2/Assign_59^save_2/Assign_6^save_2/Assign_60^save_2/Assign_61^save_2/Assign_62^save_2/Assign_63^save_2/Assign_64^save_2/Assign_65^save_2/Assign_66^save_2/Assign_67^save_2/Assign_7^save_2/Assign_8^save_2/Assign_9
1
save_2/restore_allNoOp^save_2/restore_shard
[
save_3/filename/inputConst*
_output_shapes
: *
dtype0*
valueB Bmodel
r
save_3/filenamePlaceholderWithDefaultsave_3/filename/input*
dtype0*
_output_shapes
: *
shape: 
i
save_3/ConstPlaceholderWithDefaultsave_3/filename*
dtype0*
_output_shapes
: *
shape: 

save_3/StringJoin/inputs_1Const*<
value3B1 B+_temp_8d72ce54b5e240bab45d927130614b40/part*
_output_shapes
: *
dtype0
{
save_3/StringJoin
StringJoinsave_3/Constsave_3/StringJoin/inputs_1*
_output_shapes
: *
N*
	separator 
S
save_3/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
^
save_3/ShardedFilename/shardConst*
value	B : *
_output_shapes
: *
dtype0

save_3/ShardedFilenameShardedFilenamesave_3/StringJoinsave_3/ShardedFilename/shardsave_3/num_shards*
_output_shapes
: 
÷
save_3/SaveV2/tensor_namesConst*¨
valueBDBbeta1_powerBbeta1_power_1Bbeta1_power_2Bbeta2_powerBbeta2_power_1Bbeta2_power_2Bpenalty/penalty_paramBpenalty/penalty_param/AdamBpenalty/penalty_param/Adam_1Bpenalty/penalty_param/Adam_2Bpenalty/penalty_param/Adam_3Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1B
pi/log_stdBpi/log_std/AdamBpi/log_std/Adam_1Bvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1*
_output_shapes
:D*
dtype0
đ
save_3/SaveV2/shape_and_slicesConst*
_output_shapes
:D*
valueBDB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0
Ň
save_3/SaveV2SaveV2save_3/ShardedFilenamesave_3/SaveV2/tensor_namessave_3/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta1_power_2beta2_powerbeta2_power_1beta2_power_2penalty/penalty_parampenalty/penalty_param/Adampenalty/penalty_param/Adam_1penalty/penalty_param/Adam_2penalty/penalty_param/Adam_3pi/dense/biaspi/dense/bias/Adampi/dense/bias/Adam_1pi/dense/kernelpi/dense/kernel/Adampi/dense/kernel/Adam_1pi/dense_1/biaspi/dense_1/bias/Adampi/dense_1/bias/Adam_1pi/dense_1/kernelpi/dense_1/kernel/Adampi/dense_1/kernel/Adam_1pi/dense_2/biaspi/dense_2/bias/Adampi/dense_2/bias/Adam_1pi/dense_2/kernelpi/dense_2/kernel/Adampi/dense_2/kernel/Adam_1
pi/log_stdpi/log_std/Adampi/log_std/Adam_1vc/dense/biasvc/dense/bias/Adamvc/dense/bias/Adam_1vc/dense/kernelvc/dense/kernel/Adamvc/dense/kernel/Adam_1vc/dense_1/biasvc/dense_1/bias/Adamvc/dense_1/bias/Adam_1vc/dense_1/kernelvc/dense_1/kernel/Adamvc/dense_1/kernel/Adam_1vc/dense_2/biasvc/dense_2/bias/Adamvc/dense_2/bias/Adam_1vc/dense_2/kernelvc/dense_2/kernel/Adamvc/dense_2/kernel/Adam_1vf/dense/biasvf/dense/bias/Adamvf/dense/bias/Adam_1vf/dense/kernelvf/dense/kernel/Adamvf/dense/kernel/Adam_1vf/dense_1/biasvf/dense_1/bias/Adamvf/dense_1/bias/Adam_1vf/dense_1/kernelvf/dense_1/kernel/Adamvf/dense_1/kernel/Adam_1vf/dense_2/biasvf/dense_2/bias/Adamvf/dense_2/bias/Adam_1vf/dense_2/kernelvf/dense_2/kernel/Adamvf/dense_2/kernel/Adam_1*R
dtypesH
F2D

save_3/control_dependencyIdentitysave_3/ShardedFilename^save_3/SaveV2*)
_class
loc:@save_3/ShardedFilename*
_output_shapes
: *
T0
Ł
-save_3/MergeV2Checkpoints/checkpoint_prefixesPacksave_3/ShardedFilename^save_3/control_dependency*

axis *
T0*
_output_shapes
:*
N

save_3/MergeV2CheckpointsMergeV2Checkpoints-save_3/MergeV2Checkpoints/checkpoint_prefixessave_3/Const*
delete_old_dirs(

save_3/IdentityIdentitysave_3/Const^save_3/MergeV2Checkpoints^save_3/control_dependency*
_output_shapes
: *
T0
ú
save_3/RestoreV2/tensor_namesConst*¨
valueBDBbeta1_powerBbeta1_power_1Bbeta1_power_2Bbeta2_powerBbeta2_power_1Bbeta2_power_2Bpenalty/penalty_paramBpenalty/penalty_param/AdamBpenalty/penalty_param/Adam_1Bpenalty/penalty_param/Adam_2Bpenalty/penalty_param/Adam_3Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1B
pi/log_stdBpi/log_std/AdamBpi/log_std/Adam_1Bvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1*
_output_shapes
:D*
dtype0
ó
!save_3/RestoreV2/shape_and_slicesConst*
dtype0*
valueBDB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
_output_shapes
:D
ę
save_3/RestoreV2	RestoreV2save_3/Constsave_3/RestoreV2/tensor_names!save_3/RestoreV2/shape_and_slices*R
dtypesH
F2D*Ś
_output_shapes
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
Ş
save_3/AssignAssignbeta1_powersave_3/RestoreV2*
_output_shapes
: *(
_class
loc:@penalty/penalty_param*
use_locking(*
T0*
validate_shape(
°
save_3/Assign_1Assignbeta1_power_1save_3/RestoreV2:1*
use_locking(*
T0*(
_class
loc:@penalty/penalty_param*
_output_shapes
: *
validate_shape(
¨
save_3/Assign_2Assignbeta1_power_2save_3/RestoreV2:2*
use_locking(*
_output_shapes
: * 
_class
loc:@vc/dense/bias*
validate_shape(*
T0
Ž
save_3/Assign_3Assignbeta2_powersave_3/RestoreV2:3*(
_class
loc:@penalty/penalty_param*
_output_shapes
: *
validate_shape(*
T0*
use_locking(
°
save_3/Assign_4Assignbeta2_power_1save_3/RestoreV2:4*
T0*
use_locking(*(
_class
loc:@penalty/penalty_param*
_output_shapes
: *
validate_shape(
¨
save_3/Assign_5Assignbeta2_power_2save_3/RestoreV2:5*
validate_shape(* 
_class
loc:@vc/dense/bias*
T0*
use_locking(*
_output_shapes
: 
¸
save_3/Assign_6Assignpenalty/penalty_paramsave_3/RestoreV2:6*
validate_shape(*
use_locking(*
T0*(
_class
loc:@penalty/penalty_param*
_output_shapes
: 
˝
save_3/Assign_7Assignpenalty/penalty_param/Adamsave_3/RestoreV2:7*
use_locking(*
T0*
validate_shape(*
_output_shapes
: *(
_class
loc:@penalty/penalty_param
ż
save_3/Assign_8Assignpenalty/penalty_param/Adam_1save_3/RestoreV2:8*
T0*
_output_shapes
: *
validate_shape(*
use_locking(*(
_class
loc:@penalty/penalty_param
ż
save_3/Assign_9Assignpenalty/penalty_param/Adam_2save_3/RestoreV2:9*
validate_shape(*(
_class
loc:@penalty/penalty_param*
use_locking(*
_output_shapes
: *
T0
Á
save_3/Assign_10Assignpenalty/penalty_param/Adam_3save_3/RestoreV2:10*
use_locking(*
validate_shape(*
_output_shapes
: *
T0*(
_class
loc:@penalty/penalty_param
Ż
save_3/Assign_11Assignpi/dense/biassave_3/RestoreV2:11*
T0*
_output_shapes	
:*
use_locking(*
validate_shape(* 
_class
loc:@pi/dense/bias
´
save_3/Assign_12Assignpi/dense/bias/Adamsave_3/RestoreV2:12*
use_locking(*
T0*
validate_shape(*
_output_shapes	
:* 
_class
loc:@pi/dense/bias
ś
save_3/Assign_13Assignpi/dense/bias/Adam_1save_3/RestoreV2:13*
use_locking(*
_output_shapes	
:* 
_class
loc:@pi/dense/bias*
T0*
validate_shape(
ˇ
save_3/Assign_14Assignpi/dense/kernelsave_3/RestoreV2:14*
T0*
validate_shape(*
_output_shapes
:	L*
use_locking(*"
_class
loc:@pi/dense/kernel
ź
save_3/Assign_15Assignpi/dense/kernel/Adamsave_3/RestoreV2:15*
T0*"
_class
loc:@pi/dense/kernel*
validate_shape(*
_output_shapes
:	L*
use_locking(
ž
save_3/Assign_16Assignpi/dense/kernel/Adam_1save_3/RestoreV2:16*
use_locking(*
validate_shape(*"
_class
loc:@pi/dense/kernel*
T0*
_output_shapes
:	L
ł
save_3/Assign_17Assignpi/dense_1/biassave_3/RestoreV2:17*
use_locking(*
validate_shape(*
_output_shapes	
:*"
_class
loc:@pi/dense_1/bias*
T0
¸
save_3/Assign_18Assignpi/dense_1/bias/Adamsave_3/RestoreV2:18*"
_class
loc:@pi/dense_1/bias*
T0*
_output_shapes	
:*
validate_shape(*
use_locking(
ş
save_3/Assign_19Assignpi/dense_1/bias/Adam_1save_3/RestoreV2:19*
T0*
use_locking(*
validate_shape(*"
_class
loc:@pi/dense_1/bias*
_output_shapes	
:
ź
save_3/Assign_20Assignpi/dense_1/kernelsave_3/RestoreV2:20*$
_class
loc:@pi/dense_1/kernel*
validate_shape(* 
_output_shapes
:
*
T0*
use_locking(
Á
save_3/Assign_21Assignpi/dense_1/kernel/Adamsave_3/RestoreV2:21*
T0*$
_class
loc:@pi/dense_1/kernel*
validate_shape(*
use_locking(* 
_output_shapes
:

Ă
save_3/Assign_22Assignpi/dense_1/kernel/Adam_1save_3/RestoreV2:22*
validate_shape(*
use_locking(* 
_output_shapes
:
*
T0*$
_class
loc:@pi/dense_1/kernel
˛
save_3/Assign_23Assignpi/dense_2/biassave_3/RestoreV2:23*
_output_shapes
:*
T0*"
_class
loc:@pi/dense_2/bias*
validate_shape(*
use_locking(
ˇ
save_3/Assign_24Assignpi/dense_2/bias/Adamsave_3/RestoreV2:24*
validate_shape(*
use_locking(*
T0*
_output_shapes
:*"
_class
loc:@pi/dense_2/bias
š
save_3/Assign_25Assignpi/dense_2/bias/Adam_1save_3/RestoreV2:25*
validate_shape(*
use_locking(*"
_class
loc:@pi/dense_2/bias*
_output_shapes
:*
T0
ť
save_3/Assign_26Assignpi/dense_2/kernelsave_3/RestoreV2:26*
_output_shapes
:	*
validate_shape(*$
_class
loc:@pi/dense_2/kernel*
use_locking(*
T0
Ŕ
save_3/Assign_27Assignpi/dense_2/kernel/Adamsave_3/RestoreV2:27*
use_locking(*
T0*
validate_shape(*$
_class
loc:@pi/dense_2/kernel*
_output_shapes
:	
Â
save_3/Assign_28Assignpi/dense_2/kernel/Adam_1save_3/RestoreV2:28*
_output_shapes
:	*
T0*
use_locking(*
validate_shape(*$
_class
loc:@pi/dense_2/kernel
¨
save_3/Assign_29Assign
pi/log_stdsave_3/RestoreV2:29*
_output_shapes
:*
_class
loc:@pi/log_std*
T0*
validate_shape(*
use_locking(
­
save_3/Assign_30Assignpi/log_std/Adamsave_3/RestoreV2:30*
_output_shapes
:*
_class
loc:@pi/log_std*
use_locking(*
validate_shape(*
T0
Ż
save_3/Assign_31Assignpi/log_std/Adam_1save_3/RestoreV2:31*
validate_shape(*
_class
loc:@pi/log_std*
use_locking(*
_output_shapes
:*
T0
Ż
save_3/Assign_32Assignvc/dense/biassave_3/RestoreV2:32*
_output_shapes	
:* 
_class
loc:@vc/dense/bias*
T0*
validate_shape(*
use_locking(
´
save_3/Assign_33Assignvc/dense/bias/Adamsave_3/RestoreV2:33*
_output_shapes	
:*
validate_shape(*
T0* 
_class
loc:@vc/dense/bias*
use_locking(
ś
save_3/Assign_34Assignvc/dense/bias/Adam_1save_3/RestoreV2:34*
T0*
validate_shape(*
use_locking(*
_output_shapes	
:* 
_class
loc:@vc/dense/bias
ˇ
save_3/Assign_35Assignvc/dense/kernelsave_3/RestoreV2:35*
use_locking(*
T0*"
_class
loc:@vc/dense/kernel*
validate_shape(*
_output_shapes
:	L
ź
save_3/Assign_36Assignvc/dense/kernel/Adamsave_3/RestoreV2:36*
use_locking(*
_output_shapes
:	L*"
_class
loc:@vc/dense/kernel*
validate_shape(*
T0
ž
save_3/Assign_37Assignvc/dense/kernel/Adam_1save_3/RestoreV2:37*"
_class
loc:@vc/dense/kernel*
_output_shapes
:	L*
T0*
validate_shape(*
use_locking(
ł
save_3/Assign_38Assignvc/dense_1/biassave_3/RestoreV2:38*"
_class
loc:@vc/dense_1/bias*
T0*
use_locking(*
_output_shapes	
:*
validate_shape(
¸
save_3/Assign_39Assignvc/dense_1/bias/Adamsave_3/RestoreV2:39*
validate_shape(*
T0*
use_locking(*
_output_shapes	
:*"
_class
loc:@vc/dense_1/bias
ş
save_3/Assign_40Assignvc/dense_1/bias/Adam_1save_3/RestoreV2:40*"
_class
loc:@vc/dense_1/bias*
T0*
_output_shapes	
:*
use_locking(*
validate_shape(
ź
save_3/Assign_41Assignvc/dense_1/kernelsave_3/RestoreV2:41*
use_locking(*$
_class
loc:@vc/dense_1/kernel*
validate_shape(* 
_output_shapes
:
*
T0
Á
save_3/Assign_42Assignvc/dense_1/kernel/Adamsave_3/RestoreV2:42*
use_locking(* 
_output_shapes
:
*$
_class
loc:@vc/dense_1/kernel*
T0*
validate_shape(
Ă
save_3/Assign_43Assignvc/dense_1/kernel/Adam_1save_3/RestoreV2:43*$
_class
loc:@vc/dense_1/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0
˛
save_3/Assign_44Assignvc/dense_2/biassave_3/RestoreV2:44*
validate_shape(*
T0*
_output_shapes
:*
use_locking(*"
_class
loc:@vc/dense_2/bias
ˇ
save_3/Assign_45Assignvc/dense_2/bias/Adamsave_3/RestoreV2:45*
use_locking(*
validate_shape(*
_output_shapes
:*"
_class
loc:@vc/dense_2/bias*
T0
š
save_3/Assign_46Assignvc/dense_2/bias/Adam_1save_3/RestoreV2:46*
_output_shapes
:*
validate_shape(*"
_class
loc:@vc/dense_2/bias*
use_locking(*
T0
ť
save_3/Assign_47Assignvc/dense_2/kernelsave_3/RestoreV2:47*
validate_shape(*
T0*
_output_shapes
:	*
use_locking(*$
_class
loc:@vc/dense_2/kernel
Ŕ
save_3/Assign_48Assignvc/dense_2/kernel/Adamsave_3/RestoreV2:48*
_output_shapes
:	*
validate_shape(*$
_class
loc:@vc/dense_2/kernel*
T0*
use_locking(
Â
save_3/Assign_49Assignvc/dense_2/kernel/Adam_1save_3/RestoreV2:49*
use_locking(*$
_class
loc:@vc/dense_2/kernel*
_output_shapes
:	*
T0*
validate_shape(
Ż
save_3/Assign_50Assignvf/dense/biassave_3/RestoreV2:50* 
_class
loc:@vf/dense/bias*
use_locking(*
T0*
_output_shapes	
:*
validate_shape(
´
save_3/Assign_51Assignvf/dense/bias/Adamsave_3/RestoreV2:51* 
_class
loc:@vf/dense/bias*
T0*
_output_shapes	
:*
validate_shape(*
use_locking(
ś
save_3/Assign_52Assignvf/dense/bias/Adam_1save_3/RestoreV2:52*
use_locking(* 
_class
loc:@vf/dense/bias*
_output_shapes	
:*
validate_shape(*
T0
ˇ
save_3/Assign_53Assignvf/dense/kernelsave_3/RestoreV2:53*
_output_shapes
:	L*"
_class
loc:@vf/dense/kernel*
T0*
validate_shape(*
use_locking(
ź
save_3/Assign_54Assignvf/dense/kernel/Adamsave_3/RestoreV2:54*
validate_shape(*
_output_shapes
:	L*
T0*"
_class
loc:@vf/dense/kernel*
use_locking(
ž
save_3/Assign_55Assignvf/dense/kernel/Adam_1save_3/RestoreV2:55*
T0*
_output_shapes
:	L*
use_locking(*
validate_shape(*"
_class
loc:@vf/dense/kernel
ł
save_3/Assign_56Assignvf/dense_1/biassave_3/RestoreV2:56*"
_class
loc:@vf/dense_1/bias*
T0*
use_locking(*
validate_shape(*
_output_shapes	
:
¸
save_3/Assign_57Assignvf/dense_1/bias/Adamsave_3/RestoreV2:57*
_output_shapes	
:*
use_locking(*
validate_shape(*"
_class
loc:@vf/dense_1/bias*
T0
ş
save_3/Assign_58Assignvf/dense_1/bias/Adam_1save_3/RestoreV2:58*"
_class
loc:@vf/dense_1/bias*
_output_shapes	
:*
T0*
use_locking(*
validate_shape(
ź
save_3/Assign_59Assignvf/dense_1/kernelsave_3/RestoreV2:59*
use_locking(*
T0*$
_class
loc:@vf/dense_1/kernel* 
_output_shapes
:
*
validate_shape(
Á
save_3/Assign_60Assignvf/dense_1/kernel/Adamsave_3/RestoreV2:60*
T0*$
_class
loc:@vf/dense_1/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(
Ă
save_3/Assign_61Assignvf/dense_1/kernel/Adam_1save_3/RestoreV2:61*
use_locking(* 
_output_shapes
:
*
validate_shape(*
T0*$
_class
loc:@vf/dense_1/kernel
˛
save_3/Assign_62Assignvf/dense_2/biassave_3/RestoreV2:62*
T0*
validate_shape(*"
_class
loc:@vf/dense_2/bias*
_output_shapes
:*
use_locking(
ˇ
save_3/Assign_63Assignvf/dense_2/bias/Adamsave_3/RestoreV2:63*
validate_shape(*"
_class
loc:@vf/dense_2/bias*
_output_shapes
:*
T0*
use_locking(
š
save_3/Assign_64Assignvf/dense_2/bias/Adam_1save_3/RestoreV2:64*"
_class
loc:@vf/dense_2/bias*
T0*
validate_shape(*
use_locking(*
_output_shapes
:
ť
save_3/Assign_65Assignvf/dense_2/kernelsave_3/RestoreV2:65*
_output_shapes
:	*
T0*
use_locking(*
validate_shape(*$
_class
loc:@vf/dense_2/kernel
Ŕ
save_3/Assign_66Assignvf/dense_2/kernel/Adamsave_3/RestoreV2:66*
validate_shape(*
use_locking(*
_output_shapes
:	*$
_class
loc:@vf/dense_2/kernel*
T0
Â
save_3/Assign_67Assignvf/dense_2/kernel/Adam_1save_3/RestoreV2:67*
_output_shapes
:	*
use_locking(*$
_class
loc:@vf/dense_2/kernel*
validate_shape(*
T0


save_3/restore_shardNoOp^save_3/Assign^save_3/Assign_1^save_3/Assign_10^save_3/Assign_11^save_3/Assign_12^save_3/Assign_13^save_3/Assign_14^save_3/Assign_15^save_3/Assign_16^save_3/Assign_17^save_3/Assign_18^save_3/Assign_19^save_3/Assign_2^save_3/Assign_20^save_3/Assign_21^save_3/Assign_22^save_3/Assign_23^save_3/Assign_24^save_3/Assign_25^save_3/Assign_26^save_3/Assign_27^save_3/Assign_28^save_3/Assign_29^save_3/Assign_3^save_3/Assign_30^save_3/Assign_31^save_3/Assign_32^save_3/Assign_33^save_3/Assign_34^save_3/Assign_35^save_3/Assign_36^save_3/Assign_37^save_3/Assign_38^save_3/Assign_39^save_3/Assign_4^save_3/Assign_40^save_3/Assign_41^save_3/Assign_42^save_3/Assign_43^save_3/Assign_44^save_3/Assign_45^save_3/Assign_46^save_3/Assign_47^save_3/Assign_48^save_3/Assign_49^save_3/Assign_5^save_3/Assign_50^save_3/Assign_51^save_3/Assign_52^save_3/Assign_53^save_3/Assign_54^save_3/Assign_55^save_3/Assign_56^save_3/Assign_57^save_3/Assign_58^save_3/Assign_59^save_3/Assign_6^save_3/Assign_60^save_3/Assign_61^save_3/Assign_62^save_3/Assign_63^save_3/Assign_64^save_3/Assign_65^save_3/Assign_66^save_3/Assign_67^save_3/Assign_7^save_3/Assign_8^save_3/Assign_9
1
save_3/restore_allNoOp^save_3/restore_shard
[
save_4/filename/inputConst*
valueB Bmodel*
_output_shapes
: *
dtype0
r
save_4/filenamePlaceholderWithDefaultsave_4/filename/input*
_output_shapes
: *
dtype0*
shape: 
i
save_4/ConstPlaceholderWithDefaultsave_4/filename*
dtype0*
_output_shapes
: *
shape: 

save_4/StringJoin/inputs_1Const*
dtype0*
_output_shapes
: *<
value3B1 B+_temp_03bbafc4db154e6a84abe4ef7eb71539/part
{
save_4/StringJoin
StringJoinsave_4/Constsave_4/StringJoin/inputs_1*
N*
_output_shapes
: *
	separator 
S
save_4/num_shardsConst*
dtype0*
value	B :*
_output_shapes
: 
^
save_4/ShardedFilename/shardConst*
_output_shapes
: *
dtype0*
value	B : 

save_4/ShardedFilenameShardedFilenamesave_4/StringJoinsave_4/ShardedFilename/shardsave_4/num_shards*
_output_shapes
: 
÷
save_4/SaveV2/tensor_namesConst*¨
valueBDBbeta1_powerBbeta1_power_1Bbeta1_power_2Bbeta2_powerBbeta2_power_1Bbeta2_power_2Bpenalty/penalty_paramBpenalty/penalty_param/AdamBpenalty/penalty_param/Adam_1Bpenalty/penalty_param/Adam_2Bpenalty/penalty_param/Adam_3Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1B
pi/log_stdBpi/log_std/AdamBpi/log_std/Adam_1Bvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1*
dtype0*
_output_shapes
:D
đ
save_4/SaveV2/shape_and_slicesConst*
dtype0*
_output_shapes
:D*
valueBDB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
Ň
save_4/SaveV2SaveV2save_4/ShardedFilenamesave_4/SaveV2/tensor_namessave_4/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta1_power_2beta2_powerbeta2_power_1beta2_power_2penalty/penalty_parampenalty/penalty_param/Adampenalty/penalty_param/Adam_1penalty/penalty_param/Adam_2penalty/penalty_param/Adam_3pi/dense/biaspi/dense/bias/Adampi/dense/bias/Adam_1pi/dense/kernelpi/dense/kernel/Adampi/dense/kernel/Adam_1pi/dense_1/biaspi/dense_1/bias/Adampi/dense_1/bias/Adam_1pi/dense_1/kernelpi/dense_1/kernel/Adampi/dense_1/kernel/Adam_1pi/dense_2/biaspi/dense_2/bias/Adampi/dense_2/bias/Adam_1pi/dense_2/kernelpi/dense_2/kernel/Adampi/dense_2/kernel/Adam_1
pi/log_stdpi/log_std/Adampi/log_std/Adam_1vc/dense/biasvc/dense/bias/Adamvc/dense/bias/Adam_1vc/dense/kernelvc/dense/kernel/Adamvc/dense/kernel/Adam_1vc/dense_1/biasvc/dense_1/bias/Adamvc/dense_1/bias/Adam_1vc/dense_1/kernelvc/dense_1/kernel/Adamvc/dense_1/kernel/Adam_1vc/dense_2/biasvc/dense_2/bias/Adamvc/dense_2/bias/Adam_1vc/dense_2/kernelvc/dense_2/kernel/Adamvc/dense_2/kernel/Adam_1vf/dense/biasvf/dense/bias/Adamvf/dense/bias/Adam_1vf/dense/kernelvf/dense/kernel/Adamvf/dense/kernel/Adam_1vf/dense_1/biasvf/dense_1/bias/Adamvf/dense_1/bias/Adam_1vf/dense_1/kernelvf/dense_1/kernel/Adamvf/dense_1/kernel/Adam_1vf/dense_2/biasvf/dense_2/bias/Adamvf/dense_2/bias/Adam_1vf/dense_2/kernelvf/dense_2/kernel/Adamvf/dense_2/kernel/Adam_1*R
dtypesH
F2D

save_4/control_dependencyIdentitysave_4/ShardedFilename^save_4/SaveV2*
T0*
_output_shapes
: *)
_class
loc:@save_4/ShardedFilename
Ł
-save_4/MergeV2Checkpoints/checkpoint_prefixesPacksave_4/ShardedFilename^save_4/control_dependency*

axis *
N*
T0*
_output_shapes
:

save_4/MergeV2CheckpointsMergeV2Checkpoints-save_4/MergeV2Checkpoints/checkpoint_prefixessave_4/Const*
delete_old_dirs(

save_4/IdentityIdentitysave_4/Const^save_4/MergeV2Checkpoints^save_4/control_dependency*
T0*
_output_shapes
: 
ú
save_4/RestoreV2/tensor_namesConst*
dtype0*¨
valueBDBbeta1_powerBbeta1_power_1Bbeta1_power_2Bbeta2_powerBbeta2_power_1Bbeta2_power_2Bpenalty/penalty_paramBpenalty/penalty_param/AdamBpenalty/penalty_param/Adam_1Bpenalty/penalty_param/Adam_2Bpenalty/penalty_param/Adam_3Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1B
pi/log_stdBpi/log_std/AdamBpi/log_std/Adam_1Bvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1*
_output_shapes
:D
ó
!save_4/RestoreV2/shape_and_slicesConst*
dtype0*
_output_shapes
:D*
valueBDB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
ę
save_4/RestoreV2	RestoreV2save_4/Constsave_4/RestoreV2/tensor_names!save_4/RestoreV2/shape_and_slices*R
dtypesH
F2D*Ś
_output_shapes
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
Ş
save_4/AssignAssignbeta1_powersave_4/RestoreV2*
_output_shapes
: *(
_class
loc:@penalty/penalty_param*
use_locking(*
validate_shape(*
T0
°
save_4/Assign_1Assignbeta1_power_1save_4/RestoreV2:1*(
_class
loc:@penalty/penalty_param*
_output_shapes
: *
use_locking(*
T0*
validate_shape(
¨
save_4/Assign_2Assignbeta1_power_2save_4/RestoreV2:2* 
_class
loc:@vc/dense/bias*
validate_shape(*
_output_shapes
: *
use_locking(*
T0
Ž
save_4/Assign_3Assignbeta2_powersave_4/RestoreV2:3*
_output_shapes
: *
T0*(
_class
loc:@penalty/penalty_param*
validate_shape(*
use_locking(
°
save_4/Assign_4Assignbeta2_power_1save_4/RestoreV2:4*
_output_shapes
: *(
_class
loc:@penalty/penalty_param*
validate_shape(*
T0*
use_locking(
¨
save_4/Assign_5Assignbeta2_power_2save_4/RestoreV2:5*
T0* 
_class
loc:@vc/dense/bias*
use_locking(*
validate_shape(*
_output_shapes
: 
¸
save_4/Assign_6Assignpenalty/penalty_paramsave_4/RestoreV2:6*(
_class
loc:@penalty/penalty_param*
T0*
validate_shape(*
_output_shapes
: *
use_locking(
˝
save_4/Assign_7Assignpenalty/penalty_param/Adamsave_4/RestoreV2:7*
T0*
_output_shapes
: *
validate_shape(*
use_locking(*(
_class
loc:@penalty/penalty_param
ż
save_4/Assign_8Assignpenalty/penalty_param/Adam_1save_4/RestoreV2:8*
_output_shapes
: *
use_locking(*
T0*
validate_shape(*(
_class
loc:@penalty/penalty_param
ż
save_4/Assign_9Assignpenalty/penalty_param/Adam_2save_4/RestoreV2:9*
validate_shape(*
_output_shapes
: *(
_class
loc:@penalty/penalty_param*
T0*
use_locking(
Á
save_4/Assign_10Assignpenalty/penalty_param/Adam_3save_4/RestoreV2:10*(
_class
loc:@penalty/penalty_param*
validate_shape(*
_output_shapes
: *
use_locking(*
T0
Ż
save_4/Assign_11Assignpi/dense/biassave_4/RestoreV2:11*
_output_shapes	
:*
T0*
validate_shape(*
use_locking(* 
_class
loc:@pi/dense/bias
´
save_4/Assign_12Assignpi/dense/bias/Adamsave_4/RestoreV2:12* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes	
:*
T0*
use_locking(
ś
save_4/Assign_13Assignpi/dense/bias/Adam_1save_4/RestoreV2:13*
T0*
validate_shape(*
_output_shapes	
:* 
_class
loc:@pi/dense/bias*
use_locking(
ˇ
save_4/Assign_14Assignpi/dense/kernelsave_4/RestoreV2:14*
validate_shape(*
T0*"
_class
loc:@pi/dense/kernel*
_output_shapes
:	L*
use_locking(
ź
save_4/Assign_15Assignpi/dense/kernel/Adamsave_4/RestoreV2:15*"
_class
loc:@pi/dense/kernel*
validate_shape(*
T0*
_output_shapes
:	L*
use_locking(
ž
save_4/Assign_16Assignpi/dense/kernel/Adam_1save_4/RestoreV2:16*
validate_shape(*
_output_shapes
:	L*
T0*
use_locking(*"
_class
loc:@pi/dense/kernel
ł
save_4/Assign_17Assignpi/dense_1/biassave_4/RestoreV2:17*
_output_shapes	
:*
validate_shape(*
T0*
use_locking(*"
_class
loc:@pi/dense_1/bias
¸
save_4/Assign_18Assignpi/dense_1/bias/Adamsave_4/RestoreV2:18*
validate_shape(*"
_class
loc:@pi/dense_1/bias*
_output_shapes	
:*
use_locking(*
T0
ş
save_4/Assign_19Assignpi/dense_1/bias/Adam_1save_4/RestoreV2:19*
use_locking(*
_output_shapes	
:*"
_class
loc:@pi/dense_1/bias*
T0*
validate_shape(
ź
save_4/Assign_20Assignpi/dense_1/kernelsave_4/RestoreV2:20*
use_locking(* 
_output_shapes
:
*$
_class
loc:@pi/dense_1/kernel*
validate_shape(*
T0
Á
save_4/Assign_21Assignpi/dense_1/kernel/Adamsave_4/RestoreV2:21*
T0*
validate_shape(*$
_class
loc:@pi/dense_1/kernel* 
_output_shapes
:
*
use_locking(
Ă
save_4/Assign_22Assignpi/dense_1/kernel/Adam_1save_4/RestoreV2:22* 
_output_shapes
:
*
validate_shape(*
use_locking(*
T0*$
_class
loc:@pi/dense_1/kernel
˛
save_4/Assign_23Assignpi/dense_2/biassave_4/RestoreV2:23*
use_locking(*
validate_shape(*
_output_shapes
:*"
_class
loc:@pi/dense_2/bias*
T0
ˇ
save_4/Assign_24Assignpi/dense_2/bias/Adamsave_4/RestoreV2:24*
T0*
use_locking(*"
_class
loc:@pi/dense_2/bias*
_output_shapes
:*
validate_shape(
š
save_4/Assign_25Assignpi/dense_2/bias/Adam_1save_4/RestoreV2:25*
use_locking(*
_output_shapes
:*
validate_shape(*"
_class
loc:@pi/dense_2/bias*
T0
ť
save_4/Assign_26Assignpi/dense_2/kernelsave_4/RestoreV2:26*
T0*$
_class
loc:@pi/dense_2/kernel*
_output_shapes
:	*
validate_shape(*
use_locking(
Ŕ
save_4/Assign_27Assignpi/dense_2/kernel/Adamsave_4/RestoreV2:27*
_output_shapes
:	*
T0*
validate_shape(*
use_locking(*$
_class
loc:@pi/dense_2/kernel
Â
save_4/Assign_28Assignpi/dense_2/kernel/Adam_1save_4/RestoreV2:28*
_output_shapes
:	*
T0*$
_class
loc:@pi/dense_2/kernel*
use_locking(*
validate_shape(
¨
save_4/Assign_29Assign
pi/log_stdsave_4/RestoreV2:29*
_output_shapes
:*
T0*
validate_shape(*
use_locking(*
_class
loc:@pi/log_std
­
save_4/Assign_30Assignpi/log_std/Adamsave_4/RestoreV2:30*
use_locking(*
_class
loc:@pi/log_std*
_output_shapes
:*
T0*
validate_shape(
Ż
save_4/Assign_31Assignpi/log_std/Adam_1save_4/RestoreV2:31*
validate_shape(*
use_locking(*
T0*
_class
loc:@pi/log_std*
_output_shapes
:
Ż
save_4/Assign_32Assignvc/dense/biassave_4/RestoreV2:32*
T0*
validate_shape(*
_output_shapes	
:* 
_class
loc:@vc/dense/bias*
use_locking(
´
save_4/Assign_33Assignvc/dense/bias/Adamsave_4/RestoreV2:33* 
_class
loc:@vc/dense/bias*
T0*
validate_shape(*
_output_shapes	
:*
use_locking(
ś
save_4/Assign_34Assignvc/dense/bias/Adam_1save_4/RestoreV2:34* 
_class
loc:@vc/dense/bias*
use_locking(*
validate_shape(*
_output_shapes	
:*
T0
ˇ
save_4/Assign_35Assignvc/dense/kernelsave_4/RestoreV2:35*"
_class
loc:@vc/dense/kernel*
_output_shapes
:	L*
use_locking(*
validate_shape(*
T0
ź
save_4/Assign_36Assignvc/dense/kernel/Adamsave_4/RestoreV2:36*
validate_shape(*
_output_shapes
:	L*
use_locking(*
T0*"
_class
loc:@vc/dense/kernel
ž
save_4/Assign_37Assignvc/dense/kernel/Adam_1save_4/RestoreV2:37*
T0*
validate_shape(*
use_locking(*"
_class
loc:@vc/dense/kernel*
_output_shapes
:	L
ł
save_4/Assign_38Assignvc/dense_1/biassave_4/RestoreV2:38*
T0*"
_class
loc:@vc/dense_1/bias*
_output_shapes	
:*
use_locking(*
validate_shape(
¸
save_4/Assign_39Assignvc/dense_1/bias/Adamsave_4/RestoreV2:39*
_output_shapes	
:*
T0*
validate_shape(*"
_class
loc:@vc/dense_1/bias*
use_locking(
ş
save_4/Assign_40Assignvc/dense_1/bias/Adam_1save_4/RestoreV2:40*
use_locking(*
T0*
_output_shapes	
:*"
_class
loc:@vc/dense_1/bias*
validate_shape(
ź
save_4/Assign_41Assignvc/dense_1/kernelsave_4/RestoreV2:41* 
_output_shapes
:
*$
_class
loc:@vc/dense_1/kernel*
validate_shape(*
use_locking(*
T0
Á
save_4/Assign_42Assignvc/dense_1/kernel/Adamsave_4/RestoreV2:42*
validate_shape(*
use_locking(*$
_class
loc:@vc/dense_1/kernel* 
_output_shapes
:
*
T0
Ă
save_4/Assign_43Assignvc/dense_1/kernel/Adam_1save_4/RestoreV2:43* 
_output_shapes
:
*$
_class
loc:@vc/dense_1/kernel*
validate_shape(*
T0*
use_locking(
˛
save_4/Assign_44Assignvc/dense_2/biassave_4/RestoreV2:44*"
_class
loc:@vc/dense_2/bias*
use_locking(*
_output_shapes
:*
T0*
validate_shape(
ˇ
save_4/Assign_45Assignvc/dense_2/bias/Adamsave_4/RestoreV2:45*
use_locking(*"
_class
loc:@vc/dense_2/bias*
_output_shapes
:*
T0*
validate_shape(
š
save_4/Assign_46Assignvc/dense_2/bias/Adam_1save_4/RestoreV2:46*
_output_shapes
:*
use_locking(*
T0*
validate_shape(*"
_class
loc:@vc/dense_2/bias
ť
save_4/Assign_47Assignvc/dense_2/kernelsave_4/RestoreV2:47*
T0*$
_class
loc:@vc/dense_2/kernel*
_output_shapes
:	*
use_locking(*
validate_shape(
Ŕ
save_4/Assign_48Assignvc/dense_2/kernel/Adamsave_4/RestoreV2:48*
T0*
validate_shape(*
use_locking(*
_output_shapes
:	*$
_class
loc:@vc/dense_2/kernel
Â
save_4/Assign_49Assignvc/dense_2/kernel/Adam_1save_4/RestoreV2:49*
_output_shapes
:	*
validate_shape(*$
_class
loc:@vc/dense_2/kernel*
use_locking(*
T0
Ż
save_4/Assign_50Assignvf/dense/biassave_4/RestoreV2:50*
use_locking(*
validate_shape(*
_output_shapes	
:*
T0* 
_class
loc:@vf/dense/bias
´
save_4/Assign_51Assignvf/dense/bias/Adamsave_4/RestoreV2:51*
validate_shape(*
_output_shapes	
:*
use_locking(* 
_class
loc:@vf/dense/bias*
T0
ś
save_4/Assign_52Assignvf/dense/bias/Adam_1save_4/RestoreV2:52*
T0*
use_locking(* 
_class
loc:@vf/dense/bias*
validate_shape(*
_output_shapes	
:
ˇ
save_4/Assign_53Assignvf/dense/kernelsave_4/RestoreV2:53*"
_class
loc:@vf/dense/kernel*
use_locking(*
validate_shape(*
_output_shapes
:	L*
T0
ź
save_4/Assign_54Assignvf/dense/kernel/Adamsave_4/RestoreV2:54*
use_locking(*
validate_shape(*
T0*
_output_shapes
:	L*"
_class
loc:@vf/dense/kernel
ž
save_4/Assign_55Assignvf/dense/kernel/Adam_1save_4/RestoreV2:55*
T0*
use_locking(*
validate_shape(*
_output_shapes
:	L*"
_class
loc:@vf/dense/kernel
ł
save_4/Assign_56Assignvf/dense_1/biassave_4/RestoreV2:56*
T0*
_output_shapes	
:*
use_locking(*"
_class
loc:@vf/dense_1/bias*
validate_shape(
¸
save_4/Assign_57Assignvf/dense_1/bias/Adamsave_4/RestoreV2:57*
use_locking(*
validate_shape(*
T0*
_output_shapes	
:*"
_class
loc:@vf/dense_1/bias
ş
save_4/Assign_58Assignvf/dense_1/bias/Adam_1save_4/RestoreV2:58*
_output_shapes	
:*
use_locking(*
validate_shape(*"
_class
loc:@vf/dense_1/bias*
T0
ź
save_4/Assign_59Assignvf/dense_1/kernelsave_4/RestoreV2:59*
use_locking(*
validate_shape(* 
_output_shapes
:
*$
_class
loc:@vf/dense_1/kernel*
T0
Á
save_4/Assign_60Assignvf/dense_1/kernel/Adamsave_4/RestoreV2:60*
use_locking(*$
_class
loc:@vf/dense_1/kernel* 
_output_shapes
:
*
validate_shape(*
T0
Ă
save_4/Assign_61Assignvf/dense_1/kernel/Adam_1save_4/RestoreV2:61* 
_output_shapes
:
*
validate_shape(*
T0*$
_class
loc:@vf/dense_1/kernel*
use_locking(
˛
save_4/Assign_62Assignvf/dense_2/biassave_4/RestoreV2:62*
use_locking(*
T0*
_output_shapes
:*
validate_shape(*"
_class
loc:@vf/dense_2/bias
ˇ
save_4/Assign_63Assignvf/dense_2/bias/Adamsave_4/RestoreV2:63*"
_class
loc:@vf/dense_2/bias*
_output_shapes
:*
use_locking(*
validate_shape(*
T0
š
save_4/Assign_64Assignvf/dense_2/bias/Adam_1save_4/RestoreV2:64*
_output_shapes
:*"
_class
loc:@vf/dense_2/bias*
validate_shape(*
T0*
use_locking(
ť
save_4/Assign_65Assignvf/dense_2/kernelsave_4/RestoreV2:65*
_output_shapes
:	*
validate_shape(*
use_locking(*
T0*$
_class
loc:@vf/dense_2/kernel
Ŕ
save_4/Assign_66Assignvf/dense_2/kernel/Adamsave_4/RestoreV2:66*
_output_shapes
:	*
use_locking(*
T0*
validate_shape(*$
_class
loc:@vf/dense_2/kernel
Â
save_4/Assign_67Assignvf/dense_2/kernel/Adam_1save_4/RestoreV2:67*$
_class
loc:@vf/dense_2/kernel*
validate_shape(*
T0*
use_locking(*
_output_shapes
:	


save_4/restore_shardNoOp^save_4/Assign^save_4/Assign_1^save_4/Assign_10^save_4/Assign_11^save_4/Assign_12^save_4/Assign_13^save_4/Assign_14^save_4/Assign_15^save_4/Assign_16^save_4/Assign_17^save_4/Assign_18^save_4/Assign_19^save_4/Assign_2^save_4/Assign_20^save_4/Assign_21^save_4/Assign_22^save_4/Assign_23^save_4/Assign_24^save_4/Assign_25^save_4/Assign_26^save_4/Assign_27^save_4/Assign_28^save_4/Assign_29^save_4/Assign_3^save_4/Assign_30^save_4/Assign_31^save_4/Assign_32^save_4/Assign_33^save_4/Assign_34^save_4/Assign_35^save_4/Assign_36^save_4/Assign_37^save_4/Assign_38^save_4/Assign_39^save_4/Assign_4^save_4/Assign_40^save_4/Assign_41^save_4/Assign_42^save_4/Assign_43^save_4/Assign_44^save_4/Assign_45^save_4/Assign_46^save_4/Assign_47^save_4/Assign_48^save_4/Assign_49^save_4/Assign_5^save_4/Assign_50^save_4/Assign_51^save_4/Assign_52^save_4/Assign_53^save_4/Assign_54^save_4/Assign_55^save_4/Assign_56^save_4/Assign_57^save_4/Assign_58^save_4/Assign_59^save_4/Assign_6^save_4/Assign_60^save_4/Assign_61^save_4/Assign_62^save_4/Assign_63^save_4/Assign_64^save_4/Assign_65^save_4/Assign_66^save_4/Assign_67^save_4/Assign_7^save_4/Assign_8^save_4/Assign_9
1
save_4/restore_allNoOp^save_4/restore_shard
[
save_5/filename/inputConst*
dtype0*
valueB Bmodel*
_output_shapes
: 
r
save_5/filenamePlaceholderWithDefaultsave_5/filename/input*
_output_shapes
: *
shape: *
dtype0
i
save_5/ConstPlaceholderWithDefaultsave_5/filename*
_output_shapes
: *
shape: *
dtype0

save_5/StringJoin/inputs_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+_temp_b22d832e446643fba825bf072a1a91c0/part
{
save_5/StringJoin
StringJoinsave_5/Constsave_5/StringJoin/inputs_1*
_output_shapes
: *
N*
	separator 
S
save_5/num_shardsConst*
_output_shapes
: *
value	B :*
dtype0
^
save_5/ShardedFilename/shardConst*
_output_shapes
: *
value	B : *
dtype0

save_5/ShardedFilenameShardedFilenamesave_5/StringJoinsave_5/ShardedFilename/shardsave_5/num_shards*
_output_shapes
: 
÷
save_5/SaveV2/tensor_namesConst*
dtype0*¨
valueBDBbeta1_powerBbeta1_power_1Bbeta1_power_2Bbeta2_powerBbeta2_power_1Bbeta2_power_2Bpenalty/penalty_paramBpenalty/penalty_param/AdamBpenalty/penalty_param/Adam_1Bpenalty/penalty_param/Adam_2Bpenalty/penalty_param/Adam_3Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1B
pi/log_stdBpi/log_std/AdamBpi/log_std/Adam_1Bvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1*
_output_shapes
:D
đ
save_5/SaveV2/shape_and_slicesConst*
dtype0*
valueBDB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
_output_shapes
:D
Ň
save_5/SaveV2SaveV2save_5/ShardedFilenamesave_5/SaveV2/tensor_namessave_5/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta1_power_2beta2_powerbeta2_power_1beta2_power_2penalty/penalty_parampenalty/penalty_param/Adampenalty/penalty_param/Adam_1penalty/penalty_param/Adam_2penalty/penalty_param/Adam_3pi/dense/biaspi/dense/bias/Adampi/dense/bias/Adam_1pi/dense/kernelpi/dense/kernel/Adampi/dense/kernel/Adam_1pi/dense_1/biaspi/dense_1/bias/Adampi/dense_1/bias/Adam_1pi/dense_1/kernelpi/dense_1/kernel/Adampi/dense_1/kernel/Adam_1pi/dense_2/biaspi/dense_2/bias/Adampi/dense_2/bias/Adam_1pi/dense_2/kernelpi/dense_2/kernel/Adampi/dense_2/kernel/Adam_1
pi/log_stdpi/log_std/Adampi/log_std/Adam_1vc/dense/biasvc/dense/bias/Adamvc/dense/bias/Adam_1vc/dense/kernelvc/dense/kernel/Adamvc/dense/kernel/Adam_1vc/dense_1/biasvc/dense_1/bias/Adamvc/dense_1/bias/Adam_1vc/dense_1/kernelvc/dense_1/kernel/Adamvc/dense_1/kernel/Adam_1vc/dense_2/biasvc/dense_2/bias/Adamvc/dense_2/bias/Adam_1vc/dense_2/kernelvc/dense_2/kernel/Adamvc/dense_2/kernel/Adam_1vf/dense/biasvf/dense/bias/Adamvf/dense/bias/Adam_1vf/dense/kernelvf/dense/kernel/Adamvf/dense/kernel/Adam_1vf/dense_1/biasvf/dense_1/bias/Adamvf/dense_1/bias/Adam_1vf/dense_1/kernelvf/dense_1/kernel/Adamvf/dense_1/kernel/Adam_1vf/dense_2/biasvf/dense_2/bias/Adamvf/dense_2/bias/Adam_1vf/dense_2/kernelvf/dense_2/kernel/Adamvf/dense_2/kernel/Adam_1*R
dtypesH
F2D

save_5/control_dependencyIdentitysave_5/ShardedFilename^save_5/SaveV2*)
_class
loc:@save_5/ShardedFilename*
T0*
_output_shapes
: 
Ł
-save_5/MergeV2Checkpoints/checkpoint_prefixesPacksave_5/ShardedFilename^save_5/control_dependency*
_output_shapes
:*
N*
T0*

axis 

save_5/MergeV2CheckpointsMergeV2Checkpoints-save_5/MergeV2Checkpoints/checkpoint_prefixessave_5/Const*
delete_old_dirs(

save_5/IdentityIdentitysave_5/Const^save_5/MergeV2Checkpoints^save_5/control_dependency*
T0*
_output_shapes
: 
ú
save_5/RestoreV2/tensor_namesConst*¨
valueBDBbeta1_powerBbeta1_power_1Bbeta1_power_2Bbeta2_powerBbeta2_power_1Bbeta2_power_2Bpenalty/penalty_paramBpenalty/penalty_param/AdamBpenalty/penalty_param/Adam_1Bpenalty/penalty_param/Adam_2Bpenalty/penalty_param/Adam_3Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1B
pi/log_stdBpi/log_std/AdamBpi/log_std/Adam_1Bvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1*
_output_shapes
:D*
dtype0
ó
!save_5/RestoreV2/shape_and_slicesConst*
valueBDB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
_output_shapes
:D*
dtype0
ę
save_5/RestoreV2	RestoreV2save_5/Constsave_5/RestoreV2/tensor_names!save_5/RestoreV2/shape_and_slices*R
dtypesH
F2D*Ś
_output_shapes
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
Ş
save_5/AssignAssignbeta1_powersave_5/RestoreV2*(
_class
loc:@penalty/penalty_param*
validate_shape(*
use_locking(*
T0*
_output_shapes
: 
°
save_5/Assign_1Assignbeta1_power_1save_5/RestoreV2:1*
use_locking(*
T0*
_output_shapes
: *(
_class
loc:@penalty/penalty_param*
validate_shape(
¨
save_5/Assign_2Assignbeta1_power_2save_5/RestoreV2:2*
validate_shape(*
_output_shapes
: *
T0*
use_locking(* 
_class
loc:@vc/dense/bias
Ž
save_5/Assign_3Assignbeta2_powersave_5/RestoreV2:3*
T0*(
_class
loc:@penalty/penalty_param*
validate_shape(*
_output_shapes
: *
use_locking(
°
save_5/Assign_4Assignbeta2_power_1save_5/RestoreV2:4*
use_locking(*
T0*(
_class
loc:@penalty/penalty_param*
_output_shapes
: *
validate_shape(
¨
save_5/Assign_5Assignbeta2_power_2save_5/RestoreV2:5*
use_locking(*
T0* 
_class
loc:@vc/dense/bias*
validate_shape(*
_output_shapes
: 
¸
save_5/Assign_6Assignpenalty/penalty_paramsave_5/RestoreV2:6*
T0*
validate_shape(*
_output_shapes
: *
use_locking(*(
_class
loc:@penalty/penalty_param
˝
save_5/Assign_7Assignpenalty/penalty_param/Adamsave_5/RestoreV2:7*
_output_shapes
: *
validate_shape(*
use_locking(*
T0*(
_class
loc:@penalty/penalty_param
ż
save_5/Assign_8Assignpenalty/penalty_param/Adam_1save_5/RestoreV2:8*
validate_shape(*
_output_shapes
: *
T0*(
_class
loc:@penalty/penalty_param*
use_locking(
ż
save_5/Assign_9Assignpenalty/penalty_param/Adam_2save_5/RestoreV2:9*
T0*
_output_shapes
: *
validate_shape(*
use_locking(*(
_class
loc:@penalty/penalty_param
Á
save_5/Assign_10Assignpenalty/penalty_param/Adam_3save_5/RestoreV2:10*
_output_shapes
: *(
_class
loc:@penalty/penalty_param*
validate_shape(*
T0*
use_locking(
Ż
save_5/Assign_11Assignpi/dense/biassave_5/RestoreV2:11*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0* 
_class
loc:@pi/dense/bias
´
save_5/Assign_12Assignpi/dense/bias/Adamsave_5/RestoreV2:12*
validate_shape(*
_output_shapes	
:*
T0*
use_locking(* 
_class
loc:@pi/dense/bias
ś
save_5/Assign_13Assignpi/dense/bias/Adam_1save_5/RestoreV2:13*
validate_shape(*
use_locking(*
_output_shapes	
:*
T0* 
_class
loc:@pi/dense/bias
ˇ
save_5/Assign_14Assignpi/dense/kernelsave_5/RestoreV2:14*"
_class
loc:@pi/dense/kernel*
validate_shape(*
T0*
use_locking(*
_output_shapes
:	L
ź
save_5/Assign_15Assignpi/dense/kernel/Adamsave_5/RestoreV2:15*"
_class
loc:@pi/dense/kernel*
validate_shape(*
_output_shapes
:	L*
use_locking(*
T0
ž
save_5/Assign_16Assignpi/dense/kernel/Adam_1save_5/RestoreV2:16*
_output_shapes
:	L*
use_locking(*
validate_shape(*"
_class
loc:@pi/dense/kernel*
T0
ł
save_5/Assign_17Assignpi/dense_1/biassave_5/RestoreV2:17*
validate_shape(*"
_class
loc:@pi/dense_1/bias*
use_locking(*
T0*
_output_shapes	
:
¸
save_5/Assign_18Assignpi/dense_1/bias/Adamsave_5/RestoreV2:18*
use_locking(*
validate_shape(*
T0*
_output_shapes	
:*"
_class
loc:@pi/dense_1/bias
ş
save_5/Assign_19Assignpi/dense_1/bias/Adam_1save_5/RestoreV2:19*
validate_shape(*
T0*
_output_shapes	
:*
use_locking(*"
_class
loc:@pi/dense_1/bias
ź
save_5/Assign_20Assignpi/dense_1/kernelsave_5/RestoreV2:20*
validate_shape(*$
_class
loc:@pi/dense_1/kernel*
T0* 
_output_shapes
:
*
use_locking(
Á
save_5/Assign_21Assignpi/dense_1/kernel/Adamsave_5/RestoreV2:21*$
_class
loc:@pi/dense_1/kernel*
T0* 
_output_shapes
:
*
use_locking(*
validate_shape(
Ă
save_5/Assign_22Assignpi/dense_1/kernel/Adam_1save_5/RestoreV2:22*
validate_shape(* 
_output_shapes
:
*$
_class
loc:@pi/dense_1/kernel*
T0*
use_locking(
˛
save_5/Assign_23Assignpi/dense_2/biassave_5/RestoreV2:23*
T0*
_output_shapes
:*
use_locking(*
validate_shape(*"
_class
loc:@pi/dense_2/bias
ˇ
save_5/Assign_24Assignpi/dense_2/bias/Adamsave_5/RestoreV2:24*"
_class
loc:@pi/dense_2/bias*
T0*
use_locking(*
validate_shape(*
_output_shapes
:
š
save_5/Assign_25Assignpi/dense_2/bias/Adam_1save_5/RestoreV2:25*
validate_shape(*
use_locking(*
_output_shapes
:*"
_class
loc:@pi/dense_2/bias*
T0
ť
save_5/Assign_26Assignpi/dense_2/kernelsave_5/RestoreV2:26*
_output_shapes
:	*$
_class
loc:@pi/dense_2/kernel*
T0*
validate_shape(*
use_locking(
Ŕ
save_5/Assign_27Assignpi/dense_2/kernel/Adamsave_5/RestoreV2:27*
_output_shapes
:	*
T0*
use_locking(*
validate_shape(*$
_class
loc:@pi/dense_2/kernel
Â
save_5/Assign_28Assignpi/dense_2/kernel/Adam_1save_5/RestoreV2:28*
T0*$
_class
loc:@pi/dense_2/kernel*
_output_shapes
:	*
validate_shape(*
use_locking(
¨
save_5/Assign_29Assign
pi/log_stdsave_5/RestoreV2:29*
_output_shapes
:*
_class
loc:@pi/log_std*
validate_shape(*
T0*
use_locking(
­
save_5/Assign_30Assignpi/log_std/Adamsave_5/RestoreV2:30*
use_locking(*
validate_shape(*
_output_shapes
:*
T0*
_class
loc:@pi/log_std
Ż
save_5/Assign_31Assignpi/log_std/Adam_1save_5/RestoreV2:31*
_output_shapes
:*
use_locking(*
_class
loc:@pi/log_std*
T0*
validate_shape(
Ż
save_5/Assign_32Assignvc/dense/biassave_5/RestoreV2:32* 
_class
loc:@vc/dense/bias*
validate_shape(*
use_locking(*
T0*
_output_shapes	
:
´
save_5/Assign_33Assignvc/dense/bias/Adamsave_5/RestoreV2:33*
_output_shapes	
:*
use_locking(*
T0*
validate_shape(* 
_class
loc:@vc/dense/bias
ś
save_5/Assign_34Assignvc/dense/bias/Adam_1save_5/RestoreV2:34*
validate_shape(*
_output_shapes	
:*
use_locking(* 
_class
loc:@vc/dense/bias*
T0
ˇ
save_5/Assign_35Assignvc/dense/kernelsave_5/RestoreV2:35*
use_locking(*
validate_shape(*
T0*"
_class
loc:@vc/dense/kernel*
_output_shapes
:	L
ź
save_5/Assign_36Assignvc/dense/kernel/Adamsave_5/RestoreV2:36*
use_locking(*
T0*
validate_shape(*
_output_shapes
:	L*"
_class
loc:@vc/dense/kernel
ž
save_5/Assign_37Assignvc/dense/kernel/Adam_1save_5/RestoreV2:37*
validate_shape(*
T0*
use_locking(*
_output_shapes
:	L*"
_class
loc:@vc/dense/kernel
ł
save_5/Assign_38Assignvc/dense_1/biassave_5/RestoreV2:38*
validate_shape(*
use_locking(*
T0*"
_class
loc:@vc/dense_1/bias*
_output_shapes	
:
¸
save_5/Assign_39Assignvc/dense_1/bias/Adamsave_5/RestoreV2:39*"
_class
loc:@vc/dense_1/bias*
T0*
use_locking(*
_output_shapes	
:*
validate_shape(
ş
save_5/Assign_40Assignvc/dense_1/bias/Adam_1save_5/RestoreV2:40*
use_locking(*
_output_shapes	
:*
validate_shape(*"
_class
loc:@vc/dense_1/bias*
T0
ź
save_5/Assign_41Assignvc/dense_1/kernelsave_5/RestoreV2:41*
use_locking(* 
_output_shapes
:
*$
_class
loc:@vc/dense_1/kernel*
T0*
validate_shape(
Á
save_5/Assign_42Assignvc/dense_1/kernel/Adamsave_5/RestoreV2:42* 
_output_shapes
:
*$
_class
loc:@vc/dense_1/kernel*
use_locking(*
validate_shape(*
T0
Ă
save_5/Assign_43Assignvc/dense_1/kernel/Adam_1save_5/RestoreV2:43*
T0*$
_class
loc:@vc/dense_1/kernel*
validate_shape(*
use_locking(* 
_output_shapes
:

˛
save_5/Assign_44Assignvc/dense_2/biassave_5/RestoreV2:44*
use_locking(*
T0*"
_class
loc:@vc/dense_2/bias*
validate_shape(*
_output_shapes
:
ˇ
save_5/Assign_45Assignvc/dense_2/bias/Adamsave_5/RestoreV2:45*
_output_shapes
:*
validate_shape(*
use_locking(*"
_class
loc:@vc/dense_2/bias*
T0
š
save_5/Assign_46Assignvc/dense_2/bias/Adam_1save_5/RestoreV2:46*
T0*
_output_shapes
:*
use_locking(*"
_class
loc:@vc/dense_2/bias*
validate_shape(
ť
save_5/Assign_47Assignvc/dense_2/kernelsave_5/RestoreV2:47*
validate_shape(*
_output_shapes
:	*
T0*$
_class
loc:@vc/dense_2/kernel*
use_locking(
Ŕ
save_5/Assign_48Assignvc/dense_2/kernel/Adamsave_5/RestoreV2:48*
use_locking(*$
_class
loc:@vc/dense_2/kernel*
_output_shapes
:	*
validate_shape(*
T0
Â
save_5/Assign_49Assignvc/dense_2/kernel/Adam_1save_5/RestoreV2:49*
_output_shapes
:	*$
_class
loc:@vc/dense_2/kernel*
validate_shape(*
T0*
use_locking(
Ż
save_5/Assign_50Assignvf/dense/biassave_5/RestoreV2:50*
_output_shapes	
:*
use_locking(* 
_class
loc:@vf/dense/bias*
T0*
validate_shape(
´
save_5/Assign_51Assignvf/dense/bias/Adamsave_5/RestoreV2:51*
T0*
validate_shape(*
use_locking(*
_output_shapes	
:* 
_class
loc:@vf/dense/bias
ś
save_5/Assign_52Assignvf/dense/bias/Adam_1save_5/RestoreV2:52*
T0* 
_class
loc:@vf/dense/bias*
_output_shapes	
:*
validate_shape(*
use_locking(
ˇ
save_5/Assign_53Assignvf/dense/kernelsave_5/RestoreV2:53*
T0*"
_class
loc:@vf/dense/kernel*
_output_shapes
:	L*
validate_shape(*
use_locking(
ź
save_5/Assign_54Assignvf/dense/kernel/Adamsave_5/RestoreV2:54*
_output_shapes
:	L*
validate_shape(*"
_class
loc:@vf/dense/kernel*
T0*
use_locking(
ž
save_5/Assign_55Assignvf/dense/kernel/Adam_1save_5/RestoreV2:55*
T0*
use_locking(*
validate_shape(*"
_class
loc:@vf/dense/kernel*
_output_shapes
:	L
ł
save_5/Assign_56Assignvf/dense_1/biassave_5/RestoreV2:56*
use_locking(*
T0*"
_class
loc:@vf/dense_1/bias*
_output_shapes	
:*
validate_shape(
¸
save_5/Assign_57Assignvf/dense_1/bias/Adamsave_5/RestoreV2:57*
validate_shape(*
use_locking(*"
_class
loc:@vf/dense_1/bias*
T0*
_output_shapes	
:
ş
save_5/Assign_58Assignvf/dense_1/bias/Adam_1save_5/RestoreV2:58*
T0*
validate_shape(*"
_class
loc:@vf/dense_1/bias*
use_locking(*
_output_shapes	
:
ź
save_5/Assign_59Assignvf/dense_1/kernelsave_5/RestoreV2:59*
validate_shape(*
T0*
use_locking(*$
_class
loc:@vf/dense_1/kernel* 
_output_shapes
:

Á
save_5/Assign_60Assignvf/dense_1/kernel/Adamsave_5/RestoreV2:60*
use_locking(*
T0*
validate_shape(* 
_output_shapes
:
*$
_class
loc:@vf/dense_1/kernel
Ă
save_5/Assign_61Assignvf/dense_1/kernel/Adam_1save_5/RestoreV2:61*
T0* 
_output_shapes
:
*
validate_shape(*
use_locking(*$
_class
loc:@vf/dense_1/kernel
˛
save_5/Assign_62Assignvf/dense_2/biassave_5/RestoreV2:62*
validate_shape(*
_output_shapes
:*
T0*"
_class
loc:@vf/dense_2/bias*
use_locking(
ˇ
save_5/Assign_63Assignvf/dense_2/bias/Adamsave_5/RestoreV2:63*"
_class
loc:@vf/dense_2/bias*
use_locking(*
_output_shapes
:*
validate_shape(*
T0
š
save_5/Assign_64Assignvf/dense_2/bias/Adam_1save_5/RestoreV2:64*"
_class
loc:@vf/dense_2/bias*
validate_shape(*
T0*
use_locking(*
_output_shapes
:
ť
save_5/Assign_65Assignvf/dense_2/kernelsave_5/RestoreV2:65*
validate_shape(*
use_locking(*$
_class
loc:@vf/dense_2/kernel*
_output_shapes
:	*
T0
Ŕ
save_5/Assign_66Assignvf/dense_2/kernel/Adamsave_5/RestoreV2:66*
T0*$
_class
loc:@vf/dense_2/kernel*
use_locking(*
_output_shapes
:	*
validate_shape(
Â
save_5/Assign_67Assignvf/dense_2/kernel/Adam_1save_5/RestoreV2:67*
use_locking(*
_output_shapes
:	*$
_class
loc:@vf/dense_2/kernel*
validate_shape(*
T0


save_5/restore_shardNoOp^save_5/Assign^save_5/Assign_1^save_5/Assign_10^save_5/Assign_11^save_5/Assign_12^save_5/Assign_13^save_5/Assign_14^save_5/Assign_15^save_5/Assign_16^save_5/Assign_17^save_5/Assign_18^save_5/Assign_19^save_5/Assign_2^save_5/Assign_20^save_5/Assign_21^save_5/Assign_22^save_5/Assign_23^save_5/Assign_24^save_5/Assign_25^save_5/Assign_26^save_5/Assign_27^save_5/Assign_28^save_5/Assign_29^save_5/Assign_3^save_5/Assign_30^save_5/Assign_31^save_5/Assign_32^save_5/Assign_33^save_5/Assign_34^save_5/Assign_35^save_5/Assign_36^save_5/Assign_37^save_5/Assign_38^save_5/Assign_39^save_5/Assign_4^save_5/Assign_40^save_5/Assign_41^save_5/Assign_42^save_5/Assign_43^save_5/Assign_44^save_5/Assign_45^save_5/Assign_46^save_5/Assign_47^save_5/Assign_48^save_5/Assign_49^save_5/Assign_5^save_5/Assign_50^save_5/Assign_51^save_5/Assign_52^save_5/Assign_53^save_5/Assign_54^save_5/Assign_55^save_5/Assign_56^save_5/Assign_57^save_5/Assign_58^save_5/Assign_59^save_5/Assign_6^save_5/Assign_60^save_5/Assign_61^save_5/Assign_62^save_5/Assign_63^save_5/Assign_64^save_5/Assign_65^save_5/Assign_66^save_5/Assign_67^save_5/Assign_7^save_5/Assign_8^save_5/Assign_9
1
save_5/restore_allNoOp^save_5/restore_shard
[
save_6/filename/inputConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
r
save_6/filenamePlaceholderWithDefaultsave_6/filename/input*
dtype0*
shape: *
_output_shapes
: 
i
save_6/ConstPlaceholderWithDefaultsave_6/filename*
_output_shapes
: *
dtype0*
shape: 

save_6/StringJoin/inputs_1Const*
dtype0*<
value3B1 B+_temp_d0a791dda74a4104952e3314cdf94693/part*
_output_shapes
: 
{
save_6/StringJoin
StringJoinsave_6/Constsave_6/StringJoin/inputs_1*
N*
	separator *
_output_shapes
: 
S
save_6/num_shardsConst*
dtype0*
_output_shapes
: *
value	B :
^
save_6/ShardedFilename/shardConst*
value	B : *
_output_shapes
: *
dtype0

save_6/ShardedFilenameShardedFilenamesave_6/StringJoinsave_6/ShardedFilename/shardsave_6/num_shards*
_output_shapes
: 
÷
save_6/SaveV2/tensor_namesConst*
dtype0*¨
valueBDBbeta1_powerBbeta1_power_1Bbeta1_power_2Bbeta2_powerBbeta2_power_1Bbeta2_power_2Bpenalty/penalty_paramBpenalty/penalty_param/AdamBpenalty/penalty_param/Adam_1Bpenalty/penalty_param/Adam_2Bpenalty/penalty_param/Adam_3Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1B
pi/log_stdBpi/log_std/AdamBpi/log_std/Adam_1Bvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1*
_output_shapes
:D
đ
save_6/SaveV2/shape_and_slicesConst*
valueBDB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
_output_shapes
:D*
dtype0
Ň
save_6/SaveV2SaveV2save_6/ShardedFilenamesave_6/SaveV2/tensor_namessave_6/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta1_power_2beta2_powerbeta2_power_1beta2_power_2penalty/penalty_parampenalty/penalty_param/Adampenalty/penalty_param/Adam_1penalty/penalty_param/Adam_2penalty/penalty_param/Adam_3pi/dense/biaspi/dense/bias/Adampi/dense/bias/Adam_1pi/dense/kernelpi/dense/kernel/Adampi/dense/kernel/Adam_1pi/dense_1/biaspi/dense_1/bias/Adampi/dense_1/bias/Adam_1pi/dense_1/kernelpi/dense_1/kernel/Adampi/dense_1/kernel/Adam_1pi/dense_2/biaspi/dense_2/bias/Adampi/dense_2/bias/Adam_1pi/dense_2/kernelpi/dense_2/kernel/Adampi/dense_2/kernel/Adam_1
pi/log_stdpi/log_std/Adampi/log_std/Adam_1vc/dense/biasvc/dense/bias/Adamvc/dense/bias/Adam_1vc/dense/kernelvc/dense/kernel/Adamvc/dense/kernel/Adam_1vc/dense_1/biasvc/dense_1/bias/Adamvc/dense_1/bias/Adam_1vc/dense_1/kernelvc/dense_1/kernel/Adamvc/dense_1/kernel/Adam_1vc/dense_2/biasvc/dense_2/bias/Adamvc/dense_2/bias/Adam_1vc/dense_2/kernelvc/dense_2/kernel/Adamvc/dense_2/kernel/Adam_1vf/dense/biasvf/dense/bias/Adamvf/dense/bias/Adam_1vf/dense/kernelvf/dense/kernel/Adamvf/dense/kernel/Adam_1vf/dense_1/biasvf/dense_1/bias/Adamvf/dense_1/bias/Adam_1vf/dense_1/kernelvf/dense_1/kernel/Adamvf/dense_1/kernel/Adam_1vf/dense_2/biasvf/dense_2/bias/Adamvf/dense_2/bias/Adam_1vf/dense_2/kernelvf/dense_2/kernel/Adamvf/dense_2/kernel/Adam_1*R
dtypesH
F2D

save_6/control_dependencyIdentitysave_6/ShardedFilename^save_6/SaveV2*)
_class
loc:@save_6/ShardedFilename*
T0*
_output_shapes
: 
Ł
-save_6/MergeV2Checkpoints/checkpoint_prefixesPacksave_6/ShardedFilename^save_6/control_dependency*
T0*

axis *
_output_shapes
:*
N

save_6/MergeV2CheckpointsMergeV2Checkpoints-save_6/MergeV2Checkpoints/checkpoint_prefixessave_6/Const*
delete_old_dirs(

save_6/IdentityIdentitysave_6/Const^save_6/MergeV2Checkpoints^save_6/control_dependency*
T0*
_output_shapes
: 
ú
save_6/RestoreV2/tensor_namesConst*¨
valueBDBbeta1_powerBbeta1_power_1Bbeta1_power_2Bbeta2_powerBbeta2_power_1Bbeta2_power_2Bpenalty/penalty_paramBpenalty/penalty_param/AdamBpenalty/penalty_param/Adam_1Bpenalty/penalty_param/Adam_2Bpenalty/penalty_param/Adam_3Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1B
pi/log_stdBpi/log_std/AdamBpi/log_std/Adam_1Bvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1*
dtype0*
_output_shapes
:D
ó
!save_6/RestoreV2/shape_and_slicesConst*
_output_shapes
:D*
valueBDB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0
ę
save_6/RestoreV2	RestoreV2save_6/Constsave_6/RestoreV2/tensor_names!save_6/RestoreV2/shape_and_slices*R
dtypesH
F2D*Ś
_output_shapes
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
Ş
save_6/AssignAssignbeta1_powersave_6/RestoreV2*
_output_shapes
: *
validate_shape(*(
_class
loc:@penalty/penalty_param*
T0*
use_locking(
°
save_6/Assign_1Assignbeta1_power_1save_6/RestoreV2:1*
use_locking(*
validate_shape(*(
_class
loc:@penalty/penalty_param*
_output_shapes
: *
T0
¨
save_6/Assign_2Assignbeta1_power_2save_6/RestoreV2:2*
T0*
validate_shape(*
use_locking(*
_output_shapes
: * 
_class
loc:@vc/dense/bias
Ž
save_6/Assign_3Assignbeta2_powersave_6/RestoreV2:3*
use_locking(*
validate_shape(*
T0*
_output_shapes
: *(
_class
loc:@penalty/penalty_param
°
save_6/Assign_4Assignbeta2_power_1save_6/RestoreV2:4*(
_class
loc:@penalty/penalty_param*
_output_shapes
: *
validate_shape(*
T0*
use_locking(
¨
save_6/Assign_5Assignbeta2_power_2save_6/RestoreV2:5*
_output_shapes
: * 
_class
loc:@vc/dense/bias*
use_locking(*
validate_shape(*
T0
¸
save_6/Assign_6Assignpenalty/penalty_paramsave_6/RestoreV2:6*
use_locking(*
T0*(
_class
loc:@penalty/penalty_param*
validate_shape(*
_output_shapes
: 
˝
save_6/Assign_7Assignpenalty/penalty_param/Adamsave_6/RestoreV2:7*
use_locking(*
validate_shape(*
T0*
_output_shapes
: *(
_class
loc:@penalty/penalty_param
ż
save_6/Assign_8Assignpenalty/penalty_param/Adam_1save_6/RestoreV2:8*(
_class
loc:@penalty/penalty_param*
T0*
_output_shapes
: *
validate_shape(*
use_locking(
ż
save_6/Assign_9Assignpenalty/penalty_param/Adam_2save_6/RestoreV2:9*
T0*
validate_shape(*
use_locking(*(
_class
loc:@penalty/penalty_param*
_output_shapes
: 
Á
save_6/Assign_10Assignpenalty/penalty_param/Adam_3save_6/RestoreV2:10*
use_locking(*
_output_shapes
: *
validate_shape(*
T0*(
_class
loc:@penalty/penalty_param
Ż
save_6/Assign_11Assignpi/dense/biassave_6/RestoreV2:11*
_output_shapes	
:* 
_class
loc:@pi/dense/bias*
use_locking(*
T0*
validate_shape(
´
save_6/Assign_12Assignpi/dense/bias/Adamsave_6/RestoreV2:12* 
_class
loc:@pi/dense/bias*
_output_shapes	
:*
use_locking(*
T0*
validate_shape(
ś
save_6/Assign_13Assignpi/dense/bias/Adam_1save_6/RestoreV2:13*
use_locking(*
T0*
_output_shapes	
:*
validate_shape(* 
_class
loc:@pi/dense/bias
ˇ
save_6/Assign_14Assignpi/dense/kernelsave_6/RestoreV2:14*
_output_shapes
:	L*
T0*"
_class
loc:@pi/dense/kernel*
validate_shape(*
use_locking(
ź
save_6/Assign_15Assignpi/dense/kernel/Adamsave_6/RestoreV2:15*
use_locking(*
T0*"
_class
loc:@pi/dense/kernel*
validate_shape(*
_output_shapes
:	L
ž
save_6/Assign_16Assignpi/dense/kernel/Adam_1save_6/RestoreV2:16*"
_class
loc:@pi/dense/kernel*
use_locking(*
T0*
validate_shape(*
_output_shapes
:	L
ł
save_6/Assign_17Assignpi/dense_1/biassave_6/RestoreV2:17*
T0*
_output_shapes	
:*
validate_shape(*"
_class
loc:@pi/dense_1/bias*
use_locking(
¸
save_6/Assign_18Assignpi/dense_1/bias/Adamsave_6/RestoreV2:18*"
_class
loc:@pi/dense_1/bias*
T0*
use_locking(*
_output_shapes	
:*
validate_shape(
ş
save_6/Assign_19Assignpi/dense_1/bias/Adam_1save_6/RestoreV2:19*"
_class
loc:@pi/dense_1/bias*
T0*
validate_shape(*
_output_shapes	
:*
use_locking(
ź
save_6/Assign_20Assignpi/dense_1/kernelsave_6/RestoreV2:20*
use_locking(*
T0*
validate_shape(* 
_output_shapes
:
*$
_class
loc:@pi/dense_1/kernel
Á
save_6/Assign_21Assignpi/dense_1/kernel/Adamsave_6/RestoreV2:21*
T0*
use_locking(* 
_output_shapes
:
*$
_class
loc:@pi/dense_1/kernel*
validate_shape(
Ă
save_6/Assign_22Assignpi/dense_1/kernel/Adam_1save_6/RestoreV2:22*
use_locking(*$
_class
loc:@pi/dense_1/kernel* 
_output_shapes
:
*
validate_shape(*
T0
˛
save_6/Assign_23Assignpi/dense_2/biassave_6/RestoreV2:23*
_output_shapes
:*
T0*
use_locking(*
validate_shape(*"
_class
loc:@pi/dense_2/bias
ˇ
save_6/Assign_24Assignpi/dense_2/bias/Adamsave_6/RestoreV2:24*
use_locking(*"
_class
loc:@pi/dense_2/bias*
_output_shapes
:*
T0*
validate_shape(
š
save_6/Assign_25Assignpi/dense_2/bias/Adam_1save_6/RestoreV2:25*
_output_shapes
:*
use_locking(*
validate_shape(*"
_class
loc:@pi/dense_2/bias*
T0
ť
save_6/Assign_26Assignpi/dense_2/kernelsave_6/RestoreV2:26*
validate_shape(*
use_locking(*$
_class
loc:@pi/dense_2/kernel*
T0*
_output_shapes
:	
Ŕ
save_6/Assign_27Assignpi/dense_2/kernel/Adamsave_6/RestoreV2:27*
_output_shapes
:	*
use_locking(*
validate_shape(*$
_class
loc:@pi/dense_2/kernel*
T0
Â
save_6/Assign_28Assignpi/dense_2/kernel/Adam_1save_6/RestoreV2:28*
use_locking(*$
_class
loc:@pi/dense_2/kernel*
validate_shape(*
_output_shapes
:	*
T0
¨
save_6/Assign_29Assign
pi/log_stdsave_6/RestoreV2:29*
use_locking(*
validate_shape(*
_output_shapes
:*
_class
loc:@pi/log_std*
T0
­
save_6/Assign_30Assignpi/log_std/Adamsave_6/RestoreV2:30*
T0*
_class
loc:@pi/log_std*
validate_shape(*
_output_shapes
:*
use_locking(
Ż
save_6/Assign_31Assignpi/log_std/Adam_1save_6/RestoreV2:31*
T0*
_class
loc:@pi/log_std*
validate_shape(*
_output_shapes
:*
use_locking(
Ż
save_6/Assign_32Assignvc/dense/biassave_6/RestoreV2:32*
use_locking(*
validate_shape(* 
_class
loc:@vc/dense/bias*
_output_shapes	
:*
T0
´
save_6/Assign_33Assignvc/dense/bias/Adamsave_6/RestoreV2:33*
validate_shape(*
use_locking(* 
_class
loc:@vc/dense/bias*
_output_shapes	
:*
T0
ś
save_6/Assign_34Assignvc/dense/bias/Adam_1save_6/RestoreV2:34*
use_locking(*
_output_shapes	
:*
validate_shape(* 
_class
loc:@vc/dense/bias*
T0
ˇ
save_6/Assign_35Assignvc/dense/kernelsave_6/RestoreV2:35*"
_class
loc:@vc/dense/kernel*
use_locking(*
validate_shape(*
T0*
_output_shapes
:	L
ź
save_6/Assign_36Assignvc/dense/kernel/Adamsave_6/RestoreV2:36*
_output_shapes
:	L*
use_locking(*
T0*
validate_shape(*"
_class
loc:@vc/dense/kernel
ž
save_6/Assign_37Assignvc/dense/kernel/Adam_1save_6/RestoreV2:37*
_output_shapes
:	L*
use_locking(*
validate_shape(*"
_class
loc:@vc/dense/kernel*
T0
ł
save_6/Assign_38Assignvc/dense_1/biassave_6/RestoreV2:38*
T0*
validate_shape(*
_output_shapes	
:*
use_locking(*"
_class
loc:@vc/dense_1/bias
¸
save_6/Assign_39Assignvc/dense_1/bias/Adamsave_6/RestoreV2:39*
use_locking(*"
_class
loc:@vc/dense_1/bias*
validate_shape(*
_output_shapes	
:*
T0
ş
save_6/Assign_40Assignvc/dense_1/bias/Adam_1save_6/RestoreV2:40*
T0*
_output_shapes	
:*
validate_shape(*"
_class
loc:@vc/dense_1/bias*
use_locking(
ź
save_6/Assign_41Assignvc/dense_1/kernelsave_6/RestoreV2:41*
use_locking(*
T0* 
_output_shapes
:
*$
_class
loc:@vc/dense_1/kernel*
validate_shape(
Á
save_6/Assign_42Assignvc/dense_1/kernel/Adamsave_6/RestoreV2:42*
T0*
use_locking(* 
_output_shapes
:
*
validate_shape(*$
_class
loc:@vc/dense_1/kernel
Ă
save_6/Assign_43Assignvc/dense_1/kernel/Adam_1save_6/RestoreV2:43* 
_output_shapes
:
*
validate_shape(*
T0*
use_locking(*$
_class
loc:@vc/dense_1/kernel
˛
save_6/Assign_44Assignvc/dense_2/biassave_6/RestoreV2:44*"
_class
loc:@vc/dense_2/bias*
_output_shapes
:*
validate_shape(*
T0*
use_locking(
ˇ
save_6/Assign_45Assignvc/dense_2/bias/Adamsave_6/RestoreV2:45*"
_class
loc:@vc/dense_2/bias*
_output_shapes
:*
T0*
use_locking(*
validate_shape(
š
save_6/Assign_46Assignvc/dense_2/bias/Adam_1save_6/RestoreV2:46*
validate_shape(*
use_locking(*
_output_shapes
:*"
_class
loc:@vc/dense_2/bias*
T0
ť
save_6/Assign_47Assignvc/dense_2/kernelsave_6/RestoreV2:47*
_output_shapes
:	*
use_locking(*
T0*$
_class
loc:@vc/dense_2/kernel*
validate_shape(
Ŕ
save_6/Assign_48Assignvc/dense_2/kernel/Adamsave_6/RestoreV2:48*$
_class
loc:@vc/dense_2/kernel*
T0*
validate_shape(*
_output_shapes
:	*
use_locking(
Â
save_6/Assign_49Assignvc/dense_2/kernel/Adam_1save_6/RestoreV2:49*
_output_shapes
:	*
T0*
validate_shape(*
use_locking(*$
_class
loc:@vc/dense_2/kernel
Ż
save_6/Assign_50Assignvf/dense/biassave_6/RestoreV2:50*
T0* 
_class
loc:@vf/dense/bias*
_output_shapes	
:*
use_locking(*
validate_shape(
´
save_6/Assign_51Assignvf/dense/bias/Adamsave_6/RestoreV2:51* 
_class
loc:@vf/dense/bias*
validate_shape(*
_output_shapes	
:*
T0*
use_locking(
ś
save_6/Assign_52Assignvf/dense/bias/Adam_1save_6/RestoreV2:52* 
_class
loc:@vf/dense/bias*
validate_shape(*
T0*
use_locking(*
_output_shapes	
:
ˇ
save_6/Assign_53Assignvf/dense/kernelsave_6/RestoreV2:53*
use_locking(*"
_class
loc:@vf/dense/kernel*
validate_shape(*
T0*
_output_shapes
:	L
ź
save_6/Assign_54Assignvf/dense/kernel/Adamsave_6/RestoreV2:54*
validate_shape(*
use_locking(*
_output_shapes
:	L*
T0*"
_class
loc:@vf/dense/kernel
ž
save_6/Assign_55Assignvf/dense/kernel/Adam_1save_6/RestoreV2:55*
_output_shapes
:	L*
T0*"
_class
loc:@vf/dense/kernel*
validate_shape(*
use_locking(
ł
save_6/Assign_56Assignvf/dense_1/biassave_6/RestoreV2:56*"
_class
loc:@vf/dense_1/bias*
_output_shapes	
:*
use_locking(*
validate_shape(*
T0
¸
save_6/Assign_57Assignvf/dense_1/bias/Adamsave_6/RestoreV2:57*
validate_shape(*
use_locking(*
_output_shapes	
:*"
_class
loc:@vf/dense_1/bias*
T0
ş
save_6/Assign_58Assignvf/dense_1/bias/Adam_1save_6/RestoreV2:58*
use_locking(*
validate_shape(*"
_class
loc:@vf/dense_1/bias*
T0*
_output_shapes	
:
ź
save_6/Assign_59Assignvf/dense_1/kernelsave_6/RestoreV2:59*
validate_shape(*
T0* 
_output_shapes
:
*
use_locking(*$
_class
loc:@vf/dense_1/kernel
Á
save_6/Assign_60Assignvf/dense_1/kernel/Adamsave_6/RestoreV2:60*
validate_shape(*
use_locking(*$
_class
loc:@vf/dense_1/kernel* 
_output_shapes
:
*
T0
Ă
save_6/Assign_61Assignvf/dense_1/kernel/Adam_1save_6/RestoreV2:61*$
_class
loc:@vf/dense_1/kernel* 
_output_shapes
:
*
use_locking(*
T0*
validate_shape(
˛
save_6/Assign_62Assignvf/dense_2/biassave_6/RestoreV2:62*
use_locking(*
T0*
validate_shape(*"
_class
loc:@vf/dense_2/bias*
_output_shapes
:
ˇ
save_6/Assign_63Assignvf/dense_2/bias/Adamsave_6/RestoreV2:63*
T0*
use_locking(*
validate_shape(*"
_class
loc:@vf/dense_2/bias*
_output_shapes
:
š
save_6/Assign_64Assignvf/dense_2/bias/Adam_1save_6/RestoreV2:64*
_output_shapes
:*"
_class
loc:@vf/dense_2/bias*
use_locking(*
validate_shape(*
T0
ť
save_6/Assign_65Assignvf/dense_2/kernelsave_6/RestoreV2:65*
use_locking(*
_output_shapes
:	*$
_class
loc:@vf/dense_2/kernel*
validate_shape(*
T0
Ŕ
save_6/Assign_66Assignvf/dense_2/kernel/Adamsave_6/RestoreV2:66*
use_locking(*$
_class
loc:@vf/dense_2/kernel*
T0*
_output_shapes
:	*
validate_shape(
Â
save_6/Assign_67Assignvf/dense_2/kernel/Adam_1save_6/RestoreV2:67*
use_locking(*
validate_shape(*$
_class
loc:@vf/dense_2/kernel*
T0*
_output_shapes
:	


save_6/restore_shardNoOp^save_6/Assign^save_6/Assign_1^save_6/Assign_10^save_6/Assign_11^save_6/Assign_12^save_6/Assign_13^save_6/Assign_14^save_6/Assign_15^save_6/Assign_16^save_6/Assign_17^save_6/Assign_18^save_6/Assign_19^save_6/Assign_2^save_6/Assign_20^save_6/Assign_21^save_6/Assign_22^save_6/Assign_23^save_6/Assign_24^save_6/Assign_25^save_6/Assign_26^save_6/Assign_27^save_6/Assign_28^save_6/Assign_29^save_6/Assign_3^save_6/Assign_30^save_6/Assign_31^save_6/Assign_32^save_6/Assign_33^save_6/Assign_34^save_6/Assign_35^save_6/Assign_36^save_6/Assign_37^save_6/Assign_38^save_6/Assign_39^save_6/Assign_4^save_6/Assign_40^save_6/Assign_41^save_6/Assign_42^save_6/Assign_43^save_6/Assign_44^save_6/Assign_45^save_6/Assign_46^save_6/Assign_47^save_6/Assign_48^save_6/Assign_49^save_6/Assign_5^save_6/Assign_50^save_6/Assign_51^save_6/Assign_52^save_6/Assign_53^save_6/Assign_54^save_6/Assign_55^save_6/Assign_56^save_6/Assign_57^save_6/Assign_58^save_6/Assign_59^save_6/Assign_6^save_6/Assign_60^save_6/Assign_61^save_6/Assign_62^save_6/Assign_63^save_6/Assign_64^save_6/Assign_65^save_6/Assign_66^save_6/Assign_67^save_6/Assign_7^save_6/Assign_8^save_6/Assign_9
1
save_6/restore_allNoOp^save_6/restore_shard
[
save_7/filename/inputConst*
dtype0*
valueB Bmodel*
_output_shapes
: 
r
save_7/filenamePlaceholderWithDefaultsave_7/filename/input*
shape: *
_output_shapes
: *
dtype0
i
save_7/ConstPlaceholderWithDefaultsave_7/filename*
dtype0*
shape: *
_output_shapes
: 

save_7/StringJoin/inputs_1Const*<
value3B1 B+_temp_3e7f1c799a9b488d8871f8635c821223/part*
dtype0*
_output_shapes
: 
{
save_7/StringJoin
StringJoinsave_7/Constsave_7/StringJoin/inputs_1*
N*
_output_shapes
: *
	separator 
S
save_7/num_shardsConst*
_output_shapes
: *
dtype0*
value	B :
^
save_7/ShardedFilename/shardConst*
dtype0*
_output_shapes
: *
value	B : 

save_7/ShardedFilenameShardedFilenamesave_7/StringJoinsave_7/ShardedFilename/shardsave_7/num_shards*
_output_shapes
: 
÷
save_7/SaveV2/tensor_namesConst*¨
valueBDBbeta1_powerBbeta1_power_1Bbeta1_power_2Bbeta2_powerBbeta2_power_1Bbeta2_power_2Bpenalty/penalty_paramBpenalty/penalty_param/AdamBpenalty/penalty_param/Adam_1Bpenalty/penalty_param/Adam_2Bpenalty/penalty_param/Adam_3Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1B
pi/log_stdBpi/log_std/AdamBpi/log_std/Adam_1Bvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1*
_output_shapes
:D*
dtype0
đ
save_7/SaveV2/shape_and_slicesConst*
_output_shapes
:D*
dtype0*
valueBDB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
Ň
save_7/SaveV2SaveV2save_7/ShardedFilenamesave_7/SaveV2/tensor_namessave_7/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta1_power_2beta2_powerbeta2_power_1beta2_power_2penalty/penalty_parampenalty/penalty_param/Adampenalty/penalty_param/Adam_1penalty/penalty_param/Adam_2penalty/penalty_param/Adam_3pi/dense/biaspi/dense/bias/Adampi/dense/bias/Adam_1pi/dense/kernelpi/dense/kernel/Adampi/dense/kernel/Adam_1pi/dense_1/biaspi/dense_1/bias/Adampi/dense_1/bias/Adam_1pi/dense_1/kernelpi/dense_1/kernel/Adampi/dense_1/kernel/Adam_1pi/dense_2/biaspi/dense_2/bias/Adampi/dense_2/bias/Adam_1pi/dense_2/kernelpi/dense_2/kernel/Adampi/dense_2/kernel/Adam_1
pi/log_stdpi/log_std/Adampi/log_std/Adam_1vc/dense/biasvc/dense/bias/Adamvc/dense/bias/Adam_1vc/dense/kernelvc/dense/kernel/Adamvc/dense/kernel/Adam_1vc/dense_1/biasvc/dense_1/bias/Adamvc/dense_1/bias/Adam_1vc/dense_1/kernelvc/dense_1/kernel/Adamvc/dense_1/kernel/Adam_1vc/dense_2/biasvc/dense_2/bias/Adamvc/dense_2/bias/Adam_1vc/dense_2/kernelvc/dense_2/kernel/Adamvc/dense_2/kernel/Adam_1vf/dense/biasvf/dense/bias/Adamvf/dense/bias/Adam_1vf/dense/kernelvf/dense/kernel/Adamvf/dense/kernel/Adam_1vf/dense_1/biasvf/dense_1/bias/Adamvf/dense_1/bias/Adam_1vf/dense_1/kernelvf/dense_1/kernel/Adamvf/dense_1/kernel/Adam_1vf/dense_2/biasvf/dense_2/bias/Adamvf/dense_2/bias/Adam_1vf/dense_2/kernelvf/dense_2/kernel/Adamvf/dense_2/kernel/Adam_1*R
dtypesH
F2D

save_7/control_dependencyIdentitysave_7/ShardedFilename^save_7/SaveV2*)
_class
loc:@save_7/ShardedFilename*
T0*
_output_shapes
: 
Ł
-save_7/MergeV2Checkpoints/checkpoint_prefixesPacksave_7/ShardedFilename^save_7/control_dependency*
_output_shapes
:*

axis *
T0*
N

save_7/MergeV2CheckpointsMergeV2Checkpoints-save_7/MergeV2Checkpoints/checkpoint_prefixessave_7/Const*
delete_old_dirs(

save_7/IdentityIdentitysave_7/Const^save_7/MergeV2Checkpoints^save_7/control_dependency*
_output_shapes
: *
T0
ú
save_7/RestoreV2/tensor_namesConst*
_output_shapes
:D*¨
valueBDBbeta1_powerBbeta1_power_1Bbeta1_power_2Bbeta2_powerBbeta2_power_1Bbeta2_power_2Bpenalty/penalty_paramBpenalty/penalty_param/AdamBpenalty/penalty_param/Adam_1Bpenalty/penalty_param/Adam_2Bpenalty/penalty_param/Adam_3Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1B
pi/log_stdBpi/log_std/AdamBpi/log_std/Adam_1Bvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1*
dtype0
ó
!save_7/RestoreV2/shape_and_slicesConst*
valueBDB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:D
ę
save_7/RestoreV2	RestoreV2save_7/Constsave_7/RestoreV2/tensor_names!save_7/RestoreV2/shape_and_slices*R
dtypesH
F2D*Ś
_output_shapes
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
Ş
save_7/AssignAssignbeta1_powersave_7/RestoreV2*(
_class
loc:@penalty/penalty_param*
validate_shape(*
use_locking(*
T0*
_output_shapes
: 
°
save_7/Assign_1Assignbeta1_power_1save_7/RestoreV2:1*
use_locking(*
T0*
_output_shapes
: *
validate_shape(*(
_class
loc:@penalty/penalty_param
¨
save_7/Assign_2Assignbeta1_power_2save_7/RestoreV2:2*
validate_shape(* 
_class
loc:@vc/dense/bias*
_output_shapes
: *
T0*
use_locking(
Ž
save_7/Assign_3Assignbeta2_powersave_7/RestoreV2:3*
validate_shape(*
T0*(
_class
loc:@penalty/penalty_param*
use_locking(*
_output_shapes
: 
°
save_7/Assign_4Assignbeta2_power_1save_7/RestoreV2:4*
use_locking(*
validate_shape(*(
_class
loc:@penalty/penalty_param*
_output_shapes
: *
T0
¨
save_7/Assign_5Assignbeta2_power_2save_7/RestoreV2:5*
validate_shape(*
use_locking(*
_output_shapes
: * 
_class
loc:@vc/dense/bias*
T0
¸
save_7/Assign_6Assignpenalty/penalty_paramsave_7/RestoreV2:6*(
_class
loc:@penalty/penalty_param*
T0*
validate_shape(*
_output_shapes
: *
use_locking(
˝
save_7/Assign_7Assignpenalty/penalty_param/Adamsave_7/RestoreV2:7*(
_class
loc:@penalty/penalty_param*
_output_shapes
: *
T0*
use_locking(*
validate_shape(
ż
save_7/Assign_8Assignpenalty/penalty_param/Adam_1save_7/RestoreV2:8*
_output_shapes
: *
validate_shape(*(
_class
loc:@penalty/penalty_param*
use_locking(*
T0
ż
save_7/Assign_9Assignpenalty/penalty_param/Adam_2save_7/RestoreV2:9*
validate_shape(*
_output_shapes
: *
use_locking(*(
_class
loc:@penalty/penalty_param*
T0
Á
save_7/Assign_10Assignpenalty/penalty_param/Adam_3save_7/RestoreV2:10*
_output_shapes
: *
T0*
use_locking(*
validate_shape(*(
_class
loc:@penalty/penalty_param
Ż
save_7/Assign_11Assignpi/dense/biassave_7/RestoreV2:11*
validate_shape(*
_output_shapes	
:*
use_locking(* 
_class
loc:@pi/dense/bias*
T0
´
save_7/Assign_12Assignpi/dense/bias/Adamsave_7/RestoreV2:12* 
_class
loc:@pi/dense/bias*
_output_shapes	
:*
validate_shape(*
T0*
use_locking(
ś
save_7/Assign_13Assignpi/dense/bias/Adam_1save_7/RestoreV2:13*
validate_shape(* 
_class
loc:@pi/dense/bias*
use_locking(*
_output_shapes	
:*
T0
ˇ
save_7/Assign_14Assignpi/dense/kernelsave_7/RestoreV2:14*
T0*
validate_shape(*"
_class
loc:@pi/dense/kernel*
_output_shapes
:	L*
use_locking(
ź
save_7/Assign_15Assignpi/dense/kernel/Adamsave_7/RestoreV2:15*
validate_shape(*"
_class
loc:@pi/dense/kernel*
use_locking(*
_output_shapes
:	L*
T0
ž
save_7/Assign_16Assignpi/dense/kernel/Adam_1save_7/RestoreV2:16*
validate_shape(*"
_class
loc:@pi/dense/kernel*
use_locking(*
T0*
_output_shapes
:	L
ł
save_7/Assign_17Assignpi/dense_1/biassave_7/RestoreV2:17*
use_locking(*"
_class
loc:@pi/dense_1/bias*
T0*
_output_shapes	
:*
validate_shape(
¸
save_7/Assign_18Assignpi/dense_1/bias/Adamsave_7/RestoreV2:18*
T0*
validate_shape(*
use_locking(*"
_class
loc:@pi/dense_1/bias*
_output_shapes	
:
ş
save_7/Assign_19Assignpi/dense_1/bias/Adam_1save_7/RestoreV2:19*
_output_shapes	
:*
use_locking(*
validate_shape(*"
_class
loc:@pi/dense_1/bias*
T0
ź
save_7/Assign_20Assignpi/dense_1/kernelsave_7/RestoreV2:20*
T0*
use_locking(* 
_output_shapes
:
*$
_class
loc:@pi/dense_1/kernel*
validate_shape(
Á
save_7/Assign_21Assignpi/dense_1/kernel/Adamsave_7/RestoreV2:21*
T0* 
_output_shapes
:
*
use_locking(*
validate_shape(*$
_class
loc:@pi/dense_1/kernel
Ă
save_7/Assign_22Assignpi/dense_1/kernel/Adam_1save_7/RestoreV2:22*
use_locking(* 
_output_shapes
:
*$
_class
loc:@pi/dense_1/kernel*
validate_shape(*
T0
˛
save_7/Assign_23Assignpi/dense_2/biassave_7/RestoreV2:23*"
_class
loc:@pi/dense_2/bias*
validate_shape(*
T0*
use_locking(*
_output_shapes
:
ˇ
save_7/Assign_24Assignpi/dense_2/bias/Adamsave_7/RestoreV2:24*
T0*"
_class
loc:@pi/dense_2/bias*
use_locking(*
validate_shape(*
_output_shapes
:
š
save_7/Assign_25Assignpi/dense_2/bias/Adam_1save_7/RestoreV2:25*
_output_shapes
:*
T0*
use_locking(*
validate_shape(*"
_class
loc:@pi/dense_2/bias
ť
save_7/Assign_26Assignpi/dense_2/kernelsave_7/RestoreV2:26*
T0*$
_class
loc:@pi/dense_2/kernel*
validate_shape(*
_output_shapes
:	*
use_locking(
Ŕ
save_7/Assign_27Assignpi/dense_2/kernel/Adamsave_7/RestoreV2:27*
use_locking(*
validate_shape(*
_output_shapes
:	*$
_class
loc:@pi/dense_2/kernel*
T0
Â
save_7/Assign_28Assignpi/dense_2/kernel/Adam_1save_7/RestoreV2:28*
use_locking(*$
_class
loc:@pi/dense_2/kernel*
validate_shape(*
T0*
_output_shapes
:	
¨
save_7/Assign_29Assign
pi/log_stdsave_7/RestoreV2:29*
validate_shape(*
_class
loc:@pi/log_std*
_output_shapes
:*
T0*
use_locking(
­
save_7/Assign_30Assignpi/log_std/Adamsave_7/RestoreV2:30*
use_locking(*
_class
loc:@pi/log_std*
_output_shapes
:*
validate_shape(*
T0
Ż
save_7/Assign_31Assignpi/log_std/Adam_1save_7/RestoreV2:31*
T0*
validate_shape(*
_output_shapes
:*
_class
loc:@pi/log_std*
use_locking(
Ż
save_7/Assign_32Assignvc/dense/biassave_7/RestoreV2:32*
T0*
_output_shapes	
:* 
_class
loc:@vc/dense/bias*
validate_shape(*
use_locking(
´
save_7/Assign_33Assignvc/dense/bias/Adamsave_7/RestoreV2:33*
use_locking(*
validate_shape(*
T0* 
_class
loc:@vc/dense/bias*
_output_shapes	
:
ś
save_7/Assign_34Assignvc/dense/bias/Adam_1save_7/RestoreV2:34*
validate_shape(*
T0*
use_locking(* 
_class
loc:@vc/dense/bias*
_output_shapes	
:
ˇ
save_7/Assign_35Assignvc/dense/kernelsave_7/RestoreV2:35*
validate_shape(*
use_locking(*"
_class
loc:@vc/dense/kernel*
_output_shapes
:	L*
T0
ź
save_7/Assign_36Assignvc/dense/kernel/Adamsave_7/RestoreV2:36*"
_class
loc:@vc/dense/kernel*
T0*
validate_shape(*
_output_shapes
:	L*
use_locking(
ž
save_7/Assign_37Assignvc/dense/kernel/Adam_1save_7/RestoreV2:37*
T0*
_output_shapes
:	L*
validate_shape(*
use_locking(*"
_class
loc:@vc/dense/kernel
ł
save_7/Assign_38Assignvc/dense_1/biassave_7/RestoreV2:38*"
_class
loc:@vc/dense_1/bias*
_output_shapes	
:*
validate_shape(*
use_locking(*
T0
¸
save_7/Assign_39Assignvc/dense_1/bias/Adamsave_7/RestoreV2:39*
_output_shapes	
:*
T0*
validate_shape(*
use_locking(*"
_class
loc:@vc/dense_1/bias
ş
save_7/Assign_40Assignvc/dense_1/bias/Adam_1save_7/RestoreV2:40*
_output_shapes	
:*
use_locking(*"
_class
loc:@vc/dense_1/bias*
validate_shape(*
T0
ź
save_7/Assign_41Assignvc/dense_1/kernelsave_7/RestoreV2:41*$
_class
loc:@vc/dense_1/kernel*
validate_shape(*
T0*
use_locking(* 
_output_shapes
:

Á
save_7/Assign_42Assignvc/dense_1/kernel/Adamsave_7/RestoreV2:42*$
_class
loc:@vc/dense_1/kernel* 
_output_shapes
:
*
T0*
validate_shape(*
use_locking(
Ă
save_7/Assign_43Assignvc/dense_1/kernel/Adam_1save_7/RestoreV2:43*
validate_shape(*$
_class
loc:@vc/dense_1/kernel* 
_output_shapes
:
*
use_locking(*
T0
˛
save_7/Assign_44Assignvc/dense_2/biassave_7/RestoreV2:44*"
_class
loc:@vc/dense_2/bias*
T0*
use_locking(*
validate_shape(*
_output_shapes
:
ˇ
save_7/Assign_45Assignvc/dense_2/bias/Adamsave_7/RestoreV2:45*"
_class
loc:@vc/dense_2/bias*
validate_shape(*
T0*
use_locking(*
_output_shapes
:
š
save_7/Assign_46Assignvc/dense_2/bias/Adam_1save_7/RestoreV2:46*
T0*
validate_shape(*
use_locking(*
_output_shapes
:*"
_class
loc:@vc/dense_2/bias
ť
save_7/Assign_47Assignvc/dense_2/kernelsave_7/RestoreV2:47*
_output_shapes
:	*
T0*$
_class
loc:@vc/dense_2/kernel*
use_locking(*
validate_shape(
Ŕ
save_7/Assign_48Assignvc/dense_2/kernel/Adamsave_7/RestoreV2:48*$
_class
loc:@vc/dense_2/kernel*
validate_shape(*
use_locking(*
T0*
_output_shapes
:	
Â
save_7/Assign_49Assignvc/dense_2/kernel/Adam_1save_7/RestoreV2:49*
T0*
_output_shapes
:	*
use_locking(*$
_class
loc:@vc/dense_2/kernel*
validate_shape(
Ż
save_7/Assign_50Assignvf/dense/biassave_7/RestoreV2:50*
T0*
use_locking(*
_output_shapes	
:*
validate_shape(* 
_class
loc:@vf/dense/bias
´
save_7/Assign_51Assignvf/dense/bias/Adamsave_7/RestoreV2:51*
T0*
_output_shapes	
:*
validate_shape(* 
_class
loc:@vf/dense/bias*
use_locking(
ś
save_7/Assign_52Assignvf/dense/bias/Adam_1save_7/RestoreV2:52*
use_locking(* 
_class
loc:@vf/dense/bias*
T0*
validate_shape(*
_output_shapes	
:
ˇ
save_7/Assign_53Assignvf/dense/kernelsave_7/RestoreV2:53*
_output_shapes
:	L*
T0*"
_class
loc:@vf/dense/kernel*
use_locking(*
validate_shape(
ź
save_7/Assign_54Assignvf/dense/kernel/Adamsave_7/RestoreV2:54*
_output_shapes
:	L*
T0*
use_locking(*"
_class
loc:@vf/dense/kernel*
validate_shape(
ž
save_7/Assign_55Assignvf/dense/kernel/Adam_1save_7/RestoreV2:55*
_output_shapes
:	L*
use_locking(*
validate_shape(*
T0*"
_class
loc:@vf/dense/kernel
ł
save_7/Assign_56Assignvf/dense_1/biassave_7/RestoreV2:56*
use_locking(*"
_class
loc:@vf/dense_1/bias*
_output_shapes	
:*
validate_shape(*
T0
¸
save_7/Assign_57Assignvf/dense_1/bias/Adamsave_7/RestoreV2:57*
T0*"
_class
loc:@vf/dense_1/bias*
_output_shapes	
:*
validate_shape(*
use_locking(
ş
save_7/Assign_58Assignvf/dense_1/bias/Adam_1save_7/RestoreV2:58*
use_locking(*
validate_shape(*"
_class
loc:@vf/dense_1/bias*
T0*
_output_shapes	
:
ź
save_7/Assign_59Assignvf/dense_1/kernelsave_7/RestoreV2:59*$
_class
loc:@vf/dense_1/kernel*
use_locking(* 
_output_shapes
:
*
T0*
validate_shape(
Á
save_7/Assign_60Assignvf/dense_1/kernel/Adamsave_7/RestoreV2:60* 
_output_shapes
:
*$
_class
loc:@vf/dense_1/kernel*
T0*
use_locking(*
validate_shape(
Ă
save_7/Assign_61Assignvf/dense_1/kernel/Adam_1save_7/RestoreV2:61*$
_class
loc:@vf/dense_1/kernel*
validate_shape(*
use_locking(* 
_output_shapes
:
*
T0
˛
save_7/Assign_62Assignvf/dense_2/biassave_7/RestoreV2:62*
use_locking(*
validate_shape(*
T0*
_output_shapes
:*"
_class
loc:@vf/dense_2/bias
ˇ
save_7/Assign_63Assignvf/dense_2/bias/Adamsave_7/RestoreV2:63*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*"
_class
loc:@vf/dense_2/bias
š
save_7/Assign_64Assignvf/dense_2/bias/Adam_1save_7/RestoreV2:64*
T0*
_output_shapes
:*"
_class
loc:@vf/dense_2/bias*
use_locking(*
validate_shape(
ť
save_7/Assign_65Assignvf/dense_2/kernelsave_7/RestoreV2:65*
validate_shape(*
T0*
use_locking(*
_output_shapes
:	*$
_class
loc:@vf/dense_2/kernel
Ŕ
save_7/Assign_66Assignvf/dense_2/kernel/Adamsave_7/RestoreV2:66*
_output_shapes
:	*
use_locking(*
T0*
validate_shape(*$
_class
loc:@vf/dense_2/kernel
Â
save_7/Assign_67Assignvf/dense_2/kernel/Adam_1save_7/RestoreV2:67*
validate_shape(*$
_class
loc:@vf/dense_2/kernel*
T0*
use_locking(*
_output_shapes
:	


save_7/restore_shardNoOp^save_7/Assign^save_7/Assign_1^save_7/Assign_10^save_7/Assign_11^save_7/Assign_12^save_7/Assign_13^save_7/Assign_14^save_7/Assign_15^save_7/Assign_16^save_7/Assign_17^save_7/Assign_18^save_7/Assign_19^save_7/Assign_2^save_7/Assign_20^save_7/Assign_21^save_7/Assign_22^save_7/Assign_23^save_7/Assign_24^save_7/Assign_25^save_7/Assign_26^save_7/Assign_27^save_7/Assign_28^save_7/Assign_29^save_7/Assign_3^save_7/Assign_30^save_7/Assign_31^save_7/Assign_32^save_7/Assign_33^save_7/Assign_34^save_7/Assign_35^save_7/Assign_36^save_7/Assign_37^save_7/Assign_38^save_7/Assign_39^save_7/Assign_4^save_7/Assign_40^save_7/Assign_41^save_7/Assign_42^save_7/Assign_43^save_7/Assign_44^save_7/Assign_45^save_7/Assign_46^save_7/Assign_47^save_7/Assign_48^save_7/Assign_49^save_7/Assign_5^save_7/Assign_50^save_7/Assign_51^save_7/Assign_52^save_7/Assign_53^save_7/Assign_54^save_7/Assign_55^save_7/Assign_56^save_7/Assign_57^save_7/Assign_58^save_7/Assign_59^save_7/Assign_6^save_7/Assign_60^save_7/Assign_61^save_7/Assign_62^save_7/Assign_63^save_7/Assign_64^save_7/Assign_65^save_7/Assign_66^save_7/Assign_67^save_7/Assign_7^save_7/Assign_8^save_7/Assign_9
1
save_7/restore_allNoOp^save_7/restore_shard
[
save_8/filename/inputConst*
dtype0*
valueB Bmodel*
_output_shapes
: 
r
save_8/filenamePlaceholderWithDefaultsave_8/filename/input*
dtype0*
_output_shapes
: *
shape: 
i
save_8/ConstPlaceholderWithDefaultsave_8/filename*
shape: *
dtype0*
_output_shapes
: 

save_8/StringJoin/inputs_1Const*
_output_shapes
: *<
value3B1 B+_temp_fee3aca55189462497e7167466007e74/part*
dtype0
{
save_8/StringJoin
StringJoinsave_8/Constsave_8/StringJoin/inputs_1*
N*
	separator *
_output_shapes
: 
S
save_8/num_shardsConst*
value	B :*
_output_shapes
: *
dtype0
^
save_8/ShardedFilename/shardConst*
dtype0*
_output_shapes
: *
value	B : 

save_8/ShardedFilenameShardedFilenamesave_8/StringJoinsave_8/ShardedFilename/shardsave_8/num_shards*
_output_shapes
: 
÷
save_8/SaveV2/tensor_namesConst*
dtype0*
_output_shapes
:D*¨
valueBDBbeta1_powerBbeta1_power_1Bbeta1_power_2Bbeta2_powerBbeta2_power_1Bbeta2_power_2Bpenalty/penalty_paramBpenalty/penalty_param/AdamBpenalty/penalty_param/Adam_1Bpenalty/penalty_param/Adam_2Bpenalty/penalty_param/Adam_3Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1B
pi/log_stdBpi/log_std/AdamBpi/log_std/Adam_1Bvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1
đ
save_8/SaveV2/shape_and_slicesConst*
_output_shapes
:D*
dtype0*
valueBDB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
Ň
save_8/SaveV2SaveV2save_8/ShardedFilenamesave_8/SaveV2/tensor_namessave_8/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta1_power_2beta2_powerbeta2_power_1beta2_power_2penalty/penalty_parampenalty/penalty_param/Adampenalty/penalty_param/Adam_1penalty/penalty_param/Adam_2penalty/penalty_param/Adam_3pi/dense/biaspi/dense/bias/Adampi/dense/bias/Adam_1pi/dense/kernelpi/dense/kernel/Adampi/dense/kernel/Adam_1pi/dense_1/biaspi/dense_1/bias/Adampi/dense_1/bias/Adam_1pi/dense_1/kernelpi/dense_1/kernel/Adampi/dense_1/kernel/Adam_1pi/dense_2/biaspi/dense_2/bias/Adampi/dense_2/bias/Adam_1pi/dense_2/kernelpi/dense_2/kernel/Adampi/dense_2/kernel/Adam_1
pi/log_stdpi/log_std/Adampi/log_std/Adam_1vc/dense/biasvc/dense/bias/Adamvc/dense/bias/Adam_1vc/dense/kernelvc/dense/kernel/Adamvc/dense/kernel/Adam_1vc/dense_1/biasvc/dense_1/bias/Adamvc/dense_1/bias/Adam_1vc/dense_1/kernelvc/dense_1/kernel/Adamvc/dense_1/kernel/Adam_1vc/dense_2/biasvc/dense_2/bias/Adamvc/dense_2/bias/Adam_1vc/dense_2/kernelvc/dense_2/kernel/Adamvc/dense_2/kernel/Adam_1vf/dense/biasvf/dense/bias/Adamvf/dense/bias/Adam_1vf/dense/kernelvf/dense/kernel/Adamvf/dense/kernel/Adam_1vf/dense_1/biasvf/dense_1/bias/Adamvf/dense_1/bias/Adam_1vf/dense_1/kernelvf/dense_1/kernel/Adamvf/dense_1/kernel/Adam_1vf/dense_2/biasvf/dense_2/bias/Adamvf/dense_2/bias/Adam_1vf/dense_2/kernelvf/dense_2/kernel/Adamvf/dense_2/kernel/Adam_1*R
dtypesH
F2D

save_8/control_dependencyIdentitysave_8/ShardedFilename^save_8/SaveV2*
_output_shapes
: *)
_class
loc:@save_8/ShardedFilename*
T0
Ł
-save_8/MergeV2Checkpoints/checkpoint_prefixesPacksave_8/ShardedFilename^save_8/control_dependency*
T0*
N*
_output_shapes
:*

axis 

save_8/MergeV2CheckpointsMergeV2Checkpoints-save_8/MergeV2Checkpoints/checkpoint_prefixessave_8/Const*
delete_old_dirs(

save_8/IdentityIdentitysave_8/Const^save_8/MergeV2Checkpoints^save_8/control_dependency*
_output_shapes
: *
T0
ú
save_8/RestoreV2/tensor_namesConst*
dtype0*
_output_shapes
:D*¨
valueBDBbeta1_powerBbeta1_power_1Bbeta1_power_2Bbeta2_powerBbeta2_power_1Bbeta2_power_2Bpenalty/penalty_paramBpenalty/penalty_param/AdamBpenalty/penalty_param/Adam_1Bpenalty/penalty_param/Adam_2Bpenalty/penalty_param/Adam_3Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1B
pi/log_stdBpi/log_std/AdamBpi/log_std/Adam_1Bvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1
ó
!save_8/RestoreV2/shape_and_slicesConst*
valueBDB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
_output_shapes
:D*
dtype0
ę
save_8/RestoreV2	RestoreV2save_8/Constsave_8/RestoreV2/tensor_names!save_8/RestoreV2/shape_and_slices*Ś
_output_shapes
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*R
dtypesH
F2D
Ş
save_8/AssignAssignbeta1_powersave_8/RestoreV2*
validate_shape(*
T0*
use_locking(*
_output_shapes
: *(
_class
loc:@penalty/penalty_param
°
save_8/Assign_1Assignbeta1_power_1save_8/RestoreV2:1*(
_class
loc:@penalty/penalty_param*
_output_shapes
: *
validate_shape(*
T0*
use_locking(
¨
save_8/Assign_2Assignbeta1_power_2save_8/RestoreV2:2*
use_locking(*
_output_shapes
: *
T0* 
_class
loc:@vc/dense/bias*
validate_shape(
Ž
save_8/Assign_3Assignbeta2_powersave_8/RestoreV2:3*
_output_shapes
: *
use_locking(*(
_class
loc:@penalty/penalty_param*
validate_shape(*
T0
°
save_8/Assign_4Assignbeta2_power_1save_8/RestoreV2:4*
use_locking(*
validate_shape(*(
_class
loc:@penalty/penalty_param*
T0*
_output_shapes
: 
¨
save_8/Assign_5Assignbeta2_power_2save_8/RestoreV2:5* 
_class
loc:@vc/dense/bias*
validate_shape(*
use_locking(*
_output_shapes
: *
T0
¸
save_8/Assign_6Assignpenalty/penalty_paramsave_8/RestoreV2:6*(
_class
loc:@penalty/penalty_param*
_output_shapes
: *
validate_shape(*
use_locking(*
T0
˝
save_8/Assign_7Assignpenalty/penalty_param/Adamsave_8/RestoreV2:7*
validate_shape(*
T0*
_output_shapes
: *
use_locking(*(
_class
loc:@penalty/penalty_param
ż
save_8/Assign_8Assignpenalty/penalty_param/Adam_1save_8/RestoreV2:8*
T0*
validate_shape(*
use_locking(*
_output_shapes
: *(
_class
loc:@penalty/penalty_param
ż
save_8/Assign_9Assignpenalty/penalty_param/Adam_2save_8/RestoreV2:9*
use_locking(*
validate_shape(*
_output_shapes
: *
T0*(
_class
loc:@penalty/penalty_param
Á
save_8/Assign_10Assignpenalty/penalty_param/Adam_3save_8/RestoreV2:10*
validate_shape(*
T0*
_output_shapes
: *
use_locking(*(
_class
loc:@penalty/penalty_param
Ż
save_8/Assign_11Assignpi/dense/biassave_8/RestoreV2:11*
use_locking(*
T0*
_output_shapes	
:* 
_class
loc:@pi/dense/bias*
validate_shape(
´
save_8/Assign_12Assignpi/dense/bias/Adamsave_8/RestoreV2:12*
T0*
use_locking(* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes	
:
ś
save_8/Assign_13Assignpi/dense/bias/Adam_1save_8/RestoreV2:13* 
_class
loc:@pi/dense/bias*
_output_shapes	
:*
T0*
validate_shape(*
use_locking(
ˇ
save_8/Assign_14Assignpi/dense/kernelsave_8/RestoreV2:14*
validate_shape(*
T0*
_output_shapes
:	L*"
_class
loc:@pi/dense/kernel*
use_locking(
ź
save_8/Assign_15Assignpi/dense/kernel/Adamsave_8/RestoreV2:15*
validate_shape(*
_output_shapes
:	L*"
_class
loc:@pi/dense/kernel*
use_locking(*
T0
ž
save_8/Assign_16Assignpi/dense/kernel/Adam_1save_8/RestoreV2:16*
_output_shapes
:	L*"
_class
loc:@pi/dense/kernel*
validate_shape(*
T0*
use_locking(
ł
save_8/Assign_17Assignpi/dense_1/biassave_8/RestoreV2:17*"
_class
loc:@pi/dense_1/bias*
T0*
use_locking(*
_output_shapes	
:*
validate_shape(
¸
save_8/Assign_18Assignpi/dense_1/bias/Adamsave_8/RestoreV2:18*"
_class
loc:@pi/dense_1/bias*
T0*
_output_shapes	
:*
validate_shape(*
use_locking(
ş
save_8/Assign_19Assignpi/dense_1/bias/Adam_1save_8/RestoreV2:19*
validate_shape(*
T0*
_output_shapes	
:*
use_locking(*"
_class
loc:@pi/dense_1/bias
ź
save_8/Assign_20Assignpi/dense_1/kernelsave_8/RestoreV2:20*
use_locking(* 
_output_shapes
:
*
validate_shape(*
T0*$
_class
loc:@pi/dense_1/kernel
Á
save_8/Assign_21Assignpi/dense_1/kernel/Adamsave_8/RestoreV2:21*
T0*$
_class
loc:@pi/dense_1/kernel*
use_locking(*
validate_shape(* 
_output_shapes
:

Ă
save_8/Assign_22Assignpi/dense_1/kernel/Adam_1save_8/RestoreV2:22*$
_class
loc:@pi/dense_1/kernel*
use_locking(* 
_output_shapes
:
*
T0*
validate_shape(
˛
save_8/Assign_23Assignpi/dense_2/biassave_8/RestoreV2:23*"
_class
loc:@pi/dense_2/bias*
T0*
_output_shapes
:*
use_locking(*
validate_shape(
ˇ
save_8/Assign_24Assignpi/dense_2/bias/Adamsave_8/RestoreV2:24*
_output_shapes
:*
validate_shape(*
use_locking(*"
_class
loc:@pi/dense_2/bias*
T0
š
save_8/Assign_25Assignpi/dense_2/bias/Adam_1save_8/RestoreV2:25*
_output_shapes
:*"
_class
loc:@pi/dense_2/bias*
use_locking(*
validate_shape(*
T0
ť
save_8/Assign_26Assignpi/dense_2/kernelsave_8/RestoreV2:26*$
_class
loc:@pi/dense_2/kernel*
use_locking(*
T0*
validate_shape(*
_output_shapes
:	
Ŕ
save_8/Assign_27Assignpi/dense_2/kernel/Adamsave_8/RestoreV2:27*
_output_shapes
:	*$
_class
loc:@pi/dense_2/kernel*
use_locking(*
validate_shape(*
T0
Â
save_8/Assign_28Assignpi/dense_2/kernel/Adam_1save_8/RestoreV2:28*$
_class
loc:@pi/dense_2/kernel*
T0*
use_locking(*
validate_shape(*
_output_shapes
:	
¨
save_8/Assign_29Assign
pi/log_stdsave_8/RestoreV2:29*
use_locking(*
validate_shape(*
_output_shapes
:*
T0*
_class
loc:@pi/log_std
­
save_8/Assign_30Assignpi/log_std/Adamsave_8/RestoreV2:30*
_output_shapes
:*
use_locking(*
validate_shape(*
T0*
_class
loc:@pi/log_std
Ż
save_8/Assign_31Assignpi/log_std/Adam_1save_8/RestoreV2:31*
use_locking(*
_output_shapes
:*
validate_shape(*
T0*
_class
loc:@pi/log_std
Ż
save_8/Assign_32Assignvc/dense/biassave_8/RestoreV2:32*
use_locking(*
_output_shapes	
:*
validate_shape(*
T0* 
_class
loc:@vc/dense/bias
´
save_8/Assign_33Assignvc/dense/bias/Adamsave_8/RestoreV2:33*
_output_shapes	
:*
T0*
use_locking(*
validate_shape(* 
_class
loc:@vc/dense/bias
ś
save_8/Assign_34Assignvc/dense/bias/Adam_1save_8/RestoreV2:34*
_output_shapes	
:*
validate_shape(* 
_class
loc:@vc/dense/bias*
T0*
use_locking(
ˇ
save_8/Assign_35Assignvc/dense/kernelsave_8/RestoreV2:35*"
_class
loc:@vc/dense/kernel*
T0*
_output_shapes
:	L*
use_locking(*
validate_shape(
ź
save_8/Assign_36Assignvc/dense/kernel/Adamsave_8/RestoreV2:36*
use_locking(*
_output_shapes
:	L*
T0*"
_class
loc:@vc/dense/kernel*
validate_shape(
ž
save_8/Assign_37Assignvc/dense/kernel/Adam_1save_8/RestoreV2:37*
_output_shapes
:	L*"
_class
loc:@vc/dense/kernel*
use_locking(*
validate_shape(*
T0
ł
save_8/Assign_38Assignvc/dense_1/biassave_8/RestoreV2:38*
T0*
use_locking(*"
_class
loc:@vc/dense_1/bias*
validate_shape(*
_output_shapes	
:
¸
save_8/Assign_39Assignvc/dense_1/bias/Adamsave_8/RestoreV2:39*
T0*
_output_shapes	
:*"
_class
loc:@vc/dense_1/bias*
use_locking(*
validate_shape(
ş
save_8/Assign_40Assignvc/dense_1/bias/Adam_1save_8/RestoreV2:40*
use_locking(*
validate_shape(*"
_class
loc:@vc/dense_1/bias*
T0*
_output_shapes	
:
ź
save_8/Assign_41Assignvc/dense_1/kernelsave_8/RestoreV2:41* 
_output_shapes
:
*
T0*
validate_shape(*
use_locking(*$
_class
loc:@vc/dense_1/kernel
Á
save_8/Assign_42Assignvc/dense_1/kernel/Adamsave_8/RestoreV2:42*
validate_shape(*
T0*
use_locking(*$
_class
loc:@vc/dense_1/kernel* 
_output_shapes
:

Ă
save_8/Assign_43Assignvc/dense_1/kernel/Adam_1save_8/RestoreV2:43*
use_locking(*
T0*$
_class
loc:@vc/dense_1/kernel* 
_output_shapes
:
*
validate_shape(
˛
save_8/Assign_44Assignvc/dense_2/biassave_8/RestoreV2:44*"
_class
loc:@vc/dense_2/bias*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
ˇ
save_8/Assign_45Assignvc/dense_2/bias/Adamsave_8/RestoreV2:45*"
_class
loc:@vc/dense_2/bias*
use_locking(*
validate_shape(*
_output_shapes
:*
T0
š
save_8/Assign_46Assignvc/dense_2/bias/Adam_1save_8/RestoreV2:46*
_output_shapes
:*"
_class
loc:@vc/dense_2/bias*
T0*
use_locking(*
validate_shape(
ť
save_8/Assign_47Assignvc/dense_2/kernelsave_8/RestoreV2:47*
T0*
_output_shapes
:	*$
_class
loc:@vc/dense_2/kernel*
validate_shape(*
use_locking(
Ŕ
save_8/Assign_48Assignvc/dense_2/kernel/Adamsave_8/RestoreV2:48*
validate_shape(*$
_class
loc:@vc/dense_2/kernel*
T0*
_output_shapes
:	*
use_locking(
Â
save_8/Assign_49Assignvc/dense_2/kernel/Adam_1save_8/RestoreV2:49*$
_class
loc:@vc/dense_2/kernel*
validate_shape(*
T0*
_output_shapes
:	*
use_locking(
Ż
save_8/Assign_50Assignvf/dense/biassave_8/RestoreV2:50*
_output_shapes	
:*
T0*
validate_shape(* 
_class
loc:@vf/dense/bias*
use_locking(
´
save_8/Assign_51Assignvf/dense/bias/Adamsave_8/RestoreV2:51*
T0*
use_locking(*
validate_shape(* 
_class
loc:@vf/dense/bias*
_output_shapes	
:
ś
save_8/Assign_52Assignvf/dense/bias/Adam_1save_8/RestoreV2:52*
use_locking(*
_output_shapes	
:* 
_class
loc:@vf/dense/bias*
T0*
validate_shape(
ˇ
save_8/Assign_53Assignvf/dense/kernelsave_8/RestoreV2:53*
validate_shape(*
use_locking(*
_output_shapes
:	L*
T0*"
_class
loc:@vf/dense/kernel
ź
save_8/Assign_54Assignvf/dense/kernel/Adamsave_8/RestoreV2:54*"
_class
loc:@vf/dense/kernel*
validate_shape(*
_output_shapes
:	L*
T0*
use_locking(
ž
save_8/Assign_55Assignvf/dense/kernel/Adam_1save_8/RestoreV2:55*
validate_shape(*
use_locking(*
T0*
_output_shapes
:	L*"
_class
loc:@vf/dense/kernel
ł
save_8/Assign_56Assignvf/dense_1/biassave_8/RestoreV2:56*"
_class
loc:@vf/dense_1/bias*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0
¸
save_8/Assign_57Assignvf/dense_1/bias/Adamsave_8/RestoreV2:57*
validate_shape(*
T0*
use_locking(*
_output_shapes	
:*"
_class
loc:@vf/dense_1/bias
ş
save_8/Assign_58Assignvf/dense_1/bias/Adam_1save_8/RestoreV2:58*"
_class
loc:@vf/dense_1/bias*
T0*
_output_shapes	
:*
use_locking(*
validate_shape(
ź
save_8/Assign_59Assignvf/dense_1/kernelsave_8/RestoreV2:59*
use_locking(*
T0*$
_class
loc:@vf/dense_1/kernel* 
_output_shapes
:
*
validate_shape(
Á
save_8/Assign_60Assignvf/dense_1/kernel/Adamsave_8/RestoreV2:60*
T0*$
_class
loc:@vf/dense_1/kernel*
validate_shape(*
use_locking(* 
_output_shapes
:

Ă
save_8/Assign_61Assignvf/dense_1/kernel/Adam_1save_8/RestoreV2:61*
validate_shape(*
T0*
use_locking(* 
_output_shapes
:
*$
_class
loc:@vf/dense_1/kernel
˛
save_8/Assign_62Assignvf/dense_2/biassave_8/RestoreV2:62*"
_class
loc:@vf/dense_2/bias*
T0*
validate_shape(*
_output_shapes
:*
use_locking(
ˇ
save_8/Assign_63Assignvf/dense_2/bias/Adamsave_8/RestoreV2:63*
validate_shape(*
use_locking(*"
_class
loc:@vf/dense_2/bias*
_output_shapes
:*
T0
š
save_8/Assign_64Assignvf/dense_2/bias/Adam_1save_8/RestoreV2:64*
_output_shapes
:*"
_class
loc:@vf/dense_2/bias*
T0*
use_locking(*
validate_shape(
ť
save_8/Assign_65Assignvf/dense_2/kernelsave_8/RestoreV2:65*
T0*
use_locking(*
_output_shapes
:	*$
_class
loc:@vf/dense_2/kernel*
validate_shape(
Ŕ
save_8/Assign_66Assignvf/dense_2/kernel/Adamsave_8/RestoreV2:66*$
_class
loc:@vf/dense_2/kernel*
T0*
validate_shape(*
use_locking(*
_output_shapes
:	
Â
save_8/Assign_67Assignvf/dense_2/kernel/Adam_1save_8/RestoreV2:67*$
_class
loc:@vf/dense_2/kernel*
validate_shape(*
use_locking(*
_output_shapes
:	*
T0


save_8/restore_shardNoOp^save_8/Assign^save_8/Assign_1^save_8/Assign_10^save_8/Assign_11^save_8/Assign_12^save_8/Assign_13^save_8/Assign_14^save_8/Assign_15^save_8/Assign_16^save_8/Assign_17^save_8/Assign_18^save_8/Assign_19^save_8/Assign_2^save_8/Assign_20^save_8/Assign_21^save_8/Assign_22^save_8/Assign_23^save_8/Assign_24^save_8/Assign_25^save_8/Assign_26^save_8/Assign_27^save_8/Assign_28^save_8/Assign_29^save_8/Assign_3^save_8/Assign_30^save_8/Assign_31^save_8/Assign_32^save_8/Assign_33^save_8/Assign_34^save_8/Assign_35^save_8/Assign_36^save_8/Assign_37^save_8/Assign_38^save_8/Assign_39^save_8/Assign_4^save_8/Assign_40^save_8/Assign_41^save_8/Assign_42^save_8/Assign_43^save_8/Assign_44^save_8/Assign_45^save_8/Assign_46^save_8/Assign_47^save_8/Assign_48^save_8/Assign_49^save_8/Assign_5^save_8/Assign_50^save_8/Assign_51^save_8/Assign_52^save_8/Assign_53^save_8/Assign_54^save_8/Assign_55^save_8/Assign_56^save_8/Assign_57^save_8/Assign_58^save_8/Assign_59^save_8/Assign_6^save_8/Assign_60^save_8/Assign_61^save_8/Assign_62^save_8/Assign_63^save_8/Assign_64^save_8/Assign_65^save_8/Assign_66^save_8/Assign_67^save_8/Assign_7^save_8/Assign_8^save_8/Assign_9
1
save_8/restore_allNoOp^save_8/restore_shard
[
save_9/filename/inputConst*
_output_shapes
: *
dtype0*
valueB Bmodel
r
save_9/filenamePlaceholderWithDefaultsave_9/filename/input*
shape: *
dtype0*
_output_shapes
: 
i
save_9/ConstPlaceholderWithDefaultsave_9/filename*
shape: *
dtype0*
_output_shapes
: 

save_9/StringJoin/inputs_1Const*
dtype0*
_output_shapes
: *<
value3B1 B+_temp_5f3f4aa74e4f43d6908ae693ff31dee8/part
{
save_9/StringJoin
StringJoinsave_9/Constsave_9/StringJoin/inputs_1*
	separator *
_output_shapes
: *
N
S
save_9/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
^
save_9/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 

save_9/ShardedFilenameShardedFilenamesave_9/StringJoinsave_9/ShardedFilename/shardsave_9/num_shards*
_output_shapes
: 
÷
save_9/SaveV2/tensor_namesConst*
dtype0*¨
valueBDBbeta1_powerBbeta1_power_1Bbeta1_power_2Bbeta2_powerBbeta2_power_1Bbeta2_power_2Bpenalty/penalty_paramBpenalty/penalty_param/AdamBpenalty/penalty_param/Adam_1Bpenalty/penalty_param/Adam_2Bpenalty/penalty_param/Adam_3Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1B
pi/log_stdBpi/log_std/AdamBpi/log_std/Adam_1Bvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1*
_output_shapes
:D
đ
save_9/SaveV2/shape_and_slicesConst*
dtype0*
valueBDB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
_output_shapes
:D
Ň
save_9/SaveV2SaveV2save_9/ShardedFilenamesave_9/SaveV2/tensor_namessave_9/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta1_power_2beta2_powerbeta2_power_1beta2_power_2penalty/penalty_parampenalty/penalty_param/Adampenalty/penalty_param/Adam_1penalty/penalty_param/Adam_2penalty/penalty_param/Adam_3pi/dense/biaspi/dense/bias/Adampi/dense/bias/Adam_1pi/dense/kernelpi/dense/kernel/Adampi/dense/kernel/Adam_1pi/dense_1/biaspi/dense_1/bias/Adampi/dense_1/bias/Adam_1pi/dense_1/kernelpi/dense_1/kernel/Adampi/dense_1/kernel/Adam_1pi/dense_2/biaspi/dense_2/bias/Adampi/dense_2/bias/Adam_1pi/dense_2/kernelpi/dense_2/kernel/Adampi/dense_2/kernel/Adam_1
pi/log_stdpi/log_std/Adampi/log_std/Adam_1vc/dense/biasvc/dense/bias/Adamvc/dense/bias/Adam_1vc/dense/kernelvc/dense/kernel/Adamvc/dense/kernel/Adam_1vc/dense_1/biasvc/dense_1/bias/Adamvc/dense_1/bias/Adam_1vc/dense_1/kernelvc/dense_1/kernel/Adamvc/dense_1/kernel/Adam_1vc/dense_2/biasvc/dense_2/bias/Adamvc/dense_2/bias/Adam_1vc/dense_2/kernelvc/dense_2/kernel/Adamvc/dense_2/kernel/Adam_1vf/dense/biasvf/dense/bias/Adamvf/dense/bias/Adam_1vf/dense/kernelvf/dense/kernel/Adamvf/dense/kernel/Adam_1vf/dense_1/biasvf/dense_1/bias/Adamvf/dense_1/bias/Adam_1vf/dense_1/kernelvf/dense_1/kernel/Adamvf/dense_1/kernel/Adam_1vf/dense_2/biasvf/dense_2/bias/Adamvf/dense_2/bias/Adam_1vf/dense_2/kernelvf/dense_2/kernel/Adamvf/dense_2/kernel/Adam_1*R
dtypesH
F2D

save_9/control_dependencyIdentitysave_9/ShardedFilename^save_9/SaveV2*
_output_shapes
: *)
_class
loc:@save_9/ShardedFilename*
T0
Ł
-save_9/MergeV2Checkpoints/checkpoint_prefixesPacksave_9/ShardedFilename^save_9/control_dependency*
T0*

axis *
N*
_output_shapes
:

save_9/MergeV2CheckpointsMergeV2Checkpoints-save_9/MergeV2Checkpoints/checkpoint_prefixessave_9/Const*
delete_old_dirs(

save_9/IdentityIdentitysave_9/Const^save_9/MergeV2Checkpoints^save_9/control_dependency*
_output_shapes
: *
T0
ú
save_9/RestoreV2/tensor_namesConst*
dtype0*
_output_shapes
:D*¨
valueBDBbeta1_powerBbeta1_power_1Bbeta1_power_2Bbeta2_powerBbeta2_power_1Bbeta2_power_2Bpenalty/penalty_paramBpenalty/penalty_param/AdamBpenalty/penalty_param/Adam_1Bpenalty/penalty_param/Adam_2Bpenalty/penalty_param/Adam_3Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1B
pi/log_stdBpi/log_std/AdamBpi/log_std/Adam_1Bvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1
ó
!save_9/RestoreV2/shape_and_slicesConst*
_output_shapes
:D*
dtype0*
valueBDB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
ę
save_9/RestoreV2	RestoreV2save_9/Constsave_9/RestoreV2/tensor_names!save_9/RestoreV2/shape_and_slices*R
dtypesH
F2D*Ś
_output_shapes
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
Ş
save_9/AssignAssignbeta1_powersave_9/RestoreV2*(
_class
loc:@penalty/penalty_param*
_output_shapes
: *
T0*
use_locking(*
validate_shape(
°
save_9/Assign_1Assignbeta1_power_1save_9/RestoreV2:1*
T0*(
_class
loc:@penalty/penalty_param*
use_locking(*
validate_shape(*
_output_shapes
: 
¨
save_9/Assign_2Assignbeta1_power_2save_9/RestoreV2:2*
_output_shapes
: *
validate_shape(* 
_class
loc:@vc/dense/bias*
T0*
use_locking(
Ž
save_9/Assign_3Assignbeta2_powersave_9/RestoreV2:3*(
_class
loc:@penalty/penalty_param*
_output_shapes
: *
use_locking(*
T0*
validate_shape(
°
save_9/Assign_4Assignbeta2_power_1save_9/RestoreV2:4*
T0*
_output_shapes
: *
validate_shape(*
use_locking(*(
_class
loc:@penalty/penalty_param
¨
save_9/Assign_5Assignbeta2_power_2save_9/RestoreV2:5*
_output_shapes
: * 
_class
loc:@vc/dense/bias*
T0*
use_locking(*
validate_shape(
¸
save_9/Assign_6Assignpenalty/penalty_paramsave_9/RestoreV2:6*
T0*
validate_shape(*(
_class
loc:@penalty/penalty_param*
use_locking(*
_output_shapes
: 
˝
save_9/Assign_7Assignpenalty/penalty_param/Adamsave_9/RestoreV2:7*
T0*(
_class
loc:@penalty/penalty_param*
validate_shape(*
_output_shapes
: *
use_locking(
ż
save_9/Assign_8Assignpenalty/penalty_param/Adam_1save_9/RestoreV2:8*
_output_shapes
: *
validate_shape(*
use_locking(*(
_class
loc:@penalty/penalty_param*
T0
ż
save_9/Assign_9Assignpenalty/penalty_param/Adam_2save_9/RestoreV2:9*
use_locking(*
T0*(
_class
loc:@penalty/penalty_param*
_output_shapes
: *
validate_shape(
Á
save_9/Assign_10Assignpenalty/penalty_param/Adam_3save_9/RestoreV2:10*
use_locking(*
T0*
_output_shapes
: *
validate_shape(*(
_class
loc:@penalty/penalty_param
Ż
save_9/Assign_11Assignpi/dense/biassave_9/RestoreV2:11*
_output_shapes	
:*
validate_shape(* 
_class
loc:@pi/dense/bias*
T0*
use_locking(
´
save_9/Assign_12Assignpi/dense/bias/Adamsave_9/RestoreV2:12*
_output_shapes	
:*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
use_locking(
ś
save_9/Assign_13Assignpi/dense/bias/Adam_1save_9/RestoreV2:13*
validate_shape(* 
_class
loc:@pi/dense/bias*
_output_shapes	
:*
T0*
use_locking(
ˇ
save_9/Assign_14Assignpi/dense/kernelsave_9/RestoreV2:14*
T0*
validate_shape(*
_output_shapes
:	L*
use_locking(*"
_class
loc:@pi/dense/kernel
ź
save_9/Assign_15Assignpi/dense/kernel/Adamsave_9/RestoreV2:15*
use_locking(*
validate_shape(*
T0*
_output_shapes
:	L*"
_class
loc:@pi/dense/kernel
ž
save_9/Assign_16Assignpi/dense/kernel/Adam_1save_9/RestoreV2:16*
T0*
_output_shapes
:	L*
validate_shape(*"
_class
loc:@pi/dense/kernel*
use_locking(
ł
save_9/Assign_17Assignpi/dense_1/biassave_9/RestoreV2:17*
validate_shape(*
use_locking(*
_output_shapes	
:*
T0*"
_class
loc:@pi/dense_1/bias
¸
save_9/Assign_18Assignpi/dense_1/bias/Adamsave_9/RestoreV2:18*
_output_shapes	
:*
T0*
validate_shape(*
use_locking(*"
_class
loc:@pi/dense_1/bias
ş
save_9/Assign_19Assignpi/dense_1/bias/Adam_1save_9/RestoreV2:19*
T0*"
_class
loc:@pi/dense_1/bias*
validate_shape(*
_output_shapes	
:*
use_locking(
ź
save_9/Assign_20Assignpi/dense_1/kernelsave_9/RestoreV2:20* 
_output_shapes
:
*
T0*$
_class
loc:@pi/dense_1/kernel*
use_locking(*
validate_shape(
Á
save_9/Assign_21Assignpi/dense_1/kernel/Adamsave_9/RestoreV2:21* 
_output_shapes
:
*
T0*$
_class
loc:@pi/dense_1/kernel*
validate_shape(*
use_locking(
Ă
save_9/Assign_22Assignpi/dense_1/kernel/Adam_1save_9/RestoreV2:22*$
_class
loc:@pi/dense_1/kernel*
use_locking(*
validate_shape(*
T0* 
_output_shapes
:

˛
save_9/Assign_23Assignpi/dense_2/biassave_9/RestoreV2:23*"
_class
loc:@pi/dense_2/bias*
validate_shape(*
T0*
_output_shapes
:*
use_locking(
ˇ
save_9/Assign_24Assignpi/dense_2/bias/Adamsave_9/RestoreV2:24*
validate_shape(*
_output_shapes
:*
T0*
use_locking(*"
_class
loc:@pi/dense_2/bias
š
save_9/Assign_25Assignpi/dense_2/bias/Adam_1save_9/RestoreV2:25*
validate_shape(*
T0*"
_class
loc:@pi/dense_2/bias*
_output_shapes
:*
use_locking(
ť
save_9/Assign_26Assignpi/dense_2/kernelsave_9/RestoreV2:26*
T0*
validate_shape(*
_output_shapes
:	*$
_class
loc:@pi/dense_2/kernel*
use_locking(
Ŕ
save_9/Assign_27Assignpi/dense_2/kernel/Adamsave_9/RestoreV2:27*
T0*
use_locking(*
validate_shape(*
_output_shapes
:	*$
_class
loc:@pi/dense_2/kernel
Â
save_9/Assign_28Assignpi/dense_2/kernel/Adam_1save_9/RestoreV2:28*
T0*$
_class
loc:@pi/dense_2/kernel*
validate_shape(*
_output_shapes
:	*
use_locking(
¨
save_9/Assign_29Assign
pi/log_stdsave_9/RestoreV2:29*
use_locking(*
T0*
_output_shapes
:*
_class
loc:@pi/log_std*
validate_shape(
­
save_9/Assign_30Assignpi/log_std/Adamsave_9/RestoreV2:30*
validate_shape(*
T0*
_output_shapes
:*
use_locking(*
_class
loc:@pi/log_std
Ż
save_9/Assign_31Assignpi/log_std/Adam_1save_9/RestoreV2:31*
_class
loc:@pi/log_std*
T0*
_output_shapes
:*
use_locking(*
validate_shape(
Ż
save_9/Assign_32Assignvc/dense/biassave_9/RestoreV2:32*
use_locking(*
validate_shape(* 
_class
loc:@vc/dense/bias*
_output_shapes	
:*
T0
´
save_9/Assign_33Assignvc/dense/bias/Adamsave_9/RestoreV2:33*
validate_shape(*
T0*
use_locking(*
_output_shapes	
:* 
_class
loc:@vc/dense/bias
ś
save_9/Assign_34Assignvc/dense/bias/Adam_1save_9/RestoreV2:34*
T0* 
_class
loc:@vc/dense/bias*
validate_shape(*
_output_shapes	
:*
use_locking(
ˇ
save_9/Assign_35Assignvc/dense/kernelsave_9/RestoreV2:35*
_output_shapes
:	L*
use_locking(*
validate_shape(*"
_class
loc:@vc/dense/kernel*
T0
ź
save_9/Assign_36Assignvc/dense/kernel/Adamsave_9/RestoreV2:36*"
_class
loc:@vc/dense/kernel*
_output_shapes
:	L*
validate_shape(*
use_locking(*
T0
ž
save_9/Assign_37Assignvc/dense/kernel/Adam_1save_9/RestoreV2:37*"
_class
loc:@vc/dense/kernel*
T0*
use_locking(*
_output_shapes
:	L*
validate_shape(
ł
save_9/Assign_38Assignvc/dense_1/biassave_9/RestoreV2:38*"
_class
loc:@vc/dense_1/bias*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0
¸
save_9/Assign_39Assignvc/dense_1/bias/Adamsave_9/RestoreV2:39*
T0*
_output_shapes	
:*
use_locking(*
validate_shape(*"
_class
loc:@vc/dense_1/bias
ş
save_9/Assign_40Assignvc/dense_1/bias/Adam_1save_9/RestoreV2:40*
validate_shape(*
use_locking(*"
_class
loc:@vc/dense_1/bias*
_output_shapes	
:*
T0
ź
save_9/Assign_41Assignvc/dense_1/kernelsave_9/RestoreV2:41* 
_output_shapes
:
*
use_locking(*
validate_shape(*$
_class
loc:@vc/dense_1/kernel*
T0
Á
save_9/Assign_42Assignvc/dense_1/kernel/Adamsave_9/RestoreV2:42* 
_output_shapes
:
*$
_class
loc:@vc/dense_1/kernel*
T0*
use_locking(*
validate_shape(
Ă
save_9/Assign_43Assignvc/dense_1/kernel/Adam_1save_9/RestoreV2:43*
T0* 
_output_shapes
:
*
use_locking(*
validate_shape(*$
_class
loc:@vc/dense_1/kernel
˛
save_9/Assign_44Assignvc/dense_2/biassave_9/RestoreV2:44*
use_locking(*"
_class
loc:@vc/dense_2/bias*
_output_shapes
:*
validate_shape(*
T0
ˇ
save_9/Assign_45Assignvc/dense_2/bias/Adamsave_9/RestoreV2:45*
validate_shape(*"
_class
loc:@vc/dense_2/bias*
_output_shapes
:*
use_locking(*
T0
š
save_9/Assign_46Assignvc/dense_2/bias/Adam_1save_9/RestoreV2:46*
T0*
use_locking(*
validate_shape(*"
_class
loc:@vc/dense_2/bias*
_output_shapes
:
ť
save_9/Assign_47Assignvc/dense_2/kernelsave_9/RestoreV2:47*
_output_shapes
:	*
validate_shape(*
use_locking(*
T0*$
_class
loc:@vc/dense_2/kernel
Ŕ
save_9/Assign_48Assignvc/dense_2/kernel/Adamsave_9/RestoreV2:48*
validate_shape(*
_output_shapes
:	*
use_locking(*
T0*$
_class
loc:@vc/dense_2/kernel
Â
save_9/Assign_49Assignvc/dense_2/kernel/Adam_1save_9/RestoreV2:49*
validate_shape(*
T0*
use_locking(*
_output_shapes
:	*$
_class
loc:@vc/dense_2/kernel
Ż
save_9/Assign_50Assignvf/dense/biassave_9/RestoreV2:50*
validate_shape(*
use_locking(* 
_class
loc:@vf/dense/bias*
T0*
_output_shapes	
:
´
save_9/Assign_51Assignvf/dense/bias/Adamsave_9/RestoreV2:51*
validate_shape(*
T0*
_output_shapes	
:* 
_class
loc:@vf/dense/bias*
use_locking(
ś
save_9/Assign_52Assignvf/dense/bias/Adam_1save_9/RestoreV2:52*
T0* 
_class
loc:@vf/dense/bias*
use_locking(*
_output_shapes	
:*
validate_shape(
ˇ
save_9/Assign_53Assignvf/dense/kernelsave_9/RestoreV2:53*
_output_shapes
:	L*
T0*
use_locking(*"
_class
loc:@vf/dense/kernel*
validate_shape(
ź
save_9/Assign_54Assignvf/dense/kernel/Adamsave_9/RestoreV2:54*
_output_shapes
:	L*
T0*
use_locking(*
validate_shape(*"
_class
loc:@vf/dense/kernel
ž
save_9/Assign_55Assignvf/dense/kernel/Adam_1save_9/RestoreV2:55*
T0*
validate_shape(*
use_locking(*"
_class
loc:@vf/dense/kernel*
_output_shapes
:	L
ł
save_9/Assign_56Assignvf/dense_1/biassave_9/RestoreV2:56*
T0*
validate_shape(*
use_locking(*"
_class
loc:@vf/dense_1/bias*
_output_shapes	
:
¸
save_9/Assign_57Assignvf/dense_1/bias/Adamsave_9/RestoreV2:57*
T0*
_output_shapes	
:*"
_class
loc:@vf/dense_1/bias*
validate_shape(*
use_locking(
ş
save_9/Assign_58Assignvf/dense_1/bias/Adam_1save_9/RestoreV2:58*
use_locking(*
validate_shape(*"
_class
loc:@vf/dense_1/bias*
_output_shapes	
:*
T0
ź
save_9/Assign_59Assignvf/dense_1/kernelsave_9/RestoreV2:59* 
_output_shapes
:
*
T0*
validate_shape(*$
_class
loc:@vf/dense_1/kernel*
use_locking(
Á
save_9/Assign_60Assignvf/dense_1/kernel/Adamsave_9/RestoreV2:60*
T0* 
_output_shapes
:
*
use_locking(*
validate_shape(*$
_class
loc:@vf/dense_1/kernel
Ă
save_9/Assign_61Assignvf/dense_1/kernel/Adam_1save_9/RestoreV2:61*
T0*$
_class
loc:@vf/dense_1/kernel* 
_output_shapes
:
*
use_locking(*
validate_shape(
˛
save_9/Assign_62Assignvf/dense_2/biassave_9/RestoreV2:62*
use_locking(*
validate_shape(*"
_class
loc:@vf/dense_2/bias*
T0*
_output_shapes
:
ˇ
save_9/Assign_63Assignvf/dense_2/bias/Adamsave_9/RestoreV2:63*"
_class
loc:@vf/dense_2/bias*
T0*
validate_shape(*
_output_shapes
:*
use_locking(
š
save_9/Assign_64Assignvf/dense_2/bias/Adam_1save_9/RestoreV2:64*
validate_shape(*
use_locking(*
_output_shapes
:*"
_class
loc:@vf/dense_2/bias*
T0
ť
save_9/Assign_65Assignvf/dense_2/kernelsave_9/RestoreV2:65*$
_class
loc:@vf/dense_2/kernel*
_output_shapes
:	*
use_locking(*
T0*
validate_shape(
Ŕ
save_9/Assign_66Assignvf/dense_2/kernel/Adamsave_9/RestoreV2:66*$
_class
loc:@vf/dense_2/kernel*
use_locking(*
_output_shapes
:	*
T0*
validate_shape(
Â
save_9/Assign_67Assignvf/dense_2/kernel/Adam_1save_9/RestoreV2:67*
T0*
validate_shape(*
use_locking(*$
_class
loc:@vf/dense_2/kernel*
_output_shapes
:	


save_9/restore_shardNoOp^save_9/Assign^save_9/Assign_1^save_9/Assign_10^save_9/Assign_11^save_9/Assign_12^save_9/Assign_13^save_9/Assign_14^save_9/Assign_15^save_9/Assign_16^save_9/Assign_17^save_9/Assign_18^save_9/Assign_19^save_9/Assign_2^save_9/Assign_20^save_9/Assign_21^save_9/Assign_22^save_9/Assign_23^save_9/Assign_24^save_9/Assign_25^save_9/Assign_26^save_9/Assign_27^save_9/Assign_28^save_9/Assign_29^save_9/Assign_3^save_9/Assign_30^save_9/Assign_31^save_9/Assign_32^save_9/Assign_33^save_9/Assign_34^save_9/Assign_35^save_9/Assign_36^save_9/Assign_37^save_9/Assign_38^save_9/Assign_39^save_9/Assign_4^save_9/Assign_40^save_9/Assign_41^save_9/Assign_42^save_9/Assign_43^save_9/Assign_44^save_9/Assign_45^save_9/Assign_46^save_9/Assign_47^save_9/Assign_48^save_9/Assign_49^save_9/Assign_5^save_9/Assign_50^save_9/Assign_51^save_9/Assign_52^save_9/Assign_53^save_9/Assign_54^save_9/Assign_55^save_9/Assign_56^save_9/Assign_57^save_9/Assign_58^save_9/Assign_59^save_9/Assign_6^save_9/Assign_60^save_9/Assign_61^save_9/Assign_62^save_9/Assign_63^save_9/Assign_64^save_9/Assign_65^save_9/Assign_66^save_9/Assign_67^save_9/Assign_7^save_9/Assign_8^save_9/Assign_9
1
save_9/restore_allNoOp^save_9/restore_shard
\
save_10/filename/inputConst*
_output_shapes
: *
dtype0*
valueB Bmodel
t
save_10/filenamePlaceholderWithDefaultsave_10/filename/input*
_output_shapes
: *
dtype0*
shape: 
k
save_10/ConstPlaceholderWithDefaultsave_10/filename*
shape: *
dtype0*
_output_shapes
: 

save_10/StringJoin/inputs_1Const*
dtype0*
_output_shapes
: *<
value3B1 B+_temp_f9954d57e47145129c06fce908157103/part
~
save_10/StringJoin
StringJoinsave_10/Constsave_10/StringJoin/inputs_1*
	separator *
_output_shapes
: *
N
T
save_10/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
_
save_10/ShardedFilename/shardConst*
_output_shapes
: *
dtype0*
value	B : 

save_10/ShardedFilenameShardedFilenamesave_10/StringJoinsave_10/ShardedFilename/shardsave_10/num_shards*
_output_shapes
: 
ř
save_10/SaveV2/tensor_namesConst*
_output_shapes
:D*
dtype0*¨
valueBDBbeta1_powerBbeta1_power_1Bbeta1_power_2Bbeta2_powerBbeta2_power_1Bbeta2_power_2Bpenalty/penalty_paramBpenalty/penalty_param/AdamBpenalty/penalty_param/Adam_1Bpenalty/penalty_param/Adam_2Bpenalty/penalty_param/Adam_3Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1B
pi/log_stdBpi/log_std/AdamBpi/log_std/Adam_1Bvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1
ń
save_10/SaveV2/shape_and_slicesConst*
valueBDB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:D
Ö
save_10/SaveV2SaveV2save_10/ShardedFilenamesave_10/SaveV2/tensor_namessave_10/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta1_power_2beta2_powerbeta2_power_1beta2_power_2penalty/penalty_parampenalty/penalty_param/Adampenalty/penalty_param/Adam_1penalty/penalty_param/Adam_2penalty/penalty_param/Adam_3pi/dense/biaspi/dense/bias/Adampi/dense/bias/Adam_1pi/dense/kernelpi/dense/kernel/Adampi/dense/kernel/Adam_1pi/dense_1/biaspi/dense_1/bias/Adampi/dense_1/bias/Adam_1pi/dense_1/kernelpi/dense_1/kernel/Adampi/dense_1/kernel/Adam_1pi/dense_2/biaspi/dense_2/bias/Adampi/dense_2/bias/Adam_1pi/dense_2/kernelpi/dense_2/kernel/Adampi/dense_2/kernel/Adam_1
pi/log_stdpi/log_std/Adampi/log_std/Adam_1vc/dense/biasvc/dense/bias/Adamvc/dense/bias/Adam_1vc/dense/kernelvc/dense/kernel/Adamvc/dense/kernel/Adam_1vc/dense_1/biasvc/dense_1/bias/Adamvc/dense_1/bias/Adam_1vc/dense_1/kernelvc/dense_1/kernel/Adamvc/dense_1/kernel/Adam_1vc/dense_2/biasvc/dense_2/bias/Adamvc/dense_2/bias/Adam_1vc/dense_2/kernelvc/dense_2/kernel/Adamvc/dense_2/kernel/Adam_1vf/dense/biasvf/dense/bias/Adamvf/dense/bias/Adam_1vf/dense/kernelvf/dense/kernel/Adamvf/dense/kernel/Adam_1vf/dense_1/biasvf/dense_1/bias/Adamvf/dense_1/bias/Adam_1vf/dense_1/kernelvf/dense_1/kernel/Adamvf/dense_1/kernel/Adam_1vf/dense_2/biasvf/dense_2/bias/Adamvf/dense_2/bias/Adam_1vf/dense_2/kernelvf/dense_2/kernel/Adamvf/dense_2/kernel/Adam_1*R
dtypesH
F2D

save_10/control_dependencyIdentitysave_10/ShardedFilename^save_10/SaveV2*
T0*
_output_shapes
: **
_class 
loc:@save_10/ShardedFilename
Ś
.save_10/MergeV2Checkpoints/checkpoint_prefixesPacksave_10/ShardedFilename^save_10/control_dependency*
N*
_output_shapes
:*
T0*

axis 

save_10/MergeV2CheckpointsMergeV2Checkpoints.save_10/MergeV2Checkpoints/checkpoint_prefixessave_10/Const*
delete_old_dirs(

save_10/IdentityIdentitysave_10/Const^save_10/MergeV2Checkpoints^save_10/control_dependency*
T0*
_output_shapes
: 
ű
save_10/RestoreV2/tensor_namesConst*
_output_shapes
:D*¨
valueBDBbeta1_powerBbeta1_power_1Bbeta1_power_2Bbeta2_powerBbeta2_power_1Bbeta2_power_2Bpenalty/penalty_paramBpenalty/penalty_param/AdamBpenalty/penalty_param/Adam_1Bpenalty/penalty_param/Adam_2Bpenalty/penalty_param/Adam_3Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1B
pi/log_stdBpi/log_std/AdamBpi/log_std/Adam_1Bvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1*
dtype0
ô
"save_10/RestoreV2/shape_and_slicesConst*
valueBDB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
_output_shapes
:D*
dtype0
î
save_10/RestoreV2	RestoreV2save_10/Constsave_10/RestoreV2/tensor_names"save_10/RestoreV2/shape_and_slices*Ś
_output_shapes
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*R
dtypesH
F2D
Ź
save_10/AssignAssignbeta1_powersave_10/RestoreV2*
validate_shape(*
T0*
use_locking(*
_output_shapes
: *(
_class
loc:@penalty/penalty_param
˛
save_10/Assign_1Assignbeta1_power_1save_10/RestoreV2:1*
use_locking(*
_output_shapes
: *
T0*(
_class
loc:@penalty/penalty_param*
validate_shape(
Ş
save_10/Assign_2Assignbeta1_power_2save_10/RestoreV2:2*
use_locking(*
validate_shape(*
T0* 
_class
loc:@vc/dense/bias*
_output_shapes
: 
°
save_10/Assign_3Assignbeta2_powersave_10/RestoreV2:3*
use_locking(*
validate_shape(*
T0*(
_class
loc:@penalty/penalty_param*
_output_shapes
: 
˛
save_10/Assign_4Assignbeta2_power_1save_10/RestoreV2:4*
use_locking(*
T0*(
_class
loc:@penalty/penalty_param*
_output_shapes
: *
validate_shape(
Ş
save_10/Assign_5Assignbeta2_power_2save_10/RestoreV2:5* 
_class
loc:@vc/dense/bias*
_output_shapes
: *
use_locking(*
T0*
validate_shape(
ş
save_10/Assign_6Assignpenalty/penalty_paramsave_10/RestoreV2:6*
T0*
validate_shape(*
use_locking(*
_output_shapes
: *(
_class
loc:@penalty/penalty_param
ż
save_10/Assign_7Assignpenalty/penalty_param/Adamsave_10/RestoreV2:7*
validate_shape(*
_output_shapes
: *(
_class
loc:@penalty/penalty_param*
T0*
use_locking(
Á
save_10/Assign_8Assignpenalty/penalty_param/Adam_1save_10/RestoreV2:8*
validate_shape(*(
_class
loc:@penalty/penalty_param*
use_locking(*
_output_shapes
: *
T0
Á
save_10/Assign_9Assignpenalty/penalty_param/Adam_2save_10/RestoreV2:9*
validate_shape(*
use_locking(*
T0*(
_class
loc:@penalty/penalty_param*
_output_shapes
: 
Ă
save_10/Assign_10Assignpenalty/penalty_param/Adam_3save_10/RestoreV2:10*
validate_shape(*(
_class
loc:@penalty/penalty_param*
_output_shapes
: *
use_locking(*
T0
ą
save_10/Assign_11Assignpi/dense/biassave_10/RestoreV2:11*
_output_shapes	
:*
T0* 
_class
loc:@pi/dense/bias*
use_locking(*
validate_shape(
ś
save_10/Assign_12Assignpi/dense/bias/Adamsave_10/RestoreV2:12* 
_class
loc:@pi/dense/bias*
validate_shape(*
T0*
_output_shapes	
:*
use_locking(
¸
save_10/Assign_13Assignpi/dense/bias/Adam_1save_10/RestoreV2:13*
validate_shape(*
_output_shapes	
:* 
_class
loc:@pi/dense/bias*
use_locking(*
T0
š
save_10/Assign_14Assignpi/dense/kernelsave_10/RestoreV2:14*
validate_shape(*
_output_shapes
:	L*"
_class
loc:@pi/dense/kernel*
use_locking(*
T0
ž
save_10/Assign_15Assignpi/dense/kernel/Adamsave_10/RestoreV2:15*
validate_shape(*
use_locking(*"
_class
loc:@pi/dense/kernel*
T0*
_output_shapes
:	L
Ŕ
save_10/Assign_16Assignpi/dense/kernel/Adam_1save_10/RestoreV2:16*
T0*"
_class
loc:@pi/dense/kernel*
validate_shape(*
_output_shapes
:	L*
use_locking(
ľ
save_10/Assign_17Assignpi/dense_1/biassave_10/RestoreV2:17*
use_locking(*
validate_shape(*"
_class
loc:@pi/dense_1/bias*
T0*
_output_shapes	
:
ş
save_10/Assign_18Assignpi/dense_1/bias/Adamsave_10/RestoreV2:18*
validate_shape(*
T0*
use_locking(*"
_class
loc:@pi/dense_1/bias*
_output_shapes	
:
ź
save_10/Assign_19Assignpi/dense_1/bias/Adam_1save_10/RestoreV2:19*
validate_shape(*"
_class
loc:@pi/dense_1/bias*
T0*
_output_shapes	
:*
use_locking(
ž
save_10/Assign_20Assignpi/dense_1/kernelsave_10/RestoreV2:20* 
_output_shapes
:
*
T0*
validate_shape(*
use_locking(*$
_class
loc:@pi/dense_1/kernel
Ă
save_10/Assign_21Assignpi/dense_1/kernel/Adamsave_10/RestoreV2:21*$
_class
loc:@pi/dense_1/kernel*
T0*
validate_shape(* 
_output_shapes
:
*
use_locking(
Ĺ
save_10/Assign_22Assignpi/dense_1/kernel/Adam_1save_10/RestoreV2:22*
T0*
validate_shape(* 
_output_shapes
:
*$
_class
loc:@pi/dense_1/kernel*
use_locking(
´
save_10/Assign_23Assignpi/dense_2/biassave_10/RestoreV2:23*"
_class
loc:@pi/dense_2/bias*
_output_shapes
:*
validate_shape(*
use_locking(*
T0
š
save_10/Assign_24Assignpi/dense_2/bias/Adamsave_10/RestoreV2:24*
_output_shapes
:*"
_class
loc:@pi/dense_2/bias*
use_locking(*
T0*
validate_shape(
ť
save_10/Assign_25Assignpi/dense_2/bias/Adam_1save_10/RestoreV2:25*
T0*
_output_shapes
:*
validate_shape(*"
_class
loc:@pi/dense_2/bias*
use_locking(
˝
save_10/Assign_26Assignpi/dense_2/kernelsave_10/RestoreV2:26*
use_locking(*
_output_shapes
:	*
T0*
validate_shape(*$
_class
loc:@pi/dense_2/kernel
Â
save_10/Assign_27Assignpi/dense_2/kernel/Adamsave_10/RestoreV2:27*
T0*
_output_shapes
:	*
use_locking(*$
_class
loc:@pi/dense_2/kernel*
validate_shape(
Ä
save_10/Assign_28Assignpi/dense_2/kernel/Adam_1save_10/RestoreV2:28*
validate_shape(*
T0*
use_locking(*$
_class
loc:@pi/dense_2/kernel*
_output_shapes
:	
Ş
save_10/Assign_29Assign
pi/log_stdsave_10/RestoreV2:29*
_class
loc:@pi/log_std*
use_locking(*
validate_shape(*
T0*
_output_shapes
:
Ż
save_10/Assign_30Assignpi/log_std/Adamsave_10/RestoreV2:30*
_output_shapes
:*
T0*
use_locking(*
validate_shape(*
_class
loc:@pi/log_std
ą
save_10/Assign_31Assignpi/log_std/Adam_1save_10/RestoreV2:31*
validate_shape(*
T0*
_class
loc:@pi/log_std*
_output_shapes
:*
use_locking(
ą
save_10/Assign_32Assignvc/dense/biassave_10/RestoreV2:32* 
_class
loc:@vc/dense/bias*
use_locking(*
_output_shapes	
:*
T0*
validate_shape(
ś
save_10/Assign_33Assignvc/dense/bias/Adamsave_10/RestoreV2:33*
_output_shapes	
:* 
_class
loc:@vc/dense/bias*
use_locking(*
T0*
validate_shape(
¸
save_10/Assign_34Assignvc/dense/bias/Adam_1save_10/RestoreV2:34*
_output_shapes	
:*
T0* 
_class
loc:@vc/dense/bias*
use_locking(*
validate_shape(
š
save_10/Assign_35Assignvc/dense/kernelsave_10/RestoreV2:35*
validate_shape(*
T0*
use_locking(*"
_class
loc:@vc/dense/kernel*
_output_shapes
:	L
ž
save_10/Assign_36Assignvc/dense/kernel/Adamsave_10/RestoreV2:36*"
_class
loc:@vc/dense/kernel*
use_locking(*
T0*
validate_shape(*
_output_shapes
:	L
Ŕ
save_10/Assign_37Assignvc/dense/kernel/Adam_1save_10/RestoreV2:37*
validate_shape(*
use_locking(*"
_class
loc:@vc/dense/kernel*
_output_shapes
:	L*
T0
ľ
save_10/Assign_38Assignvc/dense_1/biassave_10/RestoreV2:38*
validate_shape(*
T0*"
_class
loc:@vc/dense_1/bias*
use_locking(*
_output_shapes	
:
ş
save_10/Assign_39Assignvc/dense_1/bias/Adamsave_10/RestoreV2:39*"
_class
loc:@vc/dense_1/bias*
validate_shape(*
use_locking(*
T0*
_output_shapes	
:
ź
save_10/Assign_40Assignvc/dense_1/bias/Adam_1save_10/RestoreV2:40*
T0*
_output_shapes	
:*
validate_shape(*
use_locking(*"
_class
loc:@vc/dense_1/bias
ž
save_10/Assign_41Assignvc/dense_1/kernelsave_10/RestoreV2:41*
use_locking(*$
_class
loc:@vc/dense_1/kernel* 
_output_shapes
:
*
validate_shape(*
T0
Ă
save_10/Assign_42Assignvc/dense_1/kernel/Adamsave_10/RestoreV2:42*$
_class
loc:@vc/dense_1/kernel*
T0*
validate_shape(* 
_output_shapes
:
*
use_locking(
Ĺ
save_10/Assign_43Assignvc/dense_1/kernel/Adam_1save_10/RestoreV2:43*
T0*
validate_shape(* 
_output_shapes
:
*
use_locking(*$
_class
loc:@vc/dense_1/kernel
´
save_10/Assign_44Assignvc/dense_2/biassave_10/RestoreV2:44*
T0*
_output_shapes
:*
use_locking(*
validate_shape(*"
_class
loc:@vc/dense_2/bias
š
save_10/Assign_45Assignvc/dense_2/bias/Adamsave_10/RestoreV2:45*
use_locking(*
T0*"
_class
loc:@vc/dense_2/bias*
_output_shapes
:*
validate_shape(
ť
save_10/Assign_46Assignvc/dense_2/bias/Adam_1save_10/RestoreV2:46*
use_locking(*
validate_shape(*
_output_shapes
:*"
_class
loc:@vc/dense_2/bias*
T0
˝
save_10/Assign_47Assignvc/dense_2/kernelsave_10/RestoreV2:47*
validate_shape(*
use_locking(*
T0*$
_class
loc:@vc/dense_2/kernel*
_output_shapes
:	
Â
save_10/Assign_48Assignvc/dense_2/kernel/Adamsave_10/RestoreV2:48*
_output_shapes
:	*
use_locking(*$
_class
loc:@vc/dense_2/kernel*
T0*
validate_shape(
Ä
save_10/Assign_49Assignvc/dense_2/kernel/Adam_1save_10/RestoreV2:49*
T0*
validate_shape(*
_output_shapes
:	*
use_locking(*$
_class
loc:@vc/dense_2/kernel
ą
save_10/Assign_50Assignvf/dense/biassave_10/RestoreV2:50*
validate_shape(*
T0*
use_locking(* 
_class
loc:@vf/dense/bias*
_output_shapes	
:
ś
save_10/Assign_51Assignvf/dense/bias/Adamsave_10/RestoreV2:51*
validate_shape(*
use_locking(*
_output_shapes	
:* 
_class
loc:@vf/dense/bias*
T0
¸
save_10/Assign_52Assignvf/dense/bias/Adam_1save_10/RestoreV2:52*
_output_shapes	
:*
T0*
validate_shape(* 
_class
loc:@vf/dense/bias*
use_locking(
š
save_10/Assign_53Assignvf/dense/kernelsave_10/RestoreV2:53*
use_locking(*
T0*
validate_shape(*"
_class
loc:@vf/dense/kernel*
_output_shapes
:	L
ž
save_10/Assign_54Assignvf/dense/kernel/Adamsave_10/RestoreV2:54*
use_locking(*
validate_shape(*
T0*
_output_shapes
:	L*"
_class
loc:@vf/dense/kernel
Ŕ
save_10/Assign_55Assignvf/dense/kernel/Adam_1save_10/RestoreV2:55*
validate_shape(*
_output_shapes
:	L*
use_locking(*
T0*"
_class
loc:@vf/dense/kernel
ľ
save_10/Assign_56Assignvf/dense_1/biassave_10/RestoreV2:56*"
_class
loc:@vf/dense_1/bias*
validate_shape(*
_output_shapes	
:*
T0*
use_locking(
ş
save_10/Assign_57Assignvf/dense_1/bias/Adamsave_10/RestoreV2:57*
_output_shapes	
:*
validate_shape(*"
_class
loc:@vf/dense_1/bias*
use_locking(*
T0
ź
save_10/Assign_58Assignvf/dense_1/bias/Adam_1save_10/RestoreV2:58*
T0*
_output_shapes	
:*
validate_shape(*
use_locking(*"
_class
loc:@vf/dense_1/bias
ž
save_10/Assign_59Assignvf/dense_1/kernelsave_10/RestoreV2:59*
validate_shape(*
use_locking(*$
_class
loc:@vf/dense_1/kernel*
T0* 
_output_shapes
:

Ă
save_10/Assign_60Assignvf/dense_1/kernel/Adamsave_10/RestoreV2:60*
T0*
use_locking(*$
_class
loc:@vf/dense_1/kernel*
validate_shape(* 
_output_shapes
:

Ĺ
save_10/Assign_61Assignvf/dense_1/kernel/Adam_1save_10/RestoreV2:61*
validate_shape(*
use_locking(*$
_class
loc:@vf/dense_1/kernel* 
_output_shapes
:
*
T0
´
save_10/Assign_62Assignvf/dense_2/biassave_10/RestoreV2:62*
validate_shape(*
use_locking(*"
_class
loc:@vf/dense_2/bias*
T0*
_output_shapes
:
š
save_10/Assign_63Assignvf/dense_2/bias/Adamsave_10/RestoreV2:63*
T0*
validate_shape(*"
_class
loc:@vf/dense_2/bias*
use_locking(*
_output_shapes
:
ť
save_10/Assign_64Assignvf/dense_2/bias/Adam_1save_10/RestoreV2:64*
_output_shapes
:*
validate_shape(*
use_locking(*"
_class
loc:@vf/dense_2/bias*
T0
˝
save_10/Assign_65Assignvf/dense_2/kernelsave_10/RestoreV2:65*
_output_shapes
:	*
validate_shape(*$
_class
loc:@vf/dense_2/kernel*
T0*
use_locking(
Â
save_10/Assign_66Assignvf/dense_2/kernel/Adamsave_10/RestoreV2:66*
_output_shapes
:	*
use_locking(*
T0*
validate_shape(*$
_class
loc:@vf/dense_2/kernel
Ä
save_10/Assign_67Assignvf/dense_2/kernel/Adam_1save_10/RestoreV2:67*$
_class
loc:@vf/dense_2/kernel*
_output_shapes
:	*
use_locking(*
validate_shape(*
T0
á

save_10/restore_shardNoOp^save_10/Assign^save_10/Assign_1^save_10/Assign_10^save_10/Assign_11^save_10/Assign_12^save_10/Assign_13^save_10/Assign_14^save_10/Assign_15^save_10/Assign_16^save_10/Assign_17^save_10/Assign_18^save_10/Assign_19^save_10/Assign_2^save_10/Assign_20^save_10/Assign_21^save_10/Assign_22^save_10/Assign_23^save_10/Assign_24^save_10/Assign_25^save_10/Assign_26^save_10/Assign_27^save_10/Assign_28^save_10/Assign_29^save_10/Assign_3^save_10/Assign_30^save_10/Assign_31^save_10/Assign_32^save_10/Assign_33^save_10/Assign_34^save_10/Assign_35^save_10/Assign_36^save_10/Assign_37^save_10/Assign_38^save_10/Assign_39^save_10/Assign_4^save_10/Assign_40^save_10/Assign_41^save_10/Assign_42^save_10/Assign_43^save_10/Assign_44^save_10/Assign_45^save_10/Assign_46^save_10/Assign_47^save_10/Assign_48^save_10/Assign_49^save_10/Assign_5^save_10/Assign_50^save_10/Assign_51^save_10/Assign_52^save_10/Assign_53^save_10/Assign_54^save_10/Assign_55^save_10/Assign_56^save_10/Assign_57^save_10/Assign_58^save_10/Assign_59^save_10/Assign_6^save_10/Assign_60^save_10/Assign_61^save_10/Assign_62^save_10/Assign_63^save_10/Assign_64^save_10/Assign_65^save_10/Assign_66^save_10/Assign_67^save_10/Assign_7^save_10/Assign_8^save_10/Assign_9
3
save_10/restore_allNoOp^save_10/restore_shard
\
save_11/filename/inputConst*
dtype0*
_output_shapes
: *
valueB Bmodel
t
save_11/filenamePlaceholderWithDefaultsave_11/filename/input*
dtype0*
_output_shapes
: *
shape: 
k
save_11/ConstPlaceholderWithDefaultsave_11/filename*
shape: *
_output_shapes
: *
dtype0

save_11/StringJoin/inputs_1Const*
dtype0*<
value3B1 B+_temp_2f1c9d4a49c8494497e372584819093b/part*
_output_shapes
: 
~
save_11/StringJoin
StringJoinsave_11/Constsave_11/StringJoin/inputs_1*
	separator *
N*
_output_shapes
: 
T
save_11/num_shardsConst*
value	B :*
_output_shapes
: *
dtype0
_
save_11/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 

save_11/ShardedFilenameShardedFilenamesave_11/StringJoinsave_11/ShardedFilename/shardsave_11/num_shards*
_output_shapes
: 
ř
save_11/SaveV2/tensor_namesConst*¨
valueBDBbeta1_powerBbeta1_power_1Bbeta1_power_2Bbeta2_powerBbeta2_power_1Bbeta2_power_2Bpenalty/penalty_paramBpenalty/penalty_param/AdamBpenalty/penalty_param/Adam_1Bpenalty/penalty_param/Adam_2Bpenalty/penalty_param/Adam_3Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1B
pi/log_stdBpi/log_std/AdamBpi/log_std/Adam_1Bvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1*
_output_shapes
:D*
dtype0
ń
save_11/SaveV2/shape_and_slicesConst*
dtype0*
valueBDB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
_output_shapes
:D
Ö
save_11/SaveV2SaveV2save_11/ShardedFilenamesave_11/SaveV2/tensor_namessave_11/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta1_power_2beta2_powerbeta2_power_1beta2_power_2penalty/penalty_parampenalty/penalty_param/Adampenalty/penalty_param/Adam_1penalty/penalty_param/Adam_2penalty/penalty_param/Adam_3pi/dense/biaspi/dense/bias/Adampi/dense/bias/Adam_1pi/dense/kernelpi/dense/kernel/Adampi/dense/kernel/Adam_1pi/dense_1/biaspi/dense_1/bias/Adampi/dense_1/bias/Adam_1pi/dense_1/kernelpi/dense_1/kernel/Adampi/dense_1/kernel/Adam_1pi/dense_2/biaspi/dense_2/bias/Adampi/dense_2/bias/Adam_1pi/dense_2/kernelpi/dense_2/kernel/Adampi/dense_2/kernel/Adam_1
pi/log_stdpi/log_std/Adampi/log_std/Adam_1vc/dense/biasvc/dense/bias/Adamvc/dense/bias/Adam_1vc/dense/kernelvc/dense/kernel/Adamvc/dense/kernel/Adam_1vc/dense_1/biasvc/dense_1/bias/Adamvc/dense_1/bias/Adam_1vc/dense_1/kernelvc/dense_1/kernel/Adamvc/dense_1/kernel/Adam_1vc/dense_2/biasvc/dense_2/bias/Adamvc/dense_2/bias/Adam_1vc/dense_2/kernelvc/dense_2/kernel/Adamvc/dense_2/kernel/Adam_1vf/dense/biasvf/dense/bias/Adamvf/dense/bias/Adam_1vf/dense/kernelvf/dense/kernel/Adamvf/dense/kernel/Adam_1vf/dense_1/biasvf/dense_1/bias/Adamvf/dense_1/bias/Adam_1vf/dense_1/kernelvf/dense_1/kernel/Adamvf/dense_1/kernel/Adam_1vf/dense_2/biasvf/dense_2/bias/Adamvf/dense_2/bias/Adam_1vf/dense_2/kernelvf/dense_2/kernel/Adamvf/dense_2/kernel/Adam_1*R
dtypesH
F2D

save_11/control_dependencyIdentitysave_11/ShardedFilename^save_11/SaveV2*
T0**
_class 
loc:@save_11/ShardedFilename*
_output_shapes
: 
Ś
.save_11/MergeV2Checkpoints/checkpoint_prefixesPacksave_11/ShardedFilename^save_11/control_dependency*
N*
_output_shapes
:*

axis *
T0

save_11/MergeV2CheckpointsMergeV2Checkpoints.save_11/MergeV2Checkpoints/checkpoint_prefixessave_11/Const*
delete_old_dirs(

save_11/IdentityIdentitysave_11/Const^save_11/MergeV2Checkpoints^save_11/control_dependency*
T0*
_output_shapes
: 
ű
save_11/RestoreV2/tensor_namesConst*¨
valueBDBbeta1_powerBbeta1_power_1Bbeta1_power_2Bbeta2_powerBbeta2_power_1Bbeta2_power_2Bpenalty/penalty_paramBpenalty/penalty_param/AdamBpenalty/penalty_param/Adam_1Bpenalty/penalty_param/Adam_2Bpenalty/penalty_param/Adam_3Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1B
pi/log_stdBpi/log_std/AdamBpi/log_std/Adam_1Bvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1*
_output_shapes
:D*
dtype0
ô
"save_11/RestoreV2/shape_and_slicesConst*
_output_shapes
:D*
dtype0*
valueBDB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
î
save_11/RestoreV2	RestoreV2save_11/Constsave_11/RestoreV2/tensor_names"save_11/RestoreV2/shape_and_slices*R
dtypesH
F2D*Ś
_output_shapes
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
Ź
save_11/AssignAssignbeta1_powersave_11/RestoreV2*(
_class
loc:@penalty/penalty_param*
use_locking(*
validate_shape(*
_output_shapes
: *
T0
˛
save_11/Assign_1Assignbeta1_power_1save_11/RestoreV2:1*(
_class
loc:@penalty/penalty_param*
validate_shape(*
T0*
use_locking(*
_output_shapes
: 
Ş
save_11/Assign_2Assignbeta1_power_2save_11/RestoreV2:2*
_output_shapes
: *
use_locking(* 
_class
loc:@vc/dense/bias*
T0*
validate_shape(
°
save_11/Assign_3Assignbeta2_powersave_11/RestoreV2:3*
T0*
use_locking(*
_output_shapes
: *(
_class
loc:@penalty/penalty_param*
validate_shape(
˛
save_11/Assign_4Assignbeta2_power_1save_11/RestoreV2:4*
use_locking(*
_output_shapes
: *
validate_shape(*(
_class
loc:@penalty/penalty_param*
T0
Ş
save_11/Assign_5Assignbeta2_power_2save_11/RestoreV2:5*
T0*
use_locking(*
_output_shapes
: *
validate_shape(* 
_class
loc:@vc/dense/bias
ş
save_11/Assign_6Assignpenalty/penalty_paramsave_11/RestoreV2:6*
validate_shape(*(
_class
loc:@penalty/penalty_param*
_output_shapes
: *
use_locking(*
T0
ż
save_11/Assign_7Assignpenalty/penalty_param/Adamsave_11/RestoreV2:7*
T0*
validate_shape(*
_output_shapes
: *
use_locking(*(
_class
loc:@penalty/penalty_param
Á
save_11/Assign_8Assignpenalty/penalty_param/Adam_1save_11/RestoreV2:8*
T0*
validate_shape(*
_output_shapes
: *
use_locking(*(
_class
loc:@penalty/penalty_param
Á
save_11/Assign_9Assignpenalty/penalty_param/Adam_2save_11/RestoreV2:9*
_output_shapes
: *
validate_shape(*
T0*(
_class
loc:@penalty/penalty_param*
use_locking(
Ă
save_11/Assign_10Assignpenalty/penalty_param/Adam_3save_11/RestoreV2:10*
validate_shape(*
_output_shapes
: *(
_class
loc:@penalty/penalty_param*
T0*
use_locking(
ą
save_11/Assign_11Assignpi/dense/biassave_11/RestoreV2:11* 
_class
loc:@pi/dense/bias*
_output_shapes	
:*
validate_shape(*
use_locking(*
T0
ś
save_11/Assign_12Assignpi/dense/bias/Adamsave_11/RestoreV2:12*
_output_shapes	
:* 
_class
loc:@pi/dense/bias*
T0*
validate_shape(*
use_locking(
¸
save_11/Assign_13Assignpi/dense/bias/Adam_1save_11/RestoreV2:13*
T0*
validate_shape(* 
_class
loc:@pi/dense/bias*
_output_shapes	
:*
use_locking(
š
save_11/Assign_14Assignpi/dense/kernelsave_11/RestoreV2:14*
_output_shapes
:	L*"
_class
loc:@pi/dense/kernel*
validate_shape(*
use_locking(*
T0
ž
save_11/Assign_15Assignpi/dense/kernel/Adamsave_11/RestoreV2:15*
use_locking(*
validate_shape(*"
_class
loc:@pi/dense/kernel*
_output_shapes
:	L*
T0
Ŕ
save_11/Assign_16Assignpi/dense/kernel/Adam_1save_11/RestoreV2:16*
validate_shape(*
T0*
use_locking(*
_output_shapes
:	L*"
_class
loc:@pi/dense/kernel
ľ
save_11/Assign_17Assignpi/dense_1/biassave_11/RestoreV2:17*"
_class
loc:@pi/dense_1/bias*
T0*
use_locking(*
validate_shape(*
_output_shapes	
:
ş
save_11/Assign_18Assignpi/dense_1/bias/Adamsave_11/RestoreV2:18*
validate_shape(*
T0*
_output_shapes	
:*"
_class
loc:@pi/dense_1/bias*
use_locking(
ź
save_11/Assign_19Assignpi/dense_1/bias/Adam_1save_11/RestoreV2:19*
T0*"
_class
loc:@pi/dense_1/bias*
use_locking(*
validate_shape(*
_output_shapes	
:
ž
save_11/Assign_20Assignpi/dense_1/kernelsave_11/RestoreV2:20*$
_class
loc:@pi/dense_1/kernel* 
_output_shapes
:
*
validate_shape(*
use_locking(*
T0
Ă
save_11/Assign_21Assignpi/dense_1/kernel/Adamsave_11/RestoreV2:21* 
_output_shapes
:
*
T0*
validate_shape(*$
_class
loc:@pi/dense_1/kernel*
use_locking(
Ĺ
save_11/Assign_22Assignpi/dense_1/kernel/Adam_1save_11/RestoreV2:22*
use_locking(* 
_output_shapes
:
*
validate_shape(*$
_class
loc:@pi/dense_1/kernel*
T0
´
save_11/Assign_23Assignpi/dense_2/biassave_11/RestoreV2:23*
use_locking(*
T0*
validate_shape(*"
_class
loc:@pi/dense_2/bias*
_output_shapes
:
š
save_11/Assign_24Assignpi/dense_2/bias/Adamsave_11/RestoreV2:24*
T0*"
_class
loc:@pi/dense_2/bias*
_output_shapes
:*
use_locking(*
validate_shape(
ť
save_11/Assign_25Assignpi/dense_2/bias/Adam_1save_11/RestoreV2:25*
T0*
_output_shapes
:*
use_locking(*
validate_shape(*"
_class
loc:@pi/dense_2/bias
˝
save_11/Assign_26Assignpi/dense_2/kernelsave_11/RestoreV2:26*
validate_shape(*
_output_shapes
:	*$
_class
loc:@pi/dense_2/kernel*
use_locking(*
T0
Â
save_11/Assign_27Assignpi/dense_2/kernel/Adamsave_11/RestoreV2:27*
validate_shape(*
_output_shapes
:	*
use_locking(*$
_class
loc:@pi/dense_2/kernel*
T0
Ä
save_11/Assign_28Assignpi/dense_2/kernel/Adam_1save_11/RestoreV2:28*
validate_shape(*
_output_shapes
:	*
use_locking(*$
_class
loc:@pi/dense_2/kernel*
T0
Ş
save_11/Assign_29Assign
pi/log_stdsave_11/RestoreV2:29*
_class
loc:@pi/log_std*
T0*
validate_shape(*
use_locking(*
_output_shapes
:
Ż
save_11/Assign_30Assignpi/log_std/Adamsave_11/RestoreV2:30*
validate_shape(*
T0*
use_locking(*
_class
loc:@pi/log_std*
_output_shapes
:
ą
save_11/Assign_31Assignpi/log_std/Adam_1save_11/RestoreV2:31*
_class
loc:@pi/log_std*
T0*
_output_shapes
:*
validate_shape(*
use_locking(
ą
save_11/Assign_32Assignvc/dense/biassave_11/RestoreV2:32*
T0*
_output_shapes	
:* 
_class
loc:@vc/dense/bias*
use_locking(*
validate_shape(
ś
save_11/Assign_33Assignvc/dense/bias/Adamsave_11/RestoreV2:33*
use_locking(* 
_class
loc:@vc/dense/bias*
T0*
validate_shape(*
_output_shapes	
:
¸
save_11/Assign_34Assignvc/dense/bias/Adam_1save_11/RestoreV2:34*
_output_shapes	
:*
validate_shape(*
use_locking(*
T0* 
_class
loc:@vc/dense/bias
š
save_11/Assign_35Assignvc/dense/kernelsave_11/RestoreV2:35*
use_locking(*"
_class
loc:@vc/dense/kernel*
T0*
validate_shape(*
_output_shapes
:	L
ž
save_11/Assign_36Assignvc/dense/kernel/Adamsave_11/RestoreV2:36*
use_locking(*"
_class
loc:@vc/dense/kernel*
T0*
_output_shapes
:	L*
validate_shape(
Ŕ
save_11/Assign_37Assignvc/dense/kernel/Adam_1save_11/RestoreV2:37*
_output_shapes
:	L*"
_class
loc:@vc/dense/kernel*
validate_shape(*
T0*
use_locking(
ľ
save_11/Assign_38Assignvc/dense_1/biassave_11/RestoreV2:38*
T0*"
_class
loc:@vc/dense_1/bias*
validate_shape(*
_output_shapes	
:*
use_locking(
ş
save_11/Assign_39Assignvc/dense_1/bias/Adamsave_11/RestoreV2:39*
_output_shapes	
:*"
_class
loc:@vc/dense_1/bias*
validate_shape(*
T0*
use_locking(
ź
save_11/Assign_40Assignvc/dense_1/bias/Adam_1save_11/RestoreV2:40*
use_locking(*
_output_shapes	
:*"
_class
loc:@vc/dense_1/bias*
validate_shape(*
T0
ž
save_11/Assign_41Assignvc/dense_1/kernelsave_11/RestoreV2:41*
T0*$
_class
loc:@vc/dense_1/kernel* 
_output_shapes
:
*
validate_shape(*
use_locking(
Ă
save_11/Assign_42Assignvc/dense_1/kernel/Adamsave_11/RestoreV2:42*
validate_shape(*
use_locking(*
T0* 
_output_shapes
:
*$
_class
loc:@vc/dense_1/kernel
Ĺ
save_11/Assign_43Assignvc/dense_1/kernel/Adam_1save_11/RestoreV2:43* 
_output_shapes
:
*
validate_shape(*$
_class
loc:@vc/dense_1/kernel*
T0*
use_locking(
´
save_11/Assign_44Assignvc/dense_2/biassave_11/RestoreV2:44*
T0*
use_locking(*
validate_shape(*
_output_shapes
:*"
_class
loc:@vc/dense_2/bias
š
save_11/Assign_45Assignvc/dense_2/bias/Adamsave_11/RestoreV2:45*
T0*
_output_shapes
:*
validate_shape(*"
_class
loc:@vc/dense_2/bias*
use_locking(
ť
save_11/Assign_46Assignvc/dense_2/bias/Adam_1save_11/RestoreV2:46*
T0*"
_class
loc:@vc/dense_2/bias*
use_locking(*
_output_shapes
:*
validate_shape(
˝
save_11/Assign_47Assignvc/dense_2/kernelsave_11/RestoreV2:47*$
_class
loc:@vc/dense_2/kernel*
_output_shapes
:	*
validate_shape(*
use_locking(*
T0
Â
save_11/Assign_48Assignvc/dense_2/kernel/Adamsave_11/RestoreV2:48*
use_locking(*
validate_shape(*
_output_shapes
:	*$
_class
loc:@vc/dense_2/kernel*
T0
Ä
save_11/Assign_49Assignvc/dense_2/kernel/Adam_1save_11/RestoreV2:49*
_output_shapes
:	*
validate_shape(*$
_class
loc:@vc/dense_2/kernel*
use_locking(*
T0
ą
save_11/Assign_50Assignvf/dense/biassave_11/RestoreV2:50* 
_class
loc:@vf/dense/bias*
T0*
validate_shape(*
use_locking(*
_output_shapes	
:
ś
save_11/Assign_51Assignvf/dense/bias/Adamsave_11/RestoreV2:51*
T0*
use_locking(* 
_class
loc:@vf/dense/bias*
_output_shapes	
:*
validate_shape(
¸
save_11/Assign_52Assignvf/dense/bias/Adam_1save_11/RestoreV2:52*
validate_shape(*
use_locking(*
T0* 
_class
loc:@vf/dense/bias*
_output_shapes	
:
š
save_11/Assign_53Assignvf/dense/kernelsave_11/RestoreV2:53*
use_locking(*
T0*"
_class
loc:@vf/dense/kernel*
validate_shape(*
_output_shapes
:	L
ž
save_11/Assign_54Assignvf/dense/kernel/Adamsave_11/RestoreV2:54*
use_locking(*"
_class
loc:@vf/dense/kernel*
_output_shapes
:	L*
validate_shape(*
T0
Ŕ
save_11/Assign_55Assignvf/dense/kernel/Adam_1save_11/RestoreV2:55*
validate_shape(*"
_class
loc:@vf/dense/kernel*
_output_shapes
:	L*
use_locking(*
T0
ľ
save_11/Assign_56Assignvf/dense_1/biassave_11/RestoreV2:56*"
_class
loc:@vf/dense_1/bias*
T0*
use_locking(*
validate_shape(*
_output_shapes	
:
ş
save_11/Assign_57Assignvf/dense_1/bias/Adamsave_11/RestoreV2:57*
use_locking(*
_output_shapes	
:*
T0*"
_class
loc:@vf/dense_1/bias*
validate_shape(
ź
save_11/Assign_58Assignvf/dense_1/bias/Adam_1save_11/RestoreV2:58*
validate_shape(*"
_class
loc:@vf/dense_1/bias*
use_locking(*
T0*
_output_shapes	
:
ž
save_11/Assign_59Assignvf/dense_1/kernelsave_11/RestoreV2:59*
validate_shape(* 
_output_shapes
:
*$
_class
loc:@vf/dense_1/kernel*
use_locking(*
T0
Ă
save_11/Assign_60Assignvf/dense_1/kernel/Adamsave_11/RestoreV2:60*
validate_shape(*
T0* 
_output_shapes
:
*$
_class
loc:@vf/dense_1/kernel*
use_locking(
Ĺ
save_11/Assign_61Assignvf/dense_1/kernel/Adam_1save_11/RestoreV2:61*
T0*
use_locking(*$
_class
loc:@vf/dense_1/kernel*
validate_shape(* 
_output_shapes
:

´
save_11/Assign_62Assignvf/dense_2/biassave_11/RestoreV2:62*
use_locking(*"
_class
loc:@vf/dense_2/bias*
T0*
validate_shape(*
_output_shapes
:
š
save_11/Assign_63Assignvf/dense_2/bias/Adamsave_11/RestoreV2:63*"
_class
loc:@vf/dense_2/bias*
validate_shape(*
T0*
use_locking(*
_output_shapes
:
ť
save_11/Assign_64Assignvf/dense_2/bias/Adam_1save_11/RestoreV2:64*
use_locking(*"
_class
loc:@vf/dense_2/bias*
T0*
_output_shapes
:*
validate_shape(
˝
save_11/Assign_65Assignvf/dense_2/kernelsave_11/RestoreV2:65*
T0*
_output_shapes
:	*
validate_shape(*$
_class
loc:@vf/dense_2/kernel*
use_locking(
Â
save_11/Assign_66Assignvf/dense_2/kernel/Adamsave_11/RestoreV2:66*
_output_shapes
:	*
validate_shape(*
T0*
use_locking(*$
_class
loc:@vf/dense_2/kernel
Ä
save_11/Assign_67Assignvf/dense_2/kernel/Adam_1save_11/RestoreV2:67*$
_class
loc:@vf/dense_2/kernel*
_output_shapes
:	*
validate_shape(*
T0*
use_locking(
á

save_11/restore_shardNoOp^save_11/Assign^save_11/Assign_1^save_11/Assign_10^save_11/Assign_11^save_11/Assign_12^save_11/Assign_13^save_11/Assign_14^save_11/Assign_15^save_11/Assign_16^save_11/Assign_17^save_11/Assign_18^save_11/Assign_19^save_11/Assign_2^save_11/Assign_20^save_11/Assign_21^save_11/Assign_22^save_11/Assign_23^save_11/Assign_24^save_11/Assign_25^save_11/Assign_26^save_11/Assign_27^save_11/Assign_28^save_11/Assign_29^save_11/Assign_3^save_11/Assign_30^save_11/Assign_31^save_11/Assign_32^save_11/Assign_33^save_11/Assign_34^save_11/Assign_35^save_11/Assign_36^save_11/Assign_37^save_11/Assign_38^save_11/Assign_39^save_11/Assign_4^save_11/Assign_40^save_11/Assign_41^save_11/Assign_42^save_11/Assign_43^save_11/Assign_44^save_11/Assign_45^save_11/Assign_46^save_11/Assign_47^save_11/Assign_48^save_11/Assign_49^save_11/Assign_5^save_11/Assign_50^save_11/Assign_51^save_11/Assign_52^save_11/Assign_53^save_11/Assign_54^save_11/Assign_55^save_11/Assign_56^save_11/Assign_57^save_11/Assign_58^save_11/Assign_59^save_11/Assign_6^save_11/Assign_60^save_11/Assign_61^save_11/Assign_62^save_11/Assign_63^save_11/Assign_64^save_11/Assign_65^save_11/Assign_66^save_11/Assign_67^save_11/Assign_7^save_11/Assign_8^save_11/Assign_9
3
save_11/restore_allNoOp^save_11/restore_shard
\
save_12/filename/inputConst*
_output_shapes
: *
valueB Bmodel*
dtype0
t
save_12/filenamePlaceholderWithDefaultsave_12/filename/input*
shape: *
_output_shapes
: *
dtype0
k
save_12/ConstPlaceholderWithDefaultsave_12/filename*
shape: *
dtype0*
_output_shapes
: 

save_12/StringJoin/inputs_1Const*
dtype0*
_output_shapes
: *<
value3B1 B+_temp_ae79ebea5611485cbde720fe9a52ae54/part
~
save_12/StringJoin
StringJoinsave_12/Constsave_12/StringJoin/inputs_1*
	separator *
N*
_output_shapes
: 
T
save_12/num_shardsConst*
dtype0*
value	B :*
_output_shapes
: 
_
save_12/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 

save_12/ShardedFilenameShardedFilenamesave_12/StringJoinsave_12/ShardedFilename/shardsave_12/num_shards*
_output_shapes
: 
ř
save_12/SaveV2/tensor_namesConst*¨
valueBDBbeta1_powerBbeta1_power_1Bbeta1_power_2Bbeta2_powerBbeta2_power_1Bbeta2_power_2Bpenalty/penalty_paramBpenalty/penalty_param/AdamBpenalty/penalty_param/Adam_1Bpenalty/penalty_param/Adam_2Bpenalty/penalty_param/Adam_3Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1B
pi/log_stdBpi/log_std/AdamBpi/log_std/Adam_1Bvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1*
dtype0*
_output_shapes
:D
ń
save_12/SaveV2/shape_and_slicesConst*
valueBDB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:D
Ö
save_12/SaveV2SaveV2save_12/ShardedFilenamesave_12/SaveV2/tensor_namessave_12/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta1_power_2beta2_powerbeta2_power_1beta2_power_2penalty/penalty_parampenalty/penalty_param/Adampenalty/penalty_param/Adam_1penalty/penalty_param/Adam_2penalty/penalty_param/Adam_3pi/dense/biaspi/dense/bias/Adampi/dense/bias/Adam_1pi/dense/kernelpi/dense/kernel/Adampi/dense/kernel/Adam_1pi/dense_1/biaspi/dense_1/bias/Adampi/dense_1/bias/Adam_1pi/dense_1/kernelpi/dense_1/kernel/Adampi/dense_1/kernel/Adam_1pi/dense_2/biaspi/dense_2/bias/Adampi/dense_2/bias/Adam_1pi/dense_2/kernelpi/dense_2/kernel/Adampi/dense_2/kernel/Adam_1
pi/log_stdpi/log_std/Adampi/log_std/Adam_1vc/dense/biasvc/dense/bias/Adamvc/dense/bias/Adam_1vc/dense/kernelvc/dense/kernel/Adamvc/dense/kernel/Adam_1vc/dense_1/biasvc/dense_1/bias/Adamvc/dense_1/bias/Adam_1vc/dense_1/kernelvc/dense_1/kernel/Adamvc/dense_1/kernel/Adam_1vc/dense_2/biasvc/dense_2/bias/Adamvc/dense_2/bias/Adam_1vc/dense_2/kernelvc/dense_2/kernel/Adamvc/dense_2/kernel/Adam_1vf/dense/biasvf/dense/bias/Adamvf/dense/bias/Adam_1vf/dense/kernelvf/dense/kernel/Adamvf/dense/kernel/Adam_1vf/dense_1/biasvf/dense_1/bias/Adamvf/dense_1/bias/Adam_1vf/dense_1/kernelvf/dense_1/kernel/Adamvf/dense_1/kernel/Adam_1vf/dense_2/biasvf/dense_2/bias/Adamvf/dense_2/bias/Adam_1vf/dense_2/kernelvf/dense_2/kernel/Adamvf/dense_2/kernel/Adam_1*R
dtypesH
F2D

save_12/control_dependencyIdentitysave_12/ShardedFilename^save_12/SaveV2**
_class 
loc:@save_12/ShardedFilename*
T0*
_output_shapes
: 
Ś
.save_12/MergeV2Checkpoints/checkpoint_prefixesPacksave_12/ShardedFilename^save_12/control_dependency*
T0*

axis *
N*
_output_shapes
:

save_12/MergeV2CheckpointsMergeV2Checkpoints.save_12/MergeV2Checkpoints/checkpoint_prefixessave_12/Const*
delete_old_dirs(

save_12/IdentityIdentitysave_12/Const^save_12/MergeV2Checkpoints^save_12/control_dependency*
T0*
_output_shapes
: 
ű
save_12/RestoreV2/tensor_namesConst*
_output_shapes
:D*¨
valueBDBbeta1_powerBbeta1_power_1Bbeta1_power_2Bbeta2_powerBbeta2_power_1Bbeta2_power_2Bpenalty/penalty_paramBpenalty/penalty_param/AdamBpenalty/penalty_param/Adam_1Bpenalty/penalty_param/Adam_2Bpenalty/penalty_param/Adam_3Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1B
pi/log_stdBpi/log_std/AdamBpi/log_std/Adam_1Bvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1*
dtype0
ô
"save_12/RestoreV2/shape_and_slicesConst*
dtype0*
valueBDB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
_output_shapes
:D
î
save_12/RestoreV2	RestoreV2save_12/Constsave_12/RestoreV2/tensor_names"save_12/RestoreV2/shape_and_slices*Ś
_output_shapes
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*R
dtypesH
F2D
Ź
save_12/AssignAssignbeta1_powersave_12/RestoreV2*
validate_shape(*
use_locking(*
T0*(
_class
loc:@penalty/penalty_param*
_output_shapes
: 
˛
save_12/Assign_1Assignbeta1_power_1save_12/RestoreV2:1*(
_class
loc:@penalty/penalty_param*
use_locking(*
validate_shape(*
T0*
_output_shapes
: 
Ş
save_12/Assign_2Assignbeta1_power_2save_12/RestoreV2:2*
use_locking(*
_output_shapes
: *
T0*
validate_shape(* 
_class
loc:@vc/dense/bias
°
save_12/Assign_3Assignbeta2_powersave_12/RestoreV2:3*
_output_shapes
: *
T0*
validate_shape(*(
_class
loc:@penalty/penalty_param*
use_locking(
˛
save_12/Assign_4Assignbeta2_power_1save_12/RestoreV2:4*(
_class
loc:@penalty/penalty_param*
use_locking(*
_output_shapes
: *
validate_shape(*
T0
Ş
save_12/Assign_5Assignbeta2_power_2save_12/RestoreV2:5*
T0*
use_locking(* 
_class
loc:@vc/dense/bias*
_output_shapes
: *
validate_shape(
ş
save_12/Assign_6Assignpenalty/penalty_paramsave_12/RestoreV2:6*
validate_shape(*(
_class
loc:@penalty/penalty_param*
_output_shapes
: *
T0*
use_locking(
ż
save_12/Assign_7Assignpenalty/penalty_param/Adamsave_12/RestoreV2:7*(
_class
loc:@penalty/penalty_param*
_output_shapes
: *
use_locking(*
T0*
validate_shape(
Á
save_12/Assign_8Assignpenalty/penalty_param/Adam_1save_12/RestoreV2:8*(
_class
loc:@penalty/penalty_param*
T0*
validate_shape(*
_output_shapes
: *
use_locking(
Á
save_12/Assign_9Assignpenalty/penalty_param/Adam_2save_12/RestoreV2:9*
_output_shapes
: *
T0*
validate_shape(*
use_locking(*(
_class
loc:@penalty/penalty_param
Ă
save_12/Assign_10Assignpenalty/penalty_param/Adam_3save_12/RestoreV2:10*
_output_shapes
: *(
_class
loc:@penalty/penalty_param*
validate_shape(*
T0*
use_locking(
ą
save_12/Assign_11Assignpi/dense/biassave_12/RestoreV2:11*
_output_shapes	
:*
validate_shape(* 
_class
loc:@pi/dense/bias*
T0*
use_locking(
ś
save_12/Assign_12Assignpi/dense/bias/Adamsave_12/RestoreV2:12*
use_locking(* 
_class
loc:@pi/dense/bias*
T0*
validate_shape(*
_output_shapes	
:
¸
save_12/Assign_13Assignpi/dense/bias/Adam_1save_12/RestoreV2:13*
T0*
validate_shape(* 
_class
loc:@pi/dense/bias*
_output_shapes	
:*
use_locking(
š
save_12/Assign_14Assignpi/dense/kernelsave_12/RestoreV2:14*
_output_shapes
:	L*
T0*
use_locking(*"
_class
loc:@pi/dense/kernel*
validate_shape(
ž
save_12/Assign_15Assignpi/dense/kernel/Adamsave_12/RestoreV2:15*
T0*
validate_shape(*
use_locking(*
_output_shapes
:	L*"
_class
loc:@pi/dense/kernel
Ŕ
save_12/Assign_16Assignpi/dense/kernel/Adam_1save_12/RestoreV2:16*
validate_shape(*
T0*
_output_shapes
:	L*
use_locking(*"
_class
loc:@pi/dense/kernel
ľ
save_12/Assign_17Assignpi/dense_1/biassave_12/RestoreV2:17*
_output_shapes	
:*
use_locking(*
T0*"
_class
loc:@pi/dense_1/bias*
validate_shape(
ş
save_12/Assign_18Assignpi/dense_1/bias/Adamsave_12/RestoreV2:18*
validate_shape(*
use_locking(*"
_class
loc:@pi/dense_1/bias*
T0*
_output_shapes	
:
ź
save_12/Assign_19Assignpi/dense_1/bias/Adam_1save_12/RestoreV2:19*
use_locking(*
T0*"
_class
loc:@pi/dense_1/bias*
_output_shapes	
:*
validate_shape(
ž
save_12/Assign_20Assignpi/dense_1/kernelsave_12/RestoreV2:20*$
_class
loc:@pi/dense_1/kernel*
validate_shape(* 
_output_shapes
:
*
T0*
use_locking(
Ă
save_12/Assign_21Assignpi/dense_1/kernel/Adamsave_12/RestoreV2:21*
T0*
validate_shape(*
use_locking(*$
_class
loc:@pi/dense_1/kernel* 
_output_shapes
:

Ĺ
save_12/Assign_22Assignpi/dense_1/kernel/Adam_1save_12/RestoreV2:22*
T0*
use_locking(*
validate_shape(* 
_output_shapes
:
*$
_class
loc:@pi/dense_1/kernel
´
save_12/Assign_23Assignpi/dense_2/biassave_12/RestoreV2:23*
use_locking(*
validate_shape(*
T0*"
_class
loc:@pi/dense_2/bias*
_output_shapes
:
š
save_12/Assign_24Assignpi/dense_2/bias/Adamsave_12/RestoreV2:24*"
_class
loc:@pi/dense_2/bias*
T0*
validate_shape(*
_output_shapes
:*
use_locking(
ť
save_12/Assign_25Assignpi/dense_2/bias/Adam_1save_12/RestoreV2:25*
validate_shape(*
use_locking(*
T0*"
_class
loc:@pi/dense_2/bias*
_output_shapes
:
˝
save_12/Assign_26Assignpi/dense_2/kernelsave_12/RestoreV2:26*$
_class
loc:@pi/dense_2/kernel*
T0*
use_locking(*
_output_shapes
:	*
validate_shape(
Â
save_12/Assign_27Assignpi/dense_2/kernel/Adamsave_12/RestoreV2:27*
validate_shape(*
_output_shapes
:	*
use_locking(*
T0*$
_class
loc:@pi/dense_2/kernel
Ä
save_12/Assign_28Assignpi/dense_2/kernel/Adam_1save_12/RestoreV2:28*$
_class
loc:@pi/dense_2/kernel*
validate_shape(*
use_locking(*
T0*
_output_shapes
:	
Ş
save_12/Assign_29Assign
pi/log_stdsave_12/RestoreV2:29*
validate_shape(*
_output_shapes
:*
_class
loc:@pi/log_std*
T0*
use_locking(
Ż
save_12/Assign_30Assignpi/log_std/Adamsave_12/RestoreV2:30*
use_locking(*
validate_shape(*
_class
loc:@pi/log_std*
_output_shapes
:*
T0
ą
save_12/Assign_31Assignpi/log_std/Adam_1save_12/RestoreV2:31*
_output_shapes
:*
T0*
_class
loc:@pi/log_std*
validate_shape(*
use_locking(
ą
save_12/Assign_32Assignvc/dense/biassave_12/RestoreV2:32*
use_locking(*
_output_shapes	
:* 
_class
loc:@vc/dense/bias*
T0*
validate_shape(
ś
save_12/Assign_33Assignvc/dense/bias/Adamsave_12/RestoreV2:33*
validate_shape(*
_output_shapes	
:*
T0*
use_locking(* 
_class
loc:@vc/dense/bias
¸
save_12/Assign_34Assignvc/dense/bias/Adam_1save_12/RestoreV2:34*
T0*
_output_shapes	
:* 
_class
loc:@vc/dense/bias*
validate_shape(*
use_locking(
š
save_12/Assign_35Assignvc/dense/kernelsave_12/RestoreV2:35*
use_locking(*
validate_shape(*
_output_shapes
:	L*
T0*"
_class
loc:@vc/dense/kernel
ž
save_12/Assign_36Assignvc/dense/kernel/Adamsave_12/RestoreV2:36*
use_locking(*
_output_shapes
:	L*
validate_shape(*
T0*"
_class
loc:@vc/dense/kernel
Ŕ
save_12/Assign_37Assignvc/dense/kernel/Adam_1save_12/RestoreV2:37*
use_locking(*
_output_shapes
:	L*
T0*
validate_shape(*"
_class
loc:@vc/dense/kernel
ľ
save_12/Assign_38Assignvc/dense_1/biassave_12/RestoreV2:38*
use_locking(*
validate_shape(*
T0*
_output_shapes	
:*"
_class
loc:@vc/dense_1/bias
ş
save_12/Assign_39Assignvc/dense_1/bias/Adamsave_12/RestoreV2:39*"
_class
loc:@vc/dense_1/bias*
use_locking(*
_output_shapes	
:*
validate_shape(*
T0
ź
save_12/Assign_40Assignvc/dense_1/bias/Adam_1save_12/RestoreV2:40*
validate_shape(*
T0*"
_class
loc:@vc/dense_1/bias*
use_locking(*
_output_shapes	
:
ž
save_12/Assign_41Assignvc/dense_1/kernelsave_12/RestoreV2:41*
T0*$
_class
loc:@vc/dense_1/kernel*
validate_shape(*
use_locking(* 
_output_shapes
:

Ă
save_12/Assign_42Assignvc/dense_1/kernel/Adamsave_12/RestoreV2:42*
use_locking(* 
_output_shapes
:
*
validate_shape(*
T0*$
_class
loc:@vc/dense_1/kernel
Ĺ
save_12/Assign_43Assignvc/dense_1/kernel/Adam_1save_12/RestoreV2:43*
use_locking(*
validate_shape(* 
_output_shapes
:
*$
_class
loc:@vc/dense_1/kernel*
T0
´
save_12/Assign_44Assignvc/dense_2/biassave_12/RestoreV2:44*
_output_shapes
:*
T0*
validate_shape(*"
_class
loc:@vc/dense_2/bias*
use_locking(
š
save_12/Assign_45Assignvc/dense_2/bias/Adamsave_12/RestoreV2:45*
validate_shape(*
_output_shapes
:*"
_class
loc:@vc/dense_2/bias*
use_locking(*
T0
ť
save_12/Assign_46Assignvc/dense_2/bias/Adam_1save_12/RestoreV2:46*
validate_shape(*"
_class
loc:@vc/dense_2/bias*
_output_shapes
:*
T0*
use_locking(
˝
save_12/Assign_47Assignvc/dense_2/kernelsave_12/RestoreV2:47*
T0*
use_locking(*
validate_shape(*
_output_shapes
:	*$
_class
loc:@vc/dense_2/kernel
Â
save_12/Assign_48Assignvc/dense_2/kernel/Adamsave_12/RestoreV2:48*
use_locking(*
_output_shapes
:	*
T0*
validate_shape(*$
_class
loc:@vc/dense_2/kernel
Ä
save_12/Assign_49Assignvc/dense_2/kernel/Adam_1save_12/RestoreV2:49*$
_class
loc:@vc/dense_2/kernel*
_output_shapes
:	*
validate_shape(*
use_locking(*
T0
ą
save_12/Assign_50Assignvf/dense/biassave_12/RestoreV2:50* 
_class
loc:@vf/dense/bias*
_output_shapes	
:*
use_locking(*
T0*
validate_shape(
ś
save_12/Assign_51Assignvf/dense/bias/Adamsave_12/RestoreV2:51* 
_class
loc:@vf/dense/bias*
validate_shape(*
T0*
use_locking(*
_output_shapes	
:
¸
save_12/Assign_52Assignvf/dense/bias/Adam_1save_12/RestoreV2:52* 
_class
loc:@vf/dense/bias*
T0*
_output_shapes	
:*
use_locking(*
validate_shape(
š
save_12/Assign_53Assignvf/dense/kernelsave_12/RestoreV2:53*"
_class
loc:@vf/dense/kernel*
T0*
_output_shapes
:	L*
validate_shape(*
use_locking(
ž
save_12/Assign_54Assignvf/dense/kernel/Adamsave_12/RestoreV2:54*
use_locking(*
_output_shapes
:	L*"
_class
loc:@vf/dense/kernel*
T0*
validate_shape(
Ŕ
save_12/Assign_55Assignvf/dense/kernel/Adam_1save_12/RestoreV2:55*
use_locking(*
_output_shapes
:	L*"
_class
loc:@vf/dense/kernel*
T0*
validate_shape(
ľ
save_12/Assign_56Assignvf/dense_1/biassave_12/RestoreV2:56*
use_locking(*"
_class
loc:@vf/dense_1/bias*
_output_shapes	
:*
T0*
validate_shape(
ş
save_12/Assign_57Assignvf/dense_1/bias/Adamsave_12/RestoreV2:57*
T0*"
_class
loc:@vf/dense_1/bias*
_output_shapes	
:*
use_locking(*
validate_shape(
ź
save_12/Assign_58Assignvf/dense_1/bias/Adam_1save_12/RestoreV2:58*
T0*
validate_shape(*
use_locking(*"
_class
loc:@vf/dense_1/bias*
_output_shapes	
:
ž
save_12/Assign_59Assignvf/dense_1/kernelsave_12/RestoreV2:59*$
_class
loc:@vf/dense_1/kernel*
use_locking(*
validate_shape(* 
_output_shapes
:
*
T0
Ă
save_12/Assign_60Assignvf/dense_1/kernel/Adamsave_12/RestoreV2:60*
validate_shape(* 
_output_shapes
:
*$
_class
loc:@vf/dense_1/kernel*
T0*
use_locking(
Ĺ
save_12/Assign_61Assignvf/dense_1/kernel/Adam_1save_12/RestoreV2:61*
use_locking(*
validate_shape(* 
_output_shapes
:
*$
_class
loc:@vf/dense_1/kernel*
T0
´
save_12/Assign_62Assignvf/dense_2/biassave_12/RestoreV2:62*
_output_shapes
:*
T0*
validate_shape(*
use_locking(*"
_class
loc:@vf/dense_2/bias
š
save_12/Assign_63Assignvf/dense_2/bias/Adamsave_12/RestoreV2:63*"
_class
loc:@vf/dense_2/bias*
use_locking(*
T0*
_output_shapes
:*
validate_shape(
ť
save_12/Assign_64Assignvf/dense_2/bias/Adam_1save_12/RestoreV2:64*
T0*"
_class
loc:@vf/dense_2/bias*
_output_shapes
:*
use_locking(*
validate_shape(
˝
save_12/Assign_65Assignvf/dense_2/kernelsave_12/RestoreV2:65*
use_locking(*
_output_shapes
:	*
T0*$
_class
loc:@vf/dense_2/kernel*
validate_shape(
Â
save_12/Assign_66Assignvf/dense_2/kernel/Adamsave_12/RestoreV2:66*
T0*
use_locking(*$
_class
loc:@vf/dense_2/kernel*
_output_shapes
:	*
validate_shape(
Ä
save_12/Assign_67Assignvf/dense_2/kernel/Adam_1save_12/RestoreV2:67*
_output_shapes
:	*
use_locking(*
validate_shape(*
T0*$
_class
loc:@vf/dense_2/kernel
á

save_12/restore_shardNoOp^save_12/Assign^save_12/Assign_1^save_12/Assign_10^save_12/Assign_11^save_12/Assign_12^save_12/Assign_13^save_12/Assign_14^save_12/Assign_15^save_12/Assign_16^save_12/Assign_17^save_12/Assign_18^save_12/Assign_19^save_12/Assign_2^save_12/Assign_20^save_12/Assign_21^save_12/Assign_22^save_12/Assign_23^save_12/Assign_24^save_12/Assign_25^save_12/Assign_26^save_12/Assign_27^save_12/Assign_28^save_12/Assign_29^save_12/Assign_3^save_12/Assign_30^save_12/Assign_31^save_12/Assign_32^save_12/Assign_33^save_12/Assign_34^save_12/Assign_35^save_12/Assign_36^save_12/Assign_37^save_12/Assign_38^save_12/Assign_39^save_12/Assign_4^save_12/Assign_40^save_12/Assign_41^save_12/Assign_42^save_12/Assign_43^save_12/Assign_44^save_12/Assign_45^save_12/Assign_46^save_12/Assign_47^save_12/Assign_48^save_12/Assign_49^save_12/Assign_5^save_12/Assign_50^save_12/Assign_51^save_12/Assign_52^save_12/Assign_53^save_12/Assign_54^save_12/Assign_55^save_12/Assign_56^save_12/Assign_57^save_12/Assign_58^save_12/Assign_59^save_12/Assign_6^save_12/Assign_60^save_12/Assign_61^save_12/Assign_62^save_12/Assign_63^save_12/Assign_64^save_12/Assign_65^save_12/Assign_66^save_12/Assign_67^save_12/Assign_7^save_12/Assign_8^save_12/Assign_9
3
save_12/restore_allNoOp^save_12/restore_shard
\
save_13/filename/inputConst*
dtype0*
_output_shapes
: *
valueB Bmodel
t
save_13/filenamePlaceholderWithDefaultsave_13/filename/input*
shape: *
dtype0*
_output_shapes
: 
k
save_13/ConstPlaceholderWithDefaultsave_13/filename*
_output_shapes
: *
shape: *
dtype0

save_13/StringJoin/inputs_1Const*
_output_shapes
: *<
value3B1 B+_temp_76002d5c3c0e4aa18de21f7dd15efb23/part*
dtype0
~
save_13/StringJoin
StringJoinsave_13/Constsave_13/StringJoin/inputs_1*
_output_shapes
: *
N*
	separator 
T
save_13/num_shardsConst*
_output_shapes
: *
dtype0*
value	B :
_
save_13/ShardedFilename/shardConst*
value	B : *
_output_shapes
: *
dtype0

save_13/ShardedFilenameShardedFilenamesave_13/StringJoinsave_13/ShardedFilename/shardsave_13/num_shards*
_output_shapes
: 
ř
save_13/SaveV2/tensor_namesConst*
dtype0*¨
valueBDBbeta1_powerBbeta1_power_1Bbeta1_power_2Bbeta2_powerBbeta2_power_1Bbeta2_power_2Bpenalty/penalty_paramBpenalty/penalty_param/AdamBpenalty/penalty_param/Adam_1Bpenalty/penalty_param/Adam_2Bpenalty/penalty_param/Adam_3Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1B
pi/log_stdBpi/log_std/AdamBpi/log_std/Adam_1Bvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1*
_output_shapes
:D
ń
save_13/SaveV2/shape_and_slicesConst*
dtype0*
valueBDB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
_output_shapes
:D
Ö
save_13/SaveV2SaveV2save_13/ShardedFilenamesave_13/SaveV2/tensor_namessave_13/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta1_power_2beta2_powerbeta2_power_1beta2_power_2penalty/penalty_parampenalty/penalty_param/Adampenalty/penalty_param/Adam_1penalty/penalty_param/Adam_2penalty/penalty_param/Adam_3pi/dense/biaspi/dense/bias/Adampi/dense/bias/Adam_1pi/dense/kernelpi/dense/kernel/Adampi/dense/kernel/Adam_1pi/dense_1/biaspi/dense_1/bias/Adampi/dense_1/bias/Adam_1pi/dense_1/kernelpi/dense_1/kernel/Adampi/dense_1/kernel/Adam_1pi/dense_2/biaspi/dense_2/bias/Adampi/dense_2/bias/Adam_1pi/dense_2/kernelpi/dense_2/kernel/Adampi/dense_2/kernel/Adam_1
pi/log_stdpi/log_std/Adampi/log_std/Adam_1vc/dense/biasvc/dense/bias/Adamvc/dense/bias/Adam_1vc/dense/kernelvc/dense/kernel/Adamvc/dense/kernel/Adam_1vc/dense_1/biasvc/dense_1/bias/Adamvc/dense_1/bias/Adam_1vc/dense_1/kernelvc/dense_1/kernel/Adamvc/dense_1/kernel/Adam_1vc/dense_2/biasvc/dense_2/bias/Adamvc/dense_2/bias/Adam_1vc/dense_2/kernelvc/dense_2/kernel/Adamvc/dense_2/kernel/Adam_1vf/dense/biasvf/dense/bias/Adamvf/dense/bias/Adam_1vf/dense/kernelvf/dense/kernel/Adamvf/dense/kernel/Adam_1vf/dense_1/biasvf/dense_1/bias/Adamvf/dense_1/bias/Adam_1vf/dense_1/kernelvf/dense_1/kernel/Adamvf/dense_1/kernel/Adam_1vf/dense_2/biasvf/dense_2/bias/Adamvf/dense_2/bias/Adam_1vf/dense_2/kernelvf/dense_2/kernel/Adamvf/dense_2/kernel/Adam_1*R
dtypesH
F2D

save_13/control_dependencyIdentitysave_13/ShardedFilename^save_13/SaveV2**
_class 
loc:@save_13/ShardedFilename*
_output_shapes
: *
T0
Ś
.save_13/MergeV2Checkpoints/checkpoint_prefixesPacksave_13/ShardedFilename^save_13/control_dependency*
_output_shapes
:*
N*
T0*

axis 

save_13/MergeV2CheckpointsMergeV2Checkpoints.save_13/MergeV2Checkpoints/checkpoint_prefixessave_13/Const*
delete_old_dirs(

save_13/IdentityIdentitysave_13/Const^save_13/MergeV2Checkpoints^save_13/control_dependency*
T0*
_output_shapes
: 
ű
save_13/RestoreV2/tensor_namesConst*
_output_shapes
:D*¨
valueBDBbeta1_powerBbeta1_power_1Bbeta1_power_2Bbeta2_powerBbeta2_power_1Bbeta2_power_2Bpenalty/penalty_paramBpenalty/penalty_param/AdamBpenalty/penalty_param/Adam_1Bpenalty/penalty_param/Adam_2Bpenalty/penalty_param/Adam_3Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1B
pi/log_stdBpi/log_std/AdamBpi/log_std/Adam_1Bvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1*
dtype0
ô
"save_13/RestoreV2/shape_and_slicesConst*
_output_shapes
:D*
dtype0*
valueBDB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
î
save_13/RestoreV2	RestoreV2save_13/Constsave_13/RestoreV2/tensor_names"save_13/RestoreV2/shape_and_slices*Ś
_output_shapes
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*R
dtypesH
F2D
Ź
save_13/AssignAssignbeta1_powersave_13/RestoreV2*
use_locking(*
validate_shape(*(
_class
loc:@penalty/penalty_param*
T0*
_output_shapes
: 
˛
save_13/Assign_1Assignbeta1_power_1save_13/RestoreV2:1*
validate_shape(*
_output_shapes
: *
T0*
use_locking(*(
_class
loc:@penalty/penalty_param
Ş
save_13/Assign_2Assignbeta1_power_2save_13/RestoreV2:2*
_output_shapes
: *
use_locking(*
validate_shape(*
T0* 
_class
loc:@vc/dense/bias
°
save_13/Assign_3Assignbeta2_powersave_13/RestoreV2:3*(
_class
loc:@penalty/penalty_param*
validate_shape(*
_output_shapes
: *
T0*
use_locking(
˛
save_13/Assign_4Assignbeta2_power_1save_13/RestoreV2:4*
_output_shapes
: *
validate_shape(*
use_locking(*(
_class
loc:@penalty/penalty_param*
T0
Ş
save_13/Assign_5Assignbeta2_power_2save_13/RestoreV2:5*
use_locking(* 
_class
loc:@vc/dense/bias*
T0*
validate_shape(*
_output_shapes
: 
ş
save_13/Assign_6Assignpenalty/penalty_paramsave_13/RestoreV2:6*
T0*
use_locking(*
_output_shapes
: *
validate_shape(*(
_class
loc:@penalty/penalty_param
ż
save_13/Assign_7Assignpenalty/penalty_param/Adamsave_13/RestoreV2:7*
validate_shape(*
T0*(
_class
loc:@penalty/penalty_param*
use_locking(*
_output_shapes
: 
Á
save_13/Assign_8Assignpenalty/penalty_param/Adam_1save_13/RestoreV2:8*
validate_shape(*
use_locking(*(
_class
loc:@penalty/penalty_param*
T0*
_output_shapes
: 
Á
save_13/Assign_9Assignpenalty/penalty_param/Adam_2save_13/RestoreV2:9*(
_class
loc:@penalty/penalty_param*
use_locking(*
T0*
_output_shapes
: *
validate_shape(
Ă
save_13/Assign_10Assignpenalty/penalty_param/Adam_3save_13/RestoreV2:10*(
_class
loc:@penalty/penalty_param*
T0*
_output_shapes
: *
use_locking(*
validate_shape(
ą
save_13/Assign_11Assignpi/dense/biassave_13/RestoreV2:11*
_output_shapes	
:* 
_class
loc:@pi/dense/bias*
T0*
validate_shape(*
use_locking(
ś
save_13/Assign_12Assignpi/dense/bias/Adamsave_13/RestoreV2:12*
validate_shape(*
_output_shapes	
:*
T0*
use_locking(* 
_class
loc:@pi/dense/bias
¸
save_13/Assign_13Assignpi/dense/bias/Adam_1save_13/RestoreV2:13*
use_locking(*
_output_shapes	
:*
T0*
validate_shape(* 
_class
loc:@pi/dense/bias
š
save_13/Assign_14Assignpi/dense/kernelsave_13/RestoreV2:14*
validate_shape(*
T0*
use_locking(*"
_class
loc:@pi/dense/kernel*
_output_shapes
:	L
ž
save_13/Assign_15Assignpi/dense/kernel/Adamsave_13/RestoreV2:15*
validate_shape(*
_output_shapes
:	L*"
_class
loc:@pi/dense/kernel*
T0*
use_locking(
Ŕ
save_13/Assign_16Assignpi/dense/kernel/Adam_1save_13/RestoreV2:16*
use_locking(*
T0*
validate_shape(*"
_class
loc:@pi/dense/kernel*
_output_shapes
:	L
ľ
save_13/Assign_17Assignpi/dense_1/biassave_13/RestoreV2:17*
validate_shape(*"
_class
loc:@pi/dense_1/bias*
_output_shapes	
:*
T0*
use_locking(
ş
save_13/Assign_18Assignpi/dense_1/bias/Adamsave_13/RestoreV2:18*
T0*
use_locking(*
validate_shape(*
_output_shapes	
:*"
_class
loc:@pi/dense_1/bias
ź
save_13/Assign_19Assignpi/dense_1/bias/Adam_1save_13/RestoreV2:19*
validate_shape(*
use_locking(*
_output_shapes	
:*
T0*"
_class
loc:@pi/dense_1/bias
ž
save_13/Assign_20Assignpi/dense_1/kernelsave_13/RestoreV2:20*
validate_shape(*$
_class
loc:@pi/dense_1/kernel* 
_output_shapes
:
*
use_locking(*
T0
Ă
save_13/Assign_21Assignpi/dense_1/kernel/Adamsave_13/RestoreV2:21*
use_locking(* 
_output_shapes
:
*$
_class
loc:@pi/dense_1/kernel*
validate_shape(*
T0
Ĺ
save_13/Assign_22Assignpi/dense_1/kernel/Adam_1save_13/RestoreV2:22* 
_output_shapes
:
*
use_locking(*
validate_shape(*
T0*$
_class
loc:@pi/dense_1/kernel
´
save_13/Assign_23Assignpi/dense_2/biassave_13/RestoreV2:23*
T0*
use_locking(*
_output_shapes
:*
validate_shape(*"
_class
loc:@pi/dense_2/bias
š
save_13/Assign_24Assignpi/dense_2/bias/Adamsave_13/RestoreV2:24*
use_locking(*
_output_shapes
:*
T0*"
_class
loc:@pi/dense_2/bias*
validate_shape(
ť
save_13/Assign_25Assignpi/dense_2/bias/Adam_1save_13/RestoreV2:25*"
_class
loc:@pi/dense_2/bias*
T0*
_output_shapes
:*
use_locking(*
validate_shape(
˝
save_13/Assign_26Assignpi/dense_2/kernelsave_13/RestoreV2:26*
_output_shapes
:	*
validate_shape(*$
_class
loc:@pi/dense_2/kernel*
use_locking(*
T0
Â
save_13/Assign_27Assignpi/dense_2/kernel/Adamsave_13/RestoreV2:27*
T0*$
_class
loc:@pi/dense_2/kernel*
validate_shape(*
_output_shapes
:	*
use_locking(
Ä
save_13/Assign_28Assignpi/dense_2/kernel/Adam_1save_13/RestoreV2:28*
validate_shape(*
_output_shapes
:	*
use_locking(*$
_class
loc:@pi/dense_2/kernel*
T0
Ş
save_13/Assign_29Assign
pi/log_stdsave_13/RestoreV2:29*
_class
loc:@pi/log_std*
use_locking(*
validate_shape(*
T0*
_output_shapes
:
Ż
save_13/Assign_30Assignpi/log_std/Adamsave_13/RestoreV2:30*
validate_shape(*
_output_shapes
:*
T0*
use_locking(*
_class
loc:@pi/log_std
ą
save_13/Assign_31Assignpi/log_std/Adam_1save_13/RestoreV2:31*
T0*
_output_shapes
:*
use_locking(*
validate_shape(*
_class
loc:@pi/log_std
ą
save_13/Assign_32Assignvc/dense/biassave_13/RestoreV2:32*
use_locking(*
T0*
validate_shape(*
_output_shapes	
:* 
_class
loc:@vc/dense/bias
ś
save_13/Assign_33Assignvc/dense/bias/Adamsave_13/RestoreV2:33*
_output_shapes	
:*
validate_shape(*
T0* 
_class
loc:@vc/dense/bias*
use_locking(
¸
save_13/Assign_34Assignvc/dense/bias/Adam_1save_13/RestoreV2:34*
use_locking(*
_output_shapes	
:* 
_class
loc:@vc/dense/bias*
T0*
validate_shape(
š
save_13/Assign_35Assignvc/dense/kernelsave_13/RestoreV2:35*
use_locking(*
validate_shape(*"
_class
loc:@vc/dense/kernel*
_output_shapes
:	L*
T0
ž
save_13/Assign_36Assignvc/dense/kernel/Adamsave_13/RestoreV2:36*
validate_shape(*
use_locking(*
T0*
_output_shapes
:	L*"
_class
loc:@vc/dense/kernel
Ŕ
save_13/Assign_37Assignvc/dense/kernel/Adam_1save_13/RestoreV2:37*
use_locking(*
validate_shape(*
T0*
_output_shapes
:	L*"
_class
loc:@vc/dense/kernel
ľ
save_13/Assign_38Assignvc/dense_1/biassave_13/RestoreV2:38*
T0*"
_class
loc:@vc/dense_1/bias*
_output_shapes	
:*
use_locking(*
validate_shape(
ş
save_13/Assign_39Assignvc/dense_1/bias/Adamsave_13/RestoreV2:39*
T0*
_output_shapes	
:*"
_class
loc:@vc/dense_1/bias*
validate_shape(*
use_locking(
ź
save_13/Assign_40Assignvc/dense_1/bias/Adam_1save_13/RestoreV2:40*
validate_shape(*
_output_shapes	
:*"
_class
loc:@vc/dense_1/bias*
T0*
use_locking(
ž
save_13/Assign_41Assignvc/dense_1/kernelsave_13/RestoreV2:41*
use_locking(*
T0* 
_output_shapes
:
*$
_class
loc:@vc/dense_1/kernel*
validate_shape(
Ă
save_13/Assign_42Assignvc/dense_1/kernel/Adamsave_13/RestoreV2:42* 
_output_shapes
:
*
T0*$
_class
loc:@vc/dense_1/kernel*
use_locking(*
validate_shape(
Ĺ
save_13/Assign_43Assignvc/dense_1/kernel/Adam_1save_13/RestoreV2:43*
validate_shape(*$
_class
loc:@vc/dense_1/kernel* 
_output_shapes
:
*
use_locking(*
T0
´
save_13/Assign_44Assignvc/dense_2/biassave_13/RestoreV2:44*
use_locking(*
validate_shape(*
T0*
_output_shapes
:*"
_class
loc:@vc/dense_2/bias
š
save_13/Assign_45Assignvc/dense_2/bias/Adamsave_13/RestoreV2:45*
validate_shape(*"
_class
loc:@vc/dense_2/bias*
T0*
use_locking(*
_output_shapes
:
ť
save_13/Assign_46Assignvc/dense_2/bias/Adam_1save_13/RestoreV2:46*"
_class
loc:@vc/dense_2/bias*
validate_shape(*
use_locking(*
_output_shapes
:*
T0
˝
save_13/Assign_47Assignvc/dense_2/kernelsave_13/RestoreV2:47*
_output_shapes
:	*
T0*$
_class
loc:@vc/dense_2/kernel*
use_locking(*
validate_shape(
Â
save_13/Assign_48Assignvc/dense_2/kernel/Adamsave_13/RestoreV2:48*
_output_shapes
:	*
T0*$
_class
loc:@vc/dense_2/kernel*
use_locking(*
validate_shape(
Ä
save_13/Assign_49Assignvc/dense_2/kernel/Adam_1save_13/RestoreV2:49*$
_class
loc:@vc/dense_2/kernel*
validate_shape(*
T0*
_output_shapes
:	*
use_locking(
ą
save_13/Assign_50Assignvf/dense/biassave_13/RestoreV2:50*
T0*
validate_shape(*
use_locking(* 
_class
loc:@vf/dense/bias*
_output_shapes	
:
ś
save_13/Assign_51Assignvf/dense/bias/Adamsave_13/RestoreV2:51*
use_locking(*
validate_shape(*
_output_shapes	
:* 
_class
loc:@vf/dense/bias*
T0
¸
save_13/Assign_52Assignvf/dense/bias/Adam_1save_13/RestoreV2:52* 
_class
loc:@vf/dense/bias*
T0*
_output_shapes	
:*
use_locking(*
validate_shape(
š
save_13/Assign_53Assignvf/dense/kernelsave_13/RestoreV2:53*"
_class
loc:@vf/dense/kernel*
validate_shape(*
_output_shapes
:	L*
T0*
use_locking(
ž
save_13/Assign_54Assignvf/dense/kernel/Adamsave_13/RestoreV2:54*
use_locking(*"
_class
loc:@vf/dense/kernel*
validate_shape(*
_output_shapes
:	L*
T0
Ŕ
save_13/Assign_55Assignvf/dense/kernel/Adam_1save_13/RestoreV2:55*
use_locking(*
T0*
validate_shape(*
_output_shapes
:	L*"
_class
loc:@vf/dense/kernel
ľ
save_13/Assign_56Assignvf/dense_1/biassave_13/RestoreV2:56*
use_locking(*"
_class
loc:@vf/dense_1/bias*
validate_shape(*
T0*
_output_shapes	
:
ş
save_13/Assign_57Assignvf/dense_1/bias/Adamsave_13/RestoreV2:57*"
_class
loc:@vf/dense_1/bias*
use_locking(*
T0*
_output_shapes	
:*
validate_shape(
ź
save_13/Assign_58Assignvf/dense_1/bias/Adam_1save_13/RestoreV2:58*"
_class
loc:@vf/dense_1/bias*
T0*
_output_shapes	
:*
validate_shape(*
use_locking(
ž
save_13/Assign_59Assignvf/dense_1/kernelsave_13/RestoreV2:59*$
_class
loc:@vf/dense_1/kernel*
use_locking(* 
_output_shapes
:
*
T0*
validate_shape(
Ă
save_13/Assign_60Assignvf/dense_1/kernel/Adamsave_13/RestoreV2:60*
validate_shape(*$
_class
loc:@vf/dense_1/kernel* 
_output_shapes
:
*
T0*
use_locking(
Ĺ
save_13/Assign_61Assignvf/dense_1/kernel/Adam_1save_13/RestoreV2:61*
validate_shape(* 
_output_shapes
:
*
use_locking(*$
_class
loc:@vf/dense_1/kernel*
T0
´
save_13/Assign_62Assignvf/dense_2/biassave_13/RestoreV2:62*
use_locking(*"
_class
loc:@vf/dense_2/bias*
_output_shapes
:*
T0*
validate_shape(
š
save_13/Assign_63Assignvf/dense_2/bias/Adamsave_13/RestoreV2:63*
T0*
_output_shapes
:*
validate_shape(*"
_class
loc:@vf/dense_2/bias*
use_locking(
ť
save_13/Assign_64Assignvf/dense_2/bias/Adam_1save_13/RestoreV2:64*"
_class
loc:@vf/dense_2/bias*
_output_shapes
:*
validate_shape(*
T0*
use_locking(
˝
save_13/Assign_65Assignvf/dense_2/kernelsave_13/RestoreV2:65*
use_locking(*
validate_shape(*
T0*
_output_shapes
:	*$
_class
loc:@vf/dense_2/kernel
Â
save_13/Assign_66Assignvf/dense_2/kernel/Adamsave_13/RestoreV2:66*
T0*
validate_shape(*
use_locking(*
_output_shapes
:	*$
_class
loc:@vf/dense_2/kernel
Ä
save_13/Assign_67Assignvf/dense_2/kernel/Adam_1save_13/RestoreV2:67*$
_class
loc:@vf/dense_2/kernel*
validate_shape(*
T0*
use_locking(*
_output_shapes
:	
á

save_13/restore_shardNoOp^save_13/Assign^save_13/Assign_1^save_13/Assign_10^save_13/Assign_11^save_13/Assign_12^save_13/Assign_13^save_13/Assign_14^save_13/Assign_15^save_13/Assign_16^save_13/Assign_17^save_13/Assign_18^save_13/Assign_19^save_13/Assign_2^save_13/Assign_20^save_13/Assign_21^save_13/Assign_22^save_13/Assign_23^save_13/Assign_24^save_13/Assign_25^save_13/Assign_26^save_13/Assign_27^save_13/Assign_28^save_13/Assign_29^save_13/Assign_3^save_13/Assign_30^save_13/Assign_31^save_13/Assign_32^save_13/Assign_33^save_13/Assign_34^save_13/Assign_35^save_13/Assign_36^save_13/Assign_37^save_13/Assign_38^save_13/Assign_39^save_13/Assign_4^save_13/Assign_40^save_13/Assign_41^save_13/Assign_42^save_13/Assign_43^save_13/Assign_44^save_13/Assign_45^save_13/Assign_46^save_13/Assign_47^save_13/Assign_48^save_13/Assign_49^save_13/Assign_5^save_13/Assign_50^save_13/Assign_51^save_13/Assign_52^save_13/Assign_53^save_13/Assign_54^save_13/Assign_55^save_13/Assign_56^save_13/Assign_57^save_13/Assign_58^save_13/Assign_59^save_13/Assign_6^save_13/Assign_60^save_13/Assign_61^save_13/Assign_62^save_13/Assign_63^save_13/Assign_64^save_13/Assign_65^save_13/Assign_66^save_13/Assign_67^save_13/Assign_7^save_13/Assign_8^save_13/Assign_9
3
save_13/restore_allNoOp^save_13/restore_shard
\
save_14/filename/inputConst*
dtype0*
valueB Bmodel*
_output_shapes
: 
t
save_14/filenamePlaceholderWithDefaultsave_14/filename/input*
_output_shapes
: *
shape: *
dtype0
k
save_14/ConstPlaceholderWithDefaultsave_14/filename*
shape: *
_output_shapes
: *
dtype0

save_14/StringJoin/inputs_1Const*
dtype0*
_output_shapes
: *<
value3B1 B+_temp_256e694e22914ce1beaff0f7eaf0101f/part
~
save_14/StringJoin
StringJoinsave_14/Constsave_14/StringJoin/inputs_1*
_output_shapes
: *
	separator *
N
T
save_14/num_shardsConst*
value	B :*
_output_shapes
: *
dtype0
_
save_14/ShardedFilename/shardConst*
value	B : *
_output_shapes
: *
dtype0

save_14/ShardedFilenameShardedFilenamesave_14/StringJoinsave_14/ShardedFilename/shardsave_14/num_shards*
_output_shapes
: 
ř
save_14/SaveV2/tensor_namesConst*
dtype0*¨
valueBDBbeta1_powerBbeta1_power_1Bbeta1_power_2Bbeta2_powerBbeta2_power_1Bbeta2_power_2Bpenalty/penalty_paramBpenalty/penalty_param/AdamBpenalty/penalty_param/Adam_1Bpenalty/penalty_param/Adam_2Bpenalty/penalty_param/Adam_3Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1B
pi/log_stdBpi/log_std/AdamBpi/log_std/Adam_1Bvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1*
_output_shapes
:D
ń
save_14/SaveV2/shape_and_slicesConst*
valueBDB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
_output_shapes
:D*
dtype0
Ö
save_14/SaveV2SaveV2save_14/ShardedFilenamesave_14/SaveV2/tensor_namessave_14/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta1_power_2beta2_powerbeta2_power_1beta2_power_2penalty/penalty_parampenalty/penalty_param/Adampenalty/penalty_param/Adam_1penalty/penalty_param/Adam_2penalty/penalty_param/Adam_3pi/dense/biaspi/dense/bias/Adampi/dense/bias/Adam_1pi/dense/kernelpi/dense/kernel/Adampi/dense/kernel/Adam_1pi/dense_1/biaspi/dense_1/bias/Adampi/dense_1/bias/Adam_1pi/dense_1/kernelpi/dense_1/kernel/Adampi/dense_1/kernel/Adam_1pi/dense_2/biaspi/dense_2/bias/Adampi/dense_2/bias/Adam_1pi/dense_2/kernelpi/dense_2/kernel/Adampi/dense_2/kernel/Adam_1
pi/log_stdpi/log_std/Adampi/log_std/Adam_1vc/dense/biasvc/dense/bias/Adamvc/dense/bias/Adam_1vc/dense/kernelvc/dense/kernel/Adamvc/dense/kernel/Adam_1vc/dense_1/biasvc/dense_1/bias/Adamvc/dense_1/bias/Adam_1vc/dense_1/kernelvc/dense_1/kernel/Adamvc/dense_1/kernel/Adam_1vc/dense_2/biasvc/dense_2/bias/Adamvc/dense_2/bias/Adam_1vc/dense_2/kernelvc/dense_2/kernel/Adamvc/dense_2/kernel/Adam_1vf/dense/biasvf/dense/bias/Adamvf/dense/bias/Adam_1vf/dense/kernelvf/dense/kernel/Adamvf/dense/kernel/Adam_1vf/dense_1/biasvf/dense_1/bias/Adamvf/dense_1/bias/Adam_1vf/dense_1/kernelvf/dense_1/kernel/Adamvf/dense_1/kernel/Adam_1vf/dense_2/biasvf/dense_2/bias/Adamvf/dense_2/bias/Adam_1vf/dense_2/kernelvf/dense_2/kernel/Adamvf/dense_2/kernel/Adam_1*R
dtypesH
F2D

save_14/control_dependencyIdentitysave_14/ShardedFilename^save_14/SaveV2*
T0*
_output_shapes
: **
_class 
loc:@save_14/ShardedFilename
Ś
.save_14/MergeV2Checkpoints/checkpoint_prefixesPacksave_14/ShardedFilename^save_14/control_dependency*
_output_shapes
:*
N*

axis *
T0

save_14/MergeV2CheckpointsMergeV2Checkpoints.save_14/MergeV2Checkpoints/checkpoint_prefixessave_14/Const*
delete_old_dirs(

save_14/IdentityIdentitysave_14/Const^save_14/MergeV2Checkpoints^save_14/control_dependency*
_output_shapes
: *
T0
ű
save_14/RestoreV2/tensor_namesConst*¨
valueBDBbeta1_powerBbeta1_power_1Bbeta1_power_2Bbeta2_powerBbeta2_power_1Bbeta2_power_2Bpenalty/penalty_paramBpenalty/penalty_param/AdamBpenalty/penalty_param/Adam_1Bpenalty/penalty_param/Adam_2Bpenalty/penalty_param/Adam_3Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1B
pi/log_stdBpi/log_std/AdamBpi/log_std/Adam_1Bvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1*
dtype0*
_output_shapes
:D
ô
"save_14/RestoreV2/shape_and_slicesConst*
valueBDB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
_output_shapes
:D*
dtype0
î
save_14/RestoreV2	RestoreV2save_14/Constsave_14/RestoreV2/tensor_names"save_14/RestoreV2/shape_and_slices*R
dtypesH
F2D*Ś
_output_shapes
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
Ź
save_14/AssignAssignbeta1_powersave_14/RestoreV2*(
_class
loc:@penalty/penalty_param*
_output_shapes
: *
use_locking(*
T0*
validate_shape(
˛
save_14/Assign_1Assignbeta1_power_1save_14/RestoreV2:1*
use_locking(*
validate_shape(*
T0*(
_class
loc:@penalty/penalty_param*
_output_shapes
: 
Ş
save_14/Assign_2Assignbeta1_power_2save_14/RestoreV2:2*
validate_shape(* 
_class
loc:@vc/dense/bias*
_output_shapes
: *
use_locking(*
T0
°
save_14/Assign_3Assignbeta2_powersave_14/RestoreV2:3*
validate_shape(*
T0*(
_class
loc:@penalty/penalty_param*
_output_shapes
: *
use_locking(
˛
save_14/Assign_4Assignbeta2_power_1save_14/RestoreV2:4*
T0*
validate_shape(*(
_class
loc:@penalty/penalty_param*
use_locking(*
_output_shapes
: 
Ş
save_14/Assign_5Assignbeta2_power_2save_14/RestoreV2:5* 
_class
loc:@vc/dense/bias*
T0*
use_locking(*
_output_shapes
: *
validate_shape(
ş
save_14/Assign_6Assignpenalty/penalty_paramsave_14/RestoreV2:6*
use_locking(*
_output_shapes
: *
T0*(
_class
loc:@penalty/penalty_param*
validate_shape(
ż
save_14/Assign_7Assignpenalty/penalty_param/Adamsave_14/RestoreV2:7*
T0*(
_class
loc:@penalty/penalty_param*
_output_shapes
: *
validate_shape(*
use_locking(
Á
save_14/Assign_8Assignpenalty/penalty_param/Adam_1save_14/RestoreV2:8*(
_class
loc:@penalty/penalty_param*
_output_shapes
: *
T0*
validate_shape(*
use_locking(
Á
save_14/Assign_9Assignpenalty/penalty_param/Adam_2save_14/RestoreV2:9*
T0*
_output_shapes
: *
validate_shape(*
use_locking(*(
_class
loc:@penalty/penalty_param
Ă
save_14/Assign_10Assignpenalty/penalty_param/Adam_3save_14/RestoreV2:10*
T0*(
_class
loc:@penalty/penalty_param*
use_locking(*
validate_shape(*
_output_shapes
: 
ą
save_14/Assign_11Assignpi/dense/biassave_14/RestoreV2:11*
validate_shape(*
use_locking(* 
_class
loc:@pi/dense/bias*
T0*
_output_shapes	
:
ś
save_14/Assign_12Assignpi/dense/bias/Adamsave_14/RestoreV2:12*
_output_shapes	
:*
T0*
use_locking(* 
_class
loc:@pi/dense/bias*
validate_shape(
¸
save_14/Assign_13Assignpi/dense/bias/Adam_1save_14/RestoreV2:13*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
use_locking(*
_output_shapes	
:
š
save_14/Assign_14Assignpi/dense/kernelsave_14/RestoreV2:14*"
_class
loc:@pi/dense/kernel*
validate_shape(*
_output_shapes
:	L*
use_locking(*
T0
ž
save_14/Assign_15Assignpi/dense/kernel/Adamsave_14/RestoreV2:15*
T0*"
_class
loc:@pi/dense/kernel*
validate_shape(*
use_locking(*
_output_shapes
:	L
Ŕ
save_14/Assign_16Assignpi/dense/kernel/Adam_1save_14/RestoreV2:16*
validate_shape(*
_output_shapes
:	L*
T0*
use_locking(*"
_class
loc:@pi/dense/kernel
ľ
save_14/Assign_17Assignpi/dense_1/biassave_14/RestoreV2:17*"
_class
loc:@pi/dense_1/bias*
T0*
_output_shapes	
:*
validate_shape(*
use_locking(
ş
save_14/Assign_18Assignpi/dense_1/bias/Adamsave_14/RestoreV2:18*
validate_shape(*
use_locking(*"
_class
loc:@pi/dense_1/bias*
_output_shapes	
:*
T0
ź
save_14/Assign_19Assignpi/dense_1/bias/Adam_1save_14/RestoreV2:19*
T0*
use_locking(*"
_class
loc:@pi/dense_1/bias*
_output_shapes	
:*
validate_shape(
ž
save_14/Assign_20Assignpi/dense_1/kernelsave_14/RestoreV2:20*
use_locking(*$
_class
loc:@pi/dense_1/kernel*
T0*
validate_shape(* 
_output_shapes
:

Ă
save_14/Assign_21Assignpi/dense_1/kernel/Adamsave_14/RestoreV2:21*
validate_shape(*
use_locking(*$
_class
loc:@pi/dense_1/kernel* 
_output_shapes
:
*
T0
Ĺ
save_14/Assign_22Assignpi/dense_1/kernel/Adam_1save_14/RestoreV2:22*
validate_shape(*$
_class
loc:@pi/dense_1/kernel* 
_output_shapes
:
*
use_locking(*
T0
´
save_14/Assign_23Assignpi/dense_2/biassave_14/RestoreV2:23*
validate_shape(*
use_locking(*"
_class
loc:@pi/dense_2/bias*
T0*
_output_shapes
:
š
save_14/Assign_24Assignpi/dense_2/bias/Adamsave_14/RestoreV2:24*
use_locking(*
validate_shape(*
_output_shapes
:*"
_class
loc:@pi/dense_2/bias*
T0
ť
save_14/Assign_25Assignpi/dense_2/bias/Adam_1save_14/RestoreV2:25*
validate_shape(*
T0*
_output_shapes
:*
use_locking(*"
_class
loc:@pi/dense_2/bias
˝
save_14/Assign_26Assignpi/dense_2/kernelsave_14/RestoreV2:26*
T0*
validate_shape(*
_output_shapes
:	*$
_class
loc:@pi/dense_2/kernel*
use_locking(
Â
save_14/Assign_27Assignpi/dense_2/kernel/Adamsave_14/RestoreV2:27*$
_class
loc:@pi/dense_2/kernel*
use_locking(*
T0*
_output_shapes
:	*
validate_shape(
Ä
save_14/Assign_28Assignpi/dense_2/kernel/Adam_1save_14/RestoreV2:28*$
_class
loc:@pi/dense_2/kernel*
use_locking(*
validate_shape(*
T0*
_output_shapes
:	
Ş
save_14/Assign_29Assign
pi/log_stdsave_14/RestoreV2:29*
validate_shape(*
_class
loc:@pi/log_std*
T0*
use_locking(*
_output_shapes
:
Ż
save_14/Assign_30Assignpi/log_std/Adamsave_14/RestoreV2:30*
_class
loc:@pi/log_std*
T0*
use_locking(*
validate_shape(*
_output_shapes
:
ą
save_14/Assign_31Assignpi/log_std/Adam_1save_14/RestoreV2:31*
validate_shape(*
use_locking(*
T0*
_output_shapes
:*
_class
loc:@pi/log_std
ą
save_14/Assign_32Assignvc/dense/biassave_14/RestoreV2:32* 
_class
loc:@vc/dense/bias*
validate_shape(*
use_locking(*
_output_shapes	
:*
T0
ś
save_14/Assign_33Assignvc/dense/bias/Adamsave_14/RestoreV2:33*
validate_shape(*
_output_shapes	
:*
use_locking(* 
_class
loc:@vc/dense/bias*
T0
¸
save_14/Assign_34Assignvc/dense/bias/Adam_1save_14/RestoreV2:34*
validate_shape(*
use_locking(*
_output_shapes	
:*
T0* 
_class
loc:@vc/dense/bias
š
save_14/Assign_35Assignvc/dense/kernelsave_14/RestoreV2:35*
use_locking(*
validate_shape(*
T0*"
_class
loc:@vc/dense/kernel*
_output_shapes
:	L
ž
save_14/Assign_36Assignvc/dense/kernel/Adamsave_14/RestoreV2:36*
_output_shapes
:	L*
use_locking(*"
_class
loc:@vc/dense/kernel*
T0*
validate_shape(
Ŕ
save_14/Assign_37Assignvc/dense/kernel/Adam_1save_14/RestoreV2:37*
_output_shapes
:	L*
validate_shape(*
use_locking(*
T0*"
_class
loc:@vc/dense/kernel
ľ
save_14/Assign_38Assignvc/dense_1/biassave_14/RestoreV2:38*
_output_shapes	
:*
use_locking(*
validate_shape(*"
_class
loc:@vc/dense_1/bias*
T0
ş
save_14/Assign_39Assignvc/dense_1/bias/Adamsave_14/RestoreV2:39*"
_class
loc:@vc/dense_1/bias*
use_locking(*
_output_shapes	
:*
validate_shape(*
T0
ź
save_14/Assign_40Assignvc/dense_1/bias/Adam_1save_14/RestoreV2:40*"
_class
loc:@vc/dense_1/bias*
validate_shape(*
T0*
_output_shapes	
:*
use_locking(
ž
save_14/Assign_41Assignvc/dense_1/kernelsave_14/RestoreV2:41*
validate_shape(* 
_output_shapes
:
*
T0*$
_class
loc:@vc/dense_1/kernel*
use_locking(
Ă
save_14/Assign_42Assignvc/dense_1/kernel/Adamsave_14/RestoreV2:42*$
_class
loc:@vc/dense_1/kernel*
use_locking(* 
_output_shapes
:
*
validate_shape(*
T0
Ĺ
save_14/Assign_43Assignvc/dense_1/kernel/Adam_1save_14/RestoreV2:43*$
_class
loc:@vc/dense_1/kernel*
use_locking(* 
_output_shapes
:
*
T0*
validate_shape(
´
save_14/Assign_44Assignvc/dense_2/biassave_14/RestoreV2:44*
validate_shape(*"
_class
loc:@vc/dense_2/bias*
T0*
_output_shapes
:*
use_locking(
š
save_14/Assign_45Assignvc/dense_2/bias/Adamsave_14/RestoreV2:45*
T0*
_output_shapes
:*
use_locking(*"
_class
loc:@vc/dense_2/bias*
validate_shape(
ť
save_14/Assign_46Assignvc/dense_2/bias/Adam_1save_14/RestoreV2:46*
use_locking(*"
_class
loc:@vc/dense_2/bias*
T0*
validate_shape(*
_output_shapes
:
˝
save_14/Assign_47Assignvc/dense_2/kernelsave_14/RestoreV2:47*
_output_shapes
:	*
T0*
use_locking(*$
_class
loc:@vc/dense_2/kernel*
validate_shape(
Â
save_14/Assign_48Assignvc/dense_2/kernel/Adamsave_14/RestoreV2:48*
_output_shapes
:	*
T0*
use_locking(*$
_class
loc:@vc/dense_2/kernel*
validate_shape(
Ä
save_14/Assign_49Assignvc/dense_2/kernel/Adam_1save_14/RestoreV2:49*
T0*$
_class
loc:@vc/dense_2/kernel*
use_locking(*
_output_shapes
:	*
validate_shape(
ą
save_14/Assign_50Assignvf/dense/biassave_14/RestoreV2:50* 
_class
loc:@vf/dense/bias*
use_locking(*
_output_shapes	
:*
T0*
validate_shape(
ś
save_14/Assign_51Assignvf/dense/bias/Adamsave_14/RestoreV2:51*
validate_shape(*
use_locking(*
_output_shapes	
:* 
_class
loc:@vf/dense/bias*
T0
¸
save_14/Assign_52Assignvf/dense/bias/Adam_1save_14/RestoreV2:52*
T0*
validate_shape(*
_output_shapes	
:*
use_locking(* 
_class
loc:@vf/dense/bias
š
save_14/Assign_53Assignvf/dense/kernelsave_14/RestoreV2:53*
validate_shape(*"
_class
loc:@vf/dense/kernel*
_output_shapes
:	L*
T0*
use_locking(
ž
save_14/Assign_54Assignvf/dense/kernel/Adamsave_14/RestoreV2:54*
_output_shapes
:	L*
use_locking(*
validate_shape(*"
_class
loc:@vf/dense/kernel*
T0
Ŕ
save_14/Assign_55Assignvf/dense/kernel/Adam_1save_14/RestoreV2:55*
_output_shapes
:	L*"
_class
loc:@vf/dense/kernel*
T0*
validate_shape(*
use_locking(
ľ
save_14/Assign_56Assignvf/dense_1/biassave_14/RestoreV2:56*
_output_shapes	
:*
use_locking(*
T0*"
_class
loc:@vf/dense_1/bias*
validate_shape(
ş
save_14/Assign_57Assignvf/dense_1/bias/Adamsave_14/RestoreV2:57*
use_locking(*
_output_shapes	
:*
T0*"
_class
loc:@vf/dense_1/bias*
validate_shape(
ź
save_14/Assign_58Assignvf/dense_1/bias/Adam_1save_14/RestoreV2:58*
_output_shapes	
:*
use_locking(*
validate_shape(*
T0*"
_class
loc:@vf/dense_1/bias
ž
save_14/Assign_59Assignvf/dense_1/kernelsave_14/RestoreV2:59*
validate_shape(*
use_locking(* 
_output_shapes
:
*
T0*$
_class
loc:@vf/dense_1/kernel
Ă
save_14/Assign_60Assignvf/dense_1/kernel/Adamsave_14/RestoreV2:60*$
_class
loc:@vf/dense_1/kernel*
use_locking(*
validate_shape(* 
_output_shapes
:
*
T0
Ĺ
save_14/Assign_61Assignvf/dense_1/kernel/Adam_1save_14/RestoreV2:61* 
_output_shapes
:
*
use_locking(*
validate_shape(*
T0*$
_class
loc:@vf/dense_1/kernel
´
save_14/Assign_62Assignvf/dense_2/biassave_14/RestoreV2:62*
T0*
validate_shape(*"
_class
loc:@vf/dense_2/bias*
use_locking(*
_output_shapes
:
š
save_14/Assign_63Assignvf/dense_2/bias/Adamsave_14/RestoreV2:63*
validate_shape(*
T0*
use_locking(*"
_class
loc:@vf/dense_2/bias*
_output_shapes
:
ť
save_14/Assign_64Assignvf/dense_2/bias/Adam_1save_14/RestoreV2:64*
T0*
validate_shape(*
_output_shapes
:*
use_locking(*"
_class
loc:@vf/dense_2/bias
˝
save_14/Assign_65Assignvf/dense_2/kernelsave_14/RestoreV2:65*
validate_shape(*
_output_shapes
:	*$
_class
loc:@vf/dense_2/kernel*
T0*
use_locking(
Â
save_14/Assign_66Assignvf/dense_2/kernel/Adamsave_14/RestoreV2:66*
_output_shapes
:	*
use_locking(*
validate_shape(*
T0*$
_class
loc:@vf/dense_2/kernel
Ä
save_14/Assign_67Assignvf/dense_2/kernel/Adam_1save_14/RestoreV2:67*
_output_shapes
:	*
validate_shape(*
use_locking(*
T0*$
_class
loc:@vf/dense_2/kernel
á

save_14/restore_shardNoOp^save_14/Assign^save_14/Assign_1^save_14/Assign_10^save_14/Assign_11^save_14/Assign_12^save_14/Assign_13^save_14/Assign_14^save_14/Assign_15^save_14/Assign_16^save_14/Assign_17^save_14/Assign_18^save_14/Assign_19^save_14/Assign_2^save_14/Assign_20^save_14/Assign_21^save_14/Assign_22^save_14/Assign_23^save_14/Assign_24^save_14/Assign_25^save_14/Assign_26^save_14/Assign_27^save_14/Assign_28^save_14/Assign_29^save_14/Assign_3^save_14/Assign_30^save_14/Assign_31^save_14/Assign_32^save_14/Assign_33^save_14/Assign_34^save_14/Assign_35^save_14/Assign_36^save_14/Assign_37^save_14/Assign_38^save_14/Assign_39^save_14/Assign_4^save_14/Assign_40^save_14/Assign_41^save_14/Assign_42^save_14/Assign_43^save_14/Assign_44^save_14/Assign_45^save_14/Assign_46^save_14/Assign_47^save_14/Assign_48^save_14/Assign_49^save_14/Assign_5^save_14/Assign_50^save_14/Assign_51^save_14/Assign_52^save_14/Assign_53^save_14/Assign_54^save_14/Assign_55^save_14/Assign_56^save_14/Assign_57^save_14/Assign_58^save_14/Assign_59^save_14/Assign_6^save_14/Assign_60^save_14/Assign_61^save_14/Assign_62^save_14/Assign_63^save_14/Assign_64^save_14/Assign_65^save_14/Assign_66^save_14/Assign_67^save_14/Assign_7^save_14/Assign_8^save_14/Assign_9
3
save_14/restore_allNoOp^save_14/restore_shard
\
save_15/filename/inputConst*
_output_shapes
: *
valueB Bmodel*
dtype0
t
save_15/filenamePlaceholderWithDefaultsave_15/filename/input*
dtype0*
shape: *
_output_shapes
: 
k
save_15/ConstPlaceholderWithDefaultsave_15/filename*
_output_shapes
: *
dtype0*
shape: 

save_15/StringJoin/inputs_1Const*
_output_shapes
: *<
value3B1 B+_temp_afafa07d82a84a49b62f1804c9060cd2/part*
dtype0
~
save_15/StringJoin
StringJoinsave_15/Constsave_15/StringJoin/inputs_1*
N*
	separator *
_output_shapes
: 
T
save_15/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
_
save_15/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 

save_15/ShardedFilenameShardedFilenamesave_15/StringJoinsave_15/ShardedFilename/shardsave_15/num_shards*
_output_shapes
: 
ř
save_15/SaveV2/tensor_namesConst*
dtype0*¨
valueBDBbeta1_powerBbeta1_power_1Bbeta1_power_2Bbeta2_powerBbeta2_power_1Bbeta2_power_2Bpenalty/penalty_paramBpenalty/penalty_param/AdamBpenalty/penalty_param/Adam_1Bpenalty/penalty_param/Adam_2Bpenalty/penalty_param/Adam_3Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1B
pi/log_stdBpi/log_std/AdamBpi/log_std/Adam_1Bvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1*
_output_shapes
:D
ń
save_15/SaveV2/shape_and_slicesConst*
dtype0*
valueBDB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
_output_shapes
:D
Ö
save_15/SaveV2SaveV2save_15/ShardedFilenamesave_15/SaveV2/tensor_namessave_15/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta1_power_2beta2_powerbeta2_power_1beta2_power_2penalty/penalty_parampenalty/penalty_param/Adampenalty/penalty_param/Adam_1penalty/penalty_param/Adam_2penalty/penalty_param/Adam_3pi/dense/biaspi/dense/bias/Adampi/dense/bias/Adam_1pi/dense/kernelpi/dense/kernel/Adampi/dense/kernel/Adam_1pi/dense_1/biaspi/dense_1/bias/Adampi/dense_1/bias/Adam_1pi/dense_1/kernelpi/dense_1/kernel/Adampi/dense_1/kernel/Adam_1pi/dense_2/biaspi/dense_2/bias/Adampi/dense_2/bias/Adam_1pi/dense_2/kernelpi/dense_2/kernel/Adampi/dense_2/kernel/Adam_1
pi/log_stdpi/log_std/Adampi/log_std/Adam_1vc/dense/biasvc/dense/bias/Adamvc/dense/bias/Adam_1vc/dense/kernelvc/dense/kernel/Adamvc/dense/kernel/Adam_1vc/dense_1/biasvc/dense_1/bias/Adamvc/dense_1/bias/Adam_1vc/dense_1/kernelvc/dense_1/kernel/Adamvc/dense_1/kernel/Adam_1vc/dense_2/biasvc/dense_2/bias/Adamvc/dense_2/bias/Adam_1vc/dense_2/kernelvc/dense_2/kernel/Adamvc/dense_2/kernel/Adam_1vf/dense/biasvf/dense/bias/Adamvf/dense/bias/Adam_1vf/dense/kernelvf/dense/kernel/Adamvf/dense/kernel/Adam_1vf/dense_1/biasvf/dense_1/bias/Adamvf/dense_1/bias/Adam_1vf/dense_1/kernelvf/dense_1/kernel/Adamvf/dense_1/kernel/Adam_1vf/dense_2/biasvf/dense_2/bias/Adamvf/dense_2/bias/Adam_1vf/dense_2/kernelvf/dense_2/kernel/Adamvf/dense_2/kernel/Adam_1*R
dtypesH
F2D

save_15/control_dependencyIdentitysave_15/ShardedFilename^save_15/SaveV2**
_class 
loc:@save_15/ShardedFilename*
T0*
_output_shapes
: 
Ś
.save_15/MergeV2Checkpoints/checkpoint_prefixesPacksave_15/ShardedFilename^save_15/control_dependency*
_output_shapes
:*
N*
T0*

axis 

save_15/MergeV2CheckpointsMergeV2Checkpoints.save_15/MergeV2Checkpoints/checkpoint_prefixessave_15/Const*
delete_old_dirs(

save_15/IdentityIdentitysave_15/Const^save_15/MergeV2Checkpoints^save_15/control_dependency*
T0*
_output_shapes
: 
ű
save_15/RestoreV2/tensor_namesConst*¨
valueBDBbeta1_powerBbeta1_power_1Bbeta1_power_2Bbeta2_powerBbeta2_power_1Bbeta2_power_2Bpenalty/penalty_paramBpenalty/penalty_param/AdamBpenalty/penalty_param/Adam_1Bpenalty/penalty_param/Adam_2Bpenalty/penalty_param/Adam_3Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1B
pi/log_stdBpi/log_std/AdamBpi/log_std/Adam_1Bvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1*
_output_shapes
:D*
dtype0
ô
"save_15/RestoreV2/shape_and_slicesConst*
valueBDB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:D
î
save_15/RestoreV2	RestoreV2save_15/Constsave_15/RestoreV2/tensor_names"save_15/RestoreV2/shape_and_slices*Ś
_output_shapes
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*R
dtypesH
F2D
Ź
save_15/AssignAssignbeta1_powersave_15/RestoreV2*
use_locking(*(
_class
loc:@penalty/penalty_param*
T0*
_output_shapes
: *
validate_shape(
˛
save_15/Assign_1Assignbeta1_power_1save_15/RestoreV2:1*
validate_shape(*
_output_shapes
: *
T0*
use_locking(*(
_class
loc:@penalty/penalty_param
Ş
save_15/Assign_2Assignbeta1_power_2save_15/RestoreV2:2*
T0* 
_class
loc:@vc/dense/bias*
_output_shapes
: *
validate_shape(*
use_locking(
°
save_15/Assign_3Assignbeta2_powersave_15/RestoreV2:3*
T0*
validate_shape(*
_output_shapes
: *(
_class
loc:@penalty/penalty_param*
use_locking(
˛
save_15/Assign_4Assignbeta2_power_1save_15/RestoreV2:4*
use_locking(*
_output_shapes
: *
T0*
validate_shape(*(
_class
loc:@penalty/penalty_param
Ş
save_15/Assign_5Assignbeta2_power_2save_15/RestoreV2:5*
use_locking(* 
_class
loc:@vc/dense/bias*
_output_shapes
: *
T0*
validate_shape(
ş
save_15/Assign_6Assignpenalty/penalty_paramsave_15/RestoreV2:6*(
_class
loc:@penalty/penalty_param*
use_locking(*
validate_shape(*
T0*
_output_shapes
: 
ż
save_15/Assign_7Assignpenalty/penalty_param/Adamsave_15/RestoreV2:7*
validate_shape(*(
_class
loc:@penalty/penalty_param*
T0*
_output_shapes
: *
use_locking(
Á
save_15/Assign_8Assignpenalty/penalty_param/Adam_1save_15/RestoreV2:8*(
_class
loc:@penalty/penalty_param*
validate_shape(*
use_locking(*
_output_shapes
: *
T0
Á
save_15/Assign_9Assignpenalty/penalty_param/Adam_2save_15/RestoreV2:9*
validate_shape(*(
_class
loc:@penalty/penalty_param*
use_locking(*
_output_shapes
: *
T0
Ă
save_15/Assign_10Assignpenalty/penalty_param/Adam_3save_15/RestoreV2:10*
T0*
use_locking(*
validate_shape(*
_output_shapes
: *(
_class
loc:@penalty/penalty_param
ą
save_15/Assign_11Assignpi/dense/biassave_15/RestoreV2:11*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes	
:*
use_locking(
ś
save_15/Assign_12Assignpi/dense/bias/Adamsave_15/RestoreV2:12*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
use_locking(*
_output_shapes	
:
¸
save_15/Assign_13Assignpi/dense/bias/Adam_1save_15/RestoreV2:13* 
_class
loc:@pi/dense/bias*
use_locking(*
_output_shapes	
:*
T0*
validate_shape(
š
save_15/Assign_14Assignpi/dense/kernelsave_15/RestoreV2:14*
validate_shape(*
use_locking(*
T0*
_output_shapes
:	L*"
_class
loc:@pi/dense/kernel
ž
save_15/Assign_15Assignpi/dense/kernel/Adamsave_15/RestoreV2:15*
_output_shapes
:	L*
use_locking(*"
_class
loc:@pi/dense/kernel*
T0*
validate_shape(
Ŕ
save_15/Assign_16Assignpi/dense/kernel/Adam_1save_15/RestoreV2:16*
_output_shapes
:	L*"
_class
loc:@pi/dense/kernel*
T0*
validate_shape(*
use_locking(
ľ
save_15/Assign_17Assignpi/dense_1/biassave_15/RestoreV2:17*"
_class
loc:@pi/dense_1/bias*
use_locking(*
_output_shapes	
:*
validate_shape(*
T0
ş
save_15/Assign_18Assignpi/dense_1/bias/Adamsave_15/RestoreV2:18*
use_locking(*
_output_shapes	
:*"
_class
loc:@pi/dense_1/bias*
T0*
validate_shape(
ź
save_15/Assign_19Assignpi/dense_1/bias/Adam_1save_15/RestoreV2:19*
_output_shapes	
:*
validate_shape(*
use_locking(*
T0*"
_class
loc:@pi/dense_1/bias
ž
save_15/Assign_20Assignpi/dense_1/kernelsave_15/RestoreV2:20*
use_locking(*
validate_shape(*
T0*$
_class
loc:@pi/dense_1/kernel* 
_output_shapes
:

Ă
save_15/Assign_21Assignpi/dense_1/kernel/Adamsave_15/RestoreV2:21*
T0*$
_class
loc:@pi/dense_1/kernel*
validate_shape(*
use_locking(* 
_output_shapes
:

Ĺ
save_15/Assign_22Assignpi/dense_1/kernel/Adam_1save_15/RestoreV2:22*
validate_shape(*
use_locking(* 
_output_shapes
:
*
T0*$
_class
loc:@pi/dense_1/kernel
´
save_15/Assign_23Assignpi/dense_2/biassave_15/RestoreV2:23*
_output_shapes
:*
T0*
use_locking(*
validate_shape(*"
_class
loc:@pi/dense_2/bias
š
save_15/Assign_24Assignpi/dense_2/bias/Adamsave_15/RestoreV2:24*
use_locking(*
T0*
validate_shape(*
_output_shapes
:*"
_class
loc:@pi/dense_2/bias
ť
save_15/Assign_25Assignpi/dense_2/bias/Adam_1save_15/RestoreV2:25*
use_locking(*
validate_shape(*
_output_shapes
:*
T0*"
_class
loc:@pi/dense_2/bias
˝
save_15/Assign_26Assignpi/dense_2/kernelsave_15/RestoreV2:26*
T0*
use_locking(*$
_class
loc:@pi/dense_2/kernel*
validate_shape(*
_output_shapes
:	
Â
save_15/Assign_27Assignpi/dense_2/kernel/Adamsave_15/RestoreV2:27*
_output_shapes
:	*
validate_shape(*
T0*$
_class
loc:@pi/dense_2/kernel*
use_locking(
Ä
save_15/Assign_28Assignpi/dense_2/kernel/Adam_1save_15/RestoreV2:28*
validate_shape(*
T0*
_output_shapes
:	*$
_class
loc:@pi/dense_2/kernel*
use_locking(
Ş
save_15/Assign_29Assign
pi/log_stdsave_15/RestoreV2:29*
_output_shapes
:*
use_locking(*
_class
loc:@pi/log_std*
T0*
validate_shape(
Ż
save_15/Assign_30Assignpi/log_std/Adamsave_15/RestoreV2:30*
validate_shape(*
_output_shapes
:*
T0*
_class
loc:@pi/log_std*
use_locking(
ą
save_15/Assign_31Assignpi/log_std/Adam_1save_15/RestoreV2:31*
use_locking(*
_class
loc:@pi/log_std*
T0*
validate_shape(*
_output_shapes
:
ą
save_15/Assign_32Assignvc/dense/biassave_15/RestoreV2:32*
_output_shapes	
:* 
_class
loc:@vc/dense/bias*
use_locking(*
T0*
validate_shape(
ś
save_15/Assign_33Assignvc/dense/bias/Adamsave_15/RestoreV2:33*
validate_shape(* 
_class
loc:@vc/dense/bias*
T0*
use_locking(*
_output_shapes	
:
¸
save_15/Assign_34Assignvc/dense/bias/Adam_1save_15/RestoreV2:34*
validate_shape(*
T0* 
_class
loc:@vc/dense/bias*
use_locking(*
_output_shapes	
:
š
save_15/Assign_35Assignvc/dense/kernelsave_15/RestoreV2:35*
T0*
use_locking(*
validate_shape(*
_output_shapes
:	L*"
_class
loc:@vc/dense/kernel
ž
save_15/Assign_36Assignvc/dense/kernel/Adamsave_15/RestoreV2:36*"
_class
loc:@vc/dense/kernel*
validate_shape(*
_output_shapes
:	L*
T0*
use_locking(
Ŕ
save_15/Assign_37Assignvc/dense/kernel/Adam_1save_15/RestoreV2:37*
validate_shape(*
_output_shapes
:	L*"
_class
loc:@vc/dense/kernel*
use_locking(*
T0
ľ
save_15/Assign_38Assignvc/dense_1/biassave_15/RestoreV2:38*
use_locking(*
T0*"
_class
loc:@vc/dense_1/bias*
_output_shapes	
:*
validate_shape(
ş
save_15/Assign_39Assignvc/dense_1/bias/Adamsave_15/RestoreV2:39*
T0*
use_locking(*"
_class
loc:@vc/dense_1/bias*
validate_shape(*
_output_shapes	
:
ź
save_15/Assign_40Assignvc/dense_1/bias/Adam_1save_15/RestoreV2:40*
_output_shapes	
:*
validate_shape(*
use_locking(*"
_class
loc:@vc/dense_1/bias*
T0
ž
save_15/Assign_41Assignvc/dense_1/kernelsave_15/RestoreV2:41*
T0*
use_locking(*
validate_shape(* 
_output_shapes
:
*$
_class
loc:@vc/dense_1/kernel
Ă
save_15/Assign_42Assignvc/dense_1/kernel/Adamsave_15/RestoreV2:42*$
_class
loc:@vc/dense_1/kernel*
validate_shape(* 
_output_shapes
:
*
T0*
use_locking(
Ĺ
save_15/Assign_43Assignvc/dense_1/kernel/Adam_1save_15/RestoreV2:43*
use_locking(*$
_class
loc:@vc/dense_1/kernel*
T0*
validate_shape(* 
_output_shapes
:

´
save_15/Assign_44Assignvc/dense_2/biassave_15/RestoreV2:44*"
_class
loc:@vc/dense_2/bias*
_output_shapes
:*
validate_shape(*
use_locking(*
T0
š
save_15/Assign_45Assignvc/dense_2/bias/Adamsave_15/RestoreV2:45*
T0*
use_locking(*
_output_shapes
:*"
_class
loc:@vc/dense_2/bias*
validate_shape(
ť
save_15/Assign_46Assignvc/dense_2/bias/Adam_1save_15/RestoreV2:46*"
_class
loc:@vc/dense_2/bias*
use_locking(*
_output_shapes
:*
T0*
validate_shape(
˝
save_15/Assign_47Assignvc/dense_2/kernelsave_15/RestoreV2:47*
use_locking(*
_output_shapes
:	*
validate_shape(*$
_class
loc:@vc/dense_2/kernel*
T0
Â
save_15/Assign_48Assignvc/dense_2/kernel/Adamsave_15/RestoreV2:48*
validate_shape(*
use_locking(*
T0*$
_class
loc:@vc/dense_2/kernel*
_output_shapes
:	
Ä
save_15/Assign_49Assignvc/dense_2/kernel/Adam_1save_15/RestoreV2:49*$
_class
loc:@vc/dense_2/kernel*
_output_shapes
:	*
validate_shape(*
T0*
use_locking(
ą
save_15/Assign_50Assignvf/dense/biassave_15/RestoreV2:50*
use_locking(* 
_class
loc:@vf/dense/bias*
validate_shape(*
T0*
_output_shapes	
:
ś
save_15/Assign_51Assignvf/dense/bias/Adamsave_15/RestoreV2:51*
_output_shapes	
:*
use_locking(*
T0* 
_class
loc:@vf/dense/bias*
validate_shape(
¸
save_15/Assign_52Assignvf/dense/bias/Adam_1save_15/RestoreV2:52* 
_class
loc:@vf/dense/bias*
use_locking(*
validate_shape(*
T0*
_output_shapes	
:
š
save_15/Assign_53Assignvf/dense/kernelsave_15/RestoreV2:53*
use_locking(*
T0*
_output_shapes
:	L*
validate_shape(*"
_class
loc:@vf/dense/kernel
ž
save_15/Assign_54Assignvf/dense/kernel/Adamsave_15/RestoreV2:54*
validate_shape(*
T0*
use_locking(*
_output_shapes
:	L*"
_class
loc:@vf/dense/kernel
Ŕ
save_15/Assign_55Assignvf/dense/kernel/Adam_1save_15/RestoreV2:55*
T0*"
_class
loc:@vf/dense/kernel*
validate_shape(*
use_locking(*
_output_shapes
:	L
ľ
save_15/Assign_56Assignvf/dense_1/biassave_15/RestoreV2:56*
use_locking(*
validate_shape(*
T0*
_output_shapes	
:*"
_class
loc:@vf/dense_1/bias
ş
save_15/Assign_57Assignvf/dense_1/bias/Adamsave_15/RestoreV2:57*
use_locking(*"
_class
loc:@vf/dense_1/bias*
validate_shape(*
T0*
_output_shapes	
:
ź
save_15/Assign_58Assignvf/dense_1/bias/Adam_1save_15/RestoreV2:58*
use_locking(*"
_class
loc:@vf/dense_1/bias*
validate_shape(*
_output_shapes	
:*
T0
ž
save_15/Assign_59Assignvf/dense_1/kernelsave_15/RestoreV2:59* 
_output_shapes
:
*$
_class
loc:@vf/dense_1/kernel*
use_locking(*
T0*
validate_shape(
Ă
save_15/Assign_60Assignvf/dense_1/kernel/Adamsave_15/RestoreV2:60* 
_output_shapes
:
*
T0*
validate_shape(*$
_class
loc:@vf/dense_1/kernel*
use_locking(
Ĺ
save_15/Assign_61Assignvf/dense_1/kernel/Adam_1save_15/RestoreV2:61*
use_locking(*
T0*$
_class
loc:@vf/dense_1/kernel*
validate_shape(* 
_output_shapes
:

´
save_15/Assign_62Assignvf/dense_2/biassave_15/RestoreV2:62*
use_locking(*"
_class
loc:@vf/dense_2/bias*
_output_shapes
:*
T0*
validate_shape(
š
save_15/Assign_63Assignvf/dense_2/bias/Adamsave_15/RestoreV2:63*
use_locking(*"
_class
loc:@vf/dense_2/bias*
validate_shape(*
T0*
_output_shapes
:
ť
save_15/Assign_64Assignvf/dense_2/bias/Adam_1save_15/RestoreV2:64*
T0*
_output_shapes
:*
validate_shape(*"
_class
loc:@vf/dense_2/bias*
use_locking(
˝
save_15/Assign_65Assignvf/dense_2/kernelsave_15/RestoreV2:65*
T0*
validate_shape(*
use_locking(*
_output_shapes
:	*$
_class
loc:@vf/dense_2/kernel
Â
save_15/Assign_66Assignvf/dense_2/kernel/Adamsave_15/RestoreV2:66*
validate_shape(*$
_class
loc:@vf/dense_2/kernel*
T0*
_output_shapes
:	*
use_locking(
Ä
save_15/Assign_67Assignvf/dense_2/kernel/Adam_1save_15/RestoreV2:67*$
_class
loc:@vf/dense_2/kernel*
use_locking(*
_output_shapes
:	*
validate_shape(*
T0
á

save_15/restore_shardNoOp^save_15/Assign^save_15/Assign_1^save_15/Assign_10^save_15/Assign_11^save_15/Assign_12^save_15/Assign_13^save_15/Assign_14^save_15/Assign_15^save_15/Assign_16^save_15/Assign_17^save_15/Assign_18^save_15/Assign_19^save_15/Assign_2^save_15/Assign_20^save_15/Assign_21^save_15/Assign_22^save_15/Assign_23^save_15/Assign_24^save_15/Assign_25^save_15/Assign_26^save_15/Assign_27^save_15/Assign_28^save_15/Assign_29^save_15/Assign_3^save_15/Assign_30^save_15/Assign_31^save_15/Assign_32^save_15/Assign_33^save_15/Assign_34^save_15/Assign_35^save_15/Assign_36^save_15/Assign_37^save_15/Assign_38^save_15/Assign_39^save_15/Assign_4^save_15/Assign_40^save_15/Assign_41^save_15/Assign_42^save_15/Assign_43^save_15/Assign_44^save_15/Assign_45^save_15/Assign_46^save_15/Assign_47^save_15/Assign_48^save_15/Assign_49^save_15/Assign_5^save_15/Assign_50^save_15/Assign_51^save_15/Assign_52^save_15/Assign_53^save_15/Assign_54^save_15/Assign_55^save_15/Assign_56^save_15/Assign_57^save_15/Assign_58^save_15/Assign_59^save_15/Assign_6^save_15/Assign_60^save_15/Assign_61^save_15/Assign_62^save_15/Assign_63^save_15/Assign_64^save_15/Assign_65^save_15/Assign_66^save_15/Assign_67^save_15/Assign_7^save_15/Assign_8^save_15/Assign_9
3
save_15/restore_allNoOp^save_15/restore_shard
\
save_16/filename/inputConst*
dtype0*
valueB Bmodel*
_output_shapes
: 
t
save_16/filenamePlaceholderWithDefaultsave_16/filename/input*
_output_shapes
: *
shape: *
dtype0
k
save_16/ConstPlaceholderWithDefaultsave_16/filename*
shape: *
_output_shapes
: *
dtype0

save_16/StringJoin/inputs_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+_temp_36c9926c69d64e5ca1f736c97e291e9b/part
~
save_16/StringJoin
StringJoinsave_16/Constsave_16/StringJoin/inputs_1*
N*
_output_shapes
: *
	separator 
T
save_16/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
_
save_16/ShardedFilename/shardConst*
dtype0*
value	B : *
_output_shapes
: 

save_16/ShardedFilenameShardedFilenamesave_16/StringJoinsave_16/ShardedFilename/shardsave_16/num_shards*
_output_shapes
: 
ř
save_16/SaveV2/tensor_namesConst*
dtype0*¨
valueBDBbeta1_powerBbeta1_power_1Bbeta1_power_2Bbeta2_powerBbeta2_power_1Bbeta2_power_2Bpenalty/penalty_paramBpenalty/penalty_param/AdamBpenalty/penalty_param/Adam_1Bpenalty/penalty_param/Adam_2Bpenalty/penalty_param/Adam_3Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1B
pi/log_stdBpi/log_std/AdamBpi/log_std/Adam_1Bvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1*
_output_shapes
:D
ń
save_16/SaveV2/shape_and_slicesConst*
valueBDB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
_output_shapes
:D*
dtype0
Ö
save_16/SaveV2SaveV2save_16/ShardedFilenamesave_16/SaveV2/tensor_namessave_16/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta1_power_2beta2_powerbeta2_power_1beta2_power_2penalty/penalty_parampenalty/penalty_param/Adampenalty/penalty_param/Adam_1penalty/penalty_param/Adam_2penalty/penalty_param/Adam_3pi/dense/biaspi/dense/bias/Adampi/dense/bias/Adam_1pi/dense/kernelpi/dense/kernel/Adampi/dense/kernel/Adam_1pi/dense_1/biaspi/dense_1/bias/Adampi/dense_1/bias/Adam_1pi/dense_1/kernelpi/dense_1/kernel/Adampi/dense_1/kernel/Adam_1pi/dense_2/biaspi/dense_2/bias/Adampi/dense_2/bias/Adam_1pi/dense_2/kernelpi/dense_2/kernel/Adampi/dense_2/kernel/Adam_1
pi/log_stdpi/log_std/Adampi/log_std/Adam_1vc/dense/biasvc/dense/bias/Adamvc/dense/bias/Adam_1vc/dense/kernelvc/dense/kernel/Adamvc/dense/kernel/Adam_1vc/dense_1/biasvc/dense_1/bias/Adamvc/dense_1/bias/Adam_1vc/dense_1/kernelvc/dense_1/kernel/Adamvc/dense_1/kernel/Adam_1vc/dense_2/biasvc/dense_2/bias/Adamvc/dense_2/bias/Adam_1vc/dense_2/kernelvc/dense_2/kernel/Adamvc/dense_2/kernel/Adam_1vf/dense/biasvf/dense/bias/Adamvf/dense/bias/Adam_1vf/dense/kernelvf/dense/kernel/Adamvf/dense/kernel/Adam_1vf/dense_1/biasvf/dense_1/bias/Adamvf/dense_1/bias/Adam_1vf/dense_1/kernelvf/dense_1/kernel/Adamvf/dense_1/kernel/Adam_1vf/dense_2/biasvf/dense_2/bias/Adamvf/dense_2/bias/Adam_1vf/dense_2/kernelvf/dense_2/kernel/Adamvf/dense_2/kernel/Adam_1*R
dtypesH
F2D

save_16/control_dependencyIdentitysave_16/ShardedFilename^save_16/SaveV2**
_class 
loc:@save_16/ShardedFilename*
_output_shapes
: *
T0
Ś
.save_16/MergeV2Checkpoints/checkpoint_prefixesPacksave_16/ShardedFilename^save_16/control_dependency*

axis *
N*
T0*
_output_shapes
:

save_16/MergeV2CheckpointsMergeV2Checkpoints.save_16/MergeV2Checkpoints/checkpoint_prefixessave_16/Const*
delete_old_dirs(

save_16/IdentityIdentitysave_16/Const^save_16/MergeV2Checkpoints^save_16/control_dependency*
_output_shapes
: *
T0
ű
save_16/RestoreV2/tensor_namesConst*
_output_shapes
:D*
dtype0*¨
valueBDBbeta1_powerBbeta1_power_1Bbeta1_power_2Bbeta2_powerBbeta2_power_1Bbeta2_power_2Bpenalty/penalty_paramBpenalty/penalty_param/AdamBpenalty/penalty_param/Adam_1Bpenalty/penalty_param/Adam_2Bpenalty/penalty_param/Adam_3Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1B
pi/log_stdBpi/log_std/AdamBpi/log_std/Adam_1Bvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1
ô
"save_16/RestoreV2/shape_and_slicesConst*
valueBDB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:D
î
save_16/RestoreV2	RestoreV2save_16/Constsave_16/RestoreV2/tensor_names"save_16/RestoreV2/shape_and_slices*R
dtypesH
F2D*Ś
_output_shapes
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
Ź
save_16/AssignAssignbeta1_powersave_16/RestoreV2*
T0*
_output_shapes
: *
use_locking(*
validate_shape(*(
_class
loc:@penalty/penalty_param
˛
save_16/Assign_1Assignbeta1_power_1save_16/RestoreV2:1*(
_class
loc:@penalty/penalty_param*
validate_shape(*
T0*
_output_shapes
: *
use_locking(
Ş
save_16/Assign_2Assignbeta1_power_2save_16/RestoreV2:2*
_output_shapes
: *
use_locking(* 
_class
loc:@vc/dense/bias*
validate_shape(*
T0
°
save_16/Assign_3Assignbeta2_powersave_16/RestoreV2:3*
validate_shape(*
_output_shapes
: *
T0*(
_class
loc:@penalty/penalty_param*
use_locking(
˛
save_16/Assign_4Assignbeta2_power_1save_16/RestoreV2:4*
T0*(
_class
loc:@penalty/penalty_param*
_output_shapes
: *
validate_shape(*
use_locking(
Ş
save_16/Assign_5Assignbeta2_power_2save_16/RestoreV2:5*
use_locking(* 
_class
loc:@vc/dense/bias*
T0*
validate_shape(*
_output_shapes
: 
ş
save_16/Assign_6Assignpenalty/penalty_paramsave_16/RestoreV2:6*
use_locking(*(
_class
loc:@penalty/penalty_param*
T0*
_output_shapes
: *
validate_shape(
ż
save_16/Assign_7Assignpenalty/penalty_param/Adamsave_16/RestoreV2:7*
validate_shape(*
T0*(
_class
loc:@penalty/penalty_param*
_output_shapes
: *
use_locking(
Á
save_16/Assign_8Assignpenalty/penalty_param/Adam_1save_16/RestoreV2:8*
validate_shape(*(
_class
loc:@penalty/penalty_param*
T0*
_output_shapes
: *
use_locking(
Á
save_16/Assign_9Assignpenalty/penalty_param/Adam_2save_16/RestoreV2:9*
validate_shape(*
use_locking(*(
_class
loc:@penalty/penalty_param*
T0*
_output_shapes
: 
Ă
save_16/Assign_10Assignpenalty/penalty_param/Adam_3save_16/RestoreV2:10*
T0*
_output_shapes
: *
validate_shape(*(
_class
loc:@penalty/penalty_param*
use_locking(
ą
save_16/Assign_11Assignpi/dense/biassave_16/RestoreV2:11*
T0*
validate_shape(*
_output_shapes	
:* 
_class
loc:@pi/dense/bias*
use_locking(
ś
save_16/Assign_12Assignpi/dense/bias/Adamsave_16/RestoreV2:12*
T0*
validate_shape(*
use_locking(*
_output_shapes	
:* 
_class
loc:@pi/dense/bias
¸
save_16/Assign_13Assignpi/dense/bias/Adam_1save_16/RestoreV2:13*
use_locking(*
validate_shape(* 
_class
loc:@pi/dense/bias*
T0*
_output_shapes	
:
š
save_16/Assign_14Assignpi/dense/kernelsave_16/RestoreV2:14*
_output_shapes
:	L*
use_locking(*
validate_shape(*"
_class
loc:@pi/dense/kernel*
T0
ž
save_16/Assign_15Assignpi/dense/kernel/Adamsave_16/RestoreV2:15*
_output_shapes
:	L*"
_class
loc:@pi/dense/kernel*
validate_shape(*
use_locking(*
T0
Ŕ
save_16/Assign_16Assignpi/dense/kernel/Adam_1save_16/RestoreV2:16*"
_class
loc:@pi/dense/kernel*
use_locking(*
validate_shape(*
_output_shapes
:	L*
T0
ľ
save_16/Assign_17Assignpi/dense_1/biassave_16/RestoreV2:17*
use_locking(*"
_class
loc:@pi/dense_1/bias*
validate_shape(*
_output_shapes	
:*
T0
ş
save_16/Assign_18Assignpi/dense_1/bias/Adamsave_16/RestoreV2:18*
validate_shape(*"
_class
loc:@pi/dense_1/bias*
T0*
_output_shapes	
:*
use_locking(
ź
save_16/Assign_19Assignpi/dense_1/bias/Adam_1save_16/RestoreV2:19*
use_locking(*
T0*
validate_shape(*"
_class
loc:@pi/dense_1/bias*
_output_shapes	
:
ž
save_16/Assign_20Assignpi/dense_1/kernelsave_16/RestoreV2:20*
T0* 
_output_shapes
:
*$
_class
loc:@pi/dense_1/kernel*
validate_shape(*
use_locking(
Ă
save_16/Assign_21Assignpi/dense_1/kernel/Adamsave_16/RestoreV2:21*
validate_shape(*
T0* 
_output_shapes
:
*
use_locking(*$
_class
loc:@pi/dense_1/kernel
Ĺ
save_16/Assign_22Assignpi/dense_1/kernel/Adam_1save_16/RestoreV2:22*
T0*
use_locking(*
validate_shape(*$
_class
loc:@pi/dense_1/kernel* 
_output_shapes
:

´
save_16/Assign_23Assignpi/dense_2/biassave_16/RestoreV2:23*
T0*
validate_shape(*
_output_shapes
:*"
_class
loc:@pi/dense_2/bias*
use_locking(
š
save_16/Assign_24Assignpi/dense_2/bias/Adamsave_16/RestoreV2:24*
_output_shapes
:*"
_class
loc:@pi/dense_2/bias*
validate_shape(*
T0*
use_locking(
ť
save_16/Assign_25Assignpi/dense_2/bias/Adam_1save_16/RestoreV2:25*"
_class
loc:@pi/dense_2/bias*
use_locking(*
validate_shape(*
_output_shapes
:*
T0
˝
save_16/Assign_26Assignpi/dense_2/kernelsave_16/RestoreV2:26*
T0*
_output_shapes
:	*
use_locking(*
validate_shape(*$
_class
loc:@pi/dense_2/kernel
Â
save_16/Assign_27Assignpi/dense_2/kernel/Adamsave_16/RestoreV2:27*$
_class
loc:@pi/dense_2/kernel*
use_locking(*
validate_shape(*
_output_shapes
:	*
T0
Ä
save_16/Assign_28Assignpi/dense_2/kernel/Adam_1save_16/RestoreV2:28*
use_locking(*
validate_shape(*$
_class
loc:@pi/dense_2/kernel*
_output_shapes
:	*
T0
Ş
save_16/Assign_29Assign
pi/log_stdsave_16/RestoreV2:29*
_output_shapes
:*
use_locking(*
validate_shape(*
_class
loc:@pi/log_std*
T0
Ż
save_16/Assign_30Assignpi/log_std/Adamsave_16/RestoreV2:30*
validate_shape(*
_class
loc:@pi/log_std*
_output_shapes
:*
use_locking(*
T0
ą
save_16/Assign_31Assignpi/log_std/Adam_1save_16/RestoreV2:31*
use_locking(*
_output_shapes
:*
_class
loc:@pi/log_std*
T0*
validate_shape(
ą
save_16/Assign_32Assignvc/dense/biassave_16/RestoreV2:32* 
_class
loc:@vc/dense/bias*
validate_shape(*
use_locking(*
_output_shapes	
:*
T0
ś
save_16/Assign_33Assignvc/dense/bias/Adamsave_16/RestoreV2:33* 
_class
loc:@vc/dense/bias*
_output_shapes	
:*
use_locking(*
T0*
validate_shape(
¸
save_16/Assign_34Assignvc/dense/bias/Adam_1save_16/RestoreV2:34*
T0*
_output_shapes	
:*
validate_shape(*
use_locking(* 
_class
loc:@vc/dense/bias
š
save_16/Assign_35Assignvc/dense/kernelsave_16/RestoreV2:35*
use_locking(*"
_class
loc:@vc/dense/kernel*
validate_shape(*
T0*
_output_shapes
:	L
ž
save_16/Assign_36Assignvc/dense/kernel/Adamsave_16/RestoreV2:36*"
_class
loc:@vc/dense/kernel*
validate_shape(*
use_locking(*
T0*
_output_shapes
:	L
Ŕ
save_16/Assign_37Assignvc/dense/kernel/Adam_1save_16/RestoreV2:37*
T0*
_output_shapes
:	L*
use_locking(*
validate_shape(*"
_class
loc:@vc/dense/kernel
ľ
save_16/Assign_38Assignvc/dense_1/biassave_16/RestoreV2:38*
validate_shape(*
T0*
use_locking(*"
_class
loc:@vc/dense_1/bias*
_output_shapes	
:
ş
save_16/Assign_39Assignvc/dense_1/bias/Adamsave_16/RestoreV2:39*
use_locking(*"
_class
loc:@vc/dense_1/bias*
T0*
validate_shape(*
_output_shapes	
:
ź
save_16/Assign_40Assignvc/dense_1/bias/Adam_1save_16/RestoreV2:40*
use_locking(*
_output_shapes	
:*"
_class
loc:@vc/dense_1/bias*
T0*
validate_shape(
ž
save_16/Assign_41Assignvc/dense_1/kernelsave_16/RestoreV2:41*
validate_shape(*
use_locking(* 
_output_shapes
:
*
T0*$
_class
loc:@vc/dense_1/kernel
Ă
save_16/Assign_42Assignvc/dense_1/kernel/Adamsave_16/RestoreV2:42*$
_class
loc:@vc/dense_1/kernel* 
_output_shapes
:
*
validate_shape(*
use_locking(*
T0
Ĺ
save_16/Assign_43Assignvc/dense_1/kernel/Adam_1save_16/RestoreV2:43*
T0*
validate_shape(*
use_locking(* 
_output_shapes
:
*$
_class
loc:@vc/dense_1/kernel
´
save_16/Assign_44Assignvc/dense_2/biassave_16/RestoreV2:44*
validate_shape(*
use_locking(*
_output_shapes
:*"
_class
loc:@vc/dense_2/bias*
T0
š
save_16/Assign_45Assignvc/dense_2/bias/Adamsave_16/RestoreV2:45*
T0*
_output_shapes
:*"
_class
loc:@vc/dense_2/bias*
validate_shape(*
use_locking(
ť
save_16/Assign_46Assignvc/dense_2/bias/Adam_1save_16/RestoreV2:46*
use_locking(*"
_class
loc:@vc/dense_2/bias*
_output_shapes
:*
T0*
validate_shape(
˝
save_16/Assign_47Assignvc/dense_2/kernelsave_16/RestoreV2:47*
validate_shape(*
use_locking(*$
_class
loc:@vc/dense_2/kernel*
_output_shapes
:	*
T0
Â
save_16/Assign_48Assignvc/dense_2/kernel/Adamsave_16/RestoreV2:48*
_output_shapes
:	*
validate_shape(*
use_locking(*
T0*$
_class
loc:@vc/dense_2/kernel
Ä
save_16/Assign_49Assignvc/dense_2/kernel/Adam_1save_16/RestoreV2:49*
_output_shapes
:	*
T0*
validate_shape(*
use_locking(*$
_class
loc:@vc/dense_2/kernel
ą
save_16/Assign_50Assignvf/dense/biassave_16/RestoreV2:50* 
_class
loc:@vf/dense/bias*
T0*
use_locking(*
_output_shapes	
:*
validate_shape(
ś
save_16/Assign_51Assignvf/dense/bias/Adamsave_16/RestoreV2:51* 
_class
loc:@vf/dense/bias*
_output_shapes	
:*
T0*
use_locking(*
validate_shape(
¸
save_16/Assign_52Assignvf/dense/bias/Adam_1save_16/RestoreV2:52*
T0*
use_locking(*
_output_shapes	
:* 
_class
loc:@vf/dense/bias*
validate_shape(
š
save_16/Assign_53Assignvf/dense/kernelsave_16/RestoreV2:53*
_output_shapes
:	L*"
_class
loc:@vf/dense/kernel*
T0*
validate_shape(*
use_locking(
ž
save_16/Assign_54Assignvf/dense/kernel/Adamsave_16/RestoreV2:54*
use_locking(*
_output_shapes
:	L*
T0*
validate_shape(*"
_class
loc:@vf/dense/kernel
Ŕ
save_16/Assign_55Assignvf/dense/kernel/Adam_1save_16/RestoreV2:55*"
_class
loc:@vf/dense/kernel*
use_locking(*
validate_shape(*
_output_shapes
:	L*
T0
ľ
save_16/Assign_56Assignvf/dense_1/biassave_16/RestoreV2:56*
use_locking(*
T0*
validate_shape(*
_output_shapes	
:*"
_class
loc:@vf/dense_1/bias
ş
save_16/Assign_57Assignvf/dense_1/bias/Adamsave_16/RestoreV2:57*
use_locking(*
T0*
_output_shapes	
:*
validate_shape(*"
_class
loc:@vf/dense_1/bias
ź
save_16/Assign_58Assignvf/dense_1/bias/Adam_1save_16/RestoreV2:58*
_output_shapes	
:*
validate_shape(*"
_class
loc:@vf/dense_1/bias*
use_locking(*
T0
ž
save_16/Assign_59Assignvf/dense_1/kernelsave_16/RestoreV2:59* 
_output_shapes
:
*
use_locking(*
validate_shape(*$
_class
loc:@vf/dense_1/kernel*
T0
Ă
save_16/Assign_60Assignvf/dense_1/kernel/Adamsave_16/RestoreV2:60*
use_locking(* 
_output_shapes
:
*$
_class
loc:@vf/dense_1/kernel*
validate_shape(*
T0
Ĺ
save_16/Assign_61Assignvf/dense_1/kernel/Adam_1save_16/RestoreV2:61*
T0*
validate_shape(* 
_output_shapes
:
*
use_locking(*$
_class
loc:@vf/dense_1/kernel
´
save_16/Assign_62Assignvf/dense_2/biassave_16/RestoreV2:62*
validate_shape(*"
_class
loc:@vf/dense_2/bias*
use_locking(*
_output_shapes
:*
T0
š
save_16/Assign_63Assignvf/dense_2/bias/Adamsave_16/RestoreV2:63*
validate_shape(*"
_class
loc:@vf/dense_2/bias*
T0*
_output_shapes
:*
use_locking(
ť
save_16/Assign_64Assignvf/dense_2/bias/Adam_1save_16/RestoreV2:64*"
_class
loc:@vf/dense_2/bias*
T0*
_output_shapes
:*
validate_shape(*
use_locking(
˝
save_16/Assign_65Assignvf/dense_2/kernelsave_16/RestoreV2:65*$
_class
loc:@vf/dense_2/kernel*
_output_shapes
:	*
use_locking(*
T0*
validate_shape(
Â
save_16/Assign_66Assignvf/dense_2/kernel/Adamsave_16/RestoreV2:66*
use_locking(*
_output_shapes
:	*
T0*
validate_shape(*$
_class
loc:@vf/dense_2/kernel
Ä
save_16/Assign_67Assignvf/dense_2/kernel/Adam_1save_16/RestoreV2:67*
validate_shape(*$
_class
loc:@vf/dense_2/kernel*
T0*
_output_shapes
:	*
use_locking(
á

save_16/restore_shardNoOp^save_16/Assign^save_16/Assign_1^save_16/Assign_10^save_16/Assign_11^save_16/Assign_12^save_16/Assign_13^save_16/Assign_14^save_16/Assign_15^save_16/Assign_16^save_16/Assign_17^save_16/Assign_18^save_16/Assign_19^save_16/Assign_2^save_16/Assign_20^save_16/Assign_21^save_16/Assign_22^save_16/Assign_23^save_16/Assign_24^save_16/Assign_25^save_16/Assign_26^save_16/Assign_27^save_16/Assign_28^save_16/Assign_29^save_16/Assign_3^save_16/Assign_30^save_16/Assign_31^save_16/Assign_32^save_16/Assign_33^save_16/Assign_34^save_16/Assign_35^save_16/Assign_36^save_16/Assign_37^save_16/Assign_38^save_16/Assign_39^save_16/Assign_4^save_16/Assign_40^save_16/Assign_41^save_16/Assign_42^save_16/Assign_43^save_16/Assign_44^save_16/Assign_45^save_16/Assign_46^save_16/Assign_47^save_16/Assign_48^save_16/Assign_49^save_16/Assign_5^save_16/Assign_50^save_16/Assign_51^save_16/Assign_52^save_16/Assign_53^save_16/Assign_54^save_16/Assign_55^save_16/Assign_56^save_16/Assign_57^save_16/Assign_58^save_16/Assign_59^save_16/Assign_6^save_16/Assign_60^save_16/Assign_61^save_16/Assign_62^save_16/Assign_63^save_16/Assign_64^save_16/Assign_65^save_16/Assign_66^save_16/Assign_67^save_16/Assign_7^save_16/Assign_8^save_16/Assign_9
3
save_16/restore_allNoOp^save_16/restore_shard
\
save_17/filename/inputConst*
valueB Bmodel*
_output_shapes
: *
dtype0
t
save_17/filenamePlaceholderWithDefaultsave_17/filename/input*
dtype0*
_output_shapes
: *
shape: 
k
save_17/ConstPlaceholderWithDefaultsave_17/filename*
shape: *
_output_shapes
: *
dtype0

save_17/StringJoin/inputs_1Const*
dtype0*
_output_shapes
: *<
value3B1 B+_temp_b5939d755ef74fb1b2468019d4c650c1/part
~
save_17/StringJoin
StringJoinsave_17/Constsave_17/StringJoin/inputs_1*
N*
_output_shapes
: *
	separator 
T
save_17/num_shardsConst*
dtype0*
value	B :*
_output_shapes
: 
_
save_17/ShardedFilename/shardConst*
_output_shapes
: *
value	B : *
dtype0

save_17/ShardedFilenameShardedFilenamesave_17/StringJoinsave_17/ShardedFilename/shardsave_17/num_shards*
_output_shapes
: 
ř
save_17/SaveV2/tensor_namesConst*
_output_shapes
:D*¨
valueBDBbeta1_powerBbeta1_power_1Bbeta1_power_2Bbeta2_powerBbeta2_power_1Bbeta2_power_2Bpenalty/penalty_paramBpenalty/penalty_param/AdamBpenalty/penalty_param/Adam_1Bpenalty/penalty_param/Adam_2Bpenalty/penalty_param/Adam_3Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1B
pi/log_stdBpi/log_std/AdamBpi/log_std/Adam_1Bvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1*
dtype0
ń
save_17/SaveV2/shape_and_slicesConst*
_output_shapes
:D*
valueBDB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0
Ö
save_17/SaveV2SaveV2save_17/ShardedFilenamesave_17/SaveV2/tensor_namessave_17/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta1_power_2beta2_powerbeta2_power_1beta2_power_2penalty/penalty_parampenalty/penalty_param/Adampenalty/penalty_param/Adam_1penalty/penalty_param/Adam_2penalty/penalty_param/Adam_3pi/dense/biaspi/dense/bias/Adampi/dense/bias/Adam_1pi/dense/kernelpi/dense/kernel/Adampi/dense/kernel/Adam_1pi/dense_1/biaspi/dense_1/bias/Adampi/dense_1/bias/Adam_1pi/dense_1/kernelpi/dense_1/kernel/Adampi/dense_1/kernel/Adam_1pi/dense_2/biaspi/dense_2/bias/Adampi/dense_2/bias/Adam_1pi/dense_2/kernelpi/dense_2/kernel/Adampi/dense_2/kernel/Adam_1
pi/log_stdpi/log_std/Adampi/log_std/Adam_1vc/dense/biasvc/dense/bias/Adamvc/dense/bias/Adam_1vc/dense/kernelvc/dense/kernel/Adamvc/dense/kernel/Adam_1vc/dense_1/biasvc/dense_1/bias/Adamvc/dense_1/bias/Adam_1vc/dense_1/kernelvc/dense_1/kernel/Adamvc/dense_1/kernel/Adam_1vc/dense_2/biasvc/dense_2/bias/Adamvc/dense_2/bias/Adam_1vc/dense_2/kernelvc/dense_2/kernel/Adamvc/dense_2/kernel/Adam_1vf/dense/biasvf/dense/bias/Adamvf/dense/bias/Adam_1vf/dense/kernelvf/dense/kernel/Adamvf/dense/kernel/Adam_1vf/dense_1/biasvf/dense_1/bias/Adamvf/dense_1/bias/Adam_1vf/dense_1/kernelvf/dense_1/kernel/Adamvf/dense_1/kernel/Adam_1vf/dense_2/biasvf/dense_2/bias/Adamvf/dense_2/bias/Adam_1vf/dense_2/kernelvf/dense_2/kernel/Adamvf/dense_2/kernel/Adam_1*R
dtypesH
F2D

save_17/control_dependencyIdentitysave_17/ShardedFilename^save_17/SaveV2*
_output_shapes
: *
T0**
_class 
loc:@save_17/ShardedFilename
Ś
.save_17/MergeV2Checkpoints/checkpoint_prefixesPacksave_17/ShardedFilename^save_17/control_dependency*
T0*

axis *
N*
_output_shapes
:

save_17/MergeV2CheckpointsMergeV2Checkpoints.save_17/MergeV2Checkpoints/checkpoint_prefixessave_17/Const*
delete_old_dirs(

save_17/IdentityIdentitysave_17/Const^save_17/MergeV2Checkpoints^save_17/control_dependency*
_output_shapes
: *
T0
ű
save_17/RestoreV2/tensor_namesConst*
_output_shapes
:D*¨
valueBDBbeta1_powerBbeta1_power_1Bbeta1_power_2Bbeta2_powerBbeta2_power_1Bbeta2_power_2Bpenalty/penalty_paramBpenalty/penalty_param/AdamBpenalty/penalty_param/Adam_1Bpenalty/penalty_param/Adam_2Bpenalty/penalty_param/Adam_3Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1B
pi/log_stdBpi/log_std/AdamBpi/log_std/Adam_1Bvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1*
dtype0
ô
"save_17/RestoreV2/shape_and_slicesConst*
dtype0*
_output_shapes
:D*
valueBDB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
î
save_17/RestoreV2	RestoreV2save_17/Constsave_17/RestoreV2/tensor_names"save_17/RestoreV2/shape_and_slices*Ś
_output_shapes
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*R
dtypesH
F2D
Ź
save_17/AssignAssignbeta1_powersave_17/RestoreV2*
validate_shape(*
_output_shapes
: *(
_class
loc:@penalty/penalty_param*
use_locking(*
T0
˛
save_17/Assign_1Assignbeta1_power_1save_17/RestoreV2:1*
validate_shape(*(
_class
loc:@penalty/penalty_param*
_output_shapes
: *
use_locking(*
T0
Ş
save_17/Assign_2Assignbeta1_power_2save_17/RestoreV2:2* 
_class
loc:@vc/dense/bias*
validate_shape(*
_output_shapes
: *
T0*
use_locking(
°
save_17/Assign_3Assignbeta2_powersave_17/RestoreV2:3*
use_locking(*(
_class
loc:@penalty/penalty_param*
validate_shape(*
_output_shapes
: *
T0
˛
save_17/Assign_4Assignbeta2_power_1save_17/RestoreV2:4*
_output_shapes
: *
validate_shape(*(
_class
loc:@penalty/penalty_param*
T0*
use_locking(
Ş
save_17/Assign_5Assignbeta2_power_2save_17/RestoreV2:5*
use_locking(*
T0*
validate_shape(*
_output_shapes
: * 
_class
loc:@vc/dense/bias
ş
save_17/Assign_6Assignpenalty/penalty_paramsave_17/RestoreV2:6*
use_locking(*(
_class
loc:@penalty/penalty_param*
_output_shapes
: *
T0*
validate_shape(
ż
save_17/Assign_7Assignpenalty/penalty_param/Adamsave_17/RestoreV2:7*
use_locking(*
T0*
_output_shapes
: *(
_class
loc:@penalty/penalty_param*
validate_shape(
Á
save_17/Assign_8Assignpenalty/penalty_param/Adam_1save_17/RestoreV2:8*
T0*(
_class
loc:@penalty/penalty_param*
use_locking(*
validate_shape(*
_output_shapes
: 
Á
save_17/Assign_9Assignpenalty/penalty_param/Adam_2save_17/RestoreV2:9*
validate_shape(*
T0*(
_class
loc:@penalty/penalty_param*
_output_shapes
: *
use_locking(
Ă
save_17/Assign_10Assignpenalty/penalty_param/Adam_3save_17/RestoreV2:10*
validate_shape(*
_output_shapes
: *
T0*
use_locking(*(
_class
loc:@penalty/penalty_param
ą
save_17/Assign_11Assignpi/dense/biassave_17/RestoreV2:11*
_output_shapes	
:* 
_class
loc:@pi/dense/bias*
use_locking(*
validate_shape(*
T0
ś
save_17/Assign_12Assignpi/dense/bias/Adamsave_17/RestoreV2:12*
use_locking(*
_output_shapes	
:*
T0*
validate_shape(* 
_class
loc:@pi/dense/bias
¸
save_17/Assign_13Assignpi/dense/bias/Adam_1save_17/RestoreV2:13*
use_locking(*
validate_shape(*
T0* 
_class
loc:@pi/dense/bias*
_output_shapes	
:
š
save_17/Assign_14Assignpi/dense/kernelsave_17/RestoreV2:14*
_output_shapes
:	L*"
_class
loc:@pi/dense/kernel*
validate_shape(*
T0*
use_locking(
ž
save_17/Assign_15Assignpi/dense/kernel/Adamsave_17/RestoreV2:15*
use_locking(*
validate_shape(*"
_class
loc:@pi/dense/kernel*
_output_shapes
:	L*
T0
Ŕ
save_17/Assign_16Assignpi/dense/kernel/Adam_1save_17/RestoreV2:16*
use_locking(*
_output_shapes
:	L*
validate_shape(*"
_class
loc:@pi/dense/kernel*
T0
ľ
save_17/Assign_17Assignpi/dense_1/biassave_17/RestoreV2:17*
validate_shape(*
T0*
use_locking(*"
_class
loc:@pi/dense_1/bias*
_output_shapes	
:
ş
save_17/Assign_18Assignpi/dense_1/bias/Adamsave_17/RestoreV2:18*
T0*"
_class
loc:@pi/dense_1/bias*
use_locking(*
_output_shapes	
:*
validate_shape(
ź
save_17/Assign_19Assignpi/dense_1/bias/Adam_1save_17/RestoreV2:19*
validate_shape(*
_output_shapes	
:*
use_locking(*"
_class
loc:@pi/dense_1/bias*
T0
ž
save_17/Assign_20Assignpi/dense_1/kernelsave_17/RestoreV2:20*
validate_shape(*
T0*$
_class
loc:@pi/dense_1/kernel* 
_output_shapes
:
*
use_locking(
Ă
save_17/Assign_21Assignpi/dense_1/kernel/Adamsave_17/RestoreV2:21*$
_class
loc:@pi/dense_1/kernel*
use_locking(*
validate_shape(* 
_output_shapes
:
*
T0
Ĺ
save_17/Assign_22Assignpi/dense_1/kernel/Adam_1save_17/RestoreV2:22*$
_class
loc:@pi/dense_1/kernel*
T0* 
_output_shapes
:
*
validate_shape(*
use_locking(
´
save_17/Assign_23Assignpi/dense_2/biassave_17/RestoreV2:23*
validate_shape(*"
_class
loc:@pi/dense_2/bias*
use_locking(*
_output_shapes
:*
T0
š
save_17/Assign_24Assignpi/dense_2/bias/Adamsave_17/RestoreV2:24*"
_class
loc:@pi/dense_2/bias*
_output_shapes
:*
T0*
validate_shape(*
use_locking(
ť
save_17/Assign_25Assignpi/dense_2/bias/Adam_1save_17/RestoreV2:25*
use_locking(*
validate_shape(*"
_class
loc:@pi/dense_2/bias*
T0*
_output_shapes
:
˝
save_17/Assign_26Assignpi/dense_2/kernelsave_17/RestoreV2:26*
use_locking(*$
_class
loc:@pi/dense_2/kernel*
T0*
_output_shapes
:	*
validate_shape(
Â
save_17/Assign_27Assignpi/dense_2/kernel/Adamsave_17/RestoreV2:27*
T0*
validate_shape(*$
_class
loc:@pi/dense_2/kernel*
_output_shapes
:	*
use_locking(
Ä
save_17/Assign_28Assignpi/dense_2/kernel/Adam_1save_17/RestoreV2:28*
_output_shapes
:	*$
_class
loc:@pi/dense_2/kernel*
T0*
use_locking(*
validate_shape(
Ş
save_17/Assign_29Assign
pi/log_stdsave_17/RestoreV2:29*
use_locking(*
T0*
validate_shape(*
_output_shapes
:*
_class
loc:@pi/log_std
Ż
save_17/Assign_30Assignpi/log_std/Adamsave_17/RestoreV2:30*
use_locking(*
T0*
_class
loc:@pi/log_std*
validate_shape(*
_output_shapes
:
ą
save_17/Assign_31Assignpi/log_std/Adam_1save_17/RestoreV2:31*
_output_shapes
:*
T0*
validate_shape(*
_class
loc:@pi/log_std*
use_locking(
ą
save_17/Assign_32Assignvc/dense/biassave_17/RestoreV2:32* 
_class
loc:@vc/dense/bias*
validate_shape(*
_output_shapes	
:*
T0*
use_locking(
ś
save_17/Assign_33Assignvc/dense/bias/Adamsave_17/RestoreV2:33*
T0* 
_class
loc:@vc/dense/bias*
validate_shape(*
use_locking(*
_output_shapes	
:
¸
save_17/Assign_34Assignvc/dense/bias/Adam_1save_17/RestoreV2:34*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0* 
_class
loc:@vc/dense/bias
š
save_17/Assign_35Assignvc/dense/kernelsave_17/RestoreV2:35*
T0*"
_class
loc:@vc/dense/kernel*
validate_shape(*
use_locking(*
_output_shapes
:	L
ž
save_17/Assign_36Assignvc/dense/kernel/Adamsave_17/RestoreV2:36*
use_locking(*
_output_shapes
:	L*
T0*"
_class
loc:@vc/dense/kernel*
validate_shape(
Ŕ
save_17/Assign_37Assignvc/dense/kernel/Adam_1save_17/RestoreV2:37*
use_locking(*
validate_shape(*
_output_shapes
:	L*
T0*"
_class
loc:@vc/dense/kernel
ľ
save_17/Assign_38Assignvc/dense_1/biassave_17/RestoreV2:38*
use_locking(*
_output_shapes	
:*
T0*"
_class
loc:@vc/dense_1/bias*
validate_shape(
ş
save_17/Assign_39Assignvc/dense_1/bias/Adamsave_17/RestoreV2:39*"
_class
loc:@vc/dense_1/bias*
_output_shapes	
:*
validate_shape(*
use_locking(*
T0
ź
save_17/Assign_40Assignvc/dense_1/bias/Adam_1save_17/RestoreV2:40*
validate_shape(*"
_class
loc:@vc/dense_1/bias*
_output_shapes	
:*
T0*
use_locking(
ž
save_17/Assign_41Assignvc/dense_1/kernelsave_17/RestoreV2:41* 
_output_shapes
:
*$
_class
loc:@vc/dense_1/kernel*
validate_shape(*
use_locking(*
T0
Ă
save_17/Assign_42Assignvc/dense_1/kernel/Adamsave_17/RestoreV2:42* 
_output_shapes
:
*
T0*$
_class
loc:@vc/dense_1/kernel*
use_locking(*
validate_shape(
Ĺ
save_17/Assign_43Assignvc/dense_1/kernel/Adam_1save_17/RestoreV2:43*
validate_shape(*
T0*
use_locking(* 
_output_shapes
:
*$
_class
loc:@vc/dense_1/kernel
´
save_17/Assign_44Assignvc/dense_2/biassave_17/RestoreV2:44*"
_class
loc:@vc/dense_2/bias*
use_locking(*
_output_shapes
:*
T0*
validate_shape(
š
save_17/Assign_45Assignvc/dense_2/bias/Adamsave_17/RestoreV2:45*
T0*
_output_shapes
:*
use_locking(*
validate_shape(*"
_class
loc:@vc/dense_2/bias
ť
save_17/Assign_46Assignvc/dense_2/bias/Adam_1save_17/RestoreV2:46*
_output_shapes
:*
validate_shape(*
use_locking(*
T0*"
_class
loc:@vc/dense_2/bias
˝
save_17/Assign_47Assignvc/dense_2/kernelsave_17/RestoreV2:47*
_output_shapes
:	*
use_locking(*
validate_shape(*
T0*$
_class
loc:@vc/dense_2/kernel
Â
save_17/Assign_48Assignvc/dense_2/kernel/Adamsave_17/RestoreV2:48*
use_locking(*
T0*
_output_shapes
:	*
validate_shape(*$
_class
loc:@vc/dense_2/kernel
Ä
save_17/Assign_49Assignvc/dense_2/kernel/Adam_1save_17/RestoreV2:49*
validate_shape(*
use_locking(*$
_class
loc:@vc/dense_2/kernel*
_output_shapes
:	*
T0
ą
save_17/Assign_50Assignvf/dense/biassave_17/RestoreV2:50*
T0*
validate_shape(* 
_class
loc:@vf/dense/bias*
_output_shapes	
:*
use_locking(
ś
save_17/Assign_51Assignvf/dense/bias/Adamsave_17/RestoreV2:51*
use_locking(* 
_class
loc:@vf/dense/bias*
validate_shape(*
T0*
_output_shapes	
:
¸
save_17/Assign_52Assignvf/dense/bias/Adam_1save_17/RestoreV2:52*
T0*
_output_shapes	
:*
validate_shape(*
use_locking(* 
_class
loc:@vf/dense/bias
š
save_17/Assign_53Assignvf/dense/kernelsave_17/RestoreV2:53*
validate_shape(*
T0*
_output_shapes
:	L*
use_locking(*"
_class
loc:@vf/dense/kernel
ž
save_17/Assign_54Assignvf/dense/kernel/Adamsave_17/RestoreV2:54*
_output_shapes
:	L*
T0*
use_locking(*"
_class
loc:@vf/dense/kernel*
validate_shape(
Ŕ
save_17/Assign_55Assignvf/dense/kernel/Adam_1save_17/RestoreV2:55*
_output_shapes
:	L*"
_class
loc:@vf/dense/kernel*
validate_shape(*
use_locking(*
T0
ľ
save_17/Assign_56Assignvf/dense_1/biassave_17/RestoreV2:56*
use_locking(*
T0*"
_class
loc:@vf/dense_1/bias*
_output_shapes	
:*
validate_shape(
ş
save_17/Assign_57Assignvf/dense_1/bias/Adamsave_17/RestoreV2:57*"
_class
loc:@vf/dense_1/bias*
_output_shapes	
:*
use_locking(*
T0*
validate_shape(
ź
save_17/Assign_58Assignvf/dense_1/bias/Adam_1save_17/RestoreV2:58*
validate_shape(*"
_class
loc:@vf/dense_1/bias*
T0*
use_locking(*
_output_shapes	
:
ž
save_17/Assign_59Assignvf/dense_1/kernelsave_17/RestoreV2:59*
T0*
validate_shape(*$
_class
loc:@vf/dense_1/kernel* 
_output_shapes
:
*
use_locking(
Ă
save_17/Assign_60Assignvf/dense_1/kernel/Adamsave_17/RestoreV2:60*$
_class
loc:@vf/dense_1/kernel* 
_output_shapes
:
*
validate_shape(*
T0*
use_locking(
Ĺ
save_17/Assign_61Assignvf/dense_1/kernel/Adam_1save_17/RestoreV2:61* 
_output_shapes
:
*
use_locking(*
validate_shape(*
T0*$
_class
loc:@vf/dense_1/kernel
´
save_17/Assign_62Assignvf/dense_2/biassave_17/RestoreV2:62*"
_class
loc:@vf/dense_2/bias*
use_locking(*
_output_shapes
:*
validate_shape(*
T0
š
save_17/Assign_63Assignvf/dense_2/bias/Adamsave_17/RestoreV2:63*
_output_shapes
:*
T0*
validate_shape(*
use_locking(*"
_class
loc:@vf/dense_2/bias
ť
save_17/Assign_64Assignvf/dense_2/bias/Adam_1save_17/RestoreV2:64*
_output_shapes
:*"
_class
loc:@vf/dense_2/bias*
T0*
validate_shape(*
use_locking(
˝
save_17/Assign_65Assignvf/dense_2/kernelsave_17/RestoreV2:65*$
_class
loc:@vf/dense_2/kernel*
use_locking(*
validate_shape(*
_output_shapes
:	*
T0
Â
save_17/Assign_66Assignvf/dense_2/kernel/Adamsave_17/RestoreV2:66*
validate_shape(*$
_class
loc:@vf/dense_2/kernel*
_output_shapes
:	*
T0*
use_locking(
Ä
save_17/Assign_67Assignvf/dense_2/kernel/Adam_1save_17/RestoreV2:67*
T0*$
_class
loc:@vf/dense_2/kernel*
use_locking(*
_output_shapes
:	*
validate_shape(
á

save_17/restore_shardNoOp^save_17/Assign^save_17/Assign_1^save_17/Assign_10^save_17/Assign_11^save_17/Assign_12^save_17/Assign_13^save_17/Assign_14^save_17/Assign_15^save_17/Assign_16^save_17/Assign_17^save_17/Assign_18^save_17/Assign_19^save_17/Assign_2^save_17/Assign_20^save_17/Assign_21^save_17/Assign_22^save_17/Assign_23^save_17/Assign_24^save_17/Assign_25^save_17/Assign_26^save_17/Assign_27^save_17/Assign_28^save_17/Assign_29^save_17/Assign_3^save_17/Assign_30^save_17/Assign_31^save_17/Assign_32^save_17/Assign_33^save_17/Assign_34^save_17/Assign_35^save_17/Assign_36^save_17/Assign_37^save_17/Assign_38^save_17/Assign_39^save_17/Assign_4^save_17/Assign_40^save_17/Assign_41^save_17/Assign_42^save_17/Assign_43^save_17/Assign_44^save_17/Assign_45^save_17/Assign_46^save_17/Assign_47^save_17/Assign_48^save_17/Assign_49^save_17/Assign_5^save_17/Assign_50^save_17/Assign_51^save_17/Assign_52^save_17/Assign_53^save_17/Assign_54^save_17/Assign_55^save_17/Assign_56^save_17/Assign_57^save_17/Assign_58^save_17/Assign_59^save_17/Assign_6^save_17/Assign_60^save_17/Assign_61^save_17/Assign_62^save_17/Assign_63^save_17/Assign_64^save_17/Assign_65^save_17/Assign_66^save_17/Assign_67^save_17/Assign_7^save_17/Assign_8^save_17/Assign_9
3
save_17/restore_allNoOp^save_17/restore_shard
\
save_18/filename/inputConst*
dtype0*
_output_shapes
: *
valueB Bmodel
t
save_18/filenamePlaceholderWithDefaultsave_18/filename/input*
dtype0*
shape: *
_output_shapes
: 
k
save_18/ConstPlaceholderWithDefaultsave_18/filename*
dtype0*
shape: *
_output_shapes
: 

save_18/StringJoin/inputs_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+_temp_72163b708e194ec486e93c528913e2c8/part
~
save_18/StringJoin
StringJoinsave_18/Constsave_18/StringJoin/inputs_1*
N*
	separator *
_output_shapes
: 
T
save_18/num_shardsConst*
dtype0*
value	B :*
_output_shapes
: 
_
save_18/ShardedFilename/shardConst*
_output_shapes
: *
value	B : *
dtype0

save_18/ShardedFilenameShardedFilenamesave_18/StringJoinsave_18/ShardedFilename/shardsave_18/num_shards*
_output_shapes
: 
ř
save_18/SaveV2/tensor_namesConst*¨
valueBDBbeta1_powerBbeta1_power_1Bbeta1_power_2Bbeta2_powerBbeta2_power_1Bbeta2_power_2Bpenalty/penalty_paramBpenalty/penalty_param/AdamBpenalty/penalty_param/Adam_1Bpenalty/penalty_param/Adam_2Bpenalty/penalty_param/Adam_3Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1B
pi/log_stdBpi/log_std/AdamBpi/log_std/Adam_1Bvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1*
_output_shapes
:D*
dtype0
ń
save_18/SaveV2/shape_and_slicesConst*
dtype0*
_output_shapes
:D*
valueBDB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
Ö
save_18/SaveV2SaveV2save_18/ShardedFilenamesave_18/SaveV2/tensor_namessave_18/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta1_power_2beta2_powerbeta2_power_1beta2_power_2penalty/penalty_parampenalty/penalty_param/Adampenalty/penalty_param/Adam_1penalty/penalty_param/Adam_2penalty/penalty_param/Adam_3pi/dense/biaspi/dense/bias/Adampi/dense/bias/Adam_1pi/dense/kernelpi/dense/kernel/Adampi/dense/kernel/Adam_1pi/dense_1/biaspi/dense_1/bias/Adampi/dense_1/bias/Adam_1pi/dense_1/kernelpi/dense_1/kernel/Adampi/dense_1/kernel/Adam_1pi/dense_2/biaspi/dense_2/bias/Adampi/dense_2/bias/Adam_1pi/dense_2/kernelpi/dense_2/kernel/Adampi/dense_2/kernel/Adam_1
pi/log_stdpi/log_std/Adampi/log_std/Adam_1vc/dense/biasvc/dense/bias/Adamvc/dense/bias/Adam_1vc/dense/kernelvc/dense/kernel/Adamvc/dense/kernel/Adam_1vc/dense_1/biasvc/dense_1/bias/Adamvc/dense_1/bias/Adam_1vc/dense_1/kernelvc/dense_1/kernel/Adamvc/dense_1/kernel/Adam_1vc/dense_2/biasvc/dense_2/bias/Adamvc/dense_2/bias/Adam_1vc/dense_2/kernelvc/dense_2/kernel/Adamvc/dense_2/kernel/Adam_1vf/dense/biasvf/dense/bias/Adamvf/dense/bias/Adam_1vf/dense/kernelvf/dense/kernel/Adamvf/dense/kernel/Adam_1vf/dense_1/biasvf/dense_1/bias/Adamvf/dense_1/bias/Adam_1vf/dense_1/kernelvf/dense_1/kernel/Adamvf/dense_1/kernel/Adam_1vf/dense_2/biasvf/dense_2/bias/Adamvf/dense_2/bias/Adam_1vf/dense_2/kernelvf/dense_2/kernel/Adamvf/dense_2/kernel/Adam_1*R
dtypesH
F2D

save_18/control_dependencyIdentitysave_18/ShardedFilename^save_18/SaveV2*
_output_shapes
: *
T0**
_class 
loc:@save_18/ShardedFilename
Ś
.save_18/MergeV2Checkpoints/checkpoint_prefixesPacksave_18/ShardedFilename^save_18/control_dependency*
T0*
N*
_output_shapes
:*

axis 

save_18/MergeV2CheckpointsMergeV2Checkpoints.save_18/MergeV2Checkpoints/checkpoint_prefixessave_18/Const*
delete_old_dirs(

save_18/IdentityIdentitysave_18/Const^save_18/MergeV2Checkpoints^save_18/control_dependency*
_output_shapes
: *
T0
ű
save_18/RestoreV2/tensor_namesConst*
_output_shapes
:D*
dtype0*¨
valueBDBbeta1_powerBbeta1_power_1Bbeta1_power_2Bbeta2_powerBbeta2_power_1Bbeta2_power_2Bpenalty/penalty_paramBpenalty/penalty_param/AdamBpenalty/penalty_param/Adam_1Bpenalty/penalty_param/Adam_2Bpenalty/penalty_param/Adam_3Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1B
pi/log_stdBpi/log_std/AdamBpi/log_std/Adam_1Bvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1
ô
"save_18/RestoreV2/shape_and_slicesConst*
_output_shapes
:D*
dtype0*
valueBDB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
î
save_18/RestoreV2	RestoreV2save_18/Constsave_18/RestoreV2/tensor_names"save_18/RestoreV2/shape_and_slices*R
dtypesH
F2D*Ś
_output_shapes
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
Ź
save_18/AssignAssignbeta1_powersave_18/RestoreV2*
use_locking(*
_output_shapes
: *
validate_shape(*(
_class
loc:@penalty/penalty_param*
T0
˛
save_18/Assign_1Assignbeta1_power_1save_18/RestoreV2:1*
_output_shapes
: *(
_class
loc:@penalty/penalty_param*
T0*
use_locking(*
validate_shape(
Ş
save_18/Assign_2Assignbeta1_power_2save_18/RestoreV2:2*
T0* 
_class
loc:@vc/dense/bias*
_output_shapes
: *
validate_shape(*
use_locking(
°
save_18/Assign_3Assignbeta2_powersave_18/RestoreV2:3*
_output_shapes
: *
T0*
use_locking(*(
_class
loc:@penalty/penalty_param*
validate_shape(
˛
save_18/Assign_4Assignbeta2_power_1save_18/RestoreV2:4*(
_class
loc:@penalty/penalty_param*
_output_shapes
: *
T0*
use_locking(*
validate_shape(
Ş
save_18/Assign_5Assignbeta2_power_2save_18/RestoreV2:5*
_output_shapes
: *
validate_shape(*
use_locking(* 
_class
loc:@vc/dense/bias*
T0
ş
save_18/Assign_6Assignpenalty/penalty_paramsave_18/RestoreV2:6*
validate_shape(*
_output_shapes
: *(
_class
loc:@penalty/penalty_param*
T0*
use_locking(
ż
save_18/Assign_7Assignpenalty/penalty_param/Adamsave_18/RestoreV2:7*
use_locking(*
T0*(
_class
loc:@penalty/penalty_param*
validate_shape(*
_output_shapes
: 
Á
save_18/Assign_8Assignpenalty/penalty_param/Adam_1save_18/RestoreV2:8*
use_locking(*
T0*
_output_shapes
: *
validate_shape(*(
_class
loc:@penalty/penalty_param
Á
save_18/Assign_9Assignpenalty/penalty_param/Adam_2save_18/RestoreV2:9*
validate_shape(*
T0*
use_locking(*
_output_shapes
: *(
_class
loc:@penalty/penalty_param
Ă
save_18/Assign_10Assignpenalty/penalty_param/Adam_3save_18/RestoreV2:10*
_output_shapes
: *(
_class
loc:@penalty/penalty_param*
T0*
use_locking(*
validate_shape(
ą
save_18/Assign_11Assignpi/dense/biassave_18/RestoreV2:11*
T0*
_output_shapes	
:*
use_locking(* 
_class
loc:@pi/dense/bias*
validate_shape(
ś
save_18/Assign_12Assignpi/dense/bias/Adamsave_18/RestoreV2:12*
_output_shapes	
:*
use_locking(* 
_class
loc:@pi/dense/bias*
T0*
validate_shape(
¸
save_18/Assign_13Assignpi/dense/bias/Adam_1save_18/RestoreV2:13* 
_class
loc:@pi/dense/bias*
use_locking(*
_output_shapes	
:*
validate_shape(*
T0
š
save_18/Assign_14Assignpi/dense/kernelsave_18/RestoreV2:14*
use_locking(*
validate_shape(*
_output_shapes
:	L*"
_class
loc:@pi/dense/kernel*
T0
ž
save_18/Assign_15Assignpi/dense/kernel/Adamsave_18/RestoreV2:15*
T0*"
_class
loc:@pi/dense/kernel*
use_locking(*
_output_shapes
:	L*
validate_shape(
Ŕ
save_18/Assign_16Assignpi/dense/kernel/Adam_1save_18/RestoreV2:16*
T0*
use_locking(*
_output_shapes
:	L*"
_class
loc:@pi/dense/kernel*
validate_shape(
ľ
save_18/Assign_17Assignpi/dense_1/biassave_18/RestoreV2:17*
T0*"
_class
loc:@pi/dense_1/bias*
validate_shape(*
use_locking(*
_output_shapes	
:
ş
save_18/Assign_18Assignpi/dense_1/bias/Adamsave_18/RestoreV2:18*
T0*
_output_shapes	
:*
validate_shape(*"
_class
loc:@pi/dense_1/bias*
use_locking(
ź
save_18/Assign_19Assignpi/dense_1/bias/Adam_1save_18/RestoreV2:19*"
_class
loc:@pi/dense_1/bias*
T0*
validate_shape(*
_output_shapes	
:*
use_locking(
ž
save_18/Assign_20Assignpi/dense_1/kernelsave_18/RestoreV2:20*
validate_shape(*
use_locking(* 
_output_shapes
:
*$
_class
loc:@pi/dense_1/kernel*
T0
Ă
save_18/Assign_21Assignpi/dense_1/kernel/Adamsave_18/RestoreV2:21* 
_output_shapes
:
*
T0*$
_class
loc:@pi/dense_1/kernel*
validate_shape(*
use_locking(
Ĺ
save_18/Assign_22Assignpi/dense_1/kernel/Adam_1save_18/RestoreV2:22*$
_class
loc:@pi/dense_1/kernel*
validate_shape(*
T0* 
_output_shapes
:
*
use_locking(
´
save_18/Assign_23Assignpi/dense_2/biassave_18/RestoreV2:23*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*"
_class
loc:@pi/dense_2/bias
š
save_18/Assign_24Assignpi/dense_2/bias/Adamsave_18/RestoreV2:24*
_output_shapes
:*
T0*
use_locking(*"
_class
loc:@pi/dense_2/bias*
validate_shape(
ť
save_18/Assign_25Assignpi/dense_2/bias/Adam_1save_18/RestoreV2:25*
T0*
_output_shapes
:*
validate_shape(*
use_locking(*"
_class
loc:@pi/dense_2/bias
˝
save_18/Assign_26Assignpi/dense_2/kernelsave_18/RestoreV2:26*
_output_shapes
:	*$
_class
loc:@pi/dense_2/kernel*
validate_shape(*
T0*
use_locking(
Â
save_18/Assign_27Assignpi/dense_2/kernel/Adamsave_18/RestoreV2:27*$
_class
loc:@pi/dense_2/kernel*
use_locking(*
T0*
_output_shapes
:	*
validate_shape(
Ä
save_18/Assign_28Assignpi/dense_2/kernel/Adam_1save_18/RestoreV2:28*
use_locking(*
_output_shapes
:	*
T0*
validate_shape(*$
_class
loc:@pi/dense_2/kernel
Ş
save_18/Assign_29Assign
pi/log_stdsave_18/RestoreV2:29*
use_locking(*
_output_shapes
:*
T0*
validate_shape(*
_class
loc:@pi/log_std
Ż
save_18/Assign_30Assignpi/log_std/Adamsave_18/RestoreV2:30*
_class
loc:@pi/log_std*
validate_shape(*
use_locking(*
T0*
_output_shapes
:
ą
save_18/Assign_31Assignpi/log_std/Adam_1save_18/RestoreV2:31*
_class
loc:@pi/log_std*
_output_shapes
:*
use_locking(*
validate_shape(*
T0
ą
save_18/Assign_32Assignvc/dense/biassave_18/RestoreV2:32*
_output_shapes	
:*
validate_shape(* 
_class
loc:@vc/dense/bias*
use_locking(*
T0
ś
save_18/Assign_33Assignvc/dense/bias/Adamsave_18/RestoreV2:33*
_output_shapes	
:*
validate_shape(* 
_class
loc:@vc/dense/bias*
use_locking(*
T0
¸
save_18/Assign_34Assignvc/dense/bias/Adam_1save_18/RestoreV2:34* 
_class
loc:@vc/dense/bias*
_output_shapes	
:*
validate_shape(*
T0*
use_locking(
š
save_18/Assign_35Assignvc/dense/kernelsave_18/RestoreV2:35*
use_locking(*
validate_shape(*
_output_shapes
:	L*
T0*"
_class
loc:@vc/dense/kernel
ž
save_18/Assign_36Assignvc/dense/kernel/Adamsave_18/RestoreV2:36*
use_locking(*"
_class
loc:@vc/dense/kernel*
T0*
validate_shape(*
_output_shapes
:	L
Ŕ
save_18/Assign_37Assignvc/dense/kernel/Adam_1save_18/RestoreV2:37*
use_locking(*"
_class
loc:@vc/dense/kernel*
T0*
validate_shape(*
_output_shapes
:	L
ľ
save_18/Assign_38Assignvc/dense_1/biassave_18/RestoreV2:38*
_output_shapes	
:*"
_class
loc:@vc/dense_1/bias*
validate_shape(*
T0*
use_locking(
ş
save_18/Assign_39Assignvc/dense_1/bias/Adamsave_18/RestoreV2:39*
T0*"
_class
loc:@vc/dense_1/bias*
validate_shape(*
use_locking(*
_output_shapes	
:
ź
save_18/Assign_40Assignvc/dense_1/bias/Adam_1save_18/RestoreV2:40*"
_class
loc:@vc/dense_1/bias*
T0*
_output_shapes	
:*
validate_shape(*
use_locking(
ž
save_18/Assign_41Assignvc/dense_1/kernelsave_18/RestoreV2:41*$
_class
loc:@vc/dense_1/kernel* 
_output_shapes
:
*
use_locking(*
T0*
validate_shape(
Ă
save_18/Assign_42Assignvc/dense_1/kernel/Adamsave_18/RestoreV2:42* 
_output_shapes
:
*
T0*
use_locking(*
validate_shape(*$
_class
loc:@vc/dense_1/kernel
Ĺ
save_18/Assign_43Assignvc/dense_1/kernel/Adam_1save_18/RestoreV2:43*
use_locking(*
T0*
validate_shape(*$
_class
loc:@vc/dense_1/kernel* 
_output_shapes
:

´
save_18/Assign_44Assignvc/dense_2/biassave_18/RestoreV2:44*"
_class
loc:@vc/dense_2/bias*
use_locking(*
validate_shape(*
T0*
_output_shapes
:
š
save_18/Assign_45Assignvc/dense_2/bias/Adamsave_18/RestoreV2:45*
_output_shapes
:*"
_class
loc:@vc/dense_2/bias*
validate_shape(*
use_locking(*
T0
ť
save_18/Assign_46Assignvc/dense_2/bias/Adam_1save_18/RestoreV2:46*
T0*"
_class
loc:@vc/dense_2/bias*
_output_shapes
:*
validate_shape(*
use_locking(
˝
save_18/Assign_47Assignvc/dense_2/kernelsave_18/RestoreV2:47*
_output_shapes
:	*
use_locking(*$
_class
loc:@vc/dense_2/kernel*
validate_shape(*
T0
Â
save_18/Assign_48Assignvc/dense_2/kernel/Adamsave_18/RestoreV2:48*
validate_shape(*
_output_shapes
:	*
T0*$
_class
loc:@vc/dense_2/kernel*
use_locking(
Ä
save_18/Assign_49Assignvc/dense_2/kernel/Adam_1save_18/RestoreV2:49*
T0*
use_locking(*$
_class
loc:@vc/dense_2/kernel*
validate_shape(*
_output_shapes
:	
ą
save_18/Assign_50Assignvf/dense/biassave_18/RestoreV2:50*
validate_shape(*
use_locking(*
T0*
_output_shapes	
:* 
_class
loc:@vf/dense/bias
ś
save_18/Assign_51Assignvf/dense/bias/Adamsave_18/RestoreV2:51*
_output_shapes	
:*
use_locking(*
T0*
validate_shape(* 
_class
loc:@vf/dense/bias
¸
save_18/Assign_52Assignvf/dense/bias/Adam_1save_18/RestoreV2:52*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0* 
_class
loc:@vf/dense/bias
š
save_18/Assign_53Assignvf/dense/kernelsave_18/RestoreV2:53*
validate_shape(*
T0*
use_locking(*
_output_shapes
:	L*"
_class
loc:@vf/dense/kernel
ž
save_18/Assign_54Assignvf/dense/kernel/Adamsave_18/RestoreV2:54*
T0*
use_locking(*"
_class
loc:@vf/dense/kernel*
_output_shapes
:	L*
validate_shape(
Ŕ
save_18/Assign_55Assignvf/dense/kernel/Adam_1save_18/RestoreV2:55*"
_class
loc:@vf/dense/kernel*
use_locking(*
_output_shapes
:	L*
T0*
validate_shape(
ľ
save_18/Assign_56Assignvf/dense_1/biassave_18/RestoreV2:56*"
_class
loc:@vf/dense_1/bias*
_output_shapes	
:*
validate_shape(*
T0*
use_locking(
ş
save_18/Assign_57Assignvf/dense_1/bias/Adamsave_18/RestoreV2:57*
T0*"
_class
loc:@vf/dense_1/bias*
use_locking(*
_output_shapes	
:*
validate_shape(
ź
save_18/Assign_58Assignvf/dense_1/bias/Adam_1save_18/RestoreV2:58*
T0*
validate_shape(*
_output_shapes	
:*
use_locking(*"
_class
loc:@vf/dense_1/bias
ž
save_18/Assign_59Assignvf/dense_1/kernelsave_18/RestoreV2:59*
validate_shape(*$
_class
loc:@vf/dense_1/kernel*
T0*
use_locking(* 
_output_shapes
:

Ă
save_18/Assign_60Assignvf/dense_1/kernel/Adamsave_18/RestoreV2:60*$
_class
loc:@vf/dense_1/kernel* 
_output_shapes
:
*
validate_shape(*
T0*
use_locking(
Ĺ
save_18/Assign_61Assignvf/dense_1/kernel/Adam_1save_18/RestoreV2:61* 
_output_shapes
:
*
use_locking(*$
_class
loc:@vf/dense_1/kernel*
validate_shape(*
T0
´
save_18/Assign_62Assignvf/dense_2/biassave_18/RestoreV2:62*"
_class
loc:@vf/dense_2/bias*
T0*
validate_shape(*
use_locking(*
_output_shapes
:
š
save_18/Assign_63Assignvf/dense_2/bias/Adamsave_18/RestoreV2:63*
T0*"
_class
loc:@vf/dense_2/bias*
use_locking(*
validate_shape(*
_output_shapes
:
ť
save_18/Assign_64Assignvf/dense_2/bias/Adam_1save_18/RestoreV2:64*
validate_shape(*"
_class
loc:@vf/dense_2/bias*
_output_shapes
:*
T0*
use_locking(
˝
save_18/Assign_65Assignvf/dense_2/kernelsave_18/RestoreV2:65*
validate_shape(*
T0*$
_class
loc:@vf/dense_2/kernel*
use_locking(*
_output_shapes
:	
Â
save_18/Assign_66Assignvf/dense_2/kernel/Adamsave_18/RestoreV2:66*$
_class
loc:@vf/dense_2/kernel*
validate_shape(*
use_locking(*
T0*
_output_shapes
:	
Ä
save_18/Assign_67Assignvf/dense_2/kernel/Adam_1save_18/RestoreV2:67*$
_class
loc:@vf/dense_2/kernel*
use_locking(*
T0*
validate_shape(*
_output_shapes
:	
á

save_18/restore_shardNoOp^save_18/Assign^save_18/Assign_1^save_18/Assign_10^save_18/Assign_11^save_18/Assign_12^save_18/Assign_13^save_18/Assign_14^save_18/Assign_15^save_18/Assign_16^save_18/Assign_17^save_18/Assign_18^save_18/Assign_19^save_18/Assign_2^save_18/Assign_20^save_18/Assign_21^save_18/Assign_22^save_18/Assign_23^save_18/Assign_24^save_18/Assign_25^save_18/Assign_26^save_18/Assign_27^save_18/Assign_28^save_18/Assign_29^save_18/Assign_3^save_18/Assign_30^save_18/Assign_31^save_18/Assign_32^save_18/Assign_33^save_18/Assign_34^save_18/Assign_35^save_18/Assign_36^save_18/Assign_37^save_18/Assign_38^save_18/Assign_39^save_18/Assign_4^save_18/Assign_40^save_18/Assign_41^save_18/Assign_42^save_18/Assign_43^save_18/Assign_44^save_18/Assign_45^save_18/Assign_46^save_18/Assign_47^save_18/Assign_48^save_18/Assign_49^save_18/Assign_5^save_18/Assign_50^save_18/Assign_51^save_18/Assign_52^save_18/Assign_53^save_18/Assign_54^save_18/Assign_55^save_18/Assign_56^save_18/Assign_57^save_18/Assign_58^save_18/Assign_59^save_18/Assign_6^save_18/Assign_60^save_18/Assign_61^save_18/Assign_62^save_18/Assign_63^save_18/Assign_64^save_18/Assign_65^save_18/Assign_66^save_18/Assign_67^save_18/Assign_7^save_18/Assign_8^save_18/Assign_9
3
save_18/restore_allNoOp^save_18/restore_shard
\
save_19/filename/inputConst*
dtype0*
_output_shapes
: *
valueB Bmodel
t
save_19/filenamePlaceholderWithDefaultsave_19/filename/input*
_output_shapes
: *
dtype0*
shape: 
k
save_19/ConstPlaceholderWithDefaultsave_19/filename*
shape: *
_output_shapes
: *
dtype0

save_19/StringJoin/inputs_1Const*<
value3B1 B+_temp_451515237a3144bab8ec9f9a97700754/part*
_output_shapes
: *
dtype0
~
save_19/StringJoin
StringJoinsave_19/Constsave_19/StringJoin/inputs_1*
_output_shapes
: *
N*
	separator 
T
save_19/num_shardsConst*
_output_shapes
: *
dtype0*
value	B :
_
save_19/ShardedFilename/shardConst*
_output_shapes
: *
dtype0*
value	B : 

save_19/ShardedFilenameShardedFilenamesave_19/StringJoinsave_19/ShardedFilename/shardsave_19/num_shards*
_output_shapes
: 
ř
save_19/SaveV2/tensor_namesConst*
_output_shapes
:D*¨
valueBDBbeta1_powerBbeta1_power_1Bbeta1_power_2Bbeta2_powerBbeta2_power_1Bbeta2_power_2Bpenalty/penalty_paramBpenalty/penalty_param/AdamBpenalty/penalty_param/Adam_1Bpenalty/penalty_param/Adam_2Bpenalty/penalty_param/Adam_3Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1B
pi/log_stdBpi/log_std/AdamBpi/log_std/Adam_1Bvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1*
dtype0
ń
save_19/SaveV2/shape_and_slicesConst*
valueBDB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
_output_shapes
:D*
dtype0
Ö
save_19/SaveV2SaveV2save_19/ShardedFilenamesave_19/SaveV2/tensor_namessave_19/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta1_power_2beta2_powerbeta2_power_1beta2_power_2penalty/penalty_parampenalty/penalty_param/Adampenalty/penalty_param/Adam_1penalty/penalty_param/Adam_2penalty/penalty_param/Adam_3pi/dense/biaspi/dense/bias/Adampi/dense/bias/Adam_1pi/dense/kernelpi/dense/kernel/Adampi/dense/kernel/Adam_1pi/dense_1/biaspi/dense_1/bias/Adampi/dense_1/bias/Adam_1pi/dense_1/kernelpi/dense_1/kernel/Adampi/dense_1/kernel/Adam_1pi/dense_2/biaspi/dense_2/bias/Adampi/dense_2/bias/Adam_1pi/dense_2/kernelpi/dense_2/kernel/Adampi/dense_2/kernel/Adam_1
pi/log_stdpi/log_std/Adampi/log_std/Adam_1vc/dense/biasvc/dense/bias/Adamvc/dense/bias/Adam_1vc/dense/kernelvc/dense/kernel/Adamvc/dense/kernel/Adam_1vc/dense_1/biasvc/dense_1/bias/Adamvc/dense_1/bias/Adam_1vc/dense_1/kernelvc/dense_1/kernel/Adamvc/dense_1/kernel/Adam_1vc/dense_2/biasvc/dense_2/bias/Adamvc/dense_2/bias/Adam_1vc/dense_2/kernelvc/dense_2/kernel/Adamvc/dense_2/kernel/Adam_1vf/dense/biasvf/dense/bias/Adamvf/dense/bias/Adam_1vf/dense/kernelvf/dense/kernel/Adamvf/dense/kernel/Adam_1vf/dense_1/biasvf/dense_1/bias/Adamvf/dense_1/bias/Adam_1vf/dense_1/kernelvf/dense_1/kernel/Adamvf/dense_1/kernel/Adam_1vf/dense_2/biasvf/dense_2/bias/Adamvf/dense_2/bias/Adam_1vf/dense_2/kernelvf/dense_2/kernel/Adamvf/dense_2/kernel/Adam_1*R
dtypesH
F2D

save_19/control_dependencyIdentitysave_19/ShardedFilename^save_19/SaveV2*
T0**
_class 
loc:@save_19/ShardedFilename*
_output_shapes
: 
Ś
.save_19/MergeV2Checkpoints/checkpoint_prefixesPacksave_19/ShardedFilename^save_19/control_dependency*

axis *
N*
T0*
_output_shapes
:

save_19/MergeV2CheckpointsMergeV2Checkpoints.save_19/MergeV2Checkpoints/checkpoint_prefixessave_19/Const*
delete_old_dirs(

save_19/IdentityIdentitysave_19/Const^save_19/MergeV2Checkpoints^save_19/control_dependency*
_output_shapes
: *
T0
ű
save_19/RestoreV2/tensor_namesConst*
dtype0*¨
valueBDBbeta1_powerBbeta1_power_1Bbeta1_power_2Bbeta2_powerBbeta2_power_1Bbeta2_power_2Bpenalty/penalty_paramBpenalty/penalty_param/AdamBpenalty/penalty_param/Adam_1Bpenalty/penalty_param/Adam_2Bpenalty/penalty_param/Adam_3Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1B
pi/log_stdBpi/log_std/AdamBpi/log_std/Adam_1Bvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1*
_output_shapes
:D
ô
"save_19/RestoreV2/shape_and_slicesConst*
dtype0*
_output_shapes
:D*
valueBDB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
î
save_19/RestoreV2	RestoreV2save_19/Constsave_19/RestoreV2/tensor_names"save_19/RestoreV2/shape_and_slices*Ś
_output_shapes
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*R
dtypesH
F2D
Ź
save_19/AssignAssignbeta1_powersave_19/RestoreV2*(
_class
loc:@penalty/penalty_param*
validate_shape(*
_output_shapes
: *
T0*
use_locking(
˛
save_19/Assign_1Assignbeta1_power_1save_19/RestoreV2:1*
use_locking(*(
_class
loc:@penalty/penalty_param*
validate_shape(*
T0*
_output_shapes
: 
Ş
save_19/Assign_2Assignbeta1_power_2save_19/RestoreV2:2*
validate_shape(* 
_class
loc:@vc/dense/bias*
_output_shapes
: *
T0*
use_locking(
°
save_19/Assign_3Assignbeta2_powersave_19/RestoreV2:3*
_output_shapes
: *
validate_shape(*
use_locking(*(
_class
loc:@penalty/penalty_param*
T0
˛
save_19/Assign_4Assignbeta2_power_1save_19/RestoreV2:4*
validate_shape(*(
_class
loc:@penalty/penalty_param*
use_locking(*
T0*
_output_shapes
: 
Ş
save_19/Assign_5Assignbeta2_power_2save_19/RestoreV2:5*
T0*
use_locking(* 
_class
loc:@vc/dense/bias*
validate_shape(*
_output_shapes
: 
ş
save_19/Assign_6Assignpenalty/penalty_paramsave_19/RestoreV2:6*
validate_shape(*
T0*(
_class
loc:@penalty/penalty_param*
_output_shapes
: *
use_locking(
ż
save_19/Assign_7Assignpenalty/penalty_param/Adamsave_19/RestoreV2:7*
validate_shape(*
_output_shapes
: *
T0*(
_class
loc:@penalty/penalty_param*
use_locking(
Á
save_19/Assign_8Assignpenalty/penalty_param/Adam_1save_19/RestoreV2:8*(
_class
loc:@penalty/penalty_param*
T0*
validate_shape(*
_output_shapes
: *
use_locking(
Á
save_19/Assign_9Assignpenalty/penalty_param/Adam_2save_19/RestoreV2:9*
_output_shapes
: *(
_class
loc:@penalty/penalty_param*
use_locking(*
validate_shape(*
T0
Ă
save_19/Assign_10Assignpenalty/penalty_param/Adam_3save_19/RestoreV2:10*
_output_shapes
: *(
_class
loc:@penalty/penalty_param*
T0*
validate_shape(*
use_locking(
ą
save_19/Assign_11Assignpi/dense/biassave_19/RestoreV2:11* 
_class
loc:@pi/dense/bias*
_output_shapes	
:*
validate_shape(*
T0*
use_locking(
ś
save_19/Assign_12Assignpi/dense/bias/Adamsave_19/RestoreV2:12*
T0*
use_locking(*
validate_shape(* 
_class
loc:@pi/dense/bias*
_output_shapes	
:
¸
save_19/Assign_13Assignpi/dense/bias/Adam_1save_19/RestoreV2:13*
validate_shape(*
use_locking(* 
_class
loc:@pi/dense/bias*
T0*
_output_shapes	
:
š
save_19/Assign_14Assignpi/dense/kernelsave_19/RestoreV2:14*
validate_shape(*
T0*"
_class
loc:@pi/dense/kernel*
use_locking(*
_output_shapes
:	L
ž
save_19/Assign_15Assignpi/dense/kernel/Adamsave_19/RestoreV2:15*"
_class
loc:@pi/dense/kernel*
_output_shapes
:	L*
T0*
use_locking(*
validate_shape(
Ŕ
save_19/Assign_16Assignpi/dense/kernel/Adam_1save_19/RestoreV2:16*
T0*
validate_shape(*"
_class
loc:@pi/dense/kernel*
use_locking(*
_output_shapes
:	L
ľ
save_19/Assign_17Assignpi/dense_1/biassave_19/RestoreV2:17*
T0*
_output_shapes	
:*
validate_shape(*
use_locking(*"
_class
loc:@pi/dense_1/bias
ş
save_19/Assign_18Assignpi/dense_1/bias/Adamsave_19/RestoreV2:18*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*"
_class
loc:@pi/dense_1/bias
ź
save_19/Assign_19Assignpi/dense_1/bias/Adam_1save_19/RestoreV2:19*
validate_shape(*
_output_shapes	
:*
T0*"
_class
loc:@pi/dense_1/bias*
use_locking(
ž
save_19/Assign_20Assignpi/dense_1/kernelsave_19/RestoreV2:20*$
_class
loc:@pi/dense_1/kernel*
use_locking(*
T0*
validate_shape(* 
_output_shapes
:

Ă
save_19/Assign_21Assignpi/dense_1/kernel/Adamsave_19/RestoreV2:21*
use_locking(*$
_class
loc:@pi/dense_1/kernel*
validate_shape(* 
_output_shapes
:
*
T0
Ĺ
save_19/Assign_22Assignpi/dense_1/kernel/Adam_1save_19/RestoreV2:22*
T0*$
_class
loc:@pi/dense_1/kernel* 
_output_shapes
:
*
validate_shape(*
use_locking(
´
save_19/Assign_23Assignpi/dense_2/biassave_19/RestoreV2:23*
T0*
_output_shapes
:*"
_class
loc:@pi/dense_2/bias*
use_locking(*
validate_shape(
š
save_19/Assign_24Assignpi/dense_2/bias/Adamsave_19/RestoreV2:24*
T0*"
_class
loc:@pi/dense_2/bias*
use_locking(*
_output_shapes
:*
validate_shape(
ť
save_19/Assign_25Assignpi/dense_2/bias/Adam_1save_19/RestoreV2:25*"
_class
loc:@pi/dense_2/bias*
validate_shape(*
T0*
_output_shapes
:*
use_locking(
˝
save_19/Assign_26Assignpi/dense_2/kernelsave_19/RestoreV2:26*
_output_shapes
:	*
T0*
validate_shape(*$
_class
loc:@pi/dense_2/kernel*
use_locking(
Â
save_19/Assign_27Assignpi/dense_2/kernel/Adamsave_19/RestoreV2:27*
_output_shapes
:	*
use_locking(*
T0*
validate_shape(*$
_class
loc:@pi/dense_2/kernel
Ä
save_19/Assign_28Assignpi/dense_2/kernel/Adam_1save_19/RestoreV2:28*
_output_shapes
:	*
use_locking(*$
_class
loc:@pi/dense_2/kernel*
T0*
validate_shape(
Ş
save_19/Assign_29Assign
pi/log_stdsave_19/RestoreV2:29*
use_locking(*
_output_shapes
:*
_class
loc:@pi/log_std*
validate_shape(*
T0
Ż
save_19/Assign_30Assignpi/log_std/Adamsave_19/RestoreV2:30*
_output_shapes
:*
_class
loc:@pi/log_std*
T0*
use_locking(*
validate_shape(
ą
save_19/Assign_31Assignpi/log_std/Adam_1save_19/RestoreV2:31*
_class
loc:@pi/log_std*
T0*
_output_shapes
:*
validate_shape(*
use_locking(
ą
save_19/Assign_32Assignvc/dense/biassave_19/RestoreV2:32*
use_locking(* 
_class
loc:@vc/dense/bias*
validate_shape(*
T0*
_output_shapes	
:
ś
save_19/Assign_33Assignvc/dense/bias/Adamsave_19/RestoreV2:33* 
_class
loc:@vc/dense/bias*
T0*
validate_shape(*
use_locking(*
_output_shapes	
:
¸
save_19/Assign_34Assignvc/dense/bias/Adam_1save_19/RestoreV2:34*
validate_shape(*
_output_shapes	
:* 
_class
loc:@vc/dense/bias*
T0*
use_locking(
š
save_19/Assign_35Assignvc/dense/kernelsave_19/RestoreV2:35*
_output_shapes
:	L*
T0*
use_locking(*"
_class
loc:@vc/dense/kernel*
validate_shape(
ž
save_19/Assign_36Assignvc/dense/kernel/Adamsave_19/RestoreV2:36*
validate_shape(*
_output_shapes
:	L*"
_class
loc:@vc/dense/kernel*
use_locking(*
T0
Ŕ
save_19/Assign_37Assignvc/dense/kernel/Adam_1save_19/RestoreV2:37*
use_locking(*
validate_shape(*
T0*
_output_shapes
:	L*"
_class
loc:@vc/dense/kernel
ľ
save_19/Assign_38Assignvc/dense_1/biassave_19/RestoreV2:38*"
_class
loc:@vc/dense_1/bias*
use_locking(*
validate_shape(*
T0*
_output_shapes	
:
ş
save_19/Assign_39Assignvc/dense_1/bias/Adamsave_19/RestoreV2:39*"
_class
loc:@vc/dense_1/bias*
_output_shapes	
:*
T0*
use_locking(*
validate_shape(
ź
save_19/Assign_40Assignvc/dense_1/bias/Adam_1save_19/RestoreV2:40*
validate_shape(*
T0*
use_locking(*"
_class
loc:@vc/dense_1/bias*
_output_shapes	
:
ž
save_19/Assign_41Assignvc/dense_1/kernelsave_19/RestoreV2:41*
validate_shape(*$
_class
loc:@vc/dense_1/kernel* 
_output_shapes
:
*
use_locking(*
T0
Ă
save_19/Assign_42Assignvc/dense_1/kernel/Adamsave_19/RestoreV2:42*
T0*
validate_shape(*$
_class
loc:@vc/dense_1/kernel* 
_output_shapes
:
*
use_locking(
Ĺ
save_19/Assign_43Assignvc/dense_1/kernel/Adam_1save_19/RestoreV2:43*
T0*
use_locking(* 
_output_shapes
:
*$
_class
loc:@vc/dense_1/kernel*
validate_shape(
´
save_19/Assign_44Assignvc/dense_2/biassave_19/RestoreV2:44*"
_class
loc:@vc/dense_2/bias*
use_locking(*
T0*
_output_shapes
:*
validate_shape(
š
save_19/Assign_45Assignvc/dense_2/bias/Adamsave_19/RestoreV2:45*"
_class
loc:@vc/dense_2/bias*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
ť
save_19/Assign_46Assignvc/dense_2/bias/Adam_1save_19/RestoreV2:46*
T0*
use_locking(*
_output_shapes
:*
validate_shape(*"
_class
loc:@vc/dense_2/bias
˝
save_19/Assign_47Assignvc/dense_2/kernelsave_19/RestoreV2:47*
_output_shapes
:	*
T0*
use_locking(*
validate_shape(*$
_class
loc:@vc/dense_2/kernel
Â
save_19/Assign_48Assignvc/dense_2/kernel/Adamsave_19/RestoreV2:48*
_output_shapes
:	*
T0*
use_locking(*$
_class
loc:@vc/dense_2/kernel*
validate_shape(
Ä
save_19/Assign_49Assignvc/dense_2/kernel/Adam_1save_19/RestoreV2:49*$
_class
loc:@vc/dense_2/kernel*
T0*
validate_shape(*
_output_shapes
:	*
use_locking(
ą
save_19/Assign_50Assignvf/dense/biassave_19/RestoreV2:50*
T0*
use_locking(* 
_class
loc:@vf/dense/bias*
validate_shape(*
_output_shapes	
:
ś
save_19/Assign_51Assignvf/dense/bias/Adamsave_19/RestoreV2:51*
validate_shape(*
_output_shapes	
:* 
_class
loc:@vf/dense/bias*
T0*
use_locking(
¸
save_19/Assign_52Assignvf/dense/bias/Adam_1save_19/RestoreV2:52*
use_locking(* 
_class
loc:@vf/dense/bias*
validate_shape(*
_output_shapes	
:*
T0
š
save_19/Assign_53Assignvf/dense/kernelsave_19/RestoreV2:53*
validate_shape(*"
_class
loc:@vf/dense/kernel*
T0*
_output_shapes
:	L*
use_locking(
ž
save_19/Assign_54Assignvf/dense/kernel/Adamsave_19/RestoreV2:54*"
_class
loc:@vf/dense/kernel*
use_locking(*
validate_shape(*
T0*
_output_shapes
:	L
Ŕ
save_19/Assign_55Assignvf/dense/kernel/Adam_1save_19/RestoreV2:55*
_output_shapes
:	L*
validate_shape(*"
_class
loc:@vf/dense/kernel*
T0*
use_locking(
ľ
save_19/Assign_56Assignvf/dense_1/biassave_19/RestoreV2:56*
T0*
use_locking(*
_output_shapes	
:*
validate_shape(*"
_class
loc:@vf/dense_1/bias
ş
save_19/Assign_57Assignvf/dense_1/bias/Adamsave_19/RestoreV2:57*
validate_shape(*
T0*"
_class
loc:@vf/dense_1/bias*
use_locking(*
_output_shapes	
:
ź
save_19/Assign_58Assignvf/dense_1/bias/Adam_1save_19/RestoreV2:58*
use_locking(*"
_class
loc:@vf/dense_1/bias*
_output_shapes	
:*
T0*
validate_shape(
ž
save_19/Assign_59Assignvf/dense_1/kernelsave_19/RestoreV2:59* 
_output_shapes
:
*$
_class
loc:@vf/dense_1/kernel*
use_locking(*
T0*
validate_shape(
Ă
save_19/Assign_60Assignvf/dense_1/kernel/Adamsave_19/RestoreV2:60*
use_locking(*$
_class
loc:@vf/dense_1/kernel*
validate_shape(*
T0* 
_output_shapes
:

Ĺ
save_19/Assign_61Assignvf/dense_1/kernel/Adam_1save_19/RestoreV2:61*
T0*
validate_shape(* 
_output_shapes
:
*$
_class
loc:@vf/dense_1/kernel*
use_locking(
´
save_19/Assign_62Assignvf/dense_2/biassave_19/RestoreV2:62*"
_class
loc:@vf/dense_2/bias*
_output_shapes
:*
T0*
validate_shape(*
use_locking(
š
save_19/Assign_63Assignvf/dense_2/bias/Adamsave_19/RestoreV2:63*
_output_shapes
:*"
_class
loc:@vf/dense_2/bias*
validate_shape(*
use_locking(*
T0
ť
save_19/Assign_64Assignvf/dense_2/bias/Adam_1save_19/RestoreV2:64*
use_locking(*"
_class
loc:@vf/dense_2/bias*
T0*
_output_shapes
:*
validate_shape(
˝
save_19/Assign_65Assignvf/dense_2/kernelsave_19/RestoreV2:65*$
_class
loc:@vf/dense_2/kernel*
use_locking(*
validate_shape(*
_output_shapes
:	*
T0
Â
save_19/Assign_66Assignvf/dense_2/kernel/Adamsave_19/RestoreV2:66*
validate_shape(*
T0*
use_locking(*
_output_shapes
:	*$
_class
loc:@vf/dense_2/kernel
Ä
save_19/Assign_67Assignvf/dense_2/kernel/Adam_1save_19/RestoreV2:67*
use_locking(*
validate_shape(*
T0*
_output_shapes
:	*$
_class
loc:@vf/dense_2/kernel
á

save_19/restore_shardNoOp^save_19/Assign^save_19/Assign_1^save_19/Assign_10^save_19/Assign_11^save_19/Assign_12^save_19/Assign_13^save_19/Assign_14^save_19/Assign_15^save_19/Assign_16^save_19/Assign_17^save_19/Assign_18^save_19/Assign_19^save_19/Assign_2^save_19/Assign_20^save_19/Assign_21^save_19/Assign_22^save_19/Assign_23^save_19/Assign_24^save_19/Assign_25^save_19/Assign_26^save_19/Assign_27^save_19/Assign_28^save_19/Assign_29^save_19/Assign_3^save_19/Assign_30^save_19/Assign_31^save_19/Assign_32^save_19/Assign_33^save_19/Assign_34^save_19/Assign_35^save_19/Assign_36^save_19/Assign_37^save_19/Assign_38^save_19/Assign_39^save_19/Assign_4^save_19/Assign_40^save_19/Assign_41^save_19/Assign_42^save_19/Assign_43^save_19/Assign_44^save_19/Assign_45^save_19/Assign_46^save_19/Assign_47^save_19/Assign_48^save_19/Assign_49^save_19/Assign_5^save_19/Assign_50^save_19/Assign_51^save_19/Assign_52^save_19/Assign_53^save_19/Assign_54^save_19/Assign_55^save_19/Assign_56^save_19/Assign_57^save_19/Assign_58^save_19/Assign_59^save_19/Assign_6^save_19/Assign_60^save_19/Assign_61^save_19/Assign_62^save_19/Assign_63^save_19/Assign_64^save_19/Assign_65^save_19/Assign_66^save_19/Assign_67^save_19/Assign_7^save_19/Assign_8^save_19/Assign_9
3
save_19/restore_allNoOp^save_19/restore_shard
\
save_20/filename/inputConst*
_output_shapes
: *
dtype0*
valueB Bmodel
t
save_20/filenamePlaceholderWithDefaultsave_20/filename/input*
dtype0*
_output_shapes
: *
shape: 
k
save_20/ConstPlaceholderWithDefaultsave_20/filename*
dtype0*
_output_shapes
: *
shape: 

save_20/StringJoin/inputs_1Const*
dtype0*
_output_shapes
: *<
value3B1 B+_temp_1b41f6d41a9f466982d6bf3dd40afffb/part
~
save_20/StringJoin
StringJoinsave_20/Constsave_20/StringJoin/inputs_1*
	separator *
_output_shapes
: *
N
T
save_20/num_shardsConst*
_output_shapes
: *
value	B :*
dtype0
_
save_20/ShardedFilename/shardConst*
dtype0*
_output_shapes
: *
value	B : 

save_20/ShardedFilenameShardedFilenamesave_20/StringJoinsave_20/ShardedFilename/shardsave_20/num_shards*
_output_shapes
: 
ř
save_20/SaveV2/tensor_namesConst*
_output_shapes
:D*¨
valueBDBbeta1_powerBbeta1_power_1Bbeta1_power_2Bbeta2_powerBbeta2_power_1Bbeta2_power_2Bpenalty/penalty_paramBpenalty/penalty_param/AdamBpenalty/penalty_param/Adam_1Bpenalty/penalty_param/Adam_2Bpenalty/penalty_param/Adam_3Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1B
pi/log_stdBpi/log_std/AdamBpi/log_std/Adam_1Bvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1*
dtype0
ń
save_20/SaveV2/shape_and_slicesConst*
_output_shapes
:D*
dtype0*
valueBDB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
Ö
save_20/SaveV2SaveV2save_20/ShardedFilenamesave_20/SaveV2/tensor_namessave_20/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta1_power_2beta2_powerbeta2_power_1beta2_power_2penalty/penalty_parampenalty/penalty_param/Adampenalty/penalty_param/Adam_1penalty/penalty_param/Adam_2penalty/penalty_param/Adam_3pi/dense/biaspi/dense/bias/Adampi/dense/bias/Adam_1pi/dense/kernelpi/dense/kernel/Adampi/dense/kernel/Adam_1pi/dense_1/biaspi/dense_1/bias/Adampi/dense_1/bias/Adam_1pi/dense_1/kernelpi/dense_1/kernel/Adampi/dense_1/kernel/Adam_1pi/dense_2/biaspi/dense_2/bias/Adampi/dense_2/bias/Adam_1pi/dense_2/kernelpi/dense_2/kernel/Adampi/dense_2/kernel/Adam_1
pi/log_stdpi/log_std/Adampi/log_std/Adam_1vc/dense/biasvc/dense/bias/Adamvc/dense/bias/Adam_1vc/dense/kernelvc/dense/kernel/Adamvc/dense/kernel/Adam_1vc/dense_1/biasvc/dense_1/bias/Adamvc/dense_1/bias/Adam_1vc/dense_1/kernelvc/dense_1/kernel/Adamvc/dense_1/kernel/Adam_1vc/dense_2/biasvc/dense_2/bias/Adamvc/dense_2/bias/Adam_1vc/dense_2/kernelvc/dense_2/kernel/Adamvc/dense_2/kernel/Adam_1vf/dense/biasvf/dense/bias/Adamvf/dense/bias/Adam_1vf/dense/kernelvf/dense/kernel/Adamvf/dense/kernel/Adam_1vf/dense_1/biasvf/dense_1/bias/Adamvf/dense_1/bias/Adam_1vf/dense_1/kernelvf/dense_1/kernel/Adamvf/dense_1/kernel/Adam_1vf/dense_2/biasvf/dense_2/bias/Adamvf/dense_2/bias/Adam_1vf/dense_2/kernelvf/dense_2/kernel/Adamvf/dense_2/kernel/Adam_1*R
dtypesH
F2D

save_20/control_dependencyIdentitysave_20/ShardedFilename^save_20/SaveV2*
T0*
_output_shapes
: **
_class 
loc:@save_20/ShardedFilename
Ś
.save_20/MergeV2Checkpoints/checkpoint_prefixesPacksave_20/ShardedFilename^save_20/control_dependency*
_output_shapes
:*
T0*

axis *
N

save_20/MergeV2CheckpointsMergeV2Checkpoints.save_20/MergeV2Checkpoints/checkpoint_prefixessave_20/Const*
delete_old_dirs(

save_20/IdentityIdentitysave_20/Const^save_20/MergeV2Checkpoints^save_20/control_dependency*
_output_shapes
: *
T0
ű
save_20/RestoreV2/tensor_namesConst*
_output_shapes
:D*
dtype0*¨
valueBDBbeta1_powerBbeta1_power_1Bbeta1_power_2Bbeta2_powerBbeta2_power_1Bbeta2_power_2Bpenalty/penalty_paramBpenalty/penalty_param/AdamBpenalty/penalty_param/Adam_1Bpenalty/penalty_param/Adam_2Bpenalty/penalty_param/Adam_3Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1B
pi/log_stdBpi/log_std/AdamBpi/log_std/Adam_1Bvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1
ô
"save_20/RestoreV2/shape_and_slicesConst*
valueBDB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:D
î
save_20/RestoreV2	RestoreV2save_20/Constsave_20/RestoreV2/tensor_names"save_20/RestoreV2/shape_and_slices*Ś
_output_shapes
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*R
dtypesH
F2D
Ź
save_20/AssignAssignbeta1_powersave_20/RestoreV2*
T0*
validate_shape(*
_output_shapes
: *(
_class
loc:@penalty/penalty_param*
use_locking(
˛
save_20/Assign_1Assignbeta1_power_1save_20/RestoreV2:1*(
_class
loc:@penalty/penalty_param*
use_locking(*
validate_shape(*
T0*
_output_shapes
: 
Ş
save_20/Assign_2Assignbeta1_power_2save_20/RestoreV2:2*
use_locking(*
validate_shape(* 
_class
loc:@vc/dense/bias*
T0*
_output_shapes
: 
°
save_20/Assign_3Assignbeta2_powersave_20/RestoreV2:3*
validate_shape(*
_output_shapes
: *(
_class
loc:@penalty/penalty_param*
use_locking(*
T0
˛
save_20/Assign_4Assignbeta2_power_1save_20/RestoreV2:4*
T0*(
_class
loc:@penalty/penalty_param*
validate_shape(*
use_locking(*
_output_shapes
: 
Ş
save_20/Assign_5Assignbeta2_power_2save_20/RestoreV2:5* 
_class
loc:@vc/dense/bias*
use_locking(*
_output_shapes
: *
T0*
validate_shape(
ş
save_20/Assign_6Assignpenalty/penalty_paramsave_20/RestoreV2:6*
_output_shapes
: *
validate_shape(*
T0*
use_locking(*(
_class
loc:@penalty/penalty_param
ż
save_20/Assign_7Assignpenalty/penalty_param/Adamsave_20/RestoreV2:7*(
_class
loc:@penalty/penalty_param*
_output_shapes
: *
T0*
validate_shape(*
use_locking(
Á
save_20/Assign_8Assignpenalty/penalty_param/Adam_1save_20/RestoreV2:8*
T0*
use_locking(*
validate_shape(*(
_class
loc:@penalty/penalty_param*
_output_shapes
: 
Á
save_20/Assign_9Assignpenalty/penalty_param/Adam_2save_20/RestoreV2:9*
_output_shapes
: *(
_class
loc:@penalty/penalty_param*
validate_shape(*
T0*
use_locking(
Ă
save_20/Assign_10Assignpenalty/penalty_param/Adam_3save_20/RestoreV2:10*
validate_shape(*
use_locking(*
_output_shapes
: *
T0*(
_class
loc:@penalty/penalty_param
ą
save_20/Assign_11Assignpi/dense/biassave_20/RestoreV2:11* 
_class
loc:@pi/dense/bias*
validate_shape(*
use_locking(*
T0*
_output_shapes	
:
ś
save_20/Assign_12Assignpi/dense/bias/Adamsave_20/RestoreV2:12*
validate_shape(*
_output_shapes	
:*
use_locking(* 
_class
loc:@pi/dense/bias*
T0
¸
save_20/Assign_13Assignpi/dense/bias/Adam_1save_20/RestoreV2:13*
use_locking(*
_output_shapes	
:* 
_class
loc:@pi/dense/bias*
validate_shape(*
T0
š
save_20/Assign_14Assignpi/dense/kernelsave_20/RestoreV2:14*
use_locking(*
T0*"
_class
loc:@pi/dense/kernel*
_output_shapes
:	L*
validate_shape(
ž
save_20/Assign_15Assignpi/dense/kernel/Adamsave_20/RestoreV2:15*
_output_shapes
:	L*
validate_shape(*
use_locking(*"
_class
loc:@pi/dense/kernel*
T0
Ŕ
save_20/Assign_16Assignpi/dense/kernel/Adam_1save_20/RestoreV2:16*"
_class
loc:@pi/dense/kernel*
_output_shapes
:	L*
validate_shape(*
T0*
use_locking(
ľ
save_20/Assign_17Assignpi/dense_1/biassave_20/RestoreV2:17*
_output_shapes	
:*
validate_shape(*
T0*
use_locking(*"
_class
loc:@pi/dense_1/bias
ş
save_20/Assign_18Assignpi/dense_1/bias/Adamsave_20/RestoreV2:18*
use_locking(*
T0*
validate_shape(*
_output_shapes	
:*"
_class
loc:@pi/dense_1/bias
ź
save_20/Assign_19Assignpi/dense_1/bias/Adam_1save_20/RestoreV2:19*
T0*
_output_shapes	
:*"
_class
loc:@pi/dense_1/bias*
use_locking(*
validate_shape(
ž
save_20/Assign_20Assignpi/dense_1/kernelsave_20/RestoreV2:20* 
_output_shapes
:
*
use_locking(*
validate_shape(*
T0*$
_class
loc:@pi/dense_1/kernel
Ă
save_20/Assign_21Assignpi/dense_1/kernel/Adamsave_20/RestoreV2:21*
T0*
use_locking(* 
_output_shapes
:
*$
_class
loc:@pi/dense_1/kernel*
validate_shape(
Ĺ
save_20/Assign_22Assignpi/dense_1/kernel/Adam_1save_20/RestoreV2:22* 
_output_shapes
:
*
use_locking(*
validate_shape(*$
_class
loc:@pi/dense_1/kernel*
T0
´
save_20/Assign_23Assignpi/dense_2/biassave_20/RestoreV2:23*"
_class
loc:@pi/dense_2/bias*
T0*
validate_shape(*
_output_shapes
:*
use_locking(
š
save_20/Assign_24Assignpi/dense_2/bias/Adamsave_20/RestoreV2:24*
T0*"
_class
loc:@pi/dense_2/bias*
validate_shape(*
use_locking(*
_output_shapes
:
ť
save_20/Assign_25Assignpi/dense_2/bias/Adam_1save_20/RestoreV2:25*
validate_shape(*
T0*
use_locking(*
_output_shapes
:*"
_class
loc:@pi/dense_2/bias
˝
save_20/Assign_26Assignpi/dense_2/kernelsave_20/RestoreV2:26*
validate_shape(*
use_locking(*
T0*$
_class
loc:@pi/dense_2/kernel*
_output_shapes
:	
Â
save_20/Assign_27Assignpi/dense_2/kernel/Adamsave_20/RestoreV2:27*
_output_shapes
:	*
T0*
validate_shape(*
use_locking(*$
_class
loc:@pi/dense_2/kernel
Ä
save_20/Assign_28Assignpi/dense_2/kernel/Adam_1save_20/RestoreV2:28*
use_locking(*
validate_shape(*
T0*$
_class
loc:@pi/dense_2/kernel*
_output_shapes
:	
Ş
save_20/Assign_29Assign
pi/log_stdsave_20/RestoreV2:29*
validate_shape(*
_class
loc:@pi/log_std*
use_locking(*
_output_shapes
:*
T0
Ż
save_20/Assign_30Assignpi/log_std/Adamsave_20/RestoreV2:30*
validate_shape(*
use_locking(*
_class
loc:@pi/log_std*
_output_shapes
:*
T0
ą
save_20/Assign_31Assignpi/log_std/Adam_1save_20/RestoreV2:31*
_output_shapes
:*
use_locking(*
T0*
_class
loc:@pi/log_std*
validate_shape(
ą
save_20/Assign_32Assignvc/dense/biassave_20/RestoreV2:32*
_output_shapes	
:* 
_class
loc:@vc/dense/bias*
validate_shape(*
use_locking(*
T0
ś
save_20/Assign_33Assignvc/dense/bias/Adamsave_20/RestoreV2:33*
use_locking(* 
_class
loc:@vc/dense/bias*
validate_shape(*
_output_shapes	
:*
T0
¸
save_20/Assign_34Assignvc/dense/bias/Adam_1save_20/RestoreV2:34*
T0* 
_class
loc:@vc/dense/bias*
validate_shape(*
_output_shapes	
:*
use_locking(
š
save_20/Assign_35Assignvc/dense/kernelsave_20/RestoreV2:35*
_output_shapes
:	L*
validate_shape(*
T0*"
_class
loc:@vc/dense/kernel*
use_locking(
ž
save_20/Assign_36Assignvc/dense/kernel/Adamsave_20/RestoreV2:36*
T0*
use_locking(*
validate_shape(*
_output_shapes
:	L*"
_class
loc:@vc/dense/kernel
Ŕ
save_20/Assign_37Assignvc/dense/kernel/Adam_1save_20/RestoreV2:37*
T0*"
_class
loc:@vc/dense/kernel*
_output_shapes
:	L*
validate_shape(*
use_locking(
ľ
save_20/Assign_38Assignvc/dense_1/biassave_20/RestoreV2:38*
validate_shape(*"
_class
loc:@vc/dense_1/bias*
use_locking(*
T0*
_output_shapes	
:
ş
save_20/Assign_39Assignvc/dense_1/bias/Adamsave_20/RestoreV2:39*
T0*
_output_shapes	
:*
use_locking(*
validate_shape(*"
_class
loc:@vc/dense_1/bias
ź
save_20/Assign_40Assignvc/dense_1/bias/Adam_1save_20/RestoreV2:40*"
_class
loc:@vc/dense_1/bias*
use_locking(*
validate_shape(*
_output_shapes	
:*
T0
ž
save_20/Assign_41Assignvc/dense_1/kernelsave_20/RestoreV2:41*
validate_shape(*
use_locking(* 
_output_shapes
:
*
T0*$
_class
loc:@vc/dense_1/kernel
Ă
save_20/Assign_42Assignvc/dense_1/kernel/Adamsave_20/RestoreV2:42*
validate_shape(*
use_locking(*
T0* 
_output_shapes
:
*$
_class
loc:@vc/dense_1/kernel
Ĺ
save_20/Assign_43Assignvc/dense_1/kernel/Adam_1save_20/RestoreV2:43* 
_output_shapes
:
*
T0*
validate_shape(*$
_class
loc:@vc/dense_1/kernel*
use_locking(
´
save_20/Assign_44Assignvc/dense_2/biassave_20/RestoreV2:44*"
_class
loc:@vc/dense_2/bias*
T0*
_output_shapes
:*
use_locking(*
validate_shape(
š
save_20/Assign_45Assignvc/dense_2/bias/Adamsave_20/RestoreV2:45*
validate_shape(*"
_class
loc:@vc/dense_2/bias*
T0*
use_locking(*
_output_shapes
:
ť
save_20/Assign_46Assignvc/dense_2/bias/Adam_1save_20/RestoreV2:46*
T0*
use_locking(*
_output_shapes
:*
validate_shape(*"
_class
loc:@vc/dense_2/bias
˝
save_20/Assign_47Assignvc/dense_2/kernelsave_20/RestoreV2:47*
T0*
use_locking(*
validate_shape(*
_output_shapes
:	*$
_class
loc:@vc/dense_2/kernel
Â
save_20/Assign_48Assignvc/dense_2/kernel/Adamsave_20/RestoreV2:48*
T0*
validate_shape(*
_output_shapes
:	*$
_class
loc:@vc/dense_2/kernel*
use_locking(
Ä
save_20/Assign_49Assignvc/dense_2/kernel/Adam_1save_20/RestoreV2:49*
validate_shape(*
T0*
_output_shapes
:	*$
_class
loc:@vc/dense_2/kernel*
use_locking(
ą
save_20/Assign_50Assignvf/dense/biassave_20/RestoreV2:50* 
_class
loc:@vf/dense/bias*
T0*
use_locking(*
validate_shape(*
_output_shapes	
:
ś
save_20/Assign_51Assignvf/dense/bias/Adamsave_20/RestoreV2:51*
T0*
use_locking(*
_output_shapes	
:* 
_class
loc:@vf/dense/bias*
validate_shape(
¸
save_20/Assign_52Assignvf/dense/bias/Adam_1save_20/RestoreV2:52*
validate_shape(*
use_locking(*
_output_shapes	
:*
T0* 
_class
loc:@vf/dense/bias
š
save_20/Assign_53Assignvf/dense/kernelsave_20/RestoreV2:53*
validate_shape(*
use_locking(*
_output_shapes
:	L*
T0*"
_class
loc:@vf/dense/kernel
ž
save_20/Assign_54Assignvf/dense/kernel/Adamsave_20/RestoreV2:54*
use_locking(*
_output_shapes
:	L*"
_class
loc:@vf/dense/kernel*
T0*
validate_shape(
Ŕ
save_20/Assign_55Assignvf/dense/kernel/Adam_1save_20/RestoreV2:55*
use_locking(*
T0*
_output_shapes
:	L*"
_class
loc:@vf/dense/kernel*
validate_shape(
ľ
save_20/Assign_56Assignvf/dense_1/biassave_20/RestoreV2:56*
_output_shapes	
:*"
_class
loc:@vf/dense_1/bias*
validate_shape(*
T0*
use_locking(
ş
save_20/Assign_57Assignvf/dense_1/bias/Adamsave_20/RestoreV2:57*
use_locking(*
_output_shapes	
:*
T0*"
_class
loc:@vf/dense_1/bias*
validate_shape(
ź
save_20/Assign_58Assignvf/dense_1/bias/Adam_1save_20/RestoreV2:58*
T0*
validate_shape(*
_output_shapes	
:*"
_class
loc:@vf/dense_1/bias*
use_locking(
ž
save_20/Assign_59Assignvf/dense_1/kernelsave_20/RestoreV2:59*
validate_shape(*
T0* 
_output_shapes
:
*
use_locking(*$
_class
loc:@vf/dense_1/kernel
Ă
save_20/Assign_60Assignvf/dense_1/kernel/Adamsave_20/RestoreV2:60*
T0*
validate_shape(* 
_output_shapes
:
*
use_locking(*$
_class
loc:@vf/dense_1/kernel
Ĺ
save_20/Assign_61Assignvf/dense_1/kernel/Adam_1save_20/RestoreV2:61*$
_class
loc:@vf/dense_1/kernel*
use_locking(*
validate_shape(*
T0* 
_output_shapes
:

´
save_20/Assign_62Assignvf/dense_2/biassave_20/RestoreV2:62*
_output_shapes
:*
use_locking(*
validate_shape(*
T0*"
_class
loc:@vf/dense_2/bias
š
save_20/Assign_63Assignvf/dense_2/bias/Adamsave_20/RestoreV2:63*
T0*
validate_shape(*
use_locking(*
_output_shapes
:*"
_class
loc:@vf/dense_2/bias
ť
save_20/Assign_64Assignvf/dense_2/bias/Adam_1save_20/RestoreV2:64*
T0*
validate_shape(*
_output_shapes
:*
use_locking(*"
_class
loc:@vf/dense_2/bias
˝
save_20/Assign_65Assignvf/dense_2/kernelsave_20/RestoreV2:65*
T0*
use_locking(*
validate_shape(*$
_class
loc:@vf/dense_2/kernel*
_output_shapes
:	
Â
save_20/Assign_66Assignvf/dense_2/kernel/Adamsave_20/RestoreV2:66*
use_locking(*
validate_shape(*$
_class
loc:@vf/dense_2/kernel*
_output_shapes
:	*
T0
Ä
save_20/Assign_67Assignvf/dense_2/kernel/Adam_1save_20/RestoreV2:67*
validate_shape(*
_output_shapes
:	*$
_class
loc:@vf/dense_2/kernel*
T0*
use_locking(
á

save_20/restore_shardNoOp^save_20/Assign^save_20/Assign_1^save_20/Assign_10^save_20/Assign_11^save_20/Assign_12^save_20/Assign_13^save_20/Assign_14^save_20/Assign_15^save_20/Assign_16^save_20/Assign_17^save_20/Assign_18^save_20/Assign_19^save_20/Assign_2^save_20/Assign_20^save_20/Assign_21^save_20/Assign_22^save_20/Assign_23^save_20/Assign_24^save_20/Assign_25^save_20/Assign_26^save_20/Assign_27^save_20/Assign_28^save_20/Assign_29^save_20/Assign_3^save_20/Assign_30^save_20/Assign_31^save_20/Assign_32^save_20/Assign_33^save_20/Assign_34^save_20/Assign_35^save_20/Assign_36^save_20/Assign_37^save_20/Assign_38^save_20/Assign_39^save_20/Assign_4^save_20/Assign_40^save_20/Assign_41^save_20/Assign_42^save_20/Assign_43^save_20/Assign_44^save_20/Assign_45^save_20/Assign_46^save_20/Assign_47^save_20/Assign_48^save_20/Assign_49^save_20/Assign_5^save_20/Assign_50^save_20/Assign_51^save_20/Assign_52^save_20/Assign_53^save_20/Assign_54^save_20/Assign_55^save_20/Assign_56^save_20/Assign_57^save_20/Assign_58^save_20/Assign_59^save_20/Assign_6^save_20/Assign_60^save_20/Assign_61^save_20/Assign_62^save_20/Assign_63^save_20/Assign_64^save_20/Assign_65^save_20/Assign_66^save_20/Assign_67^save_20/Assign_7^save_20/Assign_8^save_20/Assign_9
3
save_20/restore_allNoOp^save_20/restore_shard
\
save_21/filename/inputConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
t
save_21/filenamePlaceholderWithDefaultsave_21/filename/input*
shape: *
_output_shapes
: *
dtype0
k
save_21/ConstPlaceholderWithDefaultsave_21/filename*
shape: *
_output_shapes
: *
dtype0

save_21/StringJoin/inputs_1Const*
_output_shapes
: *<
value3B1 B+_temp_ab45769ca8bb420195c00ebc7fb0fae4/part*
dtype0
~
save_21/StringJoin
StringJoinsave_21/Constsave_21/StringJoin/inputs_1*
N*
	separator *
_output_shapes
: 
T
save_21/num_shardsConst*
dtype0*
_output_shapes
: *
value	B :
_
save_21/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 

save_21/ShardedFilenameShardedFilenamesave_21/StringJoinsave_21/ShardedFilename/shardsave_21/num_shards*
_output_shapes
: 
ř
save_21/SaveV2/tensor_namesConst*
dtype0*¨
valueBDBbeta1_powerBbeta1_power_1Bbeta1_power_2Bbeta2_powerBbeta2_power_1Bbeta2_power_2Bpenalty/penalty_paramBpenalty/penalty_param/AdamBpenalty/penalty_param/Adam_1Bpenalty/penalty_param/Adam_2Bpenalty/penalty_param/Adam_3Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1B
pi/log_stdBpi/log_std/AdamBpi/log_std/Adam_1Bvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1*
_output_shapes
:D
ń
save_21/SaveV2/shape_and_slicesConst*
_output_shapes
:D*
valueBDB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0
Ö
save_21/SaveV2SaveV2save_21/ShardedFilenamesave_21/SaveV2/tensor_namessave_21/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta1_power_2beta2_powerbeta2_power_1beta2_power_2penalty/penalty_parampenalty/penalty_param/Adampenalty/penalty_param/Adam_1penalty/penalty_param/Adam_2penalty/penalty_param/Adam_3pi/dense/biaspi/dense/bias/Adampi/dense/bias/Adam_1pi/dense/kernelpi/dense/kernel/Adampi/dense/kernel/Adam_1pi/dense_1/biaspi/dense_1/bias/Adampi/dense_1/bias/Adam_1pi/dense_1/kernelpi/dense_1/kernel/Adampi/dense_1/kernel/Adam_1pi/dense_2/biaspi/dense_2/bias/Adampi/dense_2/bias/Adam_1pi/dense_2/kernelpi/dense_2/kernel/Adampi/dense_2/kernel/Adam_1
pi/log_stdpi/log_std/Adampi/log_std/Adam_1vc/dense/biasvc/dense/bias/Adamvc/dense/bias/Adam_1vc/dense/kernelvc/dense/kernel/Adamvc/dense/kernel/Adam_1vc/dense_1/biasvc/dense_1/bias/Adamvc/dense_1/bias/Adam_1vc/dense_1/kernelvc/dense_1/kernel/Adamvc/dense_1/kernel/Adam_1vc/dense_2/biasvc/dense_2/bias/Adamvc/dense_2/bias/Adam_1vc/dense_2/kernelvc/dense_2/kernel/Adamvc/dense_2/kernel/Adam_1vf/dense/biasvf/dense/bias/Adamvf/dense/bias/Adam_1vf/dense/kernelvf/dense/kernel/Adamvf/dense/kernel/Adam_1vf/dense_1/biasvf/dense_1/bias/Adamvf/dense_1/bias/Adam_1vf/dense_1/kernelvf/dense_1/kernel/Adamvf/dense_1/kernel/Adam_1vf/dense_2/biasvf/dense_2/bias/Adamvf/dense_2/bias/Adam_1vf/dense_2/kernelvf/dense_2/kernel/Adamvf/dense_2/kernel/Adam_1*R
dtypesH
F2D

save_21/control_dependencyIdentitysave_21/ShardedFilename^save_21/SaveV2*
_output_shapes
: **
_class 
loc:@save_21/ShardedFilename*
T0
Ś
.save_21/MergeV2Checkpoints/checkpoint_prefixesPacksave_21/ShardedFilename^save_21/control_dependency*
N*
_output_shapes
:*
T0*

axis 

save_21/MergeV2CheckpointsMergeV2Checkpoints.save_21/MergeV2Checkpoints/checkpoint_prefixessave_21/Const*
delete_old_dirs(

save_21/IdentityIdentitysave_21/Const^save_21/MergeV2Checkpoints^save_21/control_dependency*
T0*
_output_shapes
: 
ű
save_21/RestoreV2/tensor_namesConst*
_output_shapes
:D*¨
valueBDBbeta1_powerBbeta1_power_1Bbeta1_power_2Bbeta2_powerBbeta2_power_1Bbeta2_power_2Bpenalty/penalty_paramBpenalty/penalty_param/AdamBpenalty/penalty_param/Adam_1Bpenalty/penalty_param/Adam_2Bpenalty/penalty_param/Adam_3Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1B
pi/log_stdBpi/log_std/AdamBpi/log_std/Adam_1Bvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1*
dtype0
ô
"save_21/RestoreV2/shape_and_slicesConst*
valueBDB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:D
î
save_21/RestoreV2	RestoreV2save_21/Constsave_21/RestoreV2/tensor_names"save_21/RestoreV2/shape_and_slices*Ś
_output_shapes
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*R
dtypesH
F2D
Ź
save_21/AssignAssignbeta1_powersave_21/RestoreV2*
_output_shapes
: *
validate_shape(*
use_locking(*(
_class
loc:@penalty/penalty_param*
T0
˛
save_21/Assign_1Assignbeta1_power_1save_21/RestoreV2:1*(
_class
loc:@penalty/penalty_param*
T0*
validate_shape(*
use_locking(*
_output_shapes
: 
Ş
save_21/Assign_2Assignbeta1_power_2save_21/RestoreV2:2*
validate_shape(*
use_locking(*
_output_shapes
: *
T0* 
_class
loc:@vc/dense/bias
°
save_21/Assign_3Assignbeta2_powersave_21/RestoreV2:3*(
_class
loc:@penalty/penalty_param*
_output_shapes
: *
T0*
use_locking(*
validate_shape(
˛
save_21/Assign_4Assignbeta2_power_1save_21/RestoreV2:4*
_output_shapes
: *
validate_shape(*
use_locking(*(
_class
loc:@penalty/penalty_param*
T0
Ş
save_21/Assign_5Assignbeta2_power_2save_21/RestoreV2:5* 
_class
loc:@vc/dense/bias*
validate_shape(*
T0*
use_locking(*
_output_shapes
: 
ş
save_21/Assign_6Assignpenalty/penalty_paramsave_21/RestoreV2:6*
validate_shape(*
T0*
_output_shapes
: *
use_locking(*(
_class
loc:@penalty/penalty_param
ż
save_21/Assign_7Assignpenalty/penalty_param/Adamsave_21/RestoreV2:7*
T0*
validate_shape(*(
_class
loc:@penalty/penalty_param*
_output_shapes
: *
use_locking(
Á
save_21/Assign_8Assignpenalty/penalty_param/Adam_1save_21/RestoreV2:8*
use_locking(*
_output_shapes
: *(
_class
loc:@penalty/penalty_param*
T0*
validate_shape(
Á
save_21/Assign_9Assignpenalty/penalty_param/Adam_2save_21/RestoreV2:9*
_output_shapes
: *
T0*(
_class
loc:@penalty/penalty_param*
use_locking(*
validate_shape(
Ă
save_21/Assign_10Assignpenalty/penalty_param/Adam_3save_21/RestoreV2:10*
T0*
use_locking(*
_output_shapes
: *
validate_shape(*(
_class
loc:@penalty/penalty_param
ą
save_21/Assign_11Assignpi/dense/biassave_21/RestoreV2:11*
_output_shapes	
:*
use_locking(*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(
ś
save_21/Assign_12Assignpi/dense/bias/Adamsave_21/RestoreV2:12*
validate_shape(*
T0*
_output_shapes	
:* 
_class
loc:@pi/dense/bias*
use_locking(
¸
save_21/Assign_13Assignpi/dense/bias/Adam_1save_21/RestoreV2:13* 
_class
loc:@pi/dense/bias*
T0*
_output_shapes	
:*
use_locking(*
validate_shape(
š
save_21/Assign_14Assignpi/dense/kernelsave_21/RestoreV2:14*
T0*
validate_shape(*
_output_shapes
:	L*"
_class
loc:@pi/dense/kernel*
use_locking(
ž
save_21/Assign_15Assignpi/dense/kernel/Adamsave_21/RestoreV2:15*
_output_shapes
:	L*"
_class
loc:@pi/dense/kernel*
validate_shape(*
T0*
use_locking(
Ŕ
save_21/Assign_16Assignpi/dense/kernel/Adam_1save_21/RestoreV2:16*"
_class
loc:@pi/dense/kernel*
_output_shapes
:	L*
use_locking(*
validate_shape(*
T0
ľ
save_21/Assign_17Assignpi/dense_1/biassave_21/RestoreV2:17*
_output_shapes	
:*
use_locking(*
T0*"
_class
loc:@pi/dense_1/bias*
validate_shape(
ş
save_21/Assign_18Assignpi/dense_1/bias/Adamsave_21/RestoreV2:18*
validate_shape(*
T0*
_output_shapes	
:*
use_locking(*"
_class
loc:@pi/dense_1/bias
ź
save_21/Assign_19Assignpi/dense_1/bias/Adam_1save_21/RestoreV2:19*
_output_shapes	
:*
T0*"
_class
loc:@pi/dense_1/bias*
validate_shape(*
use_locking(
ž
save_21/Assign_20Assignpi/dense_1/kernelsave_21/RestoreV2:20*
T0*
use_locking(*$
_class
loc:@pi/dense_1/kernel* 
_output_shapes
:
*
validate_shape(
Ă
save_21/Assign_21Assignpi/dense_1/kernel/Adamsave_21/RestoreV2:21* 
_output_shapes
:
*$
_class
loc:@pi/dense_1/kernel*
validate_shape(*
use_locking(*
T0
Ĺ
save_21/Assign_22Assignpi/dense_1/kernel/Adam_1save_21/RestoreV2:22*$
_class
loc:@pi/dense_1/kernel*
T0*
use_locking(* 
_output_shapes
:
*
validate_shape(
´
save_21/Assign_23Assignpi/dense_2/biassave_21/RestoreV2:23*
use_locking(*"
_class
loc:@pi/dense_2/bias*
validate_shape(*
T0*
_output_shapes
:
š
save_21/Assign_24Assignpi/dense_2/bias/Adamsave_21/RestoreV2:24*
validate_shape(*"
_class
loc:@pi/dense_2/bias*
T0*
_output_shapes
:*
use_locking(
ť
save_21/Assign_25Assignpi/dense_2/bias/Adam_1save_21/RestoreV2:25*
_output_shapes
:*
use_locking(*"
_class
loc:@pi/dense_2/bias*
T0*
validate_shape(
˝
save_21/Assign_26Assignpi/dense_2/kernelsave_21/RestoreV2:26*
T0*
_output_shapes
:	*$
_class
loc:@pi/dense_2/kernel*
validate_shape(*
use_locking(
Â
save_21/Assign_27Assignpi/dense_2/kernel/Adamsave_21/RestoreV2:27*
validate_shape(*
_output_shapes
:	*$
_class
loc:@pi/dense_2/kernel*
T0*
use_locking(
Ä
save_21/Assign_28Assignpi/dense_2/kernel/Adam_1save_21/RestoreV2:28*
use_locking(*$
_class
loc:@pi/dense_2/kernel*
validate_shape(*
T0*
_output_shapes
:	
Ş
save_21/Assign_29Assign
pi/log_stdsave_21/RestoreV2:29*
_class
loc:@pi/log_std*
validate_shape(*
T0*
_output_shapes
:*
use_locking(
Ż
save_21/Assign_30Assignpi/log_std/Adamsave_21/RestoreV2:30*
_output_shapes
:*
_class
loc:@pi/log_std*
T0*
validate_shape(*
use_locking(
ą
save_21/Assign_31Assignpi/log_std/Adam_1save_21/RestoreV2:31*
_output_shapes
:*
use_locking(*
_class
loc:@pi/log_std*
T0*
validate_shape(
ą
save_21/Assign_32Assignvc/dense/biassave_21/RestoreV2:32*
_output_shapes	
:* 
_class
loc:@vc/dense/bias*
use_locking(*
T0*
validate_shape(
ś
save_21/Assign_33Assignvc/dense/bias/Adamsave_21/RestoreV2:33*
validate_shape(*
_output_shapes	
:* 
_class
loc:@vc/dense/bias*
use_locking(*
T0
¸
save_21/Assign_34Assignvc/dense/bias/Adam_1save_21/RestoreV2:34*
validate_shape(*
_output_shapes	
:*
T0* 
_class
loc:@vc/dense/bias*
use_locking(
š
save_21/Assign_35Assignvc/dense/kernelsave_21/RestoreV2:35*"
_class
loc:@vc/dense/kernel*
validate_shape(*
T0*
use_locking(*
_output_shapes
:	L
ž
save_21/Assign_36Assignvc/dense/kernel/Adamsave_21/RestoreV2:36*
_output_shapes
:	L*
validate_shape(*"
_class
loc:@vc/dense/kernel*
use_locking(*
T0
Ŕ
save_21/Assign_37Assignvc/dense/kernel/Adam_1save_21/RestoreV2:37*
use_locking(*"
_class
loc:@vc/dense/kernel*
validate_shape(*
_output_shapes
:	L*
T0
ľ
save_21/Assign_38Assignvc/dense_1/biassave_21/RestoreV2:38*
use_locking(*"
_class
loc:@vc/dense_1/bias*
_output_shapes	
:*
validate_shape(*
T0
ş
save_21/Assign_39Assignvc/dense_1/bias/Adamsave_21/RestoreV2:39*
use_locking(*
T0*
validate_shape(*
_output_shapes	
:*"
_class
loc:@vc/dense_1/bias
ź
save_21/Assign_40Assignvc/dense_1/bias/Adam_1save_21/RestoreV2:40*
T0*
_output_shapes	
:*
validate_shape(*
use_locking(*"
_class
loc:@vc/dense_1/bias
ž
save_21/Assign_41Assignvc/dense_1/kernelsave_21/RestoreV2:41*
validate_shape(* 
_output_shapes
:
*$
_class
loc:@vc/dense_1/kernel*
T0*
use_locking(
Ă
save_21/Assign_42Assignvc/dense_1/kernel/Adamsave_21/RestoreV2:42*
validate_shape(* 
_output_shapes
:
*
T0*$
_class
loc:@vc/dense_1/kernel*
use_locking(
Ĺ
save_21/Assign_43Assignvc/dense_1/kernel/Adam_1save_21/RestoreV2:43* 
_output_shapes
:
*
use_locking(*
validate_shape(*$
_class
loc:@vc/dense_1/kernel*
T0
´
save_21/Assign_44Assignvc/dense_2/biassave_21/RestoreV2:44*
_output_shapes
:*"
_class
loc:@vc/dense_2/bias*
use_locking(*
T0*
validate_shape(
š
save_21/Assign_45Assignvc/dense_2/bias/Adamsave_21/RestoreV2:45*"
_class
loc:@vc/dense_2/bias*
validate_shape(*
use_locking(*
T0*
_output_shapes
:
ť
save_21/Assign_46Assignvc/dense_2/bias/Adam_1save_21/RestoreV2:46*
T0*
validate_shape(*
use_locking(*"
_class
loc:@vc/dense_2/bias*
_output_shapes
:
˝
save_21/Assign_47Assignvc/dense_2/kernelsave_21/RestoreV2:47*
use_locking(*
T0*$
_class
loc:@vc/dense_2/kernel*
_output_shapes
:	*
validate_shape(
Â
save_21/Assign_48Assignvc/dense_2/kernel/Adamsave_21/RestoreV2:48*
T0*
validate_shape(*
_output_shapes
:	*
use_locking(*$
_class
loc:@vc/dense_2/kernel
Ä
save_21/Assign_49Assignvc/dense_2/kernel/Adam_1save_21/RestoreV2:49*
T0*$
_class
loc:@vc/dense_2/kernel*
validate_shape(*
use_locking(*
_output_shapes
:	
ą
save_21/Assign_50Assignvf/dense/biassave_21/RestoreV2:50*
_output_shapes	
:*
T0*
validate_shape(* 
_class
loc:@vf/dense/bias*
use_locking(
ś
save_21/Assign_51Assignvf/dense/bias/Adamsave_21/RestoreV2:51*
validate_shape(*
T0* 
_class
loc:@vf/dense/bias*
use_locking(*
_output_shapes	
:
¸
save_21/Assign_52Assignvf/dense/bias/Adam_1save_21/RestoreV2:52* 
_class
loc:@vf/dense/bias*
use_locking(*
T0*
_output_shapes	
:*
validate_shape(
š
save_21/Assign_53Assignvf/dense/kernelsave_21/RestoreV2:53*
validate_shape(*
_output_shapes
:	L*"
_class
loc:@vf/dense/kernel*
use_locking(*
T0
ž
save_21/Assign_54Assignvf/dense/kernel/Adamsave_21/RestoreV2:54*
validate_shape(*
use_locking(*"
_class
loc:@vf/dense/kernel*
_output_shapes
:	L*
T0
Ŕ
save_21/Assign_55Assignvf/dense/kernel/Adam_1save_21/RestoreV2:55*
T0*"
_class
loc:@vf/dense/kernel*
_output_shapes
:	L*
use_locking(*
validate_shape(
ľ
save_21/Assign_56Assignvf/dense_1/biassave_21/RestoreV2:56*
T0*
_output_shapes	
:*"
_class
loc:@vf/dense_1/bias*
validate_shape(*
use_locking(
ş
save_21/Assign_57Assignvf/dense_1/bias/Adamsave_21/RestoreV2:57*
_output_shapes	
:*
validate_shape(*
T0*"
_class
loc:@vf/dense_1/bias*
use_locking(
ź
save_21/Assign_58Assignvf/dense_1/bias/Adam_1save_21/RestoreV2:58*"
_class
loc:@vf/dense_1/bias*
use_locking(*
validate_shape(*
T0*
_output_shapes	
:
ž
save_21/Assign_59Assignvf/dense_1/kernelsave_21/RestoreV2:59*
T0* 
_output_shapes
:
*$
_class
loc:@vf/dense_1/kernel*
validate_shape(*
use_locking(
Ă
save_21/Assign_60Assignvf/dense_1/kernel/Adamsave_21/RestoreV2:60*$
_class
loc:@vf/dense_1/kernel*
use_locking(* 
_output_shapes
:
*
validate_shape(*
T0
Ĺ
save_21/Assign_61Assignvf/dense_1/kernel/Adam_1save_21/RestoreV2:61* 
_output_shapes
:
*
use_locking(*
T0*
validate_shape(*$
_class
loc:@vf/dense_1/kernel
´
save_21/Assign_62Assignvf/dense_2/biassave_21/RestoreV2:62*
_output_shapes
:*"
_class
loc:@vf/dense_2/bias*
T0*
use_locking(*
validate_shape(
š
save_21/Assign_63Assignvf/dense_2/bias/Adamsave_21/RestoreV2:63*
validate_shape(*
_output_shapes
:*
T0*"
_class
loc:@vf/dense_2/bias*
use_locking(
ť
save_21/Assign_64Assignvf/dense_2/bias/Adam_1save_21/RestoreV2:64*
_output_shapes
:*
T0*
validate_shape(*
use_locking(*"
_class
loc:@vf/dense_2/bias
˝
save_21/Assign_65Assignvf/dense_2/kernelsave_21/RestoreV2:65*$
_class
loc:@vf/dense_2/kernel*
_output_shapes
:	*
T0*
validate_shape(*
use_locking(
Â
save_21/Assign_66Assignvf/dense_2/kernel/Adamsave_21/RestoreV2:66*
use_locking(*
T0*$
_class
loc:@vf/dense_2/kernel*
_output_shapes
:	*
validate_shape(
Ä
save_21/Assign_67Assignvf/dense_2/kernel/Adam_1save_21/RestoreV2:67*
_output_shapes
:	*
validate_shape(*
T0*
use_locking(*$
_class
loc:@vf/dense_2/kernel
á

save_21/restore_shardNoOp^save_21/Assign^save_21/Assign_1^save_21/Assign_10^save_21/Assign_11^save_21/Assign_12^save_21/Assign_13^save_21/Assign_14^save_21/Assign_15^save_21/Assign_16^save_21/Assign_17^save_21/Assign_18^save_21/Assign_19^save_21/Assign_2^save_21/Assign_20^save_21/Assign_21^save_21/Assign_22^save_21/Assign_23^save_21/Assign_24^save_21/Assign_25^save_21/Assign_26^save_21/Assign_27^save_21/Assign_28^save_21/Assign_29^save_21/Assign_3^save_21/Assign_30^save_21/Assign_31^save_21/Assign_32^save_21/Assign_33^save_21/Assign_34^save_21/Assign_35^save_21/Assign_36^save_21/Assign_37^save_21/Assign_38^save_21/Assign_39^save_21/Assign_4^save_21/Assign_40^save_21/Assign_41^save_21/Assign_42^save_21/Assign_43^save_21/Assign_44^save_21/Assign_45^save_21/Assign_46^save_21/Assign_47^save_21/Assign_48^save_21/Assign_49^save_21/Assign_5^save_21/Assign_50^save_21/Assign_51^save_21/Assign_52^save_21/Assign_53^save_21/Assign_54^save_21/Assign_55^save_21/Assign_56^save_21/Assign_57^save_21/Assign_58^save_21/Assign_59^save_21/Assign_6^save_21/Assign_60^save_21/Assign_61^save_21/Assign_62^save_21/Assign_63^save_21/Assign_64^save_21/Assign_65^save_21/Assign_66^save_21/Assign_67^save_21/Assign_7^save_21/Assign_8^save_21/Assign_9
3
save_21/restore_allNoOp^save_21/restore_shard
\
save_22/filename/inputConst*
dtype0*
valueB Bmodel*
_output_shapes
: 
t
save_22/filenamePlaceholderWithDefaultsave_22/filename/input*
shape: *
dtype0*
_output_shapes
: 
k
save_22/ConstPlaceholderWithDefaultsave_22/filename*
dtype0*
shape: *
_output_shapes
: 

save_22/StringJoin/inputs_1Const*<
value3B1 B+_temp_1c68aecada3b4f5d9df769d0c88485f6/part*
_output_shapes
: *
dtype0
~
save_22/StringJoin
StringJoinsave_22/Constsave_22/StringJoin/inputs_1*
N*
	separator *
_output_shapes
: 
T
save_22/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
_
save_22/ShardedFilename/shardConst*
_output_shapes
: *
value	B : *
dtype0

save_22/ShardedFilenameShardedFilenamesave_22/StringJoinsave_22/ShardedFilename/shardsave_22/num_shards*
_output_shapes
: 
ř
save_22/SaveV2/tensor_namesConst*
_output_shapes
:D*
dtype0*¨
valueBDBbeta1_powerBbeta1_power_1Bbeta1_power_2Bbeta2_powerBbeta2_power_1Bbeta2_power_2Bpenalty/penalty_paramBpenalty/penalty_param/AdamBpenalty/penalty_param/Adam_1Bpenalty/penalty_param/Adam_2Bpenalty/penalty_param/Adam_3Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1B
pi/log_stdBpi/log_std/AdamBpi/log_std/Adam_1Bvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1
ń
save_22/SaveV2/shape_and_slicesConst*
dtype0*
_output_shapes
:D*
valueBDB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
Ö
save_22/SaveV2SaveV2save_22/ShardedFilenamesave_22/SaveV2/tensor_namessave_22/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta1_power_2beta2_powerbeta2_power_1beta2_power_2penalty/penalty_parampenalty/penalty_param/Adampenalty/penalty_param/Adam_1penalty/penalty_param/Adam_2penalty/penalty_param/Adam_3pi/dense/biaspi/dense/bias/Adampi/dense/bias/Adam_1pi/dense/kernelpi/dense/kernel/Adampi/dense/kernel/Adam_1pi/dense_1/biaspi/dense_1/bias/Adampi/dense_1/bias/Adam_1pi/dense_1/kernelpi/dense_1/kernel/Adampi/dense_1/kernel/Adam_1pi/dense_2/biaspi/dense_2/bias/Adampi/dense_2/bias/Adam_1pi/dense_2/kernelpi/dense_2/kernel/Adampi/dense_2/kernel/Adam_1
pi/log_stdpi/log_std/Adampi/log_std/Adam_1vc/dense/biasvc/dense/bias/Adamvc/dense/bias/Adam_1vc/dense/kernelvc/dense/kernel/Adamvc/dense/kernel/Adam_1vc/dense_1/biasvc/dense_1/bias/Adamvc/dense_1/bias/Adam_1vc/dense_1/kernelvc/dense_1/kernel/Adamvc/dense_1/kernel/Adam_1vc/dense_2/biasvc/dense_2/bias/Adamvc/dense_2/bias/Adam_1vc/dense_2/kernelvc/dense_2/kernel/Adamvc/dense_2/kernel/Adam_1vf/dense/biasvf/dense/bias/Adamvf/dense/bias/Adam_1vf/dense/kernelvf/dense/kernel/Adamvf/dense/kernel/Adam_1vf/dense_1/biasvf/dense_1/bias/Adamvf/dense_1/bias/Adam_1vf/dense_1/kernelvf/dense_1/kernel/Adamvf/dense_1/kernel/Adam_1vf/dense_2/biasvf/dense_2/bias/Adamvf/dense_2/bias/Adam_1vf/dense_2/kernelvf/dense_2/kernel/Adamvf/dense_2/kernel/Adam_1*R
dtypesH
F2D

save_22/control_dependencyIdentitysave_22/ShardedFilename^save_22/SaveV2**
_class 
loc:@save_22/ShardedFilename*
_output_shapes
: *
T0
Ś
.save_22/MergeV2Checkpoints/checkpoint_prefixesPacksave_22/ShardedFilename^save_22/control_dependency*
T0*

axis *
N*
_output_shapes
:

save_22/MergeV2CheckpointsMergeV2Checkpoints.save_22/MergeV2Checkpoints/checkpoint_prefixessave_22/Const*
delete_old_dirs(

save_22/IdentityIdentitysave_22/Const^save_22/MergeV2Checkpoints^save_22/control_dependency*
_output_shapes
: *
T0
ű
save_22/RestoreV2/tensor_namesConst*¨
valueBDBbeta1_powerBbeta1_power_1Bbeta1_power_2Bbeta2_powerBbeta2_power_1Bbeta2_power_2Bpenalty/penalty_paramBpenalty/penalty_param/AdamBpenalty/penalty_param/Adam_1Bpenalty/penalty_param/Adam_2Bpenalty/penalty_param/Adam_3Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1B
pi/log_stdBpi/log_std/AdamBpi/log_std/Adam_1Bvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1*
dtype0*
_output_shapes
:D
ô
"save_22/RestoreV2/shape_and_slicesConst*
_output_shapes
:D*
valueBDB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0
î
save_22/RestoreV2	RestoreV2save_22/Constsave_22/RestoreV2/tensor_names"save_22/RestoreV2/shape_and_slices*R
dtypesH
F2D*Ś
_output_shapes
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
Ź
save_22/AssignAssignbeta1_powersave_22/RestoreV2*
validate_shape(*
T0*(
_class
loc:@penalty/penalty_param*
use_locking(*
_output_shapes
: 
˛
save_22/Assign_1Assignbeta1_power_1save_22/RestoreV2:1*
T0*(
_class
loc:@penalty/penalty_param*
validate_shape(*
_output_shapes
: *
use_locking(
Ş
save_22/Assign_2Assignbeta1_power_2save_22/RestoreV2:2*
validate_shape(*
use_locking(* 
_class
loc:@vc/dense/bias*
T0*
_output_shapes
: 
°
save_22/Assign_3Assignbeta2_powersave_22/RestoreV2:3*
T0*
use_locking(*
validate_shape(*
_output_shapes
: *(
_class
loc:@penalty/penalty_param
˛
save_22/Assign_4Assignbeta2_power_1save_22/RestoreV2:4*
validate_shape(*
_output_shapes
: *
T0*(
_class
loc:@penalty/penalty_param*
use_locking(
Ş
save_22/Assign_5Assignbeta2_power_2save_22/RestoreV2:5*
use_locking(*
_output_shapes
: *
T0*
validate_shape(* 
_class
loc:@vc/dense/bias
ş
save_22/Assign_6Assignpenalty/penalty_paramsave_22/RestoreV2:6*(
_class
loc:@penalty/penalty_param*
validate_shape(*
use_locking(*
_output_shapes
: *
T0
ż
save_22/Assign_7Assignpenalty/penalty_param/Adamsave_22/RestoreV2:7*
validate_shape(*
use_locking(*(
_class
loc:@penalty/penalty_param*
_output_shapes
: *
T0
Á
save_22/Assign_8Assignpenalty/penalty_param/Adam_1save_22/RestoreV2:8*
validate_shape(*
use_locking(*(
_class
loc:@penalty/penalty_param*
T0*
_output_shapes
: 
Á
save_22/Assign_9Assignpenalty/penalty_param/Adam_2save_22/RestoreV2:9*
use_locking(*(
_class
loc:@penalty/penalty_param*
T0*
validate_shape(*
_output_shapes
: 
Ă
save_22/Assign_10Assignpenalty/penalty_param/Adam_3save_22/RestoreV2:10*
use_locking(*(
_class
loc:@penalty/penalty_param*
T0*
validate_shape(*
_output_shapes
: 
ą
save_22/Assign_11Assignpi/dense/biassave_22/RestoreV2:11*
validate_shape(*
_output_shapes	
:* 
_class
loc:@pi/dense/bias*
use_locking(*
T0
ś
save_22/Assign_12Assignpi/dense/bias/Adamsave_22/RestoreV2:12*
_output_shapes	
:* 
_class
loc:@pi/dense/bias*
T0*
use_locking(*
validate_shape(
¸
save_22/Assign_13Assignpi/dense/bias/Adam_1save_22/RestoreV2:13* 
_class
loc:@pi/dense/bias*
use_locking(*
T0*
validate_shape(*
_output_shapes	
:
š
save_22/Assign_14Assignpi/dense/kernelsave_22/RestoreV2:14*
_output_shapes
:	L*
use_locking(*
T0*
validate_shape(*"
_class
loc:@pi/dense/kernel
ž
save_22/Assign_15Assignpi/dense/kernel/Adamsave_22/RestoreV2:15*
_output_shapes
:	L*"
_class
loc:@pi/dense/kernel*
validate_shape(*
T0*
use_locking(
Ŕ
save_22/Assign_16Assignpi/dense/kernel/Adam_1save_22/RestoreV2:16*
T0*
use_locking(*
validate_shape(*"
_class
loc:@pi/dense/kernel*
_output_shapes
:	L
ľ
save_22/Assign_17Assignpi/dense_1/biassave_22/RestoreV2:17*"
_class
loc:@pi/dense_1/bias*
T0*
_output_shapes	
:*
validate_shape(*
use_locking(
ş
save_22/Assign_18Assignpi/dense_1/bias/Adamsave_22/RestoreV2:18*
use_locking(*
T0*
_output_shapes	
:*"
_class
loc:@pi/dense_1/bias*
validate_shape(
ź
save_22/Assign_19Assignpi/dense_1/bias/Adam_1save_22/RestoreV2:19*
_output_shapes	
:*
validate_shape(*
T0*
use_locking(*"
_class
loc:@pi/dense_1/bias
ž
save_22/Assign_20Assignpi/dense_1/kernelsave_22/RestoreV2:20*
use_locking(* 
_output_shapes
:
*
validate_shape(*
T0*$
_class
loc:@pi/dense_1/kernel
Ă
save_22/Assign_21Assignpi/dense_1/kernel/Adamsave_22/RestoreV2:21*
T0*
use_locking(*$
_class
loc:@pi/dense_1/kernel* 
_output_shapes
:
*
validate_shape(
Ĺ
save_22/Assign_22Assignpi/dense_1/kernel/Adam_1save_22/RestoreV2:22*
validate_shape(*
use_locking(* 
_output_shapes
:
*$
_class
loc:@pi/dense_1/kernel*
T0
´
save_22/Assign_23Assignpi/dense_2/biassave_22/RestoreV2:23*
T0*
_output_shapes
:*
use_locking(*"
_class
loc:@pi/dense_2/bias*
validate_shape(
š
save_22/Assign_24Assignpi/dense_2/bias/Adamsave_22/RestoreV2:24*
validate_shape(*"
_class
loc:@pi/dense_2/bias*
T0*
use_locking(*
_output_shapes
:
ť
save_22/Assign_25Assignpi/dense_2/bias/Adam_1save_22/RestoreV2:25*"
_class
loc:@pi/dense_2/bias*
validate_shape(*
T0*
use_locking(*
_output_shapes
:
˝
save_22/Assign_26Assignpi/dense_2/kernelsave_22/RestoreV2:26*$
_class
loc:@pi/dense_2/kernel*
_output_shapes
:	*
T0*
validate_shape(*
use_locking(
Â
save_22/Assign_27Assignpi/dense_2/kernel/Adamsave_22/RestoreV2:27*$
_class
loc:@pi/dense_2/kernel*
_output_shapes
:	*
use_locking(*
T0*
validate_shape(
Ä
save_22/Assign_28Assignpi/dense_2/kernel/Adam_1save_22/RestoreV2:28*$
_class
loc:@pi/dense_2/kernel*
_output_shapes
:	*
use_locking(*
T0*
validate_shape(
Ş
save_22/Assign_29Assign
pi/log_stdsave_22/RestoreV2:29*
T0*
_class
loc:@pi/log_std*
validate_shape(*
_output_shapes
:*
use_locking(
Ż
save_22/Assign_30Assignpi/log_std/Adamsave_22/RestoreV2:30*
T0*
_output_shapes
:*
use_locking(*
validate_shape(*
_class
loc:@pi/log_std
ą
save_22/Assign_31Assignpi/log_std/Adam_1save_22/RestoreV2:31*
validate_shape(*
_class
loc:@pi/log_std*
_output_shapes
:*
use_locking(*
T0
ą
save_22/Assign_32Assignvc/dense/biassave_22/RestoreV2:32*
_output_shapes	
:*
use_locking(*
T0*
validate_shape(* 
_class
loc:@vc/dense/bias
ś
save_22/Assign_33Assignvc/dense/bias/Adamsave_22/RestoreV2:33*
_output_shapes	
:* 
_class
loc:@vc/dense/bias*
T0*
use_locking(*
validate_shape(
¸
save_22/Assign_34Assignvc/dense/bias/Adam_1save_22/RestoreV2:34*
use_locking(* 
_class
loc:@vc/dense/bias*
_output_shapes	
:*
T0*
validate_shape(
š
save_22/Assign_35Assignvc/dense/kernelsave_22/RestoreV2:35*"
_class
loc:@vc/dense/kernel*
validate_shape(*
_output_shapes
:	L*
use_locking(*
T0
ž
save_22/Assign_36Assignvc/dense/kernel/Adamsave_22/RestoreV2:36*
T0*
use_locking(*"
_class
loc:@vc/dense/kernel*
_output_shapes
:	L*
validate_shape(
Ŕ
save_22/Assign_37Assignvc/dense/kernel/Adam_1save_22/RestoreV2:37*
_output_shapes
:	L*
T0*
use_locking(*
validate_shape(*"
_class
loc:@vc/dense/kernel
ľ
save_22/Assign_38Assignvc/dense_1/biassave_22/RestoreV2:38*"
_class
loc:@vc/dense_1/bias*
T0*
use_locking(*
validate_shape(*
_output_shapes	
:
ş
save_22/Assign_39Assignvc/dense_1/bias/Adamsave_22/RestoreV2:39*"
_class
loc:@vc/dense_1/bias*
T0*
use_locking(*
validate_shape(*
_output_shapes	
:
ź
save_22/Assign_40Assignvc/dense_1/bias/Adam_1save_22/RestoreV2:40*
T0*
_output_shapes	
:*
validate_shape(*
use_locking(*"
_class
loc:@vc/dense_1/bias
ž
save_22/Assign_41Assignvc/dense_1/kernelsave_22/RestoreV2:41*$
_class
loc:@vc/dense_1/kernel*
use_locking(*
validate_shape(*
T0* 
_output_shapes
:

Ă
save_22/Assign_42Assignvc/dense_1/kernel/Adamsave_22/RestoreV2:42*
use_locking(*
validate_shape(* 
_output_shapes
:
*$
_class
loc:@vc/dense_1/kernel*
T0
Ĺ
save_22/Assign_43Assignvc/dense_1/kernel/Adam_1save_22/RestoreV2:43*
T0* 
_output_shapes
:
*
use_locking(*$
_class
loc:@vc/dense_1/kernel*
validate_shape(
´
save_22/Assign_44Assignvc/dense_2/biassave_22/RestoreV2:44*
_output_shapes
:*
T0*"
_class
loc:@vc/dense_2/bias*
use_locking(*
validate_shape(
š
save_22/Assign_45Assignvc/dense_2/bias/Adamsave_22/RestoreV2:45*
_output_shapes
:*"
_class
loc:@vc/dense_2/bias*
T0*
use_locking(*
validate_shape(
ť
save_22/Assign_46Assignvc/dense_2/bias/Adam_1save_22/RestoreV2:46*
T0*
use_locking(*
_output_shapes
:*
validate_shape(*"
_class
loc:@vc/dense_2/bias
˝
save_22/Assign_47Assignvc/dense_2/kernelsave_22/RestoreV2:47*$
_class
loc:@vc/dense_2/kernel*
validate_shape(*
_output_shapes
:	*
T0*
use_locking(
Â
save_22/Assign_48Assignvc/dense_2/kernel/Adamsave_22/RestoreV2:48*
_output_shapes
:	*
T0*$
_class
loc:@vc/dense_2/kernel*
use_locking(*
validate_shape(
Ä
save_22/Assign_49Assignvc/dense_2/kernel/Adam_1save_22/RestoreV2:49*
validate_shape(*$
_class
loc:@vc/dense_2/kernel*
_output_shapes
:	*
use_locking(*
T0
ą
save_22/Assign_50Assignvf/dense/biassave_22/RestoreV2:50*
T0*
use_locking(* 
_class
loc:@vf/dense/bias*
_output_shapes	
:*
validate_shape(
ś
save_22/Assign_51Assignvf/dense/bias/Adamsave_22/RestoreV2:51*
T0*
validate_shape(*
_output_shapes	
:*
use_locking(* 
_class
loc:@vf/dense/bias
¸
save_22/Assign_52Assignvf/dense/bias/Adam_1save_22/RestoreV2:52*
validate_shape(* 
_class
loc:@vf/dense/bias*
use_locking(*
_output_shapes	
:*
T0
š
save_22/Assign_53Assignvf/dense/kernelsave_22/RestoreV2:53*
_output_shapes
:	L*
T0*
use_locking(*
validate_shape(*"
_class
loc:@vf/dense/kernel
ž
save_22/Assign_54Assignvf/dense/kernel/Adamsave_22/RestoreV2:54*
_output_shapes
:	L*
T0*
use_locking(*
validate_shape(*"
_class
loc:@vf/dense/kernel
Ŕ
save_22/Assign_55Assignvf/dense/kernel/Adam_1save_22/RestoreV2:55*"
_class
loc:@vf/dense/kernel*
_output_shapes
:	L*
use_locking(*
T0*
validate_shape(
ľ
save_22/Assign_56Assignvf/dense_1/biassave_22/RestoreV2:56*
T0*"
_class
loc:@vf/dense_1/bias*
_output_shapes	
:*
use_locking(*
validate_shape(
ş
save_22/Assign_57Assignvf/dense_1/bias/Adamsave_22/RestoreV2:57*"
_class
loc:@vf/dense_1/bias*
T0*
_output_shapes	
:*
validate_shape(*
use_locking(
ź
save_22/Assign_58Assignvf/dense_1/bias/Adam_1save_22/RestoreV2:58*
T0*
use_locking(*
validate_shape(*
_output_shapes	
:*"
_class
loc:@vf/dense_1/bias
ž
save_22/Assign_59Assignvf/dense_1/kernelsave_22/RestoreV2:59* 
_output_shapes
:
*
validate_shape(*
use_locking(*$
_class
loc:@vf/dense_1/kernel*
T0
Ă
save_22/Assign_60Assignvf/dense_1/kernel/Adamsave_22/RestoreV2:60*
validate_shape(* 
_output_shapes
:
*$
_class
loc:@vf/dense_1/kernel*
use_locking(*
T0
Ĺ
save_22/Assign_61Assignvf/dense_1/kernel/Adam_1save_22/RestoreV2:61*$
_class
loc:@vf/dense_1/kernel*
T0* 
_output_shapes
:
*
validate_shape(*
use_locking(
´
save_22/Assign_62Assignvf/dense_2/biassave_22/RestoreV2:62*"
_class
loc:@vf/dense_2/bias*
use_locking(*
validate_shape(*
_output_shapes
:*
T0
š
save_22/Assign_63Assignvf/dense_2/bias/Adamsave_22/RestoreV2:63*
use_locking(*
validate_shape(*
T0*
_output_shapes
:*"
_class
loc:@vf/dense_2/bias
ť
save_22/Assign_64Assignvf/dense_2/bias/Adam_1save_22/RestoreV2:64*
validate_shape(*"
_class
loc:@vf/dense_2/bias*
use_locking(*
T0*
_output_shapes
:
˝
save_22/Assign_65Assignvf/dense_2/kernelsave_22/RestoreV2:65*
T0*
validate_shape(*
use_locking(*$
_class
loc:@vf/dense_2/kernel*
_output_shapes
:	
Â
save_22/Assign_66Assignvf/dense_2/kernel/Adamsave_22/RestoreV2:66*
use_locking(*
_output_shapes
:	*
T0*$
_class
loc:@vf/dense_2/kernel*
validate_shape(
Ä
save_22/Assign_67Assignvf/dense_2/kernel/Adam_1save_22/RestoreV2:67*
validate_shape(*
_output_shapes
:	*$
_class
loc:@vf/dense_2/kernel*
T0*
use_locking(
á

save_22/restore_shardNoOp^save_22/Assign^save_22/Assign_1^save_22/Assign_10^save_22/Assign_11^save_22/Assign_12^save_22/Assign_13^save_22/Assign_14^save_22/Assign_15^save_22/Assign_16^save_22/Assign_17^save_22/Assign_18^save_22/Assign_19^save_22/Assign_2^save_22/Assign_20^save_22/Assign_21^save_22/Assign_22^save_22/Assign_23^save_22/Assign_24^save_22/Assign_25^save_22/Assign_26^save_22/Assign_27^save_22/Assign_28^save_22/Assign_29^save_22/Assign_3^save_22/Assign_30^save_22/Assign_31^save_22/Assign_32^save_22/Assign_33^save_22/Assign_34^save_22/Assign_35^save_22/Assign_36^save_22/Assign_37^save_22/Assign_38^save_22/Assign_39^save_22/Assign_4^save_22/Assign_40^save_22/Assign_41^save_22/Assign_42^save_22/Assign_43^save_22/Assign_44^save_22/Assign_45^save_22/Assign_46^save_22/Assign_47^save_22/Assign_48^save_22/Assign_49^save_22/Assign_5^save_22/Assign_50^save_22/Assign_51^save_22/Assign_52^save_22/Assign_53^save_22/Assign_54^save_22/Assign_55^save_22/Assign_56^save_22/Assign_57^save_22/Assign_58^save_22/Assign_59^save_22/Assign_6^save_22/Assign_60^save_22/Assign_61^save_22/Assign_62^save_22/Assign_63^save_22/Assign_64^save_22/Assign_65^save_22/Assign_66^save_22/Assign_67^save_22/Assign_7^save_22/Assign_8^save_22/Assign_9
3
save_22/restore_allNoOp^save_22/restore_shard
\
save_23/filename/inputConst*
_output_shapes
: *
dtype0*
valueB Bmodel
t
save_23/filenamePlaceholderWithDefaultsave_23/filename/input*
dtype0*
shape: *
_output_shapes
: 
k
save_23/ConstPlaceholderWithDefaultsave_23/filename*
_output_shapes
: *
dtype0*
shape: 

save_23/StringJoin/inputs_1Const*<
value3B1 B+_temp_db8f25806eaa43e8b21c23ea09d0656f/part*
_output_shapes
: *
dtype0
~
save_23/StringJoin
StringJoinsave_23/Constsave_23/StringJoin/inputs_1*
_output_shapes
: *
	separator *
N
T
save_23/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
_
save_23/ShardedFilename/shardConst*
_output_shapes
: *
dtype0*
value	B : 

save_23/ShardedFilenameShardedFilenamesave_23/StringJoinsave_23/ShardedFilename/shardsave_23/num_shards*
_output_shapes
: 
ř
save_23/SaveV2/tensor_namesConst*
_output_shapes
:D*¨
valueBDBbeta1_powerBbeta1_power_1Bbeta1_power_2Bbeta2_powerBbeta2_power_1Bbeta2_power_2Bpenalty/penalty_paramBpenalty/penalty_param/AdamBpenalty/penalty_param/Adam_1Bpenalty/penalty_param/Adam_2Bpenalty/penalty_param/Adam_3Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1B
pi/log_stdBpi/log_std/AdamBpi/log_std/Adam_1Bvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1*
dtype0
ń
save_23/SaveV2/shape_and_slicesConst*
_output_shapes
:D*
dtype0*
valueBDB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
Ö
save_23/SaveV2SaveV2save_23/ShardedFilenamesave_23/SaveV2/tensor_namessave_23/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta1_power_2beta2_powerbeta2_power_1beta2_power_2penalty/penalty_parampenalty/penalty_param/Adampenalty/penalty_param/Adam_1penalty/penalty_param/Adam_2penalty/penalty_param/Adam_3pi/dense/biaspi/dense/bias/Adampi/dense/bias/Adam_1pi/dense/kernelpi/dense/kernel/Adampi/dense/kernel/Adam_1pi/dense_1/biaspi/dense_1/bias/Adampi/dense_1/bias/Adam_1pi/dense_1/kernelpi/dense_1/kernel/Adampi/dense_1/kernel/Adam_1pi/dense_2/biaspi/dense_2/bias/Adampi/dense_2/bias/Adam_1pi/dense_2/kernelpi/dense_2/kernel/Adampi/dense_2/kernel/Adam_1
pi/log_stdpi/log_std/Adampi/log_std/Adam_1vc/dense/biasvc/dense/bias/Adamvc/dense/bias/Adam_1vc/dense/kernelvc/dense/kernel/Adamvc/dense/kernel/Adam_1vc/dense_1/biasvc/dense_1/bias/Adamvc/dense_1/bias/Adam_1vc/dense_1/kernelvc/dense_1/kernel/Adamvc/dense_1/kernel/Adam_1vc/dense_2/biasvc/dense_2/bias/Adamvc/dense_2/bias/Adam_1vc/dense_2/kernelvc/dense_2/kernel/Adamvc/dense_2/kernel/Adam_1vf/dense/biasvf/dense/bias/Adamvf/dense/bias/Adam_1vf/dense/kernelvf/dense/kernel/Adamvf/dense/kernel/Adam_1vf/dense_1/biasvf/dense_1/bias/Adamvf/dense_1/bias/Adam_1vf/dense_1/kernelvf/dense_1/kernel/Adamvf/dense_1/kernel/Adam_1vf/dense_2/biasvf/dense_2/bias/Adamvf/dense_2/bias/Adam_1vf/dense_2/kernelvf/dense_2/kernel/Adamvf/dense_2/kernel/Adam_1*R
dtypesH
F2D

save_23/control_dependencyIdentitysave_23/ShardedFilename^save_23/SaveV2*
T0**
_class 
loc:@save_23/ShardedFilename*
_output_shapes
: 
Ś
.save_23/MergeV2Checkpoints/checkpoint_prefixesPacksave_23/ShardedFilename^save_23/control_dependency*
_output_shapes
:*

axis *
T0*
N

save_23/MergeV2CheckpointsMergeV2Checkpoints.save_23/MergeV2Checkpoints/checkpoint_prefixessave_23/Const*
delete_old_dirs(

save_23/IdentityIdentitysave_23/Const^save_23/MergeV2Checkpoints^save_23/control_dependency*
_output_shapes
: *
T0
ű
save_23/RestoreV2/tensor_namesConst*¨
valueBDBbeta1_powerBbeta1_power_1Bbeta1_power_2Bbeta2_powerBbeta2_power_1Bbeta2_power_2Bpenalty/penalty_paramBpenalty/penalty_param/AdamBpenalty/penalty_param/Adam_1Bpenalty/penalty_param/Adam_2Bpenalty/penalty_param/Adam_3Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1B
pi/log_stdBpi/log_std/AdamBpi/log_std/Adam_1Bvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1*
dtype0*
_output_shapes
:D
ô
"save_23/RestoreV2/shape_and_slicesConst*
dtype0*
valueBDB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
_output_shapes
:D
î
save_23/RestoreV2	RestoreV2save_23/Constsave_23/RestoreV2/tensor_names"save_23/RestoreV2/shape_and_slices*Ś
_output_shapes
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*R
dtypesH
F2D
Ź
save_23/AssignAssignbeta1_powersave_23/RestoreV2*
validate_shape(*
T0*
_output_shapes
: *
use_locking(*(
_class
loc:@penalty/penalty_param
˛
save_23/Assign_1Assignbeta1_power_1save_23/RestoreV2:1*
validate_shape(*
T0*
_output_shapes
: *(
_class
loc:@penalty/penalty_param*
use_locking(
Ş
save_23/Assign_2Assignbeta1_power_2save_23/RestoreV2:2*
validate_shape(*
use_locking(*
T0* 
_class
loc:@vc/dense/bias*
_output_shapes
: 
°
save_23/Assign_3Assignbeta2_powersave_23/RestoreV2:3*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*(
_class
loc:@penalty/penalty_param
˛
save_23/Assign_4Assignbeta2_power_1save_23/RestoreV2:4*
_output_shapes
: *
use_locking(*(
_class
loc:@penalty/penalty_param*
validate_shape(*
T0
Ş
save_23/Assign_5Assignbeta2_power_2save_23/RestoreV2:5*
validate_shape(*
use_locking(*
T0* 
_class
loc:@vc/dense/bias*
_output_shapes
: 
ş
save_23/Assign_6Assignpenalty/penalty_paramsave_23/RestoreV2:6*
_output_shapes
: *
use_locking(*
validate_shape(*(
_class
loc:@penalty/penalty_param*
T0
ż
save_23/Assign_7Assignpenalty/penalty_param/Adamsave_23/RestoreV2:7*
use_locking(*
_output_shapes
: *(
_class
loc:@penalty/penalty_param*
T0*
validate_shape(
Á
save_23/Assign_8Assignpenalty/penalty_param/Adam_1save_23/RestoreV2:8*
use_locking(*
T0*
_output_shapes
: *
validate_shape(*(
_class
loc:@penalty/penalty_param
Á
save_23/Assign_9Assignpenalty/penalty_param/Adam_2save_23/RestoreV2:9*
T0*
use_locking(*
_output_shapes
: *(
_class
loc:@penalty/penalty_param*
validate_shape(
Ă
save_23/Assign_10Assignpenalty/penalty_param/Adam_3save_23/RestoreV2:10*
use_locking(*
T0*
validate_shape(*
_output_shapes
: *(
_class
loc:@penalty/penalty_param
ą
save_23/Assign_11Assignpi/dense/biassave_23/RestoreV2:11* 
_class
loc:@pi/dense/bias*
T0*
_output_shapes	
:*
use_locking(*
validate_shape(
ś
save_23/Assign_12Assignpi/dense/bias/Adamsave_23/RestoreV2:12* 
_class
loc:@pi/dense/bias*
validate_shape(*
T0*
_output_shapes	
:*
use_locking(
¸
save_23/Assign_13Assignpi/dense/bias/Adam_1save_23/RestoreV2:13*
_output_shapes	
:*
T0* 
_class
loc:@pi/dense/bias*
use_locking(*
validate_shape(
š
save_23/Assign_14Assignpi/dense/kernelsave_23/RestoreV2:14*"
_class
loc:@pi/dense/kernel*
T0*
_output_shapes
:	L*
use_locking(*
validate_shape(
ž
save_23/Assign_15Assignpi/dense/kernel/Adamsave_23/RestoreV2:15*
T0*"
_class
loc:@pi/dense/kernel*
_output_shapes
:	L*
use_locking(*
validate_shape(
Ŕ
save_23/Assign_16Assignpi/dense/kernel/Adam_1save_23/RestoreV2:16*
validate_shape(*"
_class
loc:@pi/dense/kernel*
T0*
_output_shapes
:	L*
use_locking(
ľ
save_23/Assign_17Assignpi/dense_1/biassave_23/RestoreV2:17*
T0*
validate_shape(*
_output_shapes	
:*"
_class
loc:@pi/dense_1/bias*
use_locking(
ş
save_23/Assign_18Assignpi/dense_1/bias/Adamsave_23/RestoreV2:18*
T0*
use_locking(*
_output_shapes	
:*"
_class
loc:@pi/dense_1/bias*
validate_shape(
ź
save_23/Assign_19Assignpi/dense_1/bias/Adam_1save_23/RestoreV2:19*
_output_shapes	
:*"
_class
loc:@pi/dense_1/bias*
use_locking(*
validate_shape(*
T0
ž
save_23/Assign_20Assignpi/dense_1/kernelsave_23/RestoreV2:20*
validate_shape(*
use_locking(*$
_class
loc:@pi/dense_1/kernel* 
_output_shapes
:
*
T0
Ă
save_23/Assign_21Assignpi/dense_1/kernel/Adamsave_23/RestoreV2:21*
validate_shape(*
use_locking(*
T0* 
_output_shapes
:
*$
_class
loc:@pi/dense_1/kernel
Ĺ
save_23/Assign_22Assignpi/dense_1/kernel/Adam_1save_23/RestoreV2:22*
T0*
use_locking(*$
_class
loc:@pi/dense_1/kernel* 
_output_shapes
:
*
validate_shape(
´
save_23/Assign_23Assignpi/dense_2/biassave_23/RestoreV2:23*
validate_shape(*
T0*
use_locking(*"
_class
loc:@pi/dense_2/bias*
_output_shapes
:
š
save_23/Assign_24Assignpi/dense_2/bias/Adamsave_23/RestoreV2:24*
use_locking(*
validate_shape(*
_output_shapes
:*
T0*"
_class
loc:@pi/dense_2/bias
ť
save_23/Assign_25Assignpi/dense_2/bias/Adam_1save_23/RestoreV2:25*
validate_shape(*
use_locking(*
T0*"
_class
loc:@pi/dense_2/bias*
_output_shapes
:
˝
save_23/Assign_26Assignpi/dense_2/kernelsave_23/RestoreV2:26*$
_class
loc:@pi/dense_2/kernel*
_output_shapes
:	*
use_locking(*
T0*
validate_shape(
Â
save_23/Assign_27Assignpi/dense_2/kernel/Adamsave_23/RestoreV2:27*
validate_shape(*
_output_shapes
:	*
T0*
use_locking(*$
_class
loc:@pi/dense_2/kernel
Ä
save_23/Assign_28Assignpi/dense_2/kernel/Adam_1save_23/RestoreV2:28*
use_locking(*
validate_shape(*$
_class
loc:@pi/dense_2/kernel*
_output_shapes
:	*
T0
Ş
save_23/Assign_29Assign
pi/log_stdsave_23/RestoreV2:29*
validate_shape(*
_output_shapes
:*
T0*
use_locking(*
_class
loc:@pi/log_std
Ż
save_23/Assign_30Assignpi/log_std/Adamsave_23/RestoreV2:30*
validate_shape(*
_class
loc:@pi/log_std*
use_locking(*
_output_shapes
:*
T0
ą
save_23/Assign_31Assignpi/log_std/Adam_1save_23/RestoreV2:31*
_output_shapes
:*
_class
loc:@pi/log_std*
validate_shape(*
T0*
use_locking(
ą
save_23/Assign_32Assignvc/dense/biassave_23/RestoreV2:32*
T0*
_output_shapes	
:* 
_class
loc:@vc/dense/bias*
use_locking(*
validate_shape(
ś
save_23/Assign_33Assignvc/dense/bias/Adamsave_23/RestoreV2:33*
validate_shape(*
use_locking(* 
_class
loc:@vc/dense/bias*
_output_shapes	
:*
T0
¸
save_23/Assign_34Assignvc/dense/bias/Adam_1save_23/RestoreV2:34*
validate_shape(*
use_locking(*
_output_shapes	
:* 
_class
loc:@vc/dense/bias*
T0
š
save_23/Assign_35Assignvc/dense/kernelsave_23/RestoreV2:35*
_output_shapes
:	L*
validate_shape(*
T0*"
_class
loc:@vc/dense/kernel*
use_locking(
ž
save_23/Assign_36Assignvc/dense/kernel/Adamsave_23/RestoreV2:36*
_output_shapes
:	L*
validate_shape(*
use_locking(*
T0*"
_class
loc:@vc/dense/kernel
Ŕ
save_23/Assign_37Assignvc/dense/kernel/Adam_1save_23/RestoreV2:37*
_output_shapes
:	L*
T0*"
_class
loc:@vc/dense/kernel*
use_locking(*
validate_shape(
ľ
save_23/Assign_38Assignvc/dense_1/biassave_23/RestoreV2:38*
use_locking(*
validate_shape(*
_output_shapes	
:*"
_class
loc:@vc/dense_1/bias*
T0
ş
save_23/Assign_39Assignvc/dense_1/bias/Adamsave_23/RestoreV2:39*
use_locking(*"
_class
loc:@vc/dense_1/bias*
validate_shape(*
_output_shapes	
:*
T0
ź
save_23/Assign_40Assignvc/dense_1/bias/Adam_1save_23/RestoreV2:40*
validate_shape(*
_output_shapes	
:*
T0*"
_class
loc:@vc/dense_1/bias*
use_locking(
ž
save_23/Assign_41Assignvc/dense_1/kernelsave_23/RestoreV2:41*
T0* 
_output_shapes
:
*
use_locking(*
validate_shape(*$
_class
loc:@vc/dense_1/kernel
Ă
save_23/Assign_42Assignvc/dense_1/kernel/Adamsave_23/RestoreV2:42*
T0* 
_output_shapes
:
*
use_locking(*$
_class
loc:@vc/dense_1/kernel*
validate_shape(
Ĺ
save_23/Assign_43Assignvc/dense_1/kernel/Adam_1save_23/RestoreV2:43* 
_output_shapes
:
*$
_class
loc:@vc/dense_1/kernel*
T0*
validate_shape(*
use_locking(
´
save_23/Assign_44Assignvc/dense_2/biassave_23/RestoreV2:44*
validate_shape(*
use_locking(*
T0*
_output_shapes
:*"
_class
loc:@vc/dense_2/bias
š
save_23/Assign_45Assignvc/dense_2/bias/Adamsave_23/RestoreV2:45*"
_class
loc:@vc/dense_2/bias*
use_locking(*
T0*
validate_shape(*
_output_shapes
:
ť
save_23/Assign_46Assignvc/dense_2/bias/Adam_1save_23/RestoreV2:46*
T0*
use_locking(*
validate_shape(*
_output_shapes
:*"
_class
loc:@vc/dense_2/bias
˝
save_23/Assign_47Assignvc/dense_2/kernelsave_23/RestoreV2:47*
_output_shapes
:	*
use_locking(*$
_class
loc:@vc/dense_2/kernel*
T0*
validate_shape(
Â
save_23/Assign_48Assignvc/dense_2/kernel/Adamsave_23/RestoreV2:48*
_output_shapes
:	*
validate_shape(*$
_class
loc:@vc/dense_2/kernel*
use_locking(*
T0
Ä
save_23/Assign_49Assignvc/dense_2/kernel/Adam_1save_23/RestoreV2:49*
_output_shapes
:	*$
_class
loc:@vc/dense_2/kernel*
T0*
use_locking(*
validate_shape(
ą
save_23/Assign_50Assignvf/dense/biassave_23/RestoreV2:50*
_output_shapes	
:*
T0* 
_class
loc:@vf/dense/bias*
use_locking(*
validate_shape(
ś
save_23/Assign_51Assignvf/dense/bias/Adamsave_23/RestoreV2:51* 
_class
loc:@vf/dense/bias*
T0*
use_locking(*
validate_shape(*
_output_shapes	
:
¸
save_23/Assign_52Assignvf/dense/bias/Adam_1save_23/RestoreV2:52*
validate_shape(*
T0*
use_locking(* 
_class
loc:@vf/dense/bias*
_output_shapes	
:
š
save_23/Assign_53Assignvf/dense/kernelsave_23/RestoreV2:53*"
_class
loc:@vf/dense/kernel*
validate_shape(*
use_locking(*
_output_shapes
:	L*
T0
ž
save_23/Assign_54Assignvf/dense/kernel/Adamsave_23/RestoreV2:54*
T0*
validate_shape(*
use_locking(*"
_class
loc:@vf/dense/kernel*
_output_shapes
:	L
Ŕ
save_23/Assign_55Assignvf/dense/kernel/Adam_1save_23/RestoreV2:55*
use_locking(*"
_class
loc:@vf/dense/kernel*
validate_shape(*
_output_shapes
:	L*
T0
ľ
save_23/Assign_56Assignvf/dense_1/biassave_23/RestoreV2:56*
validate_shape(*
T0*"
_class
loc:@vf/dense_1/bias*
use_locking(*
_output_shapes	
:
ş
save_23/Assign_57Assignvf/dense_1/bias/Adamsave_23/RestoreV2:57*
_output_shapes	
:*
T0*"
_class
loc:@vf/dense_1/bias*
use_locking(*
validate_shape(
ź
save_23/Assign_58Assignvf/dense_1/bias/Adam_1save_23/RestoreV2:58*
_output_shapes	
:*"
_class
loc:@vf/dense_1/bias*
T0*
use_locking(*
validate_shape(
ž
save_23/Assign_59Assignvf/dense_1/kernelsave_23/RestoreV2:59* 
_output_shapes
:
*$
_class
loc:@vf/dense_1/kernel*
T0*
use_locking(*
validate_shape(
Ă
save_23/Assign_60Assignvf/dense_1/kernel/Adamsave_23/RestoreV2:60*
use_locking(*$
_class
loc:@vf/dense_1/kernel*
T0*
validate_shape(* 
_output_shapes
:

Ĺ
save_23/Assign_61Assignvf/dense_1/kernel/Adam_1save_23/RestoreV2:61*
use_locking(*$
_class
loc:@vf/dense_1/kernel*
validate_shape(*
T0* 
_output_shapes
:

´
save_23/Assign_62Assignvf/dense_2/biassave_23/RestoreV2:62*
T0*"
_class
loc:@vf/dense_2/bias*
use_locking(*
_output_shapes
:*
validate_shape(
š
save_23/Assign_63Assignvf/dense_2/bias/Adamsave_23/RestoreV2:63*"
_class
loc:@vf/dense_2/bias*
T0*
validate_shape(*
_output_shapes
:*
use_locking(
ť
save_23/Assign_64Assignvf/dense_2/bias/Adam_1save_23/RestoreV2:64*
_output_shapes
:*
use_locking(*
T0*
validate_shape(*"
_class
loc:@vf/dense_2/bias
˝
save_23/Assign_65Assignvf/dense_2/kernelsave_23/RestoreV2:65*
T0*
_output_shapes
:	*
use_locking(*
validate_shape(*$
_class
loc:@vf/dense_2/kernel
Â
save_23/Assign_66Assignvf/dense_2/kernel/Adamsave_23/RestoreV2:66*
_output_shapes
:	*
validate_shape(*
use_locking(*$
_class
loc:@vf/dense_2/kernel*
T0
Ä
save_23/Assign_67Assignvf/dense_2/kernel/Adam_1save_23/RestoreV2:67*
use_locking(*
_output_shapes
:	*
T0*$
_class
loc:@vf/dense_2/kernel*
validate_shape(
á

save_23/restore_shardNoOp^save_23/Assign^save_23/Assign_1^save_23/Assign_10^save_23/Assign_11^save_23/Assign_12^save_23/Assign_13^save_23/Assign_14^save_23/Assign_15^save_23/Assign_16^save_23/Assign_17^save_23/Assign_18^save_23/Assign_19^save_23/Assign_2^save_23/Assign_20^save_23/Assign_21^save_23/Assign_22^save_23/Assign_23^save_23/Assign_24^save_23/Assign_25^save_23/Assign_26^save_23/Assign_27^save_23/Assign_28^save_23/Assign_29^save_23/Assign_3^save_23/Assign_30^save_23/Assign_31^save_23/Assign_32^save_23/Assign_33^save_23/Assign_34^save_23/Assign_35^save_23/Assign_36^save_23/Assign_37^save_23/Assign_38^save_23/Assign_39^save_23/Assign_4^save_23/Assign_40^save_23/Assign_41^save_23/Assign_42^save_23/Assign_43^save_23/Assign_44^save_23/Assign_45^save_23/Assign_46^save_23/Assign_47^save_23/Assign_48^save_23/Assign_49^save_23/Assign_5^save_23/Assign_50^save_23/Assign_51^save_23/Assign_52^save_23/Assign_53^save_23/Assign_54^save_23/Assign_55^save_23/Assign_56^save_23/Assign_57^save_23/Assign_58^save_23/Assign_59^save_23/Assign_6^save_23/Assign_60^save_23/Assign_61^save_23/Assign_62^save_23/Assign_63^save_23/Assign_64^save_23/Assign_65^save_23/Assign_66^save_23/Assign_67^save_23/Assign_7^save_23/Assign_8^save_23/Assign_9
3
save_23/restore_allNoOp^save_23/restore_shard
\
save_24/filename/inputConst*
dtype0*
valueB Bmodel*
_output_shapes
: 
t
save_24/filenamePlaceholderWithDefaultsave_24/filename/input*
dtype0*
_output_shapes
: *
shape: 
k
save_24/ConstPlaceholderWithDefaultsave_24/filename*
dtype0*
_output_shapes
: *
shape: 

save_24/StringJoin/inputs_1Const*
dtype0*<
value3B1 B+_temp_54b6bd0daf6e4f0dbf1d9724b8d60fdf/part*
_output_shapes
: 
~
save_24/StringJoin
StringJoinsave_24/Constsave_24/StringJoin/inputs_1*
	separator *
N*
_output_shapes
: 
T
save_24/num_shardsConst*
dtype0*
value	B :*
_output_shapes
: 
_
save_24/ShardedFilename/shardConst*
dtype0*
_output_shapes
: *
value	B : 

save_24/ShardedFilenameShardedFilenamesave_24/StringJoinsave_24/ShardedFilename/shardsave_24/num_shards*
_output_shapes
: 
ř
save_24/SaveV2/tensor_namesConst*
dtype0*
_output_shapes
:D*¨
valueBDBbeta1_powerBbeta1_power_1Bbeta1_power_2Bbeta2_powerBbeta2_power_1Bbeta2_power_2Bpenalty/penalty_paramBpenalty/penalty_param/AdamBpenalty/penalty_param/Adam_1Bpenalty/penalty_param/Adam_2Bpenalty/penalty_param/Adam_3Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1B
pi/log_stdBpi/log_std/AdamBpi/log_std/Adam_1Bvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1
ń
save_24/SaveV2/shape_and_slicesConst*
valueBDB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:D
Ö
save_24/SaveV2SaveV2save_24/ShardedFilenamesave_24/SaveV2/tensor_namessave_24/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta1_power_2beta2_powerbeta2_power_1beta2_power_2penalty/penalty_parampenalty/penalty_param/Adampenalty/penalty_param/Adam_1penalty/penalty_param/Adam_2penalty/penalty_param/Adam_3pi/dense/biaspi/dense/bias/Adampi/dense/bias/Adam_1pi/dense/kernelpi/dense/kernel/Adampi/dense/kernel/Adam_1pi/dense_1/biaspi/dense_1/bias/Adampi/dense_1/bias/Adam_1pi/dense_1/kernelpi/dense_1/kernel/Adampi/dense_1/kernel/Adam_1pi/dense_2/biaspi/dense_2/bias/Adampi/dense_2/bias/Adam_1pi/dense_2/kernelpi/dense_2/kernel/Adampi/dense_2/kernel/Adam_1
pi/log_stdpi/log_std/Adampi/log_std/Adam_1vc/dense/biasvc/dense/bias/Adamvc/dense/bias/Adam_1vc/dense/kernelvc/dense/kernel/Adamvc/dense/kernel/Adam_1vc/dense_1/biasvc/dense_1/bias/Adamvc/dense_1/bias/Adam_1vc/dense_1/kernelvc/dense_1/kernel/Adamvc/dense_1/kernel/Adam_1vc/dense_2/biasvc/dense_2/bias/Adamvc/dense_2/bias/Adam_1vc/dense_2/kernelvc/dense_2/kernel/Adamvc/dense_2/kernel/Adam_1vf/dense/biasvf/dense/bias/Adamvf/dense/bias/Adam_1vf/dense/kernelvf/dense/kernel/Adamvf/dense/kernel/Adam_1vf/dense_1/biasvf/dense_1/bias/Adamvf/dense_1/bias/Adam_1vf/dense_1/kernelvf/dense_1/kernel/Adamvf/dense_1/kernel/Adam_1vf/dense_2/biasvf/dense_2/bias/Adamvf/dense_2/bias/Adam_1vf/dense_2/kernelvf/dense_2/kernel/Adamvf/dense_2/kernel/Adam_1*R
dtypesH
F2D

save_24/control_dependencyIdentitysave_24/ShardedFilename^save_24/SaveV2**
_class 
loc:@save_24/ShardedFilename*
T0*
_output_shapes
: 
Ś
.save_24/MergeV2Checkpoints/checkpoint_prefixesPacksave_24/ShardedFilename^save_24/control_dependency*
_output_shapes
:*
N*
T0*

axis 

save_24/MergeV2CheckpointsMergeV2Checkpoints.save_24/MergeV2Checkpoints/checkpoint_prefixessave_24/Const*
delete_old_dirs(

save_24/IdentityIdentitysave_24/Const^save_24/MergeV2Checkpoints^save_24/control_dependency*
_output_shapes
: *
T0
ű
save_24/RestoreV2/tensor_namesConst*
dtype0*
_output_shapes
:D*¨
valueBDBbeta1_powerBbeta1_power_1Bbeta1_power_2Bbeta2_powerBbeta2_power_1Bbeta2_power_2Bpenalty/penalty_paramBpenalty/penalty_param/AdamBpenalty/penalty_param/Adam_1Bpenalty/penalty_param/Adam_2Bpenalty/penalty_param/Adam_3Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1B
pi/log_stdBpi/log_std/AdamBpi/log_std/Adam_1Bvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1
ô
"save_24/RestoreV2/shape_and_slicesConst*
dtype0*
_output_shapes
:D*
valueBDB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
î
save_24/RestoreV2	RestoreV2save_24/Constsave_24/RestoreV2/tensor_names"save_24/RestoreV2/shape_and_slices*Ś
_output_shapes
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*R
dtypesH
F2D
Ź
save_24/AssignAssignbeta1_powersave_24/RestoreV2*
validate_shape(*
use_locking(*
_output_shapes
: *
T0*(
_class
loc:@penalty/penalty_param
˛
save_24/Assign_1Assignbeta1_power_1save_24/RestoreV2:1*
T0*(
_class
loc:@penalty/penalty_param*
validate_shape(*
use_locking(*
_output_shapes
: 
Ş
save_24/Assign_2Assignbeta1_power_2save_24/RestoreV2:2*
_output_shapes
: *
validate_shape(*
T0* 
_class
loc:@vc/dense/bias*
use_locking(
°
save_24/Assign_3Assignbeta2_powersave_24/RestoreV2:3*
T0*(
_class
loc:@penalty/penalty_param*
validate_shape(*
_output_shapes
: *
use_locking(
˛
save_24/Assign_4Assignbeta2_power_1save_24/RestoreV2:4*
validate_shape(*
use_locking(*
T0*(
_class
loc:@penalty/penalty_param*
_output_shapes
: 
Ş
save_24/Assign_5Assignbeta2_power_2save_24/RestoreV2:5*
_output_shapes
: *
validate_shape(*
use_locking(*
T0* 
_class
loc:@vc/dense/bias
ş
save_24/Assign_6Assignpenalty/penalty_paramsave_24/RestoreV2:6*
validate_shape(*
T0*
use_locking(*
_output_shapes
: *(
_class
loc:@penalty/penalty_param
ż
save_24/Assign_7Assignpenalty/penalty_param/Adamsave_24/RestoreV2:7*
_output_shapes
: *
validate_shape(*
T0*
use_locking(*(
_class
loc:@penalty/penalty_param
Á
save_24/Assign_8Assignpenalty/penalty_param/Adam_1save_24/RestoreV2:8*
use_locking(*
_output_shapes
: *
validate_shape(*(
_class
loc:@penalty/penalty_param*
T0
Á
save_24/Assign_9Assignpenalty/penalty_param/Adam_2save_24/RestoreV2:9*
_output_shapes
: *(
_class
loc:@penalty/penalty_param*
T0*
use_locking(*
validate_shape(
Ă
save_24/Assign_10Assignpenalty/penalty_param/Adam_3save_24/RestoreV2:10*
T0*(
_class
loc:@penalty/penalty_param*
validate_shape(*
_output_shapes
: *
use_locking(
ą
save_24/Assign_11Assignpi/dense/biassave_24/RestoreV2:11*
_output_shapes	
:*
use_locking(*
validate_shape(*
T0* 
_class
loc:@pi/dense/bias
ś
save_24/Assign_12Assignpi/dense/bias/Adamsave_24/RestoreV2:12*
_output_shapes	
:* 
_class
loc:@pi/dense/bias*
validate_shape(*
T0*
use_locking(
¸
save_24/Assign_13Assignpi/dense/bias/Adam_1save_24/RestoreV2:13*
T0*
validate_shape(*
_output_shapes	
:*
use_locking(* 
_class
loc:@pi/dense/bias
š
save_24/Assign_14Assignpi/dense/kernelsave_24/RestoreV2:14*
validate_shape(*"
_class
loc:@pi/dense/kernel*
use_locking(*
_output_shapes
:	L*
T0
ž
save_24/Assign_15Assignpi/dense/kernel/Adamsave_24/RestoreV2:15*
validate_shape(*
use_locking(*
_output_shapes
:	L*
T0*"
_class
loc:@pi/dense/kernel
Ŕ
save_24/Assign_16Assignpi/dense/kernel/Adam_1save_24/RestoreV2:16*"
_class
loc:@pi/dense/kernel*
_output_shapes
:	L*
use_locking(*
T0*
validate_shape(
ľ
save_24/Assign_17Assignpi/dense_1/biassave_24/RestoreV2:17*"
_class
loc:@pi/dense_1/bias*
use_locking(*
T0*
_output_shapes	
:*
validate_shape(
ş
save_24/Assign_18Assignpi/dense_1/bias/Adamsave_24/RestoreV2:18*
use_locking(*
_output_shapes	
:*
T0*
validate_shape(*"
_class
loc:@pi/dense_1/bias
ź
save_24/Assign_19Assignpi/dense_1/bias/Adam_1save_24/RestoreV2:19*
_output_shapes	
:*"
_class
loc:@pi/dense_1/bias*
validate_shape(*
T0*
use_locking(
ž
save_24/Assign_20Assignpi/dense_1/kernelsave_24/RestoreV2:20*
use_locking(*$
_class
loc:@pi/dense_1/kernel* 
_output_shapes
:
*
T0*
validate_shape(
Ă
save_24/Assign_21Assignpi/dense_1/kernel/Adamsave_24/RestoreV2:21*$
_class
loc:@pi/dense_1/kernel* 
_output_shapes
:
*
use_locking(*
T0*
validate_shape(
Ĺ
save_24/Assign_22Assignpi/dense_1/kernel/Adam_1save_24/RestoreV2:22*
T0*$
_class
loc:@pi/dense_1/kernel*
use_locking(*
validate_shape(* 
_output_shapes
:

´
save_24/Assign_23Assignpi/dense_2/biassave_24/RestoreV2:23*"
_class
loc:@pi/dense_2/bias*
validate_shape(*
_output_shapes
:*
T0*
use_locking(
š
save_24/Assign_24Assignpi/dense_2/bias/Adamsave_24/RestoreV2:24*
T0*"
_class
loc:@pi/dense_2/bias*
validate_shape(*
use_locking(*
_output_shapes
:
ť
save_24/Assign_25Assignpi/dense_2/bias/Adam_1save_24/RestoreV2:25*"
_class
loc:@pi/dense_2/bias*
T0*
validate_shape(*
_output_shapes
:*
use_locking(
˝
save_24/Assign_26Assignpi/dense_2/kernelsave_24/RestoreV2:26*
T0*
_output_shapes
:	*
use_locking(*$
_class
loc:@pi/dense_2/kernel*
validate_shape(
Â
save_24/Assign_27Assignpi/dense_2/kernel/Adamsave_24/RestoreV2:27*$
_class
loc:@pi/dense_2/kernel*
use_locking(*
T0*
_output_shapes
:	*
validate_shape(
Ä
save_24/Assign_28Assignpi/dense_2/kernel/Adam_1save_24/RestoreV2:28*
validate_shape(*
_output_shapes
:	*
T0*
use_locking(*$
_class
loc:@pi/dense_2/kernel
Ş
save_24/Assign_29Assign
pi/log_stdsave_24/RestoreV2:29*
use_locking(*
_class
loc:@pi/log_std*
_output_shapes
:*
validate_shape(*
T0
Ż
save_24/Assign_30Assignpi/log_std/Adamsave_24/RestoreV2:30*
_output_shapes
:*
_class
loc:@pi/log_std*
T0*
validate_shape(*
use_locking(
ą
save_24/Assign_31Assignpi/log_std/Adam_1save_24/RestoreV2:31*
use_locking(*
_class
loc:@pi/log_std*
validate_shape(*
T0*
_output_shapes
:
ą
save_24/Assign_32Assignvc/dense/biassave_24/RestoreV2:32*
T0*
validate_shape(*
_output_shapes	
:* 
_class
loc:@vc/dense/bias*
use_locking(
ś
save_24/Assign_33Assignvc/dense/bias/Adamsave_24/RestoreV2:33*
validate_shape(* 
_class
loc:@vc/dense/bias*
_output_shapes	
:*
T0*
use_locking(
¸
save_24/Assign_34Assignvc/dense/bias/Adam_1save_24/RestoreV2:34*
use_locking(*
T0*
validate_shape(* 
_class
loc:@vc/dense/bias*
_output_shapes	
:
š
save_24/Assign_35Assignvc/dense/kernelsave_24/RestoreV2:35*
use_locking(*
T0*
_output_shapes
:	L*"
_class
loc:@vc/dense/kernel*
validate_shape(
ž
save_24/Assign_36Assignvc/dense/kernel/Adamsave_24/RestoreV2:36*
T0*
validate_shape(*
use_locking(*
_output_shapes
:	L*"
_class
loc:@vc/dense/kernel
Ŕ
save_24/Assign_37Assignvc/dense/kernel/Adam_1save_24/RestoreV2:37*
_output_shapes
:	L*
T0*"
_class
loc:@vc/dense/kernel*
use_locking(*
validate_shape(
ľ
save_24/Assign_38Assignvc/dense_1/biassave_24/RestoreV2:38*
T0*
use_locking(*
validate_shape(*"
_class
loc:@vc/dense_1/bias*
_output_shapes	
:
ş
save_24/Assign_39Assignvc/dense_1/bias/Adamsave_24/RestoreV2:39*
use_locking(*
_output_shapes	
:*
validate_shape(*
T0*"
_class
loc:@vc/dense_1/bias
ź
save_24/Assign_40Assignvc/dense_1/bias/Adam_1save_24/RestoreV2:40*
_output_shapes	
:*
validate_shape(*
use_locking(*"
_class
loc:@vc/dense_1/bias*
T0
ž
save_24/Assign_41Assignvc/dense_1/kernelsave_24/RestoreV2:41*$
_class
loc:@vc/dense_1/kernel*
validate_shape(*
T0* 
_output_shapes
:
*
use_locking(
Ă
save_24/Assign_42Assignvc/dense_1/kernel/Adamsave_24/RestoreV2:42*
validate_shape(*
T0*$
_class
loc:@vc/dense_1/kernel*
use_locking(* 
_output_shapes
:

Ĺ
save_24/Assign_43Assignvc/dense_1/kernel/Adam_1save_24/RestoreV2:43* 
_output_shapes
:
*
use_locking(*
validate_shape(*$
_class
loc:@vc/dense_1/kernel*
T0
´
save_24/Assign_44Assignvc/dense_2/biassave_24/RestoreV2:44*
T0*"
_class
loc:@vc/dense_2/bias*
use_locking(*
validate_shape(*
_output_shapes
:
š
save_24/Assign_45Assignvc/dense_2/bias/Adamsave_24/RestoreV2:45*
validate_shape(*
_output_shapes
:*"
_class
loc:@vc/dense_2/bias*
use_locking(*
T0
ť
save_24/Assign_46Assignvc/dense_2/bias/Adam_1save_24/RestoreV2:46*
_output_shapes
:*
validate_shape(*
use_locking(*"
_class
loc:@vc/dense_2/bias*
T0
˝
save_24/Assign_47Assignvc/dense_2/kernelsave_24/RestoreV2:47*
T0*$
_class
loc:@vc/dense_2/kernel*
validate_shape(*
use_locking(*
_output_shapes
:	
Â
save_24/Assign_48Assignvc/dense_2/kernel/Adamsave_24/RestoreV2:48*$
_class
loc:@vc/dense_2/kernel*
T0*
validate_shape(*
_output_shapes
:	*
use_locking(
Ä
save_24/Assign_49Assignvc/dense_2/kernel/Adam_1save_24/RestoreV2:49*
use_locking(*
_output_shapes
:	*
validate_shape(*$
_class
loc:@vc/dense_2/kernel*
T0
ą
save_24/Assign_50Assignvf/dense/biassave_24/RestoreV2:50*
validate_shape(* 
_class
loc:@vf/dense/bias*
use_locking(*
T0*
_output_shapes	
:
ś
save_24/Assign_51Assignvf/dense/bias/Adamsave_24/RestoreV2:51* 
_class
loc:@vf/dense/bias*
use_locking(*
_output_shapes	
:*
T0*
validate_shape(
¸
save_24/Assign_52Assignvf/dense/bias/Adam_1save_24/RestoreV2:52*
use_locking(*
validate_shape(*
_output_shapes	
:*
T0* 
_class
loc:@vf/dense/bias
š
save_24/Assign_53Assignvf/dense/kernelsave_24/RestoreV2:53*
validate_shape(*
_output_shapes
:	L*
T0*
use_locking(*"
_class
loc:@vf/dense/kernel
ž
save_24/Assign_54Assignvf/dense/kernel/Adamsave_24/RestoreV2:54*
_output_shapes
:	L*
validate_shape(*
use_locking(*
T0*"
_class
loc:@vf/dense/kernel
Ŕ
save_24/Assign_55Assignvf/dense/kernel/Adam_1save_24/RestoreV2:55*
validate_shape(*
_output_shapes
:	L*
T0*
use_locking(*"
_class
loc:@vf/dense/kernel
ľ
save_24/Assign_56Assignvf/dense_1/biassave_24/RestoreV2:56*
T0*
validate_shape(*
use_locking(*
_output_shapes	
:*"
_class
loc:@vf/dense_1/bias
ş
save_24/Assign_57Assignvf/dense_1/bias/Adamsave_24/RestoreV2:57*
validate_shape(*
use_locking(*"
_class
loc:@vf/dense_1/bias*
_output_shapes	
:*
T0
ź
save_24/Assign_58Assignvf/dense_1/bias/Adam_1save_24/RestoreV2:58*
use_locking(*
validate_shape(*
T0*"
_class
loc:@vf/dense_1/bias*
_output_shapes	
:
ž
save_24/Assign_59Assignvf/dense_1/kernelsave_24/RestoreV2:59*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0*$
_class
loc:@vf/dense_1/kernel
Ă
save_24/Assign_60Assignvf/dense_1/kernel/Adamsave_24/RestoreV2:60*
validate_shape(*$
_class
loc:@vf/dense_1/kernel*
use_locking(*
T0* 
_output_shapes
:

Ĺ
save_24/Assign_61Assignvf/dense_1/kernel/Adam_1save_24/RestoreV2:61* 
_output_shapes
:
*
validate_shape(*
use_locking(*
T0*$
_class
loc:@vf/dense_1/kernel
´
save_24/Assign_62Assignvf/dense_2/biassave_24/RestoreV2:62*
use_locking(*
validate_shape(*
T0*"
_class
loc:@vf/dense_2/bias*
_output_shapes
:
š
save_24/Assign_63Assignvf/dense_2/bias/Adamsave_24/RestoreV2:63*"
_class
loc:@vf/dense_2/bias*
use_locking(*
T0*
validate_shape(*
_output_shapes
:
ť
save_24/Assign_64Assignvf/dense_2/bias/Adam_1save_24/RestoreV2:64*"
_class
loc:@vf/dense_2/bias*
_output_shapes
:*
validate_shape(*
T0*
use_locking(
˝
save_24/Assign_65Assignvf/dense_2/kernelsave_24/RestoreV2:65*$
_class
loc:@vf/dense_2/kernel*
_output_shapes
:	*
T0*
validate_shape(*
use_locking(
Â
save_24/Assign_66Assignvf/dense_2/kernel/Adamsave_24/RestoreV2:66*
T0*
validate_shape(*
use_locking(*
_output_shapes
:	*$
_class
loc:@vf/dense_2/kernel
Ä
save_24/Assign_67Assignvf/dense_2/kernel/Adam_1save_24/RestoreV2:67*
_output_shapes
:	*
T0*$
_class
loc:@vf/dense_2/kernel*
validate_shape(*
use_locking(
á

save_24/restore_shardNoOp^save_24/Assign^save_24/Assign_1^save_24/Assign_10^save_24/Assign_11^save_24/Assign_12^save_24/Assign_13^save_24/Assign_14^save_24/Assign_15^save_24/Assign_16^save_24/Assign_17^save_24/Assign_18^save_24/Assign_19^save_24/Assign_2^save_24/Assign_20^save_24/Assign_21^save_24/Assign_22^save_24/Assign_23^save_24/Assign_24^save_24/Assign_25^save_24/Assign_26^save_24/Assign_27^save_24/Assign_28^save_24/Assign_29^save_24/Assign_3^save_24/Assign_30^save_24/Assign_31^save_24/Assign_32^save_24/Assign_33^save_24/Assign_34^save_24/Assign_35^save_24/Assign_36^save_24/Assign_37^save_24/Assign_38^save_24/Assign_39^save_24/Assign_4^save_24/Assign_40^save_24/Assign_41^save_24/Assign_42^save_24/Assign_43^save_24/Assign_44^save_24/Assign_45^save_24/Assign_46^save_24/Assign_47^save_24/Assign_48^save_24/Assign_49^save_24/Assign_5^save_24/Assign_50^save_24/Assign_51^save_24/Assign_52^save_24/Assign_53^save_24/Assign_54^save_24/Assign_55^save_24/Assign_56^save_24/Assign_57^save_24/Assign_58^save_24/Assign_59^save_24/Assign_6^save_24/Assign_60^save_24/Assign_61^save_24/Assign_62^save_24/Assign_63^save_24/Assign_64^save_24/Assign_65^save_24/Assign_66^save_24/Assign_67^save_24/Assign_7^save_24/Assign_8^save_24/Assign_9
3
save_24/restore_allNoOp^save_24/restore_shard
\
save_25/filename/inputConst*
dtype0*
valueB Bmodel*
_output_shapes
: 
t
save_25/filenamePlaceholderWithDefaultsave_25/filename/input*
_output_shapes
: *
dtype0*
shape: 
k
save_25/ConstPlaceholderWithDefaultsave_25/filename*
dtype0*
shape: *
_output_shapes
: 

save_25/StringJoin/inputs_1Const*
_output_shapes
: *<
value3B1 B+_temp_244d9dd81838474d81f8dc98aaa3ce76/part*
dtype0
~
save_25/StringJoin
StringJoinsave_25/Constsave_25/StringJoin/inputs_1*
	separator *
N*
_output_shapes
: 
T
save_25/num_shardsConst*
_output_shapes
: *
value	B :*
dtype0
_
save_25/ShardedFilename/shardConst*
value	B : *
_output_shapes
: *
dtype0

save_25/ShardedFilenameShardedFilenamesave_25/StringJoinsave_25/ShardedFilename/shardsave_25/num_shards*
_output_shapes
: 
ř
save_25/SaveV2/tensor_namesConst*
_output_shapes
:D*¨
valueBDBbeta1_powerBbeta1_power_1Bbeta1_power_2Bbeta2_powerBbeta2_power_1Bbeta2_power_2Bpenalty/penalty_paramBpenalty/penalty_param/AdamBpenalty/penalty_param/Adam_1Bpenalty/penalty_param/Adam_2Bpenalty/penalty_param/Adam_3Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1B
pi/log_stdBpi/log_std/AdamBpi/log_std/Adam_1Bvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1*
dtype0
ń
save_25/SaveV2/shape_and_slicesConst*
valueBDB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:D
Ö
save_25/SaveV2SaveV2save_25/ShardedFilenamesave_25/SaveV2/tensor_namessave_25/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta1_power_2beta2_powerbeta2_power_1beta2_power_2penalty/penalty_parampenalty/penalty_param/Adampenalty/penalty_param/Adam_1penalty/penalty_param/Adam_2penalty/penalty_param/Adam_3pi/dense/biaspi/dense/bias/Adampi/dense/bias/Adam_1pi/dense/kernelpi/dense/kernel/Adampi/dense/kernel/Adam_1pi/dense_1/biaspi/dense_1/bias/Adampi/dense_1/bias/Adam_1pi/dense_1/kernelpi/dense_1/kernel/Adampi/dense_1/kernel/Adam_1pi/dense_2/biaspi/dense_2/bias/Adampi/dense_2/bias/Adam_1pi/dense_2/kernelpi/dense_2/kernel/Adampi/dense_2/kernel/Adam_1
pi/log_stdpi/log_std/Adampi/log_std/Adam_1vc/dense/biasvc/dense/bias/Adamvc/dense/bias/Adam_1vc/dense/kernelvc/dense/kernel/Adamvc/dense/kernel/Adam_1vc/dense_1/biasvc/dense_1/bias/Adamvc/dense_1/bias/Adam_1vc/dense_1/kernelvc/dense_1/kernel/Adamvc/dense_1/kernel/Adam_1vc/dense_2/biasvc/dense_2/bias/Adamvc/dense_2/bias/Adam_1vc/dense_2/kernelvc/dense_2/kernel/Adamvc/dense_2/kernel/Adam_1vf/dense/biasvf/dense/bias/Adamvf/dense/bias/Adam_1vf/dense/kernelvf/dense/kernel/Adamvf/dense/kernel/Adam_1vf/dense_1/biasvf/dense_1/bias/Adamvf/dense_1/bias/Adam_1vf/dense_1/kernelvf/dense_1/kernel/Adamvf/dense_1/kernel/Adam_1vf/dense_2/biasvf/dense_2/bias/Adamvf/dense_2/bias/Adam_1vf/dense_2/kernelvf/dense_2/kernel/Adamvf/dense_2/kernel/Adam_1*R
dtypesH
F2D

save_25/control_dependencyIdentitysave_25/ShardedFilename^save_25/SaveV2*
_output_shapes
: **
_class 
loc:@save_25/ShardedFilename*
T0
Ś
.save_25/MergeV2Checkpoints/checkpoint_prefixesPacksave_25/ShardedFilename^save_25/control_dependency*
_output_shapes
:*
T0*

axis *
N

save_25/MergeV2CheckpointsMergeV2Checkpoints.save_25/MergeV2Checkpoints/checkpoint_prefixessave_25/Const*
delete_old_dirs(

save_25/IdentityIdentitysave_25/Const^save_25/MergeV2Checkpoints^save_25/control_dependency*
_output_shapes
: *
T0
ű
save_25/RestoreV2/tensor_namesConst*¨
valueBDBbeta1_powerBbeta1_power_1Bbeta1_power_2Bbeta2_powerBbeta2_power_1Bbeta2_power_2Bpenalty/penalty_paramBpenalty/penalty_param/AdamBpenalty/penalty_param/Adam_1Bpenalty/penalty_param/Adam_2Bpenalty/penalty_param/Adam_3Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1B
pi/log_stdBpi/log_std/AdamBpi/log_std/Adam_1Bvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1*
_output_shapes
:D*
dtype0
ô
"save_25/RestoreV2/shape_and_slicesConst*
valueBDB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:D
î
save_25/RestoreV2	RestoreV2save_25/Constsave_25/RestoreV2/tensor_names"save_25/RestoreV2/shape_and_slices*Ś
_output_shapes
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*R
dtypesH
F2D
Ź
save_25/AssignAssignbeta1_powersave_25/RestoreV2*
validate_shape(*
T0*
_output_shapes
: *
use_locking(*(
_class
loc:@penalty/penalty_param
˛
save_25/Assign_1Assignbeta1_power_1save_25/RestoreV2:1*
T0*
validate_shape(*
_output_shapes
: *(
_class
loc:@penalty/penalty_param*
use_locking(
Ş
save_25/Assign_2Assignbeta1_power_2save_25/RestoreV2:2*
T0*
validate_shape(* 
_class
loc:@vc/dense/bias*
use_locking(*
_output_shapes
: 
°
save_25/Assign_3Assignbeta2_powersave_25/RestoreV2:3*
_output_shapes
: *
use_locking(*(
_class
loc:@penalty/penalty_param*
T0*
validate_shape(
˛
save_25/Assign_4Assignbeta2_power_1save_25/RestoreV2:4*
use_locking(*
validate_shape(*(
_class
loc:@penalty/penalty_param*
_output_shapes
: *
T0
Ş
save_25/Assign_5Assignbeta2_power_2save_25/RestoreV2:5*
validate_shape(*
_output_shapes
: *
T0* 
_class
loc:@vc/dense/bias*
use_locking(
ş
save_25/Assign_6Assignpenalty/penalty_paramsave_25/RestoreV2:6*
T0*
validate_shape(*
use_locking(*(
_class
loc:@penalty/penalty_param*
_output_shapes
: 
ż
save_25/Assign_7Assignpenalty/penalty_param/Adamsave_25/RestoreV2:7*
_output_shapes
: *
use_locking(*
T0*
validate_shape(*(
_class
loc:@penalty/penalty_param
Á
save_25/Assign_8Assignpenalty/penalty_param/Adam_1save_25/RestoreV2:8*(
_class
loc:@penalty/penalty_param*
T0*
validate_shape(*
use_locking(*
_output_shapes
: 
Á
save_25/Assign_9Assignpenalty/penalty_param/Adam_2save_25/RestoreV2:9*
_output_shapes
: *
T0*
validate_shape(*
use_locking(*(
_class
loc:@penalty/penalty_param
Ă
save_25/Assign_10Assignpenalty/penalty_param/Adam_3save_25/RestoreV2:10*(
_class
loc:@penalty/penalty_param*
validate_shape(*
use_locking(*
T0*
_output_shapes
: 
ą
save_25/Assign_11Assignpi/dense/biassave_25/RestoreV2:11*
use_locking(*
_output_shapes	
:*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(
ś
save_25/Assign_12Assignpi/dense/bias/Adamsave_25/RestoreV2:12*
_output_shapes	
:* 
_class
loc:@pi/dense/bias*
use_locking(*
T0*
validate_shape(
¸
save_25/Assign_13Assignpi/dense/bias/Adam_1save_25/RestoreV2:13*
use_locking(*
T0*
validate_shape(* 
_class
loc:@pi/dense/bias*
_output_shapes	
:
š
save_25/Assign_14Assignpi/dense/kernelsave_25/RestoreV2:14*
_output_shapes
:	L*
use_locking(*"
_class
loc:@pi/dense/kernel*
T0*
validate_shape(
ž
save_25/Assign_15Assignpi/dense/kernel/Adamsave_25/RestoreV2:15*
validate_shape(*"
_class
loc:@pi/dense/kernel*
use_locking(*
_output_shapes
:	L*
T0
Ŕ
save_25/Assign_16Assignpi/dense/kernel/Adam_1save_25/RestoreV2:16*
_output_shapes
:	L*
validate_shape(*
use_locking(*"
_class
loc:@pi/dense/kernel*
T0
ľ
save_25/Assign_17Assignpi/dense_1/biassave_25/RestoreV2:17*"
_class
loc:@pi/dense_1/bias*
T0*
use_locking(*
validate_shape(*
_output_shapes	
:
ş
save_25/Assign_18Assignpi/dense_1/bias/Adamsave_25/RestoreV2:18*
validate_shape(*
use_locking(*
_output_shapes	
:*"
_class
loc:@pi/dense_1/bias*
T0
ź
save_25/Assign_19Assignpi/dense_1/bias/Adam_1save_25/RestoreV2:19*"
_class
loc:@pi/dense_1/bias*
_output_shapes	
:*
T0*
validate_shape(*
use_locking(
ž
save_25/Assign_20Assignpi/dense_1/kernelsave_25/RestoreV2:20*
use_locking(* 
_output_shapes
:
*
T0*$
_class
loc:@pi/dense_1/kernel*
validate_shape(
Ă
save_25/Assign_21Assignpi/dense_1/kernel/Adamsave_25/RestoreV2:21*
T0*
use_locking(*$
_class
loc:@pi/dense_1/kernel* 
_output_shapes
:
*
validate_shape(
Ĺ
save_25/Assign_22Assignpi/dense_1/kernel/Adam_1save_25/RestoreV2:22* 
_output_shapes
:
*
use_locking(*
validate_shape(*
T0*$
_class
loc:@pi/dense_1/kernel
´
save_25/Assign_23Assignpi/dense_2/biassave_25/RestoreV2:23*"
_class
loc:@pi/dense_2/bias*
_output_shapes
:*
use_locking(*
T0*
validate_shape(
š
save_25/Assign_24Assignpi/dense_2/bias/Adamsave_25/RestoreV2:24*
use_locking(*"
_class
loc:@pi/dense_2/bias*
validate_shape(*
_output_shapes
:*
T0
ť
save_25/Assign_25Assignpi/dense_2/bias/Adam_1save_25/RestoreV2:25*
_output_shapes
:*
validate_shape(*
use_locking(*
T0*"
_class
loc:@pi/dense_2/bias
˝
save_25/Assign_26Assignpi/dense_2/kernelsave_25/RestoreV2:26*
_output_shapes
:	*
validate_shape(*$
_class
loc:@pi/dense_2/kernel*
use_locking(*
T0
Â
save_25/Assign_27Assignpi/dense_2/kernel/Adamsave_25/RestoreV2:27*$
_class
loc:@pi/dense_2/kernel*
_output_shapes
:	*
validate_shape(*
use_locking(*
T0
Ä
save_25/Assign_28Assignpi/dense_2/kernel/Adam_1save_25/RestoreV2:28*$
_class
loc:@pi/dense_2/kernel*
_output_shapes
:	*
T0*
use_locking(*
validate_shape(
Ş
save_25/Assign_29Assign
pi/log_stdsave_25/RestoreV2:29*
validate_shape(*
_output_shapes
:*
_class
loc:@pi/log_std*
T0*
use_locking(
Ż
save_25/Assign_30Assignpi/log_std/Adamsave_25/RestoreV2:30*
T0*
use_locking(*
validate_shape(*
_class
loc:@pi/log_std*
_output_shapes
:
ą
save_25/Assign_31Assignpi/log_std/Adam_1save_25/RestoreV2:31*
_class
loc:@pi/log_std*
_output_shapes
:*
validate_shape(*
T0*
use_locking(
ą
save_25/Assign_32Assignvc/dense/biassave_25/RestoreV2:32*
_output_shapes	
:*
validate_shape(*
T0* 
_class
loc:@vc/dense/bias*
use_locking(
ś
save_25/Assign_33Assignvc/dense/bias/Adamsave_25/RestoreV2:33*
T0* 
_class
loc:@vc/dense/bias*
_output_shapes	
:*
validate_shape(*
use_locking(
¸
save_25/Assign_34Assignvc/dense/bias/Adam_1save_25/RestoreV2:34*
T0*
use_locking(*
_output_shapes	
:* 
_class
loc:@vc/dense/bias*
validate_shape(
š
save_25/Assign_35Assignvc/dense/kernelsave_25/RestoreV2:35*
T0*"
_class
loc:@vc/dense/kernel*
use_locking(*
validate_shape(*
_output_shapes
:	L
ž
save_25/Assign_36Assignvc/dense/kernel/Adamsave_25/RestoreV2:36*"
_class
loc:@vc/dense/kernel*
use_locking(*
_output_shapes
:	L*
validate_shape(*
T0
Ŕ
save_25/Assign_37Assignvc/dense/kernel/Adam_1save_25/RestoreV2:37*"
_class
loc:@vc/dense/kernel*
use_locking(*
T0*
validate_shape(*
_output_shapes
:	L
ľ
save_25/Assign_38Assignvc/dense_1/biassave_25/RestoreV2:38*
use_locking(*
validate_shape(*"
_class
loc:@vc/dense_1/bias*
_output_shapes	
:*
T0
ş
save_25/Assign_39Assignvc/dense_1/bias/Adamsave_25/RestoreV2:39*"
_class
loc:@vc/dense_1/bias*
T0*
validate_shape(*
_output_shapes	
:*
use_locking(
ź
save_25/Assign_40Assignvc/dense_1/bias/Adam_1save_25/RestoreV2:40*"
_class
loc:@vc/dense_1/bias*
T0*
_output_shapes	
:*
validate_shape(*
use_locking(
ž
save_25/Assign_41Assignvc/dense_1/kernelsave_25/RestoreV2:41*$
_class
loc:@vc/dense_1/kernel*
T0*
validate_shape(*
use_locking(* 
_output_shapes
:

Ă
save_25/Assign_42Assignvc/dense_1/kernel/Adamsave_25/RestoreV2:42* 
_output_shapes
:
*
validate_shape(*$
_class
loc:@vc/dense_1/kernel*
T0*
use_locking(
Ĺ
save_25/Assign_43Assignvc/dense_1/kernel/Adam_1save_25/RestoreV2:43*
use_locking(*
validate_shape(*$
_class
loc:@vc/dense_1/kernel* 
_output_shapes
:
*
T0
´
save_25/Assign_44Assignvc/dense_2/biassave_25/RestoreV2:44*
validate_shape(*
_output_shapes
:*
T0*"
_class
loc:@vc/dense_2/bias*
use_locking(
š
save_25/Assign_45Assignvc/dense_2/bias/Adamsave_25/RestoreV2:45*
validate_shape(*"
_class
loc:@vc/dense_2/bias*
T0*
use_locking(*
_output_shapes
:
ť
save_25/Assign_46Assignvc/dense_2/bias/Adam_1save_25/RestoreV2:46*
_output_shapes
:*
validate_shape(*
use_locking(*"
_class
loc:@vc/dense_2/bias*
T0
˝
save_25/Assign_47Assignvc/dense_2/kernelsave_25/RestoreV2:47*
_output_shapes
:	*
T0*
use_locking(*$
_class
loc:@vc/dense_2/kernel*
validate_shape(
Â
save_25/Assign_48Assignvc/dense_2/kernel/Adamsave_25/RestoreV2:48*$
_class
loc:@vc/dense_2/kernel*
T0*
validate_shape(*
use_locking(*
_output_shapes
:	
Ä
save_25/Assign_49Assignvc/dense_2/kernel/Adam_1save_25/RestoreV2:49*
_output_shapes
:	*
use_locking(*
validate_shape(*
T0*$
_class
loc:@vc/dense_2/kernel
ą
save_25/Assign_50Assignvf/dense/biassave_25/RestoreV2:50*
validate_shape(*
T0* 
_class
loc:@vf/dense/bias*
_output_shapes	
:*
use_locking(
ś
save_25/Assign_51Assignvf/dense/bias/Adamsave_25/RestoreV2:51* 
_class
loc:@vf/dense/bias*
T0*
use_locking(*
validate_shape(*
_output_shapes	
:
¸
save_25/Assign_52Assignvf/dense/bias/Adam_1save_25/RestoreV2:52*
validate_shape(*
T0*
use_locking(* 
_class
loc:@vf/dense/bias*
_output_shapes	
:
š
save_25/Assign_53Assignvf/dense/kernelsave_25/RestoreV2:53*
use_locking(*
validate_shape(*
T0*"
_class
loc:@vf/dense/kernel*
_output_shapes
:	L
ž
save_25/Assign_54Assignvf/dense/kernel/Adamsave_25/RestoreV2:54*
_output_shapes
:	L*
validate_shape(*
use_locking(*
T0*"
_class
loc:@vf/dense/kernel
Ŕ
save_25/Assign_55Assignvf/dense/kernel/Adam_1save_25/RestoreV2:55*
_output_shapes
:	L*
validate_shape(*"
_class
loc:@vf/dense/kernel*
T0*
use_locking(
ľ
save_25/Assign_56Assignvf/dense_1/biassave_25/RestoreV2:56*"
_class
loc:@vf/dense_1/bias*
use_locking(*
T0*
_output_shapes	
:*
validate_shape(
ş
save_25/Assign_57Assignvf/dense_1/bias/Adamsave_25/RestoreV2:57*
T0*"
_class
loc:@vf/dense_1/bias*
_output_shapes	
:*
use_locking(*
validate_shape(
ź
save_25/Assign_58Assignvf/dense_1/bias/Adam_1save_25/RestoreV2:58*"
_class
loc:@vf/dense_1/bias*
validate_shape(*
T0*
_output_shapes	
:*
use_locking(
ž
save_25/Assign_59Assignvf/dense_1/kernelsave_25/RestoreV2:59* 
_output_shapes
:
*
validate_shape(*$
_class
loc:@vf/dense_1/kernel*
use_locking(*
T0
Ă
save_25/Assign_60Assignvf/dense_1/kernel/Adamsave_25/RestoreV2:60* 
_output_shapes
:
*$
_class
loc:@vf/dense_1/kernel*
validate_shape(*
use_locking(*
T0
Ĺ
save_25/Assign_61Assignvf/dense_1/kernel/Adam_1save_25/RestoreV2:61* 
_output_shapes
:
*
T0*
use_locking(*$
_class
loc:@vf/dense_1/kernel*
validate_shape(
´
save_25/Assign_62Assignvf/dense_2/biassave_25/RestoreV2:62*
_output_shapes
:*
validate_shape(*
use_locking(*
T0*"
_class
loc:@vf/dense_2/bias
š
save_25/Assign_63Assignvf/dense_2/bias/Adamsave_25/RestoreV2:63*
use_locking(*
_output_shapes
:*
validate_shape(*
T0*"
_class
loc:@vf/dense_2/bias
ť
save_25/Assign_64Assignvf/dense_2/bias/Adam_1save_25/RestoreV2:64*
validate_shape(*
use_locking(*
T0*
_output_shapes
:*"
_class
loc:@vf/dense_2/bias
˝
save_25/Assign_65Assignvf/dense_2/kernelsave_25/RestoreV2:65*$
_class
loc:@vf/dense_2/kernel*
use_locking(*
T0*
_output_shapes
:	*
validate_shape(
Â
save_25/Assign_66Assignvf/dense_2/kernel/Adamsave_25/RestoreV2:66*$
_class
loc:@vf/dense_2/kernel*
validate_shape(*
T0*
use_locking(*
_output_shapes
:	
Ä
save_25/Assign_67Assignvf/dense_2/kernel/Adam_1save_25/RestoreV2:67*
_output_shapes
:	*
T0*
validate_shape(*
use_locking(*$
_class
loc:@vf/dense_2/kernel
á

save_25/restore_shardNoOp^save_25/Assign^save_25/Assign_1^save_25/Assign_10^save_25/Assign_11^save_25/Assign_12^save_25/Assign_13^save_25/Assign_14^save_25/Assign_15^save_25/Assign_16^save_25/Assign_17^save_25/Assign_18^save_25/Assign_19^save_25/Assign_2^save_25/Assign_20^save_25/Assign_21^save_25/Assign_22^save_25/Assign_23^save_25/Assign_24^save_25/Assign_25^save_25/Assign_26^save_25/Assign_27^save_25/Assign_28^save_25/Assign_29^save_25/Assign_3^save_25/Assign_30^save_25/Assign_31^save_25/Assign_32^save_25/Assign_33^save_25/Assign_34^save_25/Assign_35^save_25/Assign_36^save_25/Assign_37^save_25/Assign_38^save_25/Assign_39^save_25/Assign_4^save_25/Assign_40^save_25/Assign_41^save_25/Assign_42^save_25/Assign_43^save_25/Assign_44^save_25/Assign_45^save_25/Assign_46^save_25/Assign_47^save_25/Assign_48^save_25/Assign_49^save_25/Assign_5^save_25/Assign_50^save_25/Assign_51^save_25/Assign_52^save_25/Assign_53^save_25/Assign_54^save_25/Assign_55^save_25/Assign_56^save_25/Assign_57^save_25/Assign_58^save_25/Assign_59^save_25/Assign_6^save_25/Assign_60^save_25/Assign_61^save_25/Assign_62^save_25/Assign_63^save_25/Assign_64^save_25/Assign_65^save_25/Assign_66^save_25/Assign_67^save_25/Assign_7^save_25/Assign_8^save_25/Assign_9
3
save_25/restore_allNoOp^save_25/restore_shard
\
save_26/filename/inputConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
t
save_26/filenamePlaceholderWithDefaultsave_26/filename/input*
dtype0*
_output_shapes
: *
shape: 
k
save_26/ConstPlaceholderWithDefaultsave_26/filename*
dtype0*
shape: *
_output_shapes
: 

save_26/StringJoin/inputs_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+_temp_c35f19a4fdba4092b35f71cc7acc4217/part
~
save_26/StringJoin
StringJoinsave_26/Constsave_26/StringJoin/inputs_1*
_output_shapes
: *
	separator *
N
T
save_26/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
_
save_26/ShardedFilename/shardConst*
_output_shapes
: *
value	B : *
dtype0

save_26/ShardedFilenameShardedFilenamesave_26/StringJoinsave_26/ShardedFilename/shardsave_26/num_shards*
_output_shapes
: 
ř
save_26/SaveV2/tensor_namesConst*¨
valueBDBbeta1_powerBbeta1_power_1Bbeta1_power_2Bbeta2_powerBbeta2_power_1Bbeta2_power_2Bpenalty/penalty_paramBpenalty/penalty_param/AdamBpenalty/penalty_param/Adam_1Bpenalty/penalty_param/Adam_2Bpenalty/penalty_param/Adam_3Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1B
pi/log_stdBpi/log_std/AdamBpi/log_std/Adam_1Bvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1*
dtype0*
_output_shapes
:D
ń
save_26/SaveV2/shape_and_slicesConst*
valueBDB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:D
Ö
save_26/SaveV2SaveV2save_26/ShardedFilenamesave_26/SaveV2/tensor_namessave_26/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta1_power_2beta2_powerbeta2_power_1beta2_power_2penalty/penalty_parampenalty/penalty_param/Adampenalty/penalty_param/Adam_1penalty/penalty_param/Adam_2penalty/penalty_param/Adam_3pi/dense/biaspi/dense/bias/Adampi/dense/bias/Adam_1pi/dense/kernelpi/dense/kernel/Adampi/dense/kernel/Adam_1pi/dense_1/biaspi/dense_1/bias/Adampi/dense_1/bias/Adam_1pi/dense_1/kernelpi/dense_1/kernel/Adampi/dense_1/kernel/Adam_1pi/dense_2/biaspi/dense_2/bias/Adampi/dense_2/bias/Adam_1pi/dense_2/kernelpi/dense_2/kernel/Adampi/dense_2/kernel/Adam_1
pi/log_stdpi/log_std/Adampi/log_std/Adam_1vc/dense/biasvc/dense/bias/Adamvc/dense/bias/Adam_1vc/dense/kernelvc/dense/kernel/Adamvc/dense/kernel/Adam_1vc/dense_1/biasvc/dense_1/bias/Adamvc/dense_1/bias/Adam_1vc/dense_1/kernelvc/dense_1/kernel/Adamvc/dense_1/kernel/Adam_1vc/dense_2/biasvc/dense_2/bias/Adamvc/dense_2/bias/Adam_1vc/dense_2/kernelvc/dense_2/kernel/Adamvc/dense_2/kernel/Adam_1vf/dense/biasvf/dense/bias/Adamvf/dense/bias/Adam_1vf/dense/kernelvf/dense/kernel/Adamvf/dense/kernel/Adam_1vf/dense_1/biasvf/dense_1/bias/Adamvf/dense_1/bias/Adam_1vf/dense_1/kernelvf/dense_1/kernel/Adamvf/dense_1/kernel/Adam_1vf/dense_2/biasvf/dense_2/bias/Adamvf/dense_2/bias/Adam_1vf/dense_2/kernelvf/dense_2/kernel/Adamvf/dense_2/kernel/Adam_1*R
dtypesH
F2D

save_26/control_dependencyIdentitysave_26/ShardedFilename^save_26/SaveV2**
_class 
loc:@save_26/ShardedFilename*
T0*
_output_shapes
: 
Ś
.save_26/MergeV2Checkpoints/checkpoint_prefixesPacksave_26/ShardedFilename^save_26/control_dependency*
N*

axis *
T0*
_output_shapes
:

save_26/MergeV2CheckpointsMergeV2Checkpoints.save_26/MergeV2Checkpoints/checkpoint_prefixessave_26/Const*
delete_old_dirs(

save_26/IdentityIdentitysave_26/Const^save_26/MergeV2Checkpoints^save_26/control_dependency*
_output_shapes
: *
T0
ű
save_26/RestoreV2/tensor_namesConst*
dtype0*¨
valueBDBbeta1_powerBbeta1_power_1Bbeta1_power_2Bbeta2_powerBbeta2_power_1Bbeta2_power_2Bpenalty/penalty_paramBpenalty/penalty_param/AdamBpenalty/penalty_param/Adam_1Bpenalty/penalty_param/Adam_2Bpenalty/penalty_param/Adam_3Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1B
pi/log_stdBpi/log_std/AdamBpi/log_std/Adam_1Bvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1*
_output_shapes
:D
ô
"save_26/RestoreV2/shape_and_slicesConst*
dtype0*
_output_shapes
:D*
valueBDB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
î
save_26/RestoreV2	RestoreV2save_26/Constsave_26/RestoreV2/tensor_names"save_26/RestoreV2/shape_and_slices*R
dtypesH
F2D*Ś
_output_shapes
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
Ź
save_26/AssignAssignbeta1_powersave_26/RestoreV2*
_output_shapes
: *(
_class
loc:@penalty/penalty_param*
T0*
use_locking(*
validate_shape(
˛
save_26/Assign_1Assignbeta1_power_1save_26/RestoreV2:1*
_output_shapes
: *
validate_shape(*
use_locking(*(
_class
loc:@penalty/penalty_param*
T0
Ş
save_26/Assign_2Assignbeta1_power_2save_26/RestoreV2:2*
T0*
validate_shape(*
_output_shapes
: * 
_class
loc:@vc/dense/bias*
use_locking(
°
save_26/Assign_3Assignbeta2_powersave_26/RestoreV2:3*
validate_shape(*(
_class
loc:@penalty/penalty_param*
_output_shapes
: *
T0*
use_locking(
˛
save_26/Assign_4Assignbeta2_power_1save_26/RestoreV2:4*
validate_shape(*
_output_shapes
: *
T0*(
_class
loc:@penalty/penalty_param*
use_locking(
Ş
save_26/Assign_5Assignbeta2_power_2save_26/RestoreV2:5*
T0*
use_locking(*
_output_shapes
: * 
_class
loc:@vc/dense/bias*
validate_shape(
ş
save_26/Assign_6Assignpenalty/penalty_paramsave_26/RestoreV2:6*(
_class
loc:@penalty/penalty_param*
T0*
use_locking(*
validate_shape(*
_output_shapes
: 
ż
save_26/Assign_7Assignpenalty/penalty_param/Adamsave_26/RestoreV2:7*(
_class
loc:@penalty/penalty_param*
validate_shape(*
use_locking(*
T0*
_output_shapes
: 
Á
save_26/Assign_8Assignpenalty/penalty_param/Adam_1save_26/RestoreV2:8*
validate_shape(*
_output_shapes
: *
T0*
use_locking(*(
_class
loc:@penalty/penalty_param
Á
save_26/Assign_9Assignpenalty/penalty_param/Adam_2save_26/RestoreV2:9*
T0*(
_class
loc:@penalty/penalty_param*
_output_shapes
: *
use_locking(*
validate_shape(
Ă
save_26/Assign_10Assignpenalty/penalty_param/Adam_3save_26/RestoreV2:10*
T0*
_output_shapes
: *
validate_shape(*
use_locking(*(
_class
loc:@penalty/penalty_param
ą
save_26/Assign_11Assignpi/dense/biassave_26/RestoreV2:11*
T0* 
_class
loc:@pi/dense/bias*
use_locking(*
_output_shapes	
:*
validate_shape(
ś
save_26/Assign_12Assignpi/dense/bias/Adamsave_26/RestoreV2:12*
T0*
validate_shape(* 
_class
loc:@pi/dense/bias*
_output_shapes	
:*
use_locking(
¸
save_26/Assign_13Assignpi/dense/bias/Adam_1save_26/RestoreV2:13*
_output_shapes	
:* 
_class
loc:@pi/dense/bias*
T0*
use_locking(*
validate_shape(
š
save_26/Assign_14Assignpi/dense/kernelsave_26/RestoreV2:14*"
_class
loc:@pi/dense/kernel*
validate_shape(*
_output_shapes
:	L*
T0*
use_locking(
ž
save_26/Assign_15Assignpi/dense/kernel/Adamsave_26/RestoreV2:15*
use_locking(*
_output_shapes
:	L*"
_class
loc:@pi/dense/kernel*
T0*
validate_shape(
Ŕ
save_26/Assign_16Assignpi/dense/kernel/Adam_1save_26/RestoreV2:16*
_output_shapes
:	L*
T0*
use_locking(*"
_class
loc:@pi/dense/kernel*
validate_shape(
ľ
save_26/Assign_17Assignpi/dense_1/biassave_26/RestoreV2:17*"
_class
loc:@pi/dense_1/bias*
use_locking(*
T0*
_output_shapes	
:*
validate_shape(
ş
save_26/Assign_18Assignpi/dense_1/bias/Adamsave_26/RestoreV2:18*"
_class
loc:@pi/dense_1/bias*
_output_shapes	
:*
validate_shape(*
T0*
use_locking(
ź
save_26/Assign_19Assignpi/dense_1/bias/Adam_1save_26/RestoreV2:19*
use_locking(*"
_class
loc:@pi/dense_1/bias*
T0*
validate_shape(*
_output_shapes	
:
ž
save_26/Assign_20Assignpi/dense_1/kernelsave_26/RestoreV2:20*
use_locking(* 
_output_shapes
:
*$
_class
loc:@pi/dense_1/kernel*
T0*
validate_shape(
Ă
save_26/Assign_21Assignpi/dense_1/kernel/Adamsave_26/RestoreV2:21*
use_locking(*$
_class
loc:@pi/dense_1/kernel*
T0* 
_output_shapes
:
*
validate_shape(
Ĺ
save_26/Assign_22Assignpi/dense_1/kernel/Adam_1save_26/RestoreV2:22*
validate_shape(*
use_locking(*
T0* 
_output_shapes
:
*$
_class
loc:@pi/dense_1/kernel
´
save_26/Assign_23Assignpi/dense_2/biassave_26/RestoreV2:23*
validate_shape(*
T0*
use_locking(*
_output_shapes
:*"
_class
loc:@pi/dense_2/bias
š
save_26/Assign_24Assignpi/dense_2/bias/Adamsave_26/RestoreV2:24*
validate_shape(*
use_locking(*
_output_shapes
:*
T0*"
_class
loc:@pi/dense_2/bias
ť
save_26/Assign_25Assignpi/dense_2/bias/Adam_1save_26/RestoreV2:25*
validate_shape(*
T0*
use_locking(*
_output_shapes
:*"
_class
loc:@pi/dense_2/bias
˝
save_26/Assign_26Assignpi/dense_2/kernelsave_26/RestoreV2:26*
_output_shapes
:	*$
_class
loc:@pi/dense_2/kernel*
T0*
validate_shape(*
use_locking(
Â
save_26/Assign_27Assignpi/dense_2/kernel/Adamsave_26/RestoreV2:27*
validate_shape(*
T0*
_output_shapes
:	*
use_locking(*$
_class
loc:@pi/dense_2/kernel
Ä
save_26/Assign_28Assignpi/dense_2/kernel/Adam_1save_26/RestoreV2:28*
validate_shape(*
_output_shapes
:	*$
_class
loc:@pi/dense_2/kernel*
use_locking(*
T0
Ş
save_26/Assign_29Assign
pi/log_stdsave_26/RestoreV2:29*
_output_shapes
:*
T0*
_class
loc:@pi/log_std*
use_locking(*
validate_shape(
Ż
save_26/Assign_30Assignpi/log_std/Adamsave_26/RestoreV2:30*
use_locking(*
validate_shape(*
T0*
_output_shapes
:*
_class
loc:@pi/log_std
ą
save_26/Assign_31Assignpi/log_std/Adam_1save_26/RestoreV2:31*
_output_shapes
:*
T0*
use_locking(*
_class
loc:@pi/log_std*
validate_shape(
ą
save_26/Assign_32Assignvc/dense/biassave_26/RestoreV2:32*
use_locking(*
validate_shape(*
_output_shapes	
:*
T0* 
_class
loc:@vc/dense/bias
ś
save_26/Assign_33Assignvc/dense/bias/Adamsave_26/RestoreV2:33*
validate_shape(*
_output_shapes	
:*
use_locking(* 
_class
loc:@vc/dense/bias*
T0
¸
save_26/Assign_34Assignvc/dense/bias/Adam_1save_26/RestoreV2:34*
_output_shapes	
:*
use_locking(*
validate_shape(* 
_class
loc:@vc/dense/bias*
T0
š
save_26/Assign_35Assignvc/dense/kernelsave_26/RestoreV2:35*
validate_shape(*
use_locking(*"
_class
loc:@vc/dense/kernel*
_output_shapes
:	L*
T0
ž
save_26/Assign_36Assignvc/dense/kernel/Adamsave_26/RestoreV2:36*
use_locking(*
validate_shape(*"
_class
loc:@vc/dense/kernel*
T0*
_output_shapes
:	L
Ŕ
save_26/Assign_37Assignvc/dense/kernel/Adam_1save_26/RestoreV2:37*
T0*
_output_shapes
:	L*"
_class
loc:@vc/dense/kernel*
validate_shape(*
use_locking(
ľ
save_26/Assign_38Assignvc/dense_1/biassave_26/RestoreV2:38*
_output_shapes	
:*
use_locking(*
validate_shape(*
T0*"
_class
loc:@vc/dense_1/bias
ş
save_26/Assign_39Assignvc/dense_1/bias/Adamsave_26/RestoreV2:39*
validate_shape(*
_output_shapes	
:*
T0*"
_class
loc:@vc/dense_1/bias*
use_locking(
ź
save_26/Assign_40Assignvc/dense_1/bias/Adam_1save_26/RestoreV2:40*
use_locking(*
T0*"
_class
loc:@vc/dense_1/bias*
validate_shape(*
_output_shapes	
:
ž
save_26/Assign_41Assignvc/dense_1/kernelsave_26/RestoreV2:41*
use_locking(* 
_output_shapes
:
*
validate_shape(*$
_class
loc:@vc/dense_1/kernel*
T0
Ă
save_26/Assign_42Assignvc/dense_1/kernel/Adamsave_26/RestoreV2:42*$
_class
loc:@vc/dense_1/kernel*
validate_shape(*
use_locking(* 
_output_shapes
:
*
T0
Ĺ
save_26/Assign_43Assignvc/dense_1/kernel/Adam_1save_26/RestoreV2:43*
validate_shape(*$
_class
loc:@vc/dense_1/kernel*
T0* 
_output_shapes
:
*
use_locking(
´
save_26/Assign_44Assignvc/dense_2/biassave_26/RestoreV2:44*
validate_shape(*"
_class
loc:@vc/dense_2/bias*
use_locking(*
_output_shapes
:*
T0
š
save_26/Assign_45Assignvc/dense_2/bias/Adamsave_26/RestoreV2:45*
T0*
use_locking(*
_output_shapes
:*"
_class
loc:@vc/dense_2/bias*
validate_shape(
ť
save_26/Assign_46Assignvc/dense_2/bias/Adam_1save_26/RestoreV2:46*
T0*"
_class
loc:@vc/dense_2/bias*
_output_shapes
:*
use_locking(*
validate_shape(
˝
save_26/Assign_47Assignvc/dense_2/kernelsave_26/RestoreV2:47*
T0*
_output_shapes
:	*
validate_shape(*
use_locking(*$
_class
loc:@vc/dense_2/kernel
Â
save_26/Assign_48Assignvc/dense_2/kernel/Adamsave_26/RestoreV2:48*
use_locking(*
_output_shapes
:	*$
_class
loc:@vc/dense_2/kernel*
validate_shape(*
T0
Ä
save_26/Assign_49Assignvc/dense_2/kernel/Adam_1save_26/RestoreV2:49*$
_class
loc:@vc/dense_2/kernel*
use_locking(*
_output_shapes
:	*
validate_shape(*
T0
ą
save_26/Assign_50Assignvf/dense/biassave_26/RestoreV2:50*
T0*
_output_shapes	
:*
use_locking(*
validate_shape(* 
_class
loc:@vf/dense/bias
ś
save_26/Assign_51Assignvf/dense/bias/Adamsave_26/RestoreV2:51*
T0*
_output_shapes	
:*
use_locking(*
validate_shape(* 
_class
loc:@vf/dense/bias
¸
save_26/Assign_52Assignvf/dense/bias/Adam_1save_26/RestoreV2:52*
T0*
use_locking(* 
_class
loc:@vf/dense/bias*
validate_shape(*
_output_shapes	
:
š
save_26/Assign_53Assignvf/dense/kernelsave_26/RestoreV2:53*
use_locking(*
_output_shapes
:	L*
T0*"
_class
loc:@vf/dense/kernel*
validate_shape(
ž
save_26/Assign_54Assignvf/dense/kernel/Adamsave_26/RestoreV2:54*
use_locking(*
T0*
_output_shapes
:	L*
validate_shape(*"
_class
loc:@vf/dense/kernel
Ŕ
save_26/Assign_55Assignvf/dense/kernel/Adam_1save_26/RestoreV2:55*
validate_shape(*
T0*
_output_shapes
:	L*"
_class
loc:@vf/dense/kernel*
use_locking(
ľ
save_26/Assign_56Assignvf/dense_1/biassave_26/RestoreV2:56*
validate_shape(*
T0*
_output_shapes	
:*
use_locking(*"
_class
loc:@vf/dense_1/bias
ş
save_26/Assign_57Assignvf/dense_1/bias/Adamsave_26/RestoreV2:57*
use_locking(*
validate_shape(*
_output_shapes	
:*"
_class
loc:@vf/dense_1/bias*
T0
ź
save_26/Assign_58Assignvf/dense_1/bias/Adam_1save_26/RestoreV2:58*
_output_shapes	
:*"
_class
loc:@vf/dense_1/bias*
validate_shape(*
T0*
use_locking(
ž
save_26/Assign_59Assignvf/dense_1/kernelsave_26/RestoreV2:59*$
_class
loc:@vf/dense_1/kernel* 
_output_shapes
:
*
validate_shape(*
T0*
use_locking(
Ă
save_26/Assign_60Assignvf/dense_1/kernel/Adamsave_26/RestoreV2:60*
validate_shape(* 
_output_shapes
:
*
use_locking(*$
_class
loc:@vf/dense_1/kernel*
T0
Ĺ
save_26/Assign_61Assignvf/dense_1/kernel/Adam_1save_26/RestoreV2:61* 
_output_shapes
:
*
T0*$
_class
loc:@vf/dense_1/kernel*
validate_shape(*
use_locking(
´
save_26/Assign_62Assignvf/dense_2/biassave_26/RestoreV2:62*"
_class
loc:@vf/dense_2/bias*
_output_shapes
:*
use_locking(*
T0*
validate_shape(
š
save_26/Assign_63Assignvf/dense_2/bias/Adamsave_26/RestoreV2:63*
_output_shapes
:*
T0*
use_locking(*"
_class
loc:@vf/dense_2/bias*
validate_shape(
ť
save_26/Assign_64Assignvf/dense_2/bias/Adam_1save_26/RestoreV2:64*
_output_shapes
:*
use_locking(*
T0*"
_class
loc:@vf/dense_2/bias*
validate_shape(
˝
save_26/Assign_65Assignvf/dense_2/kernelsave_26/RestoreV2:65*
_output_shapes
:	*
T0*$
_class
loc:@vf/dense_2/kernel*
validate_shape(*
use_locking(
Â
save_26/Assign_66Assignvf/dense_2/kernel/Adamsave_26/RestoreV2:66*
_output_shapes
:	*$
_class
loc:@vf/dense_2/kernel*
use_locking(*
T0*
validate_shape(
Ä
save_26/Assign_67Assignvf/dense_2/kernel/Adam_1save_26/RestoreV2:67*$
_class
loc:@vf/dense_2/kernel*
T0*
_output_shapes
:	*
validate_shape(*
use_locking(
á

save_26/restore_shardNoOp^save_26/Assign^save_26/Assign_1^save_26/Assign_10^save_26/Assign_11^save_26/Assign_12^save_26/Assign_13^save_26/Assign_14^save_26/Assign_15^save_26/Assign_16^save_26/Assign_17^save_26/Assign_18^save_26/Assign_19^save_26/Assign_2^save_26/Assign_20^save_26/Assign_21^save_26/Assign_22^save_26/Assign_23^save_26/Assign_24^save_26/Assign_25^save_26/Assign_26^save_26/Assign_27^save_26/Assign_28^save_26/Assign_29^save_26/Assign_3^save_26/Assign_30^save_26/Assign_31^save_26/Assign_32^save_26/Assign_33^save_26/Assign_34^save_26/Assign_35^save_26/Assign_36^save_26/Assign_37^save_26/Assign_38^save_26/Assign_39^save_26/Assign_4^save_26/Assign_40^save_26/Assign_41^save_26/Assign_42^save_26/Assign_43^save_26/Assign_44^save_26/Assign_45^save_26/Assign_46^save_26/Assign_47^save_26/Assign_48^save_26/Assign_49^save_26/Assign_5^save_26/Assign_50^save_26/Assign_51^save_26/Assign_52^save_26/Assign_53^save_26/Assign_54^save_26/Assign_55^save_26/Assign_56^save_26/Assign_57^save_26/Assign_58^save_26/Assign_59^save_26/Assign_6^save_26/Assign_60^save_26/Assign_61^save_26/Assign_62^save_26/Assign_63^save_26/Assign_64^save_26/Assign_65^save_26/Assign_66^save_26/Assign_67^save_26/Assign_7^save_26/Assign_8^save_26/Assign_9
3
save_26/restore_allNoOp^save_26/restore_shard
\
save_27/filename/inputConst*
_output_shapes
: *
dtype0*
valueB Bmodel
t
save_27/filenamePlaceholderWithDefaultsave_27/filename/input*
_output_shapes
: *
dtype0*
shape: 
k
save_27/ConstPlaceholderWithDefaultsave_27/filename*
_output_shapes
: *
shape: *
dtype0

save_27/StringJoin/inputs_1Const*
_output_shapes
: *<
value3B1 B+_temp_8a45b78cc3114ae3884dda4482e1b4d5/part*
dtype0
~
save_27/StringJoin
StringJoinsave_27/Constsave_27/StringJoin/inputs_1*
N*
	separator *
_output_shapes
: 
T
save_27/num_shardsConst*
dtype0*
value	B :*
_output_shapes
: 
_
save_27/ShardedFilename/shardConst*
_output_shapes
: *
dtype0*
value	B : 

save_27/ShardedFilenameShardedFilenamesave_27/StringJoinsave_27/ShardedFilename/shardsave_27/num_shards*
_output_shapes
: 
ř
save_27/SaveV2/tensor_namesConst*
_output_shapes
:D*¨
valueBDBbeta1_powerBbeta1_power_1Bbeta1_power_2Bbeta2_powerBbeta2_power_1Bbeta2_power_2Bpenalty/penalty_paramBpenalty/penalty_param/AdamBpenalty/penalty_param/Adam_1Bpenalty/penalty_param/Adam_2Bpenalty/penalty_param/Adam_3Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1B
pi/log_stdBpi/log_std/AdamBpi/log_std/Adam_1Bvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1*
dtype0
ń
save_27/SaveV2/shape_and_slicesConst*
_output_shapes
:D*
dtype0*
valueBDB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
Ö
save_27/SaveV2SaveV2save_27/ShardedFilenamesave_27/SaveV2/tensor_namessave_27/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta1_power_2beta2_powerbeta2_power_1beta2_power_2penalty/penalty_parampenalty/penalty_param/Adampenalty/penalty_param/Adam_1penalty/penalty_param/Adam_2penalty/penalty_param/Adam_3pi/dense/biaspi/dense/bias/Adampi/dense/bias/Adam_1pi/dense/kernelpi/dense/kernel/Adampi/dense/kernel/Adam_1pi/dense_1/biaspi/dense_1/bias/Adampi/dense_1/bias/Adam_1pi/dense_1/kernelpi/dense_1/kernel/Adampi/dense_1/kernel/Adam_1pi/dense_2/biaspi/dense_2/bias/Adampi/dense_2/bias/Adam_1pi/dense_2/kernelpi/dense_2/kernel/Adampi/dense_2/kernel/Adam_1
pi/log_stdpi/log_std/Adampi/log_std/Adam_1vc/dense/biasvc/dense/bias/Adamvc/dense/bias/Adam_1vc/dense/kernelvc/dense/kernel/Adamvc/dense/kernel/Adam_1vc/dense_1/biasvc/dense_1/bias/Adamvc/dense_1/bias/Adam_1vc/dense_1/kernelvc/dense_1/kernel/Adamvc/dense_1/kernel/Adam_1vc/dense_2/biasvc/dense_2/bias/Adamvc/dense_2/bias/Adam_1vc/dense_2/kernelvc/dense_2/kernel/Adamvc/dense_2/kernel/Adam_1vf/dense/biasvf/dense/bias/Adamvf/dense/bias/Adam_1vf/dense/kernelvf/dense/kernel/Adamvf/dense/kernel/Adam_1vf/dense_1/biasvf/dense_1/bias/Adamvf/dense_1/bias/Adam_1vf/dense_1/kernelvf/dense_1/kernel/Adamvf/dense_1/kernel/Adam_1vf/dense_2/biasvf/dense_2/bias/Adamvf/dense_2/bias/Adam_1vf/dense_2/kernelvf/dense_2/kernel/Adamvf/dense_2/kernel/Adam_1*R
dtypesH
F2D

save_27/control_dependencyIdentitysave_27/ShardedFilename^save_27/SaveV2**
_class 
loc:@save_27/ShardedFilename*
_output_shapes
: *
T0
Ś
.save_27/MergeV2Checkpoints/checkpoint_prefixesPacksave_27/ShardedFilename^save_27/control_dependency*
_output_shapes
:*
N*

axis *
T0

save_27/MergeV2CheckpointsMergeV2Checkpoints.save_27/MergeV2Checkpoints/checkpoint_prefixessave_27/Const*
delete_old_dirs(

save_27/IdentityIdentitysave_27/Const^save_27/MergeV2Checkpoints^save_27/control_dependency*
_output_shapes
: *
T0
ű
save_27/RestoreV2/tensor_namesConst*¨
valueBDBbeta1_powerBbeta1_power_1Bbeta1_power_2Bbeta2_powerBbeta2_power_1Bbeta2_power_2Bpenalty/penalty_paramBpenalty/penalty_param/AdamBpenalty/penalty_param/Adam_1Bpenalty/penalty_param/Adam_2Bpenalty/penalty_param/Adam_3Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1B
pi/log_stdBpi/log_std/AdamBpi/log_std/Adam_1Bvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1*
dtype0*
_output_shapes
:D
ô
"save_27/RestoreV2/shape_and_slicesConst*
_output_shapes
:D*
dtype0*
valueBDB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
î
save_27/RestoreV2	RestoreV2save_27/Constsave_27/RestoreV2/tensor_names"save_27/RestoreV2/shape_and_slices*R
dtypesH
F2D*Ś
_output_shapes
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
Ź
save_27/AssignAssignbeta1_powersave_27/RestoreV2*
_output_shapes
: *
validate_shape(*
use_locking(*(
_class
loc:@penalty/penalty_param*
T0
˛
save_27/Assign_1Assignbeta1_power_1save_27/RestoreV2:1*(
_class
loc:@penalty/penalty_param*
use_locking(*
_output_shapes
: *
validate_shape(*
T0
Ş
save_27/Assign_2Assignbeta1_power_2save_27/RestoreV2:2*
use_locking(*
validate_shape(* 
_class
loc:@vc/dense/bias*
_output_shapes
: *
T0
°
save_27/Assign_3Assignbeta2_powersave_27/RestoreV2:3*
T0*
validate_shape(*
_output_shapes
: *(
_class
loc:@penalty/penalty_param*
use_locking(
˛
save_27/Assign_4Assignbeta2_power_1save_27/RestoreV2:4*
validate_shape(*
use_locking(*
_output_shapes
: *
T0*(
_class
loc:@penalty/penalty_param
Ş
save_27/Assign_5Assignbeta2_power_2save_27/RestoreV2:5* 
_class
loc:@vc/dense/bias*
use_locking(*
T0*
_output_shapes
: *
validate_shape(
ş
save_27/Assign_6Assignpenalty/penalty_paramsave_27/RestoreV2:6*
T0*
validate_shape(*
_output_shapes
: *(
_class
loc:@penalty/penalty_param*
use_locking(
ż
save_27/Assign_7Assignpenalty/penalty_param/Adamsave_27/RestoreV2:7*
validate_shape(*(
_class
loc:@penalty/penalty_param*
use_locking(*
_output_shapes
: *
T0
Á
save_27/Assign_8Assignpenalty/penalty_param/Adam_1save_27/RestoreV2:8*
validate_shape(*(
_class
loc:@penalty/penalty_param*
use_locking(*
T0*
_output_shapes
: 
Á
save_27/Assign_9Assignpenalty/penalty_param/Adam_2save_27/RestoreV2:9*
use_locking(*
_output_shapes
: *
validate_shape(*(
_class
loc:@penalty/penalty_param*
T0
Ă
save_27/Assign_10Assignpenalty/penalty_param/Adam_3save_27/RestoreV2:10*
T0*(
_class
loc:@penalty/penalty_param*
_output_shapes
: *
use_locking(*
validate_shape(
ą
save_27/Assign_11Assignpi/dense/biassave_27/RestoreV2:11*
validate_shape(*
use_locking(*
T0* 
_class
loc:@pi/dense/bias*
_output_shapes	
:
ś
save_27/Assign_12Assignpi/dense/bias/Adamsave_27/RestoreV2:12*
validate_shape(*
_output_shapes	
:*
T0* 
_class
loc:@pi/dense/bias*
use_locking(
¸
save_27/Assign_13Assignpi/dense/bias/Adam_1save_27/RestoreV2:13*
validate_shape(*
_output_shapes	
:*
use_locking(* 
_class
loc:@pi/dense/bias*
T0
š
save_27/Assign_14Assignpi/dense/kernelsave_27/RestoreV2:14*
use_locking(*"
_class
loc:@pi/dense/kernel*
validate_shape(*
T0*
_output_shapes
:	L
ž
save_27/Assign_15Assignpi/dense/kernel/Adamsave_27/RestoreV2:15*
T0*"
_class
loc:@pi/dense/kernel*
validate_shape(*
use_locking(*
_output_shapes
:	L
Ŕ
save_27/Assign_16Assignpi/dense/kernel/Adam_1save_27/RestoreV2:16*"
_class
loc:@pi/dense/kernel*
_output_shapes
:	L*
use_locking(*
T0*
validate_shape(
ľ
save_27/Assign_17Assignpi/dense_1/biassave_27/RestoreV2:17*"
_class
loc:@pi/dense_1/bias*
use_locking(*
_output_shapes	
:*
T0*
validate_shape(
ş
save_27/Assign_18Assignpi/dense_1/bias/Adamsave_27/RestoreV2:18*
use_locking(*
validate_shape(*
_output_shapes	
:*
T0*"
_class
loc:@pi/dense_1/bias
ź
save_27/Assign_19Assignpi/dense_1/bias/Adam_1save_27/RestoreV2:19*
_output_shapes	
:*
validate_shape(*"
_class
loc:@pi/dense_1/bias*
T0*
use_locking(
ž
save_27/Assign_20Assignpi/dense_1/kernelsave_27/RestoreV2:20*
T0*
validate_shape(*
use_locking(*$
_class
loc:@pi/dense_1/kernel* 
_output_shapes
:

Ă
save_27/Assign_21Assignpi/dense_1/kernel/Adamsave_27/RestoreV2:21*
validate_shape(*
T0*$
_class
loc:@pi/dense_1/kernel*
use_locking(* 
_output_shapes
:

Ĺ
save_27/Assign_22Assignpi/dense_1/kernel/Adam_1save_27/RestoreV2:22*
use_locking(* 
_output_shapes
:
*
validate_shape(*
T0*$
_class
loc:@pi/dense_1/kernel
´
save_27/Assign_23Assignpi/dense_2/biassave_27/RestoreV2:23*"
_class
loc:@pi/dense_2/bias*
validate_shape(*
_output_shapes
:*
T0*
use_locking(
š
save_27/Assign_24Assignpi/dense_2/bias/Adamsave_27/RestoreV2:24*"
_class
loc:@pi/dense_2/bias*
validate_shape(*
T0*
_output_shapes
:*
use_locking(
ť
save_27/Assign_25Assignpi/dense_2/bias/Adam_1save_27/RestoreV2:25*
validate_shape(*
use_locking(*
_output_shapes
:*"
_class
loc:@pi/dense_2/bias*
T0
˝
save_27/Assign_26Assignpi/dense_2/kernelsave_27/RestoreV2:26*
use_locking(*
_output_shapes
:	*$
_class
loc:@pi/dense_2/kernel*
validate_shape(*
T0
Â
save_27/Assign_27Assignpi/dense_2/kernel/Adamsave_27/RestoreV2:27*
validate_shape(*
_output_shapes
:	*$
_class
loc:@pi/dense_2/kernel*
use_locking(*
T0
Ä
save_27/Assign_28Assignpi/dense_2/kernel/Adam_1save_27/RestoreV2:28*
use_locking(*$
_class
loc:@pi/dense_2/kernel*
_output_shapes
:	*
T0*
validate_shape(
Ş
save_27/Assign_29Assign
pi/log_stdsave_27/RestoreV2:29*
use_locking(*
validate_shape(*
_output_shapes
:*
T0*
_class
loc:@pi/log_std
Ż
save_27/Assign_30Assignpi/log_std/Adamsave_27/RestoreV2:30*
use_locking(*
T0*
_output_shapes
:*
validate_shape(*
_class
loc:@pi/log_std
ą
save_27/Assign_31Assignpi/log_std/Adam_1save_27/RestoreV2:31*
_class
loc:@pi/log_std*
T0*
_output_shapes
:*
use_locking(*
validate_shape(
ą
save_27/Assign_32Assignvc/dense/biassave_27/RestoreV2:32*
T0*
use_locking(*
validate_shape(* 
_class
loc:@vc/dense/bias*
_output_shapes	
:
ś
save_27/Assign_33Assignvc/dense/bias/Adamsave_27/RestoreV2:33* 
_class
loc:@vc/dense/bias*
T0*
_output_shapes	
:*
validate_shape(*
use_locking(
¸
save_27/Assign_34Assignvc/dense/bias/Adam_1save_27/RestoreV2:34*
use_locking(*
_output_shapes	
:*
T0* 
_class
loc:@vc/dense/bias*
validate_shape(
š
save_27/Assign_35Assignvc/dense/kernelsave_27/RestoreV2:35*
validate_shape(*
_output_shapes
:	L*
T0*
use_locking(*"
_class
loc:@vc/dense/kernel
ž
save_27/Assign_36Assignvc/dense/kernel/Adamsave_27/RestoreV2:36*
_output_shapes
:	L*
validate_shape(*"
_class
loc:@vc/dense/kernel*
T0*
use_locking(
Ŕ
save_27/Assign_37Assignvc/dense/kernel/Adam_1save_27/RestoreV2:37*
use_locking(*
_output_shapes
:	L*"
_class
loc:@vc/dense/kernel*
validate_shape(*
T0
ľ
save_27/Assign_38Assignvc/dense_1/biassave_27/RestoreV2:38*
validate_shape(*
use_locking(*"
_class
loc:@vc/dense_1/bias*
T0*
_output_shapes	
:
ş
save_27/Assign_39Assignvc/dense_1/bias/Adamsave_27/RestoreV2:39*
validate_shape(*
T0*
_output_shapes	
:*
use_locking(*"
_class
loc:@vc/dense_1/bias
ź
save_27/Assign_40Assignvc/dense_1/bias/Adam_1save_27/RestoreV2:40*
use_locking(*"
_class
loc:@vc/dense_1/bias*
T0*
_output_shapes	
:*
validate_shape(
ž
save_27/Assign_41Assignvc/dense_1/kernelsave_27/RestoreV2:41*$
_class
loc:@vc/dense_1/kernel*
use_locking(* 
_output_shapes
:
*
validate_shape(*
T0
Ă
save_27/Assign_42Assignvc/dense_1/kernel/Adamsave_27/RestoreV2:42*$
_class
loc:@vc/dense_1/kernel*
T0*
use_locking(* 
_output_shapes
:
*
validate_shape(
Ĺ
save_27/Assign_43Assignvc/dense_1/kernel/Adam_1save_27/RestoreV2:43*
validate_shape(* 
_output_shapes
:
*
use_locking(*$
_class
loc:@vc/dense_1/kernel*
T0
´
save_27/Assign_44Assignvc/dense_2/biassave_27/RestoreV2:44*
validate_shape(*
_output_shapes
:*
T0*
use_locking(*"
_class
loc:@vc/dense_2/bias
š
save_27/Assign_45Assignvc/dense_2/bias/Adamsave_27/RestoreV2:45*"
_class
loc:@vc/dense_2/bias*
use_locking(*
validate_shape(*
T0*
_output_shapes
:
ť
save_27/Assign_46Assignvc/dense_2/bias/Adam_1save_27/RestoreV2:46*
T0*
validate_shape(*
_output_shapes
:*"
_class
loc:@vc/dense_2/bias*
use_locking(
˝
save_27/Assign_47Assignvc/dense_2/kernelsave_27/RestoreV2:47*
T0*$
_class
loc:@vc/dense_2/kernel*
_output_shapes
:	*
validate_shape(*
use_locking(
Â
save_27/Assign_48Assignvc/dense_2/kernel/Adamsave_27/RestoreV2:48*
use_locking(*$
_class
loc:@vc/dense_2/kernel*
_output_shapes
:	*
validate_shape(*
T0
Ä
save_27/Assign_49Assignvc/dense_2/kernel/Adam_1save_27/RestoreV2:49*
use_locking(*
_output_shapes
:	*
T0*$
_class
loc:@vc/dense_2/kernel*
validate_shape(
ą
save_27/Assign_50Assignvf/dense/biassave_27/RestoreV2:50*
use_locking(*
_output_shapes	
:*
validate_shape(*
T0* 
_class
loc:@vf/dense/bias
ś
save_27/Assign_51Assignvf/dense/bias/Adamsave_27/RestoreV2:51*
validate_shape(* 
_class
loc:@vf/dense/bias*
use_locking(*
T0*
_output_shapes	
:
¸
save_27/Assign_52Assignvf/dense/bias/Adam_1save_27/RestoreV2:52*
use_locking(*
validate_shape(*
_output_shapes	
:* 
_class
loc:@vf/dense/bias*
T0
š
save_27/Assign_53Assignvf/dense/kernelsave_27/RestoreV2:53*
validate_shape(*
use_locking(*
T0*"
_class
loc:@vf/dense/kernel*
_output_shapes
:	L
ž
save_27/Assign_54Assignvf/dense/kernel/Adamsave_27/RestoreV2:54*
use_locking(*"
_class
loc:@vf/dense/kernel*
_output_shapes
:	L*
validate_shape(*
T0
Ŕ
save_27/Assign_55Assignvf/dense/kernel/Adam_1save_27/RestoreV2:55*
use_locking(*
_output_shapes
:	L*
validate_shape(*
T0*"
_class
loc:@vf/dense/kernel
ľ
save_27/Assign_56Assignvf/dense_1/biassave_27/RestoreV2:56*
use_locking(*
_output_shapes	
:*
T0*"
_class
loc:@vf/dense_1/bias*
validate_shape(
ş
save_27/Assign_57Assignvf/dense_1/bias/Adamsave_27/RestoreV2:57*
T0*
validate_shape(*"
_class
loc:@vf/dense_1/bias*
_output_shapes	
:*
use_locking(
ź
save_27/Assign_58Assignvf/dense_1/bias/Adam_1save_27/RestoreV2:58*
T0*"
_class
loc:@vf/dense_1/bias*
_output_shapes	
:*
validate_shape(*
use_locking(
ž
save_27/Assign_59Assignvf/dense_1/kernelsave_27/RestoreV2:59*
validate_shape(* 
_output_shapes
:
*$
_class
loc:@vf/dense_1/kernel*
use_locking(*
T0
Ă
save_27/Assign_60Assignvf/dense_1/kernel/Adamsave_27/RestoreV2:60*
validate_shape(*
T0*$
_class
loc:@vf/dense_1/kernel*
use_locking(* 
_output_shapes
:

Ĺ
save_27/Assign_61Assignvf/dense_1/kernel/Adam_1save_27/RestoreV2:61*$
_class
loc:@vf/dense_1/kernel*
validate_shape(*
use_locking(* 
_output_shapes
:
*
T0
´
save_27/Assign_62Assignvf/dense_2/biassave_27/RestoreV2:62*
validate_shape(*
T0*
use_locking(*
_output_shapes
:*"
_class
loc:@vf/dense_2/bias
š
save_27/Assign_63Assignvf/dense_2/bias/Adamsave_27/RestoreV2:63*
T0*
_output_shapes
:*
use_locking(*"
_class
loc:@vf/dense_2/bias*
validate_shape(
ť
save_27/Assign_64Assignvf/dense_2/bias/Adam_1save_27/RestoreV2:64*
_output_shapes
:*
use_locking(*"
_class
loc:@vf/dense_2/bias*
validate_shape(*
T0
˝
save_27/Assign_65Assignvf/dense_2/kernelsave_27/RestoreV2:65*$
_class
loc:@vf/dense_2/kernel*
_output_shapes
:	*
T0*
validate_shape(*
use_locking(
Â
save_27/Assign_66Assignvf/dense_2/kernel/Adamsave_27/RestoreV2:66*
validate_shape(*
use_locking(*$
_class
loc:@vf/dense_2/kernel*
_output_shapes
:	*
T0
Ä
save_27/Assign_67Assignvf/dense_2/kernel/Adam_1save_27/RestoreV2:67*
validate_shape(*$
_class
loc:@vf/dense_2/kernel*
_output_shapes
:	*
T0*
use_locking(
á

save_27/restore_shardNoOp^save_27/Assign^save_27/Assign_1^save_27/Assign_10^save_27/Assign_11^save_27/Assign_12^save_27/Assign_13^save_27/Assign_14^save_27/Assign_15^save_27/Assign_16^save_27/Assign_17^save_27/Assign_18^save_27/Assign_19^save_27/Assign_2^save_27/Assign_20^save_27/Assign_21^save_27/Assign_22^save_27/Assign_23^save_27/Assign_24^save_27/Assign_25^save_27/Assign_26^save_27/Assign_27^save_27/Assign_28^save_27/Assign_29^save_27/Assign_3^save_27/Assign_30^save_27/Assign_31^save_27/Assign_32^save_27/Assign_33^save_27/Assign_34^save_27/Assign_35^save_27/Assign_36^save_27/Assign_37^save_27/Assign_38^save_27/Assign_39^save_27/Assign_4^save_27/Assign_40^save_27/Assign_41^save_27/Assign_42^save_27/Assign_43^save_27/Assign_44^save_27/Assign_45^save_27/Assign_46^save_27/Assign_47^save_27/Assign_48^save_27/Assign_49^save_27/Assign_5^save_27/Assign_50^save_27/Assign_51^save_27/Assign_52^save_27/Assign_53^save_27/Assign_54^save_27/Assign_55^save_27/Assign_56^save_27/Assign_57^save_27/Assign_58^save_27/Assign_59^save_27/Assign_6^save_27/Assign_60^save_27/Assign_61^save_27/Assign_62^save_27/Assign_63^save_27/Assign_64^save_27/Assign_65^save_27/Assign_66^save_27/Assign_67^save_27/Assign_7^save_27/Assign_8^save_27/Assign_9
3
save_27/restore_allNoOp^save_27/restore_shard
\
save_28/filename/inputConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
t
save_28/filenamePlaceholderWithDefaultsave_28/filename/input*
_output_shapes
: *
dtype0*
shape: 
k
save_28/ConstPlaceholderWithDefaultsave_28/filename*
dtype0*
shape: *
_output_shapes
: 

save_28/StringJoin/inputs_1Const*
dtype0*
_output_shapes
: *<
value3B1 B+_temp_881a3837557e409fa17d3424098683fd/part
~
save_28/StringJoin
StringJoinsave_28/Constsave_28/StringJoin/inputs_1*
	separator *
N*
_output_shapes
: 
T
save_28/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
_
save_28/ShardedFilename/shardConst*
dtype0*
_output_shapes
: *
value	B : 

save_28/ShardedFilenameShardedFilenamesave_28/StringJoinsave_28/ShardedFilename/shardsave_28/num_shards*
_output_shapes
: 
ř
save_28/SaveV2/tensor_namesConst*
_output_shapes
:D*¨
valueBDBbeta1_powerBbeta1_power_1Bbeta1_power_2Bbeta2_powerBbeta2_power_1Bbeta2_power_2Bpenalty/penalty_paramBpenalty/penalty_param/AdamBpenalty/penalty_param/Adam_1Bpenalty/penalty_param/Adam_2Bpenalty/penalty_param/Adam_3Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1B
pi/log_stdBpi/log_std/AdamBpi/log_std/Adam_1Bvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1*
dtype0
ń
save_28/SaveV2/shape_and_slicesConst*
valueBDB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
_output_shapes
:D*
dtype0
Ö
save_28/SaveV2SaveV2save_28/ShardedFilenamesave_28/SaveV2/tensor_namessave_28/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta1_power_2beta2_powerbeta2_power_1beta2_power_2penalty/penalty_parampenalty/penalty_param/Adampenalty/penalty_param/Adam_1penalty/penalty_param/Adam_2penalty/penalty_param/Adam_3pi/dense/biaspi/dense/bias/Adampi/dense/bias/Adam_1pi/dense/kernelpi/dense/kernel/Adampi/dense/kernel/Adam_1pi/dense_1/biaspi/dense_1/bias/Adampi/dense_1/bias/Adam_1pi/dense_1/kernelpi/dense_1/kernel/Adampi/dense_1/kernel/Adam_1pi/dense_2/biaspi/dense_2/bias/Adampi/dense_2/bias/Adam_1pi/dense_2/kernelpi/dense_2/kernel/Adampi/dense_2/kernel/Adam_1
pi/log_stdpi/log_std/Adampi/log_std/Adam_1vc/dense/biasvc/dense/bias/Adamvc/dense/bias/Adam_1vc/dense/kernelvc/dense/kernel/Adamvc/dense/kernel/Adam_1vc/dense_1/biasvc/dense_1/bias/Adamvc/dense_1/bias/Adam_1vc/dense_1/kernelvc/dense_1/kernel/Adamvc/dense_1/kernel/Adam_1vc/dense_2/biasvc/dense_2/bias/Adamvc/dense_2/bias/Adam_1vc/dense_2/kernelvc/dense_2/kernel/Adamvc/dense_2/kernel/Adam_1vf/dense/biasvf/dense/bias/Adamvf/dense/bias/Adam_1vf/dense/kernelvf/dense/kernel/Adamvf/dense/kernel/Adam_1vf/dense_1/biasvf/dense_1/bias/Adamvf/dense_1/bias/Adam_1vf/dense_1/kernelvf/dense_1/kernel/Adamvf/dense_1/kernel/Adam_1vf/dense_2/biasvf/dense_2/bias/Adamvf/dense_2/bias/Adam_1vf/dense_2/kernelvf/dense_2/kernel/Adamvf/dense_2/kernel/Adam_1*R
dtypesH
F2D

save_28/control_dependencyIdentitysave_28/ShardedFilename^save_28/SaveV2*
T0**
_class 
loc:@save_28/ShardedFilename*
_output_shapes
: 
Ś
.save_28/MergeV2Checkpoints/checkpoint_prefixesPacksave_28/ShardedFilename^save_28/control_dependency*

axis *
_output_shapes
:*
T0*
N

save_28/MergeV2CheckpointsMergeV2Checkpoints.save_28/MergeV2Checkpoints/checkpoint_prefixessave_28/Const*
delete_old_dirs(

save_28/IdentityIdentitysave_28/Const^save_28/MergeV2Checkpoints^save_28/control_dependency*
T0*
_output_shapes
: 
ű
save_28/RestoreV2/tensor_namesConst*
_output_shapes
:D*¨
valueBDBbeta1_powerBbeta1_power_1Bbeta1_power_2Bbeta2_powerBbeta2_power_1Bbeta2_power_2Bpenalty/penalty_paramBpenalty/penalty_param/AdamBpenalty/penalty_param/Adam_1Bpenalty/penalty_param/Adam_2Bpenalty/penalty_param/Adam_3Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1B
pi/log_stdBpi/log_std/AdamBpi/log_std/Adam_1Bvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1*
dtype0
ô
"save_28/RestoreV2/shape_and_slicesConst*
valueBDB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
_output_shapes
:D*
dtype0
î
save_28/RestoreV2	RestoreV2save_28/Constsave_28/RestoreV2/tensor_names"save_28/RestoreV2/shape_and_slices*Ś
_output_shapes
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*R
dtypesH
F2D
Ź
save_28/AssignAssignbeta1_powersave_28/RestoreV2*
_output_shapes
: *
use_locking(*(
_class
loc:@penalty/penalty_param*
validate_shape(*
T0
˛
save_28/Assign_1Assignbeta1_power_1save_28/RestoreV2:1*
validate_shape(*
_output_shapes
: *(
_class
loc:@penalty/penalty_param*
T0*
use_locking(
Ş
save_28/Assign_2Assignbeta1_power_2save_28/RestoreV2:2*
_output_shapes
: *
T0*
use_locking(*
validate_shape(* 
_class
loc:@vc/dense/bias
°
save_28/Assign_3Assignbeta2_powersave_28/RestoreV2:3*
_output_shapes
: *
use_locking(*(
_class
loc:@penalty/penalty_param*
T0*
validate_shape(
˛
save_28/Assign_4Assignbeta2_power_1save_28/RestoreV2:4*
use_locking(*(
_class
loc:@penalty/penalty_param*
_output_shapes
: *
validate_shape(*
T0
Ş
save_28/Assign_5Assignbeta2_power_2save_28/RestoreV2:5*
validate_shape(* 
_class
loc:@vc/dense/bias*
_output_shapes
: *
T0*
use_locking(
ş
save_28/Assign_6Assignpenalty/penalty_paramsave_28/RestoreV2:6*
_output_shapes
: *
T0*
validate_shape(*
use_locking(*(
_class
loc:@penalty/penalty_param
ż
save_28/Assign_7Assignpenalty/penalty_param/Adamsave_28/RestoreV2:7*
_output_shapes
: *
T0*
use_locking(*(
_class
loc:@penalty/penalty_param*
validate_shape(
Á
save_28/Assign_8Assignpenalty/penalty_param/Adam_1save_28/RestoreV2:8*(
_class
loc:@penalty/penalty_param*
T0*
_output_shapes
: *
validate_shape(*
use_locking(
Á
save_28/Assign_9Assignpenalty/penalty_param/Adam_2save_28/RestoreV2:9*
validate_shape(*(
_class
loc:@penalty/penalty_param*
_output_shapes
: *
T0*
use_locking(
Ă
save_28/Assign_10Assignpenalty/penalty_param/Adam_3save_28/RestoreV2:10*(
_class
loc:@penalty/penalty_param*
validate_shape(*
_output_shapes
: *
use_locking(*
T0
ą
save_28/Assign_11Assignpi/dense/biassave_28/RestoreV2:11*
use_locking(* 
_class
loc:@pi/dense/bias*
T0*
validate_shape(*
_output_shapes	
:
ś
save_28/Assign_12Assignpi/dense/bias/Adamsave_28/RestoreV2:12*
_output_shapes	
:*
validate_shape(*
T0* 
_class
loc:@pi/dense/bias*
use_locking(
¸
save_28/Assign_13Assignpi/dense/bias/Adam_1save_28/RestoreV2:13*
_output_shapes	
:*
T0*
use_locking(*
validate_shape(* 
_class
loc:@pi/dense/bias
š
save_28/Assign_14Assignpi/dense/kernelsave_28/RestoreV2:14*
T0*
validate_shape(*"
_class
loc:@pi/dense/kernel*
use_locking(*
_output_shapes
:	L
ž
save_28/Assign_15Assignpi/dense/kernel/Adamsave_28/RestoreV2:15*"
_class
loc:@pi/dense/kernel*
T0*
_output_shapes
:	L*
validate_shape(*
use_locking(
Ŕ
save_28/Assign_16Assignpi/dense/kernel/Adam_1save_28/RestoreV2:16*
T0*
_output_shapes
:	L*
use_locking(*
validate_shape(*"
_class
loc:@pi/dense/kernel
ľ
save_28/Assign_17Assignpi/dense_1/biassave_28/RestoreV2:17*
use_locking(*
_output_shapes	
:*
T0*
validate_shape(*"
_class
loc:@pi/dense_1/bias
ş
save_28/Assign_18Assignpi/dense_1/bias/Adamsave_28/RestoreV2:18*
_output_shapes	
:*
use_locking(*"
_class
loc:@pi/dense_1/bias*
validate_shape(*
T0
ź
save_28/Assign_19Assignpi/dense_1/bias/Adam_1save_28/RestoreV2:19*
validate_shape(*
use_locking(*"
_class
loc:@pi/dense_1/bias*
_output_shapes	
:*
T0
ž
save_28/Assign_20Assignpi/dense_1/kernelsave_28/RestoreV2:20*
validate_shape(*
T0*$
_class
loc:@pi/dense_1/kernel*
use_locking(* 
_output_shapes
:

Ă
save_28/Assign_21Assignpi/dense_1/kernel/Adamsave_28/RestoreV2:21* 
_output_shapes
:
*
validate_shape(*$
_class
loc:@pi/dense_1/kernel*
use_locking(*
T0
Ĺ
save_28/Assign_22Assignpi/dense_1/kernel/Adam_1save_28/RestoreV2:22*
use_locking(*
T0* 
_output_shapes
:
*$
_class
loc:@pi/dense_1/kernel*
validate_shape(
´
save_28/Assign_23Assignpi/dense_2/biassave_28/RestoreV2:23*
validate_shape(*"
_class
loc:@pi/dense_2/bias*
_output_shapes
:*
T0*
use_locking(
š
save_28/Assign_24Assignpi/dense_2/bias/Adamsave_28/RestoreV2:24*
use_locking(*
_output_shapes
:*"
_class
loc:@pi/dense_2/bias*
T0*
validate_shape(
ť
save_28/Assign_25Assignpi/dense_2/bias/Adam_1save_28/RestoreV2:25*
validate_shape(*
T0*"
_class
loc:@pi/dense_2/bias*
_output_shapes
:*
use_locking(
˝
save_28/Assign_26Assignpi/dense_2/kernelsave_28/RestoreV2:26*
_output_shapes
:	*
T0*
use_locking(*
validate_shape(*$
_class
loc:@pi/dense_2/kernel
Â
save_28/Assign_27Assignpi/dense_2/kernel/Adamsave_28/RestoreV2:27*$
_class
loc:@pi/dense_2/kernel*
use_locking(*
_output_shapes
:	*
validate_shape(*
T0
Ä
save_28/Assign_28Assignpi/dense_2/kernel/Adam_1save_28/RestoreV2:28*$
_class
loc:@pi/dense_2/kernel*
T0*
_output_shapes
:	*
use_locking(*
validate_shape(
Ş
save_28/Assign_29Assign
pi/log_stdsave_28/RestoreV2:29*
T0*
_output_shapes
:*
_class
loc:@pi/log_std*
use_locking(*
validate_shape(
Ż
save_28/Assign_30Assignpi/log_std/Adamsave_28/RestoreV2:30*
validate_shape(*
_output_shapes
:*
T0*
use_locking(*
_class
loc:@pi/log_std
ą
save_28/Assign_31Assignpi/log_std/Adam_1save_28/RestoreV2:31*
T0*
_output_shapes
:*
_class
loc:@pi/log_std*
use_locking(*
validate_shape(
ą
save_28/Assign_32Assignvc/dense/biassave_28/RestoreV2:32*
use_locking(*
validate_shape(*
_output_shapes	
:* 
_class
loc:@vc/dense/bias*
T0
ś
save_28/Assign_33Assignvc/dense/bias/Adamsave_28/RestoreV2:33*
T0*
_output_shapes	
:* 
_class
loc:@vc/dense/bias*
use_locking(*
validate_shape(
¸
save_28/Assign_34Assignvc/dense/bias/Adam_1save_28/RestoreV2:34*
use_locking(*
validate_shape(*
_output_shapes	
:* 
_class
loc:@vc/dense/bias*
T0
š
save_28/Assign_35Assignvc/dense/kernelsave_28/RestoreV2:35*
validate_shape(*
use_locking(*
_output_shapes
:	L*
T0*"
_class
loc:@vc/dense/kernel
ž
save_28/Assign_36Assignvc/dense/kernel/Adamsave_28/RestoreV2:36*"
_class
loc:@vc/dense/kernel*
_output_shapes
:	L*
use_locking(*
T0*
validate_shape(
Ŕ
save_28/Assign_37Assignvc/dense/kernel/Adam_1save_28/RestoreV2:37*
T0*
_output_shapes
:	L*
validate_shape(*"
_class
loc:@vc/dense/kernel*
use_locking(
ľ
save_28/Assign_38Assignvc/dense_1/biassave_28/RestoreV2:38*
use_locking(*
T0*"
_class
loc:@vc/dense_1/bias*
_output_shapes	
:*
validate_shape(
ş
save_28/Assign_39Assignvc/dense_1/bias/Adamsave_28/RestoreV2:39*
validate_shape(*
_output_shapes	
:*"
_class
loc:@vc/dense_1/bias*
T0*
use_locking(
ź
save_28/Assign_40Assignvc/dense_1/bias/Adam_1save_28/RestoreV2:40*
T0*
validate_shape(*
use_locking(*"
_class
loc:@vc/dense_1/bias*
_output_shapes	
:
ž
save_28/Assign_41Assignvc/dense_1/kernelsave_28/RestoreV2:41*
use_locking(*
T0*
validate_shape(*$
_class
loc:@vc/dense_1/kernel* 
_output_shapes
:

Ă
save_28/Assign_42Assignvc/dense_1/kernel/Adamsave_28/RestoreV2:42*
validate_shape(* 
_output_shapes
:
*
use_locking(*$
_class
loc:@vc/dense_1/kernel*
T0
Ĺ
save_28/Assign_43Assignvc/dense_1/kernel/Adam_1save_28/RestoreV2:43*$
_class
loc:@vc/dense_1/kernel*
T0*
validate_shape(* 
_output_shapes
:
*
use_locking(
´
save_28/Assign_44Assignvc/dense_2/biassave_28/RestoreV2:44*
use_locking(*
_output_shapes
:*
validate_shape(*
T0*"
_class
loc:@vc/dense_2/bias
š
save_28/Assign_45Assignvc/dense_2/bias/Adamsave_28/RestoreV2:45*"
_class
loc:@vc/dense_2/bias*
_output_shapes
:*
use_locking(*
T0*
validate_shape(
ť
save_28/Assign_46Assignvc/dense_2/bias/Adam_1save_28/RestoreV2:46*
T0*
validate_shape(*
_output_shapes
:*"
_class
loc:@vc/dense_2/bias*
use_locking(
˝
save_28/Assign_47Assignvc/dense_2/kernelsave_28/RestoreV2:47*
_output_shapes
:	*$
_class
loc:@vc/dense_2/kernel*
validate_shape(*
T0*
use_locking(
Â
save_28/Assign_48Assignvc/dense_2/kernel/Adamsave_28/RestoreV2:48*$
_class
loc:@vc/dense_2/kernel*
T0*
use_locking(*
validate_shape(*
_output_shapes
:	
Ä
save_28/Assign_49Assignvc/dense_2/kernel/Adam_1save_28/RestoreV2:49*
T0*
validate_shape(*$
_class
loc:@vc/dense_2/kernel*
use_locking(*
_output_shapes
:	
ą
save_28/Assign_50Assignvf/dense/biassave_28/RestoreV2:50*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0* 
_class
loc:@vf/dense/bias
ś
save_28/Assign_51Assignvf/dense/bias/Adamsave_28/RestoreV2:51* 
_class
loc:@vf/dense/bias*
T0*
use_locking(*
_output_shapes	
:*
validate_shape(
¸
save_28/Assign_52Assignvf/dense/bias/Adam_1save_28/RestoreV2:52*
validate_shape(*
T0*
use_locking(* 
_class
loc:@vf/dense/bias*
_output_shapes	
:
š
save_28/Assign_53Assignvf/dense/kernelsave_28/RestoreV2:53*
validate_shape(*"
_class
loc:@vf/dense/kernel*
_output_shapes
:	L*
T0*
use_locking(
ž
save_28/Assign_54Assignvf/dense/kernel/Adamsave_28/RestoreV2:54*
T0*
validate_shape(*
use_locking(*
_output_shapes
:	L*"
_class
loc:@vf/dense/kernel
Ŕ
save_28/Assign_55Assignvf/dense/kernel/Adam_1save_28/RestoreV2:55*
validate_shape(*
T0*"
_class
loc:@vf/dense/kernel*
use_locking(*
_output_shapes
:	L
ľ
save_28/Assign_56Assignvf/dense_1/biassave_28/RestoreV2:56*
validate_shape(*
_output_shapes	
:*"
_class
loc:@vf/dense_1/bias*
T0*
use_locking(
ş
save_28/Assign_57Assignvf/dense_1/bias/Adamsave_28/RestoreV2:57*
T0*
use_locking(*
_output_shapes	
:*"
_class
loc:@vf/dense_1/bias*
validate_shape(
ź
save_28/Assign_58Assignvf/dense_1/bias/Adam_1save_28/RestoreV2:58*
use_locking(*
_output_shapes	
:*
validate_shape(*
T0*"
_class
loc:@vf/dense_1/bias
ž
save_28/Assign_59Assignvf/dense_1/kernelsave_28/RestoreV2:59*
use_locking(* 
_output_shapes
:
*
T0*$
_class
loc:@vf/dense_1/kernel*
validate_shape(
Ă
save_28/Assign_60Assignvf/dense_1/kernel/Adamsave_28/RestoreV2:60* 
_output_shapes
:
*
validate_shape(*$
_class
loc:@vf/dense_1/kernel*
T0*
use_locking(
Ĺ
save_28/Assign_61Assignvf/dense_1/kernel/Adam_1save_28/RestoreV2:61*
validate_shape(*
use_locking(*
T0* 
_output_shapes
:
*$
_class
loc:@vf/dense_1/kernel
´
save_28/Assign_62Assignvf/dense_2/biassave_28/RestoreV2:62*
_output_shapes
:*
use_locking(*
T0*
validate_shape(*"
_class
loc:@vf/dense_2/bias
š
save_28/Assign_63Assignvf/dense_2/bias/Adamsave_28/RestoreV2:63*
_output_shapes
:*
T0*"
_class
loc:@vf/dense_2/bias*
use_locking(*
validate_shape(
ť
save_28/Assign_64Assignvf/dense_2/bias/Adam_1save_28/RestoreV2:64*
use_locking(*
T0*
validate_shape(*"
_class
loc:@vf/dense_2/bias*
_output_shapes
:
˝
save_28/Assign_65Assignvf/dense_2/kernelsave_28/RestoreV2:65*
validate_shape(*
use_locking(*
T0*
_output_shapes
:	*$
_class
loc:@vf/dense_2/kernel
Â
save_28/Assign_66Assignvf/dense_2/kernel/Adamsave_28/RestoreV2:66*
use_locking(*$
_class
loc:@vf/dense_2/kernel*
T0*
validate_shape(*
_output_shapes
:	
Ä
save_28/Assign_67Assignvf/dense_2/kernel/Adam_1save_28/RestoreV2:67*$
_class
loc:@vf/dense_2/kernel*
use_locking(*
_output_shapes
:	*
T0*
validate_shape(
á

save_28/restore_shardNoOp^save_28/Assign^save_28/Assign_1^save_28/Assign_10^save_28/Assign_11^save_28/Assign_12^save_28/Assign_13^save_28/Assign_14^save_28/Assign_15^save_28/Assign_16^save_28/Assign_17^save_28/Assign_18^save_28/Assign_19^save_28/Assign_2^save_28/Assign_20^save_28/Assign_21^save_28/Assign_22^save_28/Assign_23^save_28/Assign_24^save_28/Assign_25^save_28/Assign_26^save_28/Assign_27^save_28/Assign_28^save_28/Assign_29^save_28/Assign_3^save_28/Assign_30^save_28/Assign_31^save_28/Assign_32^save_28/Assign_33^save_28/Assign_34^save_28/Assign_35^save_28/Assign_36^save_28/Assign_37^save_28/Assign_38^save_28/Assign_39^save_28/Assign_4^save_28/Assign_40^save_28/Assign_41^save_28/Assign_42^save_28/Assign_43^save_28/Assign_44^save_28/Assign_45^save_28/Assign_46^save_28/Assign_47^save_28/Assign_48^save_28/Assign_49^save_28/Assign_5^save_28/Assign_50^save_28/Assign_51^save_28/Assign_52^save_28/Assign_53^save_28/Assign_54^save_28/Assign_55^save_28/Assign_56^save_28/Assign_57^save_28/Assign_58^save_28/Assign_59^save_28/Assign_6^save_28/Assign_60^save_28/Assign_61^save_28/Assign_62^save_28/Assign_63^save_28/Assign_64^save_28/Assign_65^save_28/Assign_66^save_28/Assign_67^save_28/Assign_7^save_28/Assign_8^save_28/Assign_9
3
save_28/restore_allNoOp^save_28/restore_shard
\
save_29/filename/inputConst*
_output_shapes
: *
valueB Bmodel*
dtype0
t
save_29/filenamePlaceholderWithDefaultsave_29/filename/input*
shape: *
_output_shapes
: *
dtype0
k
save_29/ConstPlaceholderWithDefaultsave_29/filename*
dtype0*
shape: *
_output_shapes
: 

save_29/StringJoin/inputs_1Const*<
value3B1 B+_temp_7293e55fa9554665a1a184dd7f07a85d/part*
dtype0*
_output_shapes
: 
~
save_29/StringJoin
StringJoinsave_29/Constsave_29/StringJoin/inputs_1*
N*
	separator *
_output_shapes
: 
T
save_29/num_shardsConst*
value	B :*
_output_shapes
: *
dtype0
_
save_29/ShardedFilename/shardConst*
value	B : *
_output_shapes
: *
dtype0

save_29/ShardedFilenameShardedFilenamesave_29/StringJoinsave_29/ShardedFilename/shardsave_29/num_shards*
_output_shapes
: 
ř
save_29/SaveV2/tensor_namesConst*
_output_shapes
:D*
dtype0*¨
valueBDBbeta1_powerBbeta1_power_1Bbeta1_power_2Bbeta2_powerBbeta2_power_1Bbeta2_power_2Bpenalty/penalty_paramBpenalty/penalty_param/AdamBpenalty/penalty_param/Adam_1Bpenalty/penalty_param/Adam_2Bpenalty/penalty_param/Adam_3Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1B
pi/log_stdBpi/log_std/AdamBpi/log_std/Adam_1Bvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1
ń
save_29/SaveV2/shape_and_slicesConst*
dtype0*
valueBDB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
_output_shapes
:D
Ö
save_29/SaveV2SaveV2save_29/ShardedFilenamesave_29/SaveV2/tensor_namessave_29/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta1_power_2beta2_powerbeta2_power_1beta2_power_2penalty/penalty_parampenalty/penalty_param/Adampenalty/penalty_param/Adam_1penalty/penalty_param/Adam_2penalty/penalty_param/Adam_3pi/dense/biaspi/dense/bias/Adampi/dense/bias/Adam_1pi/dense/kernelpi/dense/kernel/Adampi/dense/kernel/Adam_1pi/dense_1/biaspi/dense_1/bias/Adampi/dense_1/bias/Adam_1pi/dense_1/kernelpi/dense_1/kernel/Adampi/dense_1/kernel/Adam_1pi/dense_2/biaspi/dense_2/bias/Adampi/dense_2/bias/Adam_1pi/dense_2/kernelpi/dense_2/kernel/Adampi/dense_2/kernel/Adam_1
pi/log_stdpi/log_std/Adampi/log_std/Adam_1vc/dense/biasvc/dense/bias/Adamvc/dense/bias/Adam_1vc/dense/kernelvc/dense/kernel/Adamvc/dense/kernel/Adam_1vc/dense_1/biasvc/dense_1/bias/Adamvc/dense_1/bias/Adam_1vc/dense_1/kernelvc/dense_1/kernel/Adamvc/dense_1/kernel/Adam_1vc/dense_2/biasvc/dense_2/bias/Adamvc/dense_2/bias/Adam_1vc/dense_2/kernelvc/dense_2/kernel/Adamvc/dense_2/kernel/Adam_1vf/dense/biasvf/dense/bias/Adamvf/dense/bias/Adam_1vf/dense/kernelvf/dense/kernel/Adamvf/dense/kernel/Adam_1vf/dense_1/biasvf/dense_1/bias/Adamvf/dense_1/bias/Adam_1vf/dense_1/kernelvf/dense_1/kernel/Adamvf/dense_1/kernel/Adam_1vf/dense_2/biasvf/dense_2/bias/Adamvf/dense_2/bias/Adam_1vf/dense_2/kernelvf/dense_2/kernel/Adamvf/dense_2/kernel/Adam_1*R
dtypesH
F2D

save_29/control_dependencyIdentitysave_29/ShardedFilename^save_29/SaveV2**
_class 
loc:@save_29/ShardedFilename*
T0*
_output_shapes
: 
Ś
.save_29/MergeV2Checkpoints/checkpoint_prefixesPacksave_29/ShardedFilename^save_29/control_dependency*
N*

axis *
_output_shapes
:*
T0

save_29/MergeV2CheckpointsMergeV2Checkpoints.save_29/MergeV2Checkpoints/checkpoint_prefixessave_29/Const*
delete_old_dirs(

save_29/IdentityIdentitysave_29/Const^save_29/MergeV2Checkpoints^save_29/control_dependency*
_output_shapes
: *
T0
ű
save_29/RestoreV2/tensor_namesConst*
dtype0*¨
valueBDBbeta1_powerBbeta1_power_1Bbeta1_power_2Bbeta2_powerBbeta2_power_1Bbeta2_power_2Bpenalty/penalty_paramBpenalty/penalty_param/AdamBpenalty/penalty_param/Adam_1Bpenalty/penalty_param/Adam_2Bpenalty/penalty_param/Adam_3Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1B
pi/log_stdBpi/log_std/AdamBpi/log_std/Adam_1Bvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1*
_output_shapes
:D
ô
"save_29/RestoreV2/shape_and_slicesConst*
valueBDB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
_output_shapes
:D*
dtype0
î
save_29/RestoreV2	RestoreV2save_29/Constsave_29/RestoreV2/tensor_names"save_29/RestoreV2/shape_and_slices*Ś
_output_shapes
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*R
dtypesH
F2D
Ź
save_29/AssignAssignbeta1_powersave_29/RestoreV2*
_output_shapes
: *
T0*(
_class
loc:@penalty/penalty_param*
use_locking(*
validate_shape(
˛
save_29/Assign_1Assignbeta1_power_1save_29/RestoreV2:1*
T0*(
_class
loc:@penalty/penalty_param*
validate_shape(*
use_locking(*
_output_shapes
: 
Ş
save_29/Assign_2Assignbeta1_power_2save_29/RestoreV2:2*
use_locking(* 
_class
loc:@vc/dense/bias*
T0*
validate_shape(*
_output_shapes
: 
°
save_29/Assign_3Assignbeta2_powersave_29/RestoreV2:3*(
_class
loc:@penalty/penalty_param*
use_locking(*
T0*
_output_shapes
: *
validate_shape(
˛
save_29/Assign_4Assignbeta2_power_1save_29/RestoreV2:4*
use_locking(*
validate_shape(*
T0*(
_class
loc:@penalty/penalty_param*
_output_shapes
: 
Ş
save_29/Assign_5Assignbeta2_power_2save_29/RestoreV2:5*
T0*
use_locking(* 
_class
loc:@vc/dense/bias*
validate_shape(*
_output_shapes
: 
ş
save_29/Assign_6Assignpenalty/penalty_paramsave_29/RestoreV2:6*
validate_shape(*
T0*
_output_shapes
: *(
_class
loc:@penalty/penalty_param*
use_locking(
ż
save_29/Assign_7Assignpenalty/penalty_param/Adamsave_29/RestoreV2:7*(
_class
loc:@penalty/penalty_param*
T0*
validate_shape(*
_output_shapes
: *
use_locking(
Á
save_29/Assign_8Assignpenalty/penalty_param/Adam_1save_29/RestoreV2:8*
use_locking(*
validate_shape(*
T0*
_output_shapes
: *(
_class
loc:@penalty/penalty_param
Á
save_29/Assign_9Assignpenalty/penalty_param/Adam_2save_29/RestoreV2:9*
_output_shapes
: *
use_locking(*(
_class
loc:@penalty/penalty_param*
T0*
validate_shape(
Ă
save_29/Assign_10Assignpenalty/penalty_param/Adam_3save_29/RestoreV2:10*
use_locking(*
T0*
_output_shapes
: *(
_class
loc:@penalty/penalty_param*
validate_shape(
ą
save_29/Assign_11Assignpi/dense/biassave_29/RestoreV2:11* 
_class
loc:@pi/dense/bias*
_output_shapes	
:*
validate_shape(*
use_locking(*
T0
ś
save_29/Assign_12Assignpi/dense/bias/Adamsave_29/RestoreV2:12*
use_locking(*
T0*
_output_shapes	
:*
validate_shape(* 
_class
loc:@pi/dense/bias
¸
save_29/Assign_13Assignpi/dense/bias/Adam_1save_29/RestoreV2:13*
T0*
use_locking(* 
_class
loc:@pi/dense/bias*
_output_shapes	
:*
validate_shape(
š
save_29/Assign_14Assignpi/dense/kernelsave_29/RestoreV2:14*
use_locking(*
T0*
_output_shapes
:	L*"
_class
loc:@pi/dense/kernel*
validate_shape(
ž
save_29/Assign_15Assignpi/dense/kernel/Adamsave_29/RestoreV2:15*
use_locking(*
T0*
_output_shapes
:	L*
validate_shape(*"
_class
loc:@pi/dense/kernel
Ŕ
save_29/Assign_16Assignpi/dense/kernel/Adam_1save_29/RestoreV2:16*"
_class
loc:@pi/dense/kernel*
_output_shapes
:	L*
use_locking(*
validate_shape(*
T0
ľ
save_29/Assign_17Assignpi/dense_1/biassave_29/RestoreV2:17*
_output_shapes	
:*
validate_shape(*"
_class
loc:@pi/dense_1/bias*
use_locking(*
T0
ş
save_29/Assign_18Assignpi/dense_1/bias/Adamsave_29/RestoreV2:18*
validate_shape(*
_output_shapes	
:*"
_class
loc:@pi/dense_1/bias*
use_locking(*
T0
ź
save_29/Assign_19Assignpi/dense_1/bias/Adam_1save_29/RestoreV2:19*"
_class
loc:@pi/dense_1/bias*
_output_shapes	
:*
validate_shape(*
use_locking(*
T0
ž
save_29/Assign_20Assignpi/dense_1/kernelsave_29/RestoreV2:20* 
_output_shapes
:
*
validate_shape(*
use_locking(*
T0*$
_class
loc:@pi/dense_1/kernel
Ă
save_29/Assign_21Assignpi/dense_1/kernel/Adamsave_29/RestoreV2:21*
use_locking(*$
_class
loc:@pi/dense_1/kernel* 
_output_shapes
:
*
T0*
validate_shape(
Ĺ
save_29/Assign_22Assignpi/dense_1/kernel/Adam_1save_29/RestoreV2:22* 
_output_shapes
:
*$
_class
loc:@pi/dense_1/kernel*
use_locking(*
validate_shape(*
T0
´
save_29/Assign_23Assignpi/dense_2/biassave_29/RestoreV2:23*
T0*
_output_shapes
:*
use_locking(*
validate_shape(*"
_class
loc:@pi/dense_2/bias
š
save_29/Assign_24Assignpi/dense_2/bias/Adamsave_29/RestoreV2:24*
T0*
validate_shape(*"
_class
loc:@pi/dense_2/bias*
use_locking(*
_output_shapes
:
ť
save_29/Assign_25Assignpi/dense_2/bias/Adam_1save_29/RestoreV2:25*
use_locking(*
_output_shapes
:*
T0*"
_class
loc:@pi/dense_2/bias*
validate_shape(
˝
save_29/Assign_26Assignpi/dense_2/kernelsave_29/RestoreV2:26*
use_locking(*$
_class
loc:@pi/dense_2/kernel*
_output_shapes
:	*
T0*
validate_shape(
Â
save_29/Assign_27Assignpi/dense_2/kernel/Adamsave_29/RestoreV2:27*
validate_shape(*$
_class
loc:@pi/dense_2/kernel*
use_locking(*
_output_shapes
:	*
T0
Ä
save_29/Assign_28Assignpi/dense_2/kernel/Adam_1save_29/RestoreV2:28*
_output_shapes
:	*$
_class
loc:@pi/dense_2/kernel*
validate_shape(*
use_locking(*
T0
Ş
save_29/Assign_29Assign
pi/log_stdsave_29/RestoreV2:29*
use_locking(*
T0*
validate_shape(*
_class
loc:@pi/log_std*
_output_shapes
:
Ż
save_29/Assign_30Assignpi/log_std/Adamsave_29/RestoreV2:30*
_class
loc:@pi/log_std*
T0*
validate_shape(*
use_locking(*
_output_shapes
:
ą
save_29/Assign_31Assignpi/log_std/Adam_1save_29/RestoreV2:31*
validate_shape(*
use_locking(*
T0*
_class
loc:@pi/log_std*
_output_shapes
:
ą
save_29/Assign_32Assignvc/dense/biassave_29/RestoreV2:32*
use_locking(*
T0* 
_class
loc:@vc/dense/bias*
validate_shape(*
_output_shapes	
:
ś
save_29/Assign_33Assignvc/dense/bias/Adamsave_29/RestoreV2:33* 
_class
loc:@vc/dense/bias*
T0*
_output_shapes	
:*
validate_shape(*
use_locking(
¸
save_29/Assign_34Assignvc/dense/bias/Adam_1save_29/RestoreV2:34*
T0*
_output_shapes	
:*
use_locking(*
validate_shape(* 
_class
loc:@vc/dense/bias
š
save_29/Assign_35Assignvc/dense/kernelsave_29/RestoreV2:35*
_output_shapes
:	L*
validate_shape(*"
_class
loc:@vc/dense/kernel*
use_locking(*
T0
ž
save_29/Assign_36Assignvc/dense/kernel/Adamsave_29/RestoreV2:36*
T0*
use_locking(*"
_class
loc:@vc/dense/kernel*
validate_shape(*
_output_shapes
:	L
Ŕ
save_29/Assign_37Assignvc/dense/kernel/Adam_1save_29/RestoreV2:37*"
_class
loc:@vc/dense/kernel*
validate_shape(*
_output_shapes
:	L*
use_locking(*
T0
ľ
save_29/Assign_38Assignvc/dense_1/biassave_29/RestoreV2:38*
validate_shape(*
T0*"
_class
loc:@vc/dense_1/bias*
use_locking(*
_output_shapes	
:
ş
save_29/Assign_39Assignvc/dense_1/bias/Adamsave_29/RestoreV2:39*"
_class
loc:@vc/dense_1/bias*
use_locking(*
_output_shapes	
:*
validate_shape(*
T0
ź
save_29/Assign_40Assignvc/dense_1/bias/Adam_1save_29/RestoreV2:40*
_output_shapes	
:*
use_locking(*"
_class
loc:@vc/dense_1/bias*
T0*
validate_shape(
ž
save_29/Assign_41Assignvc/dense_1/kernelsave_29/RestoreV2:41*
T0*$
_class
loc:@vc/dense_1/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(
Ă
save_29/Assign_42Assignvc/dense_1/kernel/Adamsave_29/RestoreV2:42* 
_output_shapes
:
*
T0*$
_class
loc:@vc/dense_1/kernel*
validate_shape(*
use_locking(
Ĺ
save_29/Assign_43Assignvc/dense_1/kernel/Adam_1save_29/RestoreV2:43*
validate_shape(*
T0*$
_class
loc:@vc/dense_1/kernel*
use_locking(* 
_output_shapes
:

´
save_29/Assign_44Assignvc/dense_2/biassave_29/RestoreV2:44*
use_locking(*"
_class
loc:@vc/dense_2/bias*
validate_shape(*
_output_shapes
:*
T0
š
save_29/Assign_45Assignvc/dense_2/bias/Adamsave_29/RestoreV2:45*
_output_shapes
:*
T0*
use_locking(*"
_class
loc:@vc/dense_2/bias*
validate_shape(
ť
save_29/Assign_46Assignvc/dense_2/bias/Adam_1save_29/RestoreV2:46*
validate_shape(*
use_locking(*"
_class
loc:@vc/dense_2/bias*
_output_shapes
:*
T0
˝
save_29/Assign_47Assignvc/dense_2/kernelsave_29/RestoreV2:47*
validate_shape(*$
_class
loc:@vc/dense_2/kernel*
_output_shapes
:	*
T0*
use_locking(
Â
save_29/Assign_48Assignvc/dense_2/kernel/Adamsave_29/RestoreV2:48*
use_locking(*
validate_shape(*
_output_shapes
:	*
T0*$
_class
loc:@vc/dense_2/kernel
Ä
save_29/Assign_49Assignvc/dense_2/kernel/Adam_1save_29/RestoreV2:49*
use_locking(*
validate_shape(*$
_class
loc:@vc/dense_2/kernel*
T0*
_output_shapes
:	
ą
save_29/Assign_50Assignvf/dense/biassave_29/RestoreV2:50*
T0*
validate_shape(* 
_class
loc:@vf/dense/bias*
_output_shapes	
:*
use_locking(
ś
save_29/Assign_51Assignvf/dense/bias/Adamsave_29/RestoreV2:51* 
_class
loc:@vf/dense/bias*
validate_shape(*
use_locking(*
T0*
_output_shapes	
:
¸
save_29/Assign_52Assignvf/dense/bias/Adam_1save_29/RestoreV2:52*
T0*
use_locking(*
_output_shapes	
:*
validate_shape(* 
_class
loc:@vf/dense/bias
š
save_29/Assign_53Assignvf/dense/kernelsave_29/RestoreV2:53*
use_locking(*
validate_shape(*
_output_shapes
:	L*
T0*"
_class
loc:@vf/dense/kernel
ž
save_29/Assign_54Assignvf/dense/kernel/Adamsave_29/RestoreV2:54*"
_class
loc:@vf/dense/kernel*
T0*
validate_shape(*
_output_shapes
:	L*
use_locking(
Ŕ
save_29/Assign_55Assignvf/dense/kernel/Adam_1save_29/RestoreV2:55*
validate_shape(*"
_class
loc:@vf/dense/kernel*
_output_shapes
:	L*
T0*
use_locking(
ľ
save_29/Assign_56Assignvf/dense_1/biassave_29/RestoreV2:56*
use_locking(*
validate_shape(*
_output_shapes	
:*
T0*"
_class
loc:@vf/dense_1/bias
ş
save_29/Assign_57Assignvf/dense_1/bias/Adamsave_29/RestoreV2:57*
T0*
use_locking(*"
_class
loc:@vf/dense_1/bias*
validate_shape(*
_output_shapes	
:
ź
save_29/Assign_58Assignvf/dense_1/bias/Adam_1save_29/RestoreV2:58*
validate_shape(*
use_locking(*"
_class
loc:@vf/dense_1/bias*
_output_shapes	
:*
T0
ž
save_29/Assign_59Assignvf/dense_1/kernelsave_29/RestoreV2:59*
validate_shape(*$
_class
loc:@vf/dense_1/kernel*
T0*
use_locking(* 
_output_shapes
:

Ă
save_29/Assign_60Assignvf/dense_1/kernel/Adamsave_29/RestoreV2:60*$
_class
loc:@vf/dense_1/kernel* 
_output_shapes
:
*
validate_shape(*
use_locking(*
T0
Ĺ
save_29/Assign_61Assignvf/dense_1/kernel/Adam_1save_29/RestoreV2:61*
use_locking(*
T0*
validate_shape(* 
_output_shapes
:
*$
_class
loc:@vf/dense_1/kernel
´
save_29/Assign_62Assignvf/dense_2/biassave_29/RestoreV2:62*
validate_shape(*"
_class
loc:@vf/dense_2/bias*
use_locking(*
T0*
_output_shapes
:
š
save_29/Assign_63Assignvf/dense_2/bias/Adamsave_29/RestoreV2:63*"
_class
loc:@vf/dense_2/bias*
T0*
_output_shapes
:*
validate_shape(*
use_locking(
ť
save_29/Assign_64Assignvf/dense_2/bias/Adam_1save_29/RestoreV2:64*
use_locking(*
T0*
validate_shape(*"
_class
loc:@vf/dense_2/bias*
_output_shapes
:
˝
save_29/Assign_65Assignvf/dense_2/kernelsave_29/RestoreV2:65*
_output_shapes
:	*$
_class
loc:@vf/dense_2/kernel*
use_locking(*
validate_shape(*
T0
Â
save_29/Assign_66Assignvf/dense_2/kernel/Adamsave_29/RestoreV2:66*
use_locking(*
T0*$
_class
loc:@vf/dense_2/kernel*
validate_shape(*
_output_shapes
:	
Ä
save_29/Assign_67Assignvf/dense_2/kernel/Adam_1save_29/RestoreV2:67*
T0*
_output_shapes
:	*$
_class
loc:@vf/dense_2/kernel*
validate_shape(*
use_locking(
á

save_29/restore_shardNoOp^save_29/Assign^save_29/Assign_1^save_29/Assign_10^save_29/Assign_11^save_29/Assign_12^save_29/Assign_13^save_29/Assign_14^save_29/Assign_15^save_29/Assign_16^save_29/Assign_17^save_29/Assign_18^save_29/Assign_19^save_29/Assign_2^save_29/Assign_20^save_29/Assign_21^save_29/Assign_22^save_29/Assign_23^save_29/Assign_24^save_29/Assign_25^save_29/Assign_26^save_29/Assign_27^save_29/Assign_28^save_29/Assign_29^save_29/Assign_3^save_29/Assign_30^save_29/Assign_31^save_29/Assign_32^save_29/Assign_33^save_29/Assign_34^save_29/Assign_35^save_29/Assign_36^save_29/Assign_37^save_29/Assign_38^save_29/Assign_39^save_29/Assign_4^save_29/Assign_40^save_29/Assign_41^save_29/Assign_42^save_29/Assign_43^save_29/Assign_44^save_29/Assign_45^save_29/Assign_46^save_29/Assign_47^save_29/Assign_48^save_29/Assign_49^save_29/Assign_5^save_29/Assign_50^save_29/Assign_51^save_29/Assign_52^save_29/Assign_53^save_29/Assign_54^save_29/Assign_55^save_29/Assign_56^save_29/Assign_57^save_29/Assign_58^save_29/Assign_59^save_29/Assign_6^save_29/Assign_60^save_29/Assign_61^save_29/Assign_62^save_29/Assign_63^save_29/Assign_64^save_29/Assign_65^save_29/Assign_66^save_29/Assign_67^save_29/Assign_7^save_29/Assign_8^save_29/Assign_9
3
save_29/restore_allNoOp^save_29/restore_shard
\
save_30/filename/inputConst*
dtype0*
_output_shapes
: *
valueB Bmodel
t
save_30/filenamePlaceholderWithDefaultsave_30/filename/input*
_output_shapes
: *
dtype0*
shape: 
k
save_30/ConstPlaceholderWithDefaultsave_30/filename*
shape: *
_output_shapes
: *
dtype0

save_30/StringJoin/inputs_1Const*
dtype0*
_output_shapes
: *<
value3B1 B+_temp_34a876973983442b81aa623df5738477/part
~
save_30/StringJoin
StringJoinsave_30/Constsave_30/StringJoin/inputs_1*
_output_shapes
: *
	separator *
N
T
save_30/num_shardsConst*
_output_shapes
: *
dtype0*
value	B :
_
save_30/ShardedFilename/shardConst*
dtype0*
_output_shapes
: *
value	B : 

save_30/ShardedFilenameShardedFilenamesave_30/StringJoinsave_30/ShardedFilename/shardsave_30/num_shards*
_output_shapes
: 
ř
save_30/SaveV2/tensor_namesConst*
_output_shapes
:D*¨
valueBDBbeta1_powerBbeta1_power_1Bbeta1_power_2Bbeta2_powerBbeta2_power_1Bbeta2_power_2Bpenalty/penalty_paramBpenalty/penalty_param/AdamBpenalty/penalty_param/Adam_1Bpenalty/penalty_param/Adam_2Bpenalty/penalty_param/Adam_3Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1B
pi/log_stdBpi/log_std/AdamBpi/log_std/Adam_1Bvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1*
dtype0
ń
save_30/SaveV2/shape_and_slicesConst*
valueBDB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:D
Ö
save_30/SaveV2SaveV2save_30/ShardedFilenamesave_30/SaveV2/tensor_namessave_30/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta1_power_2beta2_powerbeta2_power_1beta2_power_2penalty/penalty_parampenalty/penalty_param/Adampenalty/penalty_param/Adam_1penalty/penalty_param/Adam_2penalty/penalty_param/Adam_3pi/dense/biaspi/dense/bias/Adampi/dense/bias/Adam_1pi/dense/kernelpi/dense/kernel/Adampi/dense/kernel/Adam_1pi/dense_1/biaspi/dense_1/bias/Adampi/dense_1/bias/Adam_1pi/dense_1/kernelpi/dense_1/kernel/Adampi/dense_1/kernel/Adam_1pi/dense_2/biaspi/dense_2/bias/Adampi/dense_2/bias/Adam_1pi/dense_2/kernelpi/dense_2/kernel/Adampi/dense_2/kernel/Adam_1
pi/log_stdpi/log_std/Adampi/log_std/Adam_1vc/dense/biasvc/dense/bias/Adamvc/dense/bias/Adam_1vc/dense/kernelvc/dense/kernel/Adamvc/dense/kernel/Adam_1vc/dense_1/biasvc/dense_1/bias/Adamvc/dense_1/bias/Adam_1vc/dense_1/kernelvc/dense_1/kernel/Adamvc/dense_1/kernel/Adam_1vc/dense_2/biasvc/dense_2/bias/Adamvc/dense_2/bias/Adam_1vc/dense_2/kernelvc/dense_2/kernel/Adamvc/dense_2/kernel/Adam_1vf/dense/biasvf/dense/bias/Adamvf/dense/bias/Adam_1vf/dense/kernelvf/dense/kernel/Adamvf/dense/kernel/Adam_1vf/dense_1/biasvf/dense_1/bias/Adamvf/dense_1/bias/Adam_1vf/dense_1/kernelvf/dense_1/kernel/Adamvf/dense_1/kernel/Adam_1vf/dense_2/biasvf/dense_2/bias/Adamvf/dense_2/bias/Adam_1vf/dense_2/kernelvf/dense_2/kernel/Adamvf/dense_2/kernel/Adam_1*R
dtypesH
F2D

save_30/control_dependencyIdentitysave_30/ShardedFilename^save_30/SaveV2**
_class 
loc:@save_30/ShardedFilename*
_output_shapes
: *
T0
Ś
.save_30/MergeV2Checkpoints/checkpoint_prefixesPacksave_30/ShardedFilename^save_30/control_dependency*
T0*
_output_shapes
:*
N*

axis 

save_30/MergeV2CheckpointsMergeV2Checkpoints.save_30/MergeV2Checkpoints/checkpoint_prefixessave_30/Const*
delete_old_dirs(

save_30/IdentityIdentitysave_30/Const^save_30/MergeV2Checkpoints^save_30/control_dependency*
T0*
_output_shapes
: 
ű
save_30/RestoreV2/tensor_namesConst*
_output_shapes
:D*
dtype0*¨
valueBDBbeta1_powerBbeta1_power_1Bbeta1_power_2Bbeta2_powerBbeta2_power_1Bbeta2_power_2Bpenalty/penalty_paramBpenalty/penalty_param/AdamBpenalty/penalty_param/Adam_1Bpenalty/penalty_param/Adam_2Bpenalty/penalty_param/Adam_3Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1B
pi/log_stdBpi/log_std/AdamBpi/log_std/Adam_1Bvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1
ô
"save_30/RestoreV2/shape_and_slicesConst*
dtype0*
valueBDB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
_output_shapes
:D
î
save_30/RestoreV2	RestoreV2save_30/Constsave_30/RestoreV2/tensor_names"save_30/RestoreV2/shape_and_slices*Ś
_output_shapes
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*R
dtypesH
F2D
Ź
save_30/AssignAssignbeta1_powersave_30/RestoreV2*
T0*
validate_shape(*
use_locking(*(
_class
loc:@penalty/penalty_param*
_output_shapes
: 
˛
save_30/Assign_1Assignbeta1_power_1save_30/RestoreV2:1*
validate_shape(*
T0*
use_locking(*
_output_shapes
: *(
_class
loc:@penalty/penalty_param
Ş
save_30/Assign_2Assignbeta1_power_2save_30/RestoreV2:2* 
_class
loc:@vc/dense/bias*
use_locking(*
validate_shape(*
_output_shapes
: *
T0
°
save_30/Assign_3Assignbeta2_powersave_30/RestoreV2:3*(
_class
loc:@penalty/penalty_param*
use_locking(*
T0*
validate_shape(*
_output_shapes
: 
˛
save_30/Assign_4Assignbeta2_power_1save_30/RestoreV2:4*
validate_shape(*(
_class
loc:@penalty/penalty_param*
T0*
_output_shapes
: *
use_locking(
Ş
save_30/Assign_5Assignbeta2_power_2save_30/RestoreV2:5*
validate_shape(* 
_class
loc:@vc/dense/bias*
_output_shapes
: *
T0*
use_locking(
ş
save_30/Assign_6Assignpenalty/penalty_paramsave_30/RestoreV2:6*
use_locking(*
_output_shapes
: *(
_class
loc:@penalty/penalty_param*
T0*
validate_shape(
ż
save_30/Assign_7Assignpenalty/penalty_param/Adamsave_30/RestoreV2:7*
T0*(
_class
loc:@penalty/penalty_param*
_output_shapes
: *
use_locking(*
validate_shape(
Á
save_30/Assign_8Assignpenalty/penalty_param/Adam_1save_30/RestoreV2:8*
_output_shapes
: *
T0*(
_class
loc:@penalty/penalty_param*
use_locking(*
validate_shape(
Á
save_30/Assign_9Assignpenalty/penalty_param/Adam_2save_30/RestoreV2:9*
T0*
_output_shapes
: *(
_class
loc:@penalty/penalty_param*
validate_shape(*
use_locking(
Ă
save_30/Assign_10Assignpenalty/penalty_param/Adam_3save_30/RestoreV2:10*(
_class
loc:@penalty/penalty_param*
use_locking(*
T0*
_output_shapes
: *
validate_shape(
ą
save_30/Assign_11Assignpi/dense/biassave_30/RestoreV2:11* 
_class
loc:@pi/dense/bias*
T0*
_output_shapes	
:*
use_locking(*
validate_shape(
ś
save_30/Assign_12Assignpi/dense/bias/Adamsave_30/RestoreV2:12*
validate_shape(* 
_class
loc:@pi/dense/bias*
T0*
use_locking(*
_output_shapes	
:
¸
save_30/Assign_13Assignpi/dense/bias/Adam_1save_30/RestoreV2:13*
T0*
validate_shape(*
_output_shapes	
:*
use_locking(* 
_class
loc:@pi/dense/bias
š
save_30/Assign_14Assignpi/dense/kernelsave_30/RestoreV2:14*
T0*
use_locking(*
validate_shape(*
_output_shapes
:	L*"
_class
loc:@pi/dense/kernel
ž
save_30/Assign_15Assignpi/dense/kernel/Adamsave_30/RestoreV2:15*
_output_shapes
:	L*"
_class
loc:@pi/dense/kernel*
T0*
use_locking(*
validate_shape(
Ŕ
save_30/Assign_16Assignpi/dense/kernel/Adam_1save_30/RestoreV2:16*
T0*"
_class
loc:@pi/dense/kernel*
use_locking(*
_output_shapes
:	L*
validate_shape(
ľ
save_30/Assign_17Assignpi/dense_1/biassave_30/RestoreV2:17*
use_locking(*"
_class
loc:@pi/dense_1/bias*
T0*
validate_shape(*
_output_shapes	
:
ş
save_30/Assign_18Assignpi/dense_1/bias/Adamsave_30/RestoreV2:18*
use_locking(*
validate_shape(*"
_class
loc:@pi/dense_1/bias*
_output_shapes	
:*
T0
ź
save_30/Assign_19Assignpi/dense_1/bias/Adam_1save_30/RestoreV2:19*
_output_shapes	
:*"
_class
loc:@pi/dense_1/bias*
use_locking(*
validate_shape(*
T0
ž
save_30/Assign_20Assignpi/dense_1/kernelsave_30/RestoreV2:20*$
_class
loc:@pi/dense_1/kernel*
use_locking(* 
_output_shapes
:
*
validate_shape(*
T0
Ă
save_30/Assign_21Assignpi/dense_1/kernel/Adamsave_30/RestoreV2:21* 
_output_shapes
:
*
T0*
use_locking(*$
_class
loc:@pi/dense_1/kernel*
validate_shape(
Ĺ
save_30/Assign_22Assignpi/dense_1/kernel/Adam_1save_30/RestoreV2:22*
T0*$
_class
loc:@pi/dense_1/kernel* 
_output_shapes
:
*
validate_shape(*
use_locking(
´
save_30/Assign_23Assignpi/dense_2/biassave_30/RestoreV2:23*
validate_shape(*
T0*
use_locking(*"
_class
loc:@pi/dense_2/bias*
_output_shapes
:
š
save_30/Assign_24Assignpi/dense_2/bias/Adamsave_30/RestoreV2:24*"
_class
loc:@pi/dense_2/bias*
_output_shapes
:*
validate_shape(*
T0*
use_locking(
ť
save_30/Assign_25Assignpi/dense_2/bias/Adam_1save_30/RestoreV2:25*"
_class
loc:@pi/dense_2/bias*
T0*
validate_shape(*
_output_shapes
:*
use_locking(
˝
save_30/Assign_26Assignpi/dense_2/kernelsave_30/RestoreV2:26*
validate_shape(*
_output_shapes
:	*$
_class
loc:@pi/dense_2/kernel*
use_locking(*
T0
Â
save_30/Assign_27Assignpi/dense_2/kernel/Adamsave_30/RestoreV2:27*
T0*
validate_shape(*
use_locking(*
_output_shapes
:	*$
_class
loc:@pi/dense_2/kernel
Ä
save_30/Assign_28Assignpi/dense_2/kernel/Adam_1save_30/RestoreV2:28*
_output_shapes
:	*
use_locking(*
T0*
validate_shape(*$
_class
loc:@pi/dense_2/kernel
Ş
save_30/Assign_29Assign
pi/log_stdsave_30/RestoreV2:29*
validate_shape(*
_class
loc:@pi/log_std*
T0*
_output_shapes
:*
use_locking(
Ż
save_30/Assign_30Assignpi/log_std/Adamsave_30/RestoreV2:30*
validate_shape(*
use_locking(*
_class
loc:@pi/log_std*
T0*
_output_shapes
:
ą
save_30/Assign_31Assignpi/log_std/Adam_1save_30/RestoreV2:31*
T0*
_class
loc:@pi/log_std*
validate_shape(*
use_locking(*
_output_shapes
:
ą
save_30/Assign_32Assignvc/dense/biassave_30/RestoreV2:32* 
_class
loc:@vc/dense/bias*
validate_shape(*
T0*
_output_shapes	
:*
use_locking(
ś
save_30/Assign_33Assignvc/dense/bias/Adamsave_30/RestoreV2:33*
use_locking(*
validate_shape(*
_output_shapes	
:*
T0* 
_class
loc:@vc/dense/bias
¸
save_30/Assign_34Assignvc/dense/bias/Adam_1save_30/RestoreV2:34* 
_class
loc:@vc/dense/bias*
use_locking(*
validate_shape(*
_output_shapes	
:*
T0
š
save_30/Assign_35Assignvc/dense/kernelsave_30/RestoreV2:35*
use_locking(*
validate_shape(*
_output_shapes
:	L*"
_class
loc:@vc/dense/kernel*
T0
ž
save_30/Assign_36Assignvc/dense/kernel/Adamsave_30/RestoreV2:36*"
_class
loc:@vc/dense/kernel*
T0*
use_locking(*
_output_shapes
:	L*
validate_shape(
Ŕ
save_30/Assign_37Assignvc/dense/kernel/Adam_1save_30/RestoreV2:37*
validate_shape(*
T0*"
_class
loc:@vc/dense/kernel*
use_locking(*
_output_shapes
:	L
ľ
save_30/Assign_38Assignvc/dense_1/biassave_30/RestoreV2:38*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*"
_class
loc:@vc/dense_1/bias
ş
save_30/Assign_39Assignvc/dense_1/bias/Adamsave_30/RestoreV2:39*
_output_shapes	
:*
T0*
use_locking(*
validate_shape(*"
_class
loc:@vc/dense_1/bias
ź
save_30/Assign_40Assignvc/dense_1/bias/Adam_1save_30/RestoreV2:40*
validate_shape(*
T0*"
_class
loc:@vc/dense_1/bias*
_output_shapes	
:*
use_locking(
ž
save_30/Assign_41Assignvc/dense_1/kernelsave_30/RestoreV2:41*
use_locking(* 
_output_shapes
:
*$
_class
loc:@vc/dense_1/kernel*
T0*
validate_shape(
Ă
save_30/Assign_42Assignvc/dense_1/kernel/Adamsave_30/RestoreV2:42*
validate_shape(*
T0* 
_output_shapes
:
*$
_class
loc:@vc/dense_1/kernel*
use_locking(
Ĺ
save_30/Assign_43Assignvc/dense_1/kernel/Adam_1save_30/RestoreV2:43*
validate_shape(*
T0*$
_class
loc:@vc/dense_1/kernel*
use_locking(* 
_output_shapes
:

´
save_30/Assign_44Assignvc/dense_2/biassave_30/RestoreV2:44*
_output_shapes
:*
validate_shape(*
T0*
use_locking(*"
_class
loc:@vc/dense_2/bias
š
save_30/Assign_45Assignvc/dense_2/bias/Adamsave_30/RestoreV2:45*
use_locking(*
_output_shapes
:*
T0*"
_class
loc:@vc/dense_2/bias*
validate_shape(
ť
save_30/Assign_46Assignvc/dense_2/bias/Adam_1save_30/RestoreV2:46*"
_class
loc:@vc/dense_2/bias*
use_locking(*
_output_shapes
:*
validate_shape(*
T0
˝
save_30/Assign_47Assignvc/dense_2/kernelsave_30/RestoreV2:47*
T0*$
_class
loc:@vc/dense_2/kernel*
use_locking(*
_output_shapes
:	*
validate_shape(
Â
save_30/Assign_48Assignvc/dense_2/kernel/Adamsave_30/RestoreV2:48*
validate_shape(*
use_locking(*$
_class
loc:@vc/dense_2/kernel*
_output_shapes
:	*
T0
Ä
save_30/Assign_49Assignvc/dense_2/kernel/Adam_1save_30/RestoreV2:49*
_output_shapes
:	*
validate_shape(*
use_locking(*$
_class
loc:@vc/dense_2/kernel*
T0
ą
save_30/Assign_50Assignvf/dense/biassave_30/RestoreV2:50* 
_class
loc:@vf/dense/bias*
use_locking(*
validate_shape(*
_output_shapes	
:*
T0
ś
save_30/Assign_51Assignvf/dense/bias/Adamsave_30/RestoreV2:51* 
_class
loc:@vf/dense/bias*
use_locking(*
_output_shapes	
:*
validate_shape(*
T0
¸
save_30/Assign_52Assignvf/dense/bias/Adam_1save_30/RestoreV2:52* 
_class
loc:@vf/dense/bias*
use_locking(*
T0*
validate_shape(*
_output_shapes	
:
š
save_30/Assign_53Assignvf/dense/kernelsave_30/RestoreV2:53*
T0*
_output_shapes
:	L*"
_class
loc:@vf/dense/kernel*
use_locking(*
validate_shape(
ž
save_30/Assign_54Assignvf/dense/kernel/Adamsave_30/RestoreV2:54*
T0*"
_class
loc:@vf/dense/kernel*
validate_shape(*
_output_shapes
:	L*
use_locking(
Ŕ
save_30/Assign_55Assignvf/dense/kernel/Adam_1save_30/RestoreV2:55*
T0*"
_class
loc:@vf/dense/kernel*
_output_shapes
:	L*
use_locking(*
validate_shape(
ľ
save_30/Assign_56Assignvf/dense_1/biassave_30/RestoreV2:56*
_output_shapes	
:*
validate_shape(*
use_locking(*"
_class
loc:@vf/dense_1/bias*
T0
ş
save_30/Assign_57Assignvf/dense_1/bias/Adamsave_30/RestoreV2:57*
use_locking(*
T0*
_output_shapes	
:*"
_class
loc:@vf/dense_1/bias*
validate_shape(
ź
save_30/Assign_58Assignvf/dense_1/bias/Adam_1save_30/RestoreV2:58*
use_locking(*
_output_shapes	
:*
validate_shape(*"
_class
loc:@vf/dense_1/bias*
T0
ž
save_30/Assign_59Assignvf/dense_1/kernelsave_30/RestoreV2:59*
validate_shape(*$
_class
loc:@vf/dense_1/kernel*
T0* 
_output_shapes
:
*
use_locking(
Ă
save_30/Assign_60Assignvf/dense_1/kernel/Adamsave_30/RestoreV2:60*$
_class
loc:@vf/dense_1/kernel* 
_output_shapes
:
*
use_locking(*
T0*
validate_shape(
Ĺ
save_30/Assign_61Assignvf/dense_1/kernel/Adam_1save_30/RestoreV2:61* 
_output_shapes
:
*
validate_shape(*
T0*$
_class
loc:@vf/dense_1/kernel*
use_locking(
´
save_30/Assign_62Assignvf/dense_2/biassave_30/RestoreV2:62*
validate_shape(*
_output_shapes
:*"
_class
loc:@vf/dense_2/bias*
T0*
use_locking(
š
save_30/Assign_63Assignvf/dense_2/bias/Adamsave_30/RestoreV2:63*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*"
_class
loc:@vf/dense_2/bias
ť
save_30/Assign_64Assignvf/dense_2/bias/Adam_1save_30/RestoreV2:64*
_output_shapes
:*
use_locking(*"
_class
loc:@vf/dense_2/bias*
T0*
validate_shape(
˝
save_30/Assign_65Assignvf/dense_2/kernelsave_30/RestoreV2:65*$
_class
loc:@vf/dense_2/kernel*
use_locking(*
validate_shape(*
_output_shapes
:	*
T0
Â
save_30/Assign_66Assignvf/dense_2/kernel/Adamsave_30/RestoreV2:66*
use_locking(*
T0*
_output_shapes
:	*$
_class
loc:@vf/dense_2/kernel*
validate_shape(
Ä
save_30/Assign_67Assignvf/dense_2/kernel/Adam_1save_30/RestoreV2:67*
T0*
validate_shape(*
_output_shapes
:	*$
_class
loc:@vf/dense_2/kernel*
use_locking(
á

save_30/restore_shardNoOp^save_30/Assign^save_30/Assign_1^save_30/Assign_10^save_30/Assign_11^save_30/Assign_12^save_30/Assign_13^save_30/Assign_14^save_30/Assign_15^save_30/Assign_16^save_30/Assign_17^save_30/Assign_18^save_30/Assign_19^save_30/Assign_2^save_30/Assign_20^save_30/Assign_21^save_30/Assign_22^save_30/Assign_23^save_30/Assign_24^save_30/Assign_25^save_30/Assign_26^save_30/Assign_27^save_30/Assign_28^save_30/Assign_29^save_30/Assign_3^save_30/Assign_30^save_30/Assign_31^save_30/Assign_32^save_30/Assign_33^save_30/Assign_34^save_30/Assign_35^save_30/Assign_36^save_30/Assign_37^save_30/Assign_38^save_30/Assign_39^save_30/Assign_4^save_30/Assign_40^save_30/Assign_41^save_30/Assign_42^save_30/Assign_43^save_30/Assign_44^save_30/Assign_45^save_30/Assign_46^save_30/Assign_47^save_30/Assign_48^save_30/Assign_49^save_30/Assign_5^save_30/Assign_50^save_30/Assign_51^save_30/Assign_52^save_30/Assign_53^save_30/Assign_54^save_30/Assign_55^save_30/Assign_56^save_30/Assign_57^save_30/Assign_58^save_30/Assign_59^save_30/Assign_6^save_30/Assign_60^save_30/Assign_61^save_30/Assign_62^save_30/Assign_63^save_30/Assign_64^save_30/Assign_65^save_30/Assign_66^save_30/Assign_67^save_30/Assign_7^save_30/Assign_8^save_30/Assign_9
3
save_30/restore_allNoOp^save_30/restore_shard
\
save_31/filename/inputConst*
valueB Bmodel*
_output_shapes
: *
dtype0
t
save_31/filenamePlaceholderWithDefaultsave_31/filename/input*
dtype0*
_output_shapes
: *
shape: 
k
save_31/ConstPlaceholderWithDefaultsave_31/filename*
dtype0*
_output_shapes
: *
shape: 

save_31/StringJoin/inputs_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+_temp_2737274c6655446a84fbd911e2455cbd/part
~
save_31/StringJoin
StringJoinsave_31/Constsave_31/StringJoin/inputs_1*
N*
_output_shapes
: *
	separator 
T
save_31/num_shardsConst*
dtype0*
value	B :*
_output_shapes
: 
_
save_31/ShardedFilename/shardConst*
_output_shapes
: *
value	B : *
dtype0

save_31/ShardedFilenameShardedFilenamesave_31/StringJoinsave_31/ShardedFilename/shardsave_31/num_shards*
_output_shapes
: 
ř
save_31/SaveV2/tensor_namesConst*
_output_shapes
:D*
dtype0*¨
valueBDBbeta1_powerBbeta1_power_1Bbeta1_power_2Bbeta2_powerBbeta2_power_1Bbeta2_power_2Bpenalty/penalty_paramBpenalty/penalty_param/AdamBpenalty/penalty_param/Adam_1Bpenalty/penalty_param/Adam_2Bpenalty/penalty_param/Adam_3Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1B
pi/log_stdBpi/log_std/AdamBpi/log_std/Adam_1Bvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1
ń
save_31/SaveV2/shape_and_slicesConst*
_output_shapes
:D*
dtype0*
valueBDB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
Ö
save_31/SaveV2SaveV2save_31/ShardedFilenamesave_31/SaveV2/tensor_namessave_31/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta1_power_2beta2_powerbeta2_power_1beta2_power_2penalty/penalty_parampenalty/penalty_param/Adampenalty/penalty_param/Adam_1penalty/penalty_param/Adam_2penalty/penalty_param/Adam_3pi/dense/biaspi/dense/bias/Adampi/dense/bias/Adam_1pi/dense/kernelpi/dense/kernel/Adampi/dense/kernel/Adam_1pi/dense_1/biaspi/dense_1/bias/Adampi/dense_1/bias/Adam_1pi/dense_1/kernelpi/dense_1/kernel/Adampi/dense_1/kernel/Adam_1pi/dense_2/biaspi/dense_2/bias/Adampi/dense_2/bias/Adam_1pi/dense_2/kernelpi/dense_2/kernel/Adampi/dense_2/kernel/Adam_1
pi/log_stdpi/log_std/Adampi/log_std/Adam_1vc/dense/biasvc/dense/bias/Adamvc/dense/bias/Adam_1vc/dense/kernelvc/dense/kernel/Adamvc/dense/kernel/Adam_1vc/dense_1/biasvc/dense_1/bias/Adamvc/dense_1/bias/Adam_1vc/dense_1/kernelvc/dense_1/kernel/Adamvc/dense_1/kernel/Adam_1vc/dense_2/biasvc/dense_2/bias/Adamvc/dense_2/bias/Adam_1vc/dense_2/kernelvc/dense_2/kernel/Adamvc/dense_2/kernel/Adam_1vf/dense/biasvf/dense/bias/Adamvf/dense/bias/Adam_1vf/dense/kernelvf/dense/kernel/Adamvf/dense/kernel/Adam_1vf/dense_1/biasvf/dense_1/bias/Adamvf/dense_1/bias/Adam_1vf/dense_1/kernelvf/dense_1/kernel/Adamvf/dense_1/kernel/Adam_1vf/dense_2/biasvf/dense_2/bias/Adamvf/dense_2/bias/Adam_1vf/dense_2/kernelvf/dense_2/kernel/Adamvf/dense_2/kernel/Adam_1*R
dtypesH
F2D

save_31/control_dependencyIdentitysave_31/ShardedFilename^save_31/SaveV2*
_output_shapes
: **
_class 
loc:@save_31/ShardedFilename*
T0
Ś
.save_31/MergeV2Checkpoints/checkpoint_prefixesPacksave_31/ShardedFilename^save_31/control_dependency*
_output_shapes
:*

axis *
T0*
N

save_31/MergeV2CheckpointsMergeV2Checkpoints.save_31/MergeV2Checkpoints/checkpoint_prefixessave_31/Const*
delete_old_dirs(

save_31/IdentityIdentitysave_31/Const^save_31/MergeV2Checkpoints^save_31/control_dependency*
_output_shapes
: *
T0
ű
save_31/RestoreV2/tensor_namesConst*¨
valueBDBbeta1_powerBbeta1_power_1Bbeta1_power_2Bbeta2_powerBbeta2_power_1Bbeta2_power_2Bpenalty/penalty_paramBpenalty/penalty_param/AdamBpenalty/penalty_param/Adam_1Bpenalty/penalty_param/Adam_2Bpenalty/penalty_param/Adam_3Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1B
pi/log_stdBpi/log_std/AdamBpi/log_std/Adam_1Bvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1*
dtype0*
_output_shapes
:D
ô
"save_31/RestoreV2/shape_and_slicesConst*
valueBDB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:D
î
save_31/RestoreV2	RestoreV2save_31/Constsave_31/RestoreV2/tensor_names"save_31/RestoreV2/shape_and_slices*Ś
_output_shapes
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*R
dtypesH
F2D
Ź
save_31/AssignAssignbeta1_powersave_31/RestoreV2*
T0*
_output_shapes
: *
use_locking(*(
_class
loc:@penalty/penalty_param*
validate_shape(
˛
save_31/Assign_1Assignbeta1_power_1save_31/RestoreV2:1*
use_locking(*
_output_shapes
: *(
_class
loc:@penalty/penalty_param*
validate_shape(*
T0
Ş
save_31/Assign_2Assignbeta1_power_2save_31/RestoreV2:2*
validate_shape(*
use_locking(* 
_class
loc:@vc/dense/bias*
T0*
_output_shapes
: 
°
save_31/Assign_3Assignbeta2_powersave_31/RestoreV2:3*(
_class
loc:@penalty/penalty_param*
validate_shape(*
use_locking(*
T0*
_output_shapes
: 
˛
save_31/Assign_4Assignbeta2_power_1save_31/RestoreV2:4*(
_class
loc:@penalty/penalty_param*
T0*
_output_shapes
: *
use_locking(*
validate_shape(
Ş
save_31/Assign_5Assignbeta2_power_2save_31/RestoreV2:5*
T0*
use_locking(* 
_class
loc:@vc/dense/bias*
_output_shapes
: *
validate_shape(
ş
save_31/Assign_6Assignpenalty/penalty_paramsave_31/RestoreV2:6*
_output_shapes
: *
use_locking(*
T0*
validate_shape(*(
_class
loc:@penalty/penalty_param
ż
save_31/Assign_7Assignpenalty/penalty_param/Adamsave_31/RestoreV2:7*
use_locking(*(
_class
loc:@penalty/penalty_param*
validate_shape(*
T0*
_output_shapes
: 
Á
save_31/Assign_8Assignpenalty/penalty_param/Adam_1save_31/RestoreV2:8*
T0*
use_locking(*(
_class
loc:@penalty/penalty_param*
_output_shapes
: *
validate_shape(
Á
save_31/Assign_9Assignpenalty/penalty_param/Adam_2save_31/RestoreV2:9*
T0*
use_locking(*(
_class
loc:@penalty/penalty_param*
validate_shape(*
_output_shapes
: 
Ă
save_31/Assign_10Assignpenalty/penalty_param/Adam_3save_31/RestoreV2:10*
_output_shapes
: *
use_locking(*
T0*
validate_shape(*(
_class
loc:@penalty/penalty_param
ą
save_31/Assign_11Assignpi/dense/biassave_31/RestoreV2:11*
use_locking(* 
_class
loc:@pi/dense/bias*
_output_shapes	
:*
T0*
validate_shape(
ś
save_31/Assign_12Assignpi/dense/bias/Adamsave_31/RestoreV2:12*
T0*
use_locking(*
_output_shapes	
:* 
_class
loc:@pi/dense/bias*
validate_shape(
¸
save_31/Assign_13Assignpi/dense/bias/Adam_1save_31/RestoreV2:13*
validate_shape(* 
_class
loc:@pi/dense/bias*
T0*
use_locking(*
_output_shapes	
:
š
save_31/Assign_14Assignpi/dense/kernelsave_31/RestoreV2:14*
T0*
validate_shape(*
use_locking(*
_output_shapes
:	L*"
_class
loc:@pi/dense/kernel
ž
save_31/Assign_15Assignpi/dense/kernel/Adamsave_31/RestoreV2:15*
use_locking(*"
_class
loc:@pi/dense/kernel*
_output_shapes
:	L*
T0*
validate_shape(
Ŕ
save_31/Assign_16Assignpi/dense/kernel/Adam_1save_31/RestoreV2:16*
validate_shape(*
use_locking(*
_output_shapes
:	L*"
_class
loc:@pi/dense/kernel*
T0
ľ
save_31/Assign_17Assignpi/dense_1/biassave_31/RestoreV2:17*"
_class
loc:@pi/dense_1/bias*
T0*
_output_shapes	
:*
validate_shape(*
use_locking(
ş
save_31/Assign_18Assignpi/dense_1/bias/Adamsave_31/RestoreV2:18*
validate_shape(*
_output_shapes	
:*"
_class
loc:@pi/dense_1/bias*
use_locking(*
T0
ź
save_31/Assign_19Assignpi/dense_1/bias/Adam_1save_31/RestoreV2:19*
use_locking(*"
_class
loc:@pi/dense_1/bias*
validate_shape(*
T0*
_output_shapes	
:
ž
save_31/Assign_20Assignpi/dense_1/kernelsave_31/RestoreV2:20*
validate_shape(*$
_class
loc:@pi/dense_1/kernel* 
_output_shapes
:
*
T0*
use_locking(
Ă
save_31/Assign_21Assignpi/dense_1/kernel/Adamsave_31/RestoreV2:21*
validate_shape(*
T0*
use_locking(*$
_class
loc:@pi/dense_1/kernel* 
_output_shapes
:

Ĺ
save_31/Assign_22Assignpi/dense_1/kernel/Adam_1save_31/RestoreV2:22*$
_class
loc:@pi/dense_1/kernel*
use_locking(*
validate_shape(* 
_output_shapes
:
*
T0
´
save_31/Assign_23Assignpi/dense_2/biassave_31/RestoreV2:23*
use_locking(*"
_class
loc:@pi/dense_2/bias*
validate_shape(*
_output_shapes
:*
T0
š
save_31/Assign_24Assignpi/dense_2/bias/Adamsave_31/RestoreV2:24*
T0*
use_locking(*"
_class
loc:@pi/dense_2/bias*
validate_shape(*
_output_shapes
:
ť
save_31/Assign_25Assignpi/dense_2/bias/Adam_1save_31/RestoreV2:25*
validate_shape(*
T0*"
_class
loc:@pi/dense_2/bias*
use_locking(*
_output_shapes
:
˝
save_31/Assign_26Assignpi/dense_2/kernelsave_31/RestoreV2:26*
T0*
use_locking(*$
_class
loc:@pi/dense_2/kernel*
_output_shapes
:	*
validate_shape(
Â
save_31/Assign_27Assignpi/dense_2/kernel/Adamsave_31/RestoreV2:27*
use_locking(*
T0*
validate_shape(*$
_class
loc:@pi/dense_2/kernel*
_output_shapes
:	
Ä
save_31/Assign_28Assignpi/dense_2/kernel/Adam_1save_31/RestoreV2:28*
T0*
_output_shapes
:	*$
_class
loc:@pi/dense_2/kernel*
validate_shape(*
use_locking(
Ş
save_31/Assign_29Assign
pi/log_stdsave_31/RestoreV2:29*
_class
loc:@pi/log_std*
use_locking(*
_output_shapes
:*
validate_shape(*
T0
Ż
save_31/Assign_30Assignpi/log_std/Adamsave_31/RestoreV2:30*
T0*
_output_shapes
:*
validate_shape(*
_class
loc:@pi/log_std*
use_locking(
ą
save_31/Assign_31Assignpi/log_std/Adam_1save_31/RestoreV2:31*
T0*
_output_shapes
:*
use_locking(*
validate_shape(*
_class
loc:@pi/log_std
ą
save_31/Assign_32Assignvc/dense/biassave_31/RestoreV2:32*
validate_shape(*
_output_shapes	
:* 
_class
loc:@vc/dense/bias*
use_locking(*
T0
ś
save_31/Assign_33Assignvc/dense/bias/Adamsave_31/RestoreV2:33*
T0*
use_locking(*
_output_shapes	
:* 
_class
loc:@vc/dense/bias*
validate_shape(
¸
save_31/Assign_34Assignvc/dense/bias/Adam_1save_31/RestoreV2:34*
use_locking(*
T0*
validate_shape(* 
_class
loc:@vc/dense/bias*
_output_shapes	
:
š
save_31/Assign_35Assignvc/dense/kernelsave_31/RestoreV2:35*
T0*
validate_shape(*
_output_shapes
:	L*"
_class
loc:@vc/dense/kernel*
use_locking(
ž
save_31/Assign_36Assignvc/dense/kernel/Adamsave_31/RestoreV2:36*"
_class
loc:@vc/dense/kernel*
_output_shapes
:	L*
T0*
use_locking(*
validate_shape(
Ŕ
save_31/Assign_37Assignvc/dense/kernel/Adam_1save_31/RestoreV2:37*"
_class
loc:@vc/dense/kernel*
_output_shapes
:	L*
validate_shape(*
T0*
use_locking(
ľ
save_31/Assign_38Assignvc/dense_1/biassave_31/RestoreV2:38*
validate_shape(*"
_class
loc:@vc/dense_1/bias*
use_locking(*
_output_shapes	
:*
T0
ş
save_31/Assign_39Assignvc/dense_1/bias/Adamsave_31/RestoreV2:39*
_output_shapes	
:*
validate_shape(*
T0*
use_locking(*"
_class
loc:@vc/dense_1/bias
ź
save_31/Assign_40Assignvc/dense_1/bias/Adam_1save_31/RestoreV2:40*
_output_shapes	
:*"
_class
loc:@vc/dense_1/bias*
validate_shape(*
use_locking(*
T0
ž
save_31/Assign_41Assignvc/dense_1/kernelsave_31/RestoreV2:41*$
_class
loc:@vc/dense_1/kernel* 
_output_shapes
:
*
use_locking(*
T0*
validate_shape(
Ă
save_31/Assign_42Assignvc/dense_1/kernel/Adamsave_31/RestoreV2:42*
validate_shape(*$
_class
loc:@vc/dense_1/kernel* 
_output_shapes
:
*
use_locking(*
T0
Ĺ
save_31/Assign_43Assignvc/dense_1/kernel/Adam_1save_31/RestoreV2:43* 
_output_shapes
:
*
T0*
validate_shape(*
use_locking(*$
_class
loc:@vc/dense_1/kernel
´
save_31/Assign_44Assignvc/dense_2/biassave_31/RestoreV2:44*
validate_shape(*
_output_shapes
:*"
_class
loc:@vc/dense_2/bias*
use_locking(*
T0
š
save_31/Assign_45Assignvc/dense_2/bias/Adamsave_31/RestoreV2:45*"
_class
loc:@vc/dense_2/bias*
T0*
use_locking(*
_output_shapes
:*
validate_shape(
ť
save_31/Assign_46Assignvc/dense_2/bias/Adam_1save_31/RestoreV2:46*"
_class
loc:@vc/dense_2/bias*
T0*
use_locking(*
validate_shape(*
_output_shapes
:
˝
save_31/Assign_47Assignvc/dense_2/kernelsave_31/RestoreV2:47*
use_locking(*
T0*
validate_shape(*
_output_shapes
:	*$
_class
loc:@vc/dense_2/kernel
Â
save_31/Assign_48Assignvc/dense_2/kernel/Adamsave_31/RestoreV2:48*
T0*
_output_shapes
:	*
use_locking(*$
_class
loc:@vc/dense_2/kernel*
validate_shape(
Ä
save_31/Assign_49Assignvc/dense_2/kernel/Adam_1save_31/RestoreV2:49*
use_locking(*
T0*
validate_shape(*
_output_shapes
:	*$
_class
loc:@vc/dense_2/kernel
ą
save_31/Assign_50Assignvf/dense/biassave_31/RestoreV2:50* 
_class
loc:@vf/dense/bias*
use_locking(*
validate_shape(*
T0*
_output_shapes	
:
ś
save_31/Assign_51Assignvf/dense/bias/Adamsave_31/RestoreV2:51*
_output_shapes	
:*
validate_shape(*
T0* 
_class
loc:@vf/dense/bias*
use_locking(
¸
save_31/Assign_52Assignvf/dense/bias/Adam_1save_31/RestoreV2:52* 
_class
loc:@vf/dense/bias*
_output_shapes	
:*
validate_shape(*
T0*
use_locking(
š
save_31/Assign_53Assignvf/dense/kernelsave_31/RestoreV2:53*
validate_shape(*
T0*"
_class
loc:@vf/dense/kernel*
_output_shapes
:	L*
use_locking(
ž
save_31/Assign_54Assignvf/dense/kernel/Adamsave_31/RestoreV2:54*"
_class
loc:@vf/dense/kernel*
validate_shape(*
use_locking(*
T0*
_output_shapes
:	L
Ŕ
save_31/Assign_55Assignvf/dense/kernel/Adam_1save_31/RestoreV2:55*
use_locking(*
validate_shape(*
T0*"
_class
loc:@vf/dense/kernel*
_output_shapes
:	L
ľ
save_31/Assign_56Assignvf/dense_1/biassave_31/RestoreV2:56*"
_class
loc:@vf/dense_1/bias*
use_locking(*
_output_shapes	
:*
validate_shape(*
T0
ş
save_31/Assign_57Assignvf/dense_1/bias/Adamsave_31/RestoreV2:57*
use_locking(*
T0*"
_class
loc:@vf/dense_1/bias*
validate_shape(*
_output_shapes	
:
ź
save_31/Assign_58Assignvf/dense_1/bias/Adam_1save_31/RestoreV2:58*"
_class
loc:@vf/dense_1/bias*
use_locking(*
T0*
validate_shape(*
_output_shapes	
:
ž
save_31/Assign_59Assignvf/dense_1/kernelsave_31/RestoreV2:59* 
_output_shapes
:
*
validate_shape(*
T0*
use_locking(*$
_class
loc:@vf/dense_1/kernel
Ă
save_31/Assign_60Assignvf/dense_1/kernel/Adamsave_31/RestoreV2:60*$
_class
loc:@vf/dense_1/kernel*
use_locking(* 
_output_shapes
:
*
T0*
validate_shape(
Ĺ
save_31/Assign_61Assignvf/dense_1/kernel/Adam_1save_31/RestoreV2:61* 
_output_shapes
:
*
T0*
use_locking(*$
_class
loc:@vf/dense_1/kernel*
validate_shape(
´
save_31/Assign_62Assignvf/dense_2/biassave_31/RestoreV2:62*
T0*"
_class
loc:@vf/dense_2/bias*
use_locking(*
validate_shape(*
_output_shapes
:
š
save_31/Assign_63Assignvf/dense_2/bias/Adamsave_31/RestoreV2:63*"
_class
loc:@vf/dense_2/bias*
validate_shape(*
T0*
_output_shapes
:*
use_locking(
ť
save_31/Assign_64Assignvf/dense_2/bias/Adam_1save_31/RestoreV2:64*
_output_shapes
:*
T0*"
_class
loc:@vf/dense_2/bias*
validate_shape(*
use_locking(
˝
save_31/Assign_65Assignvf/dense_2/kernelsave_31/RestoreV2:65*
validate_shape(*$
_class
loc:@vf/dense_2/kernel*
_output_shapes
:	*
use_locking(*
T0
Â
save_31/Assign_66Assignvf/dense_2/kernel/Adamsave_31/RestoreV2:66*
_output_shapes
:	*$
_class
loc:@vf/dense_2/kernel*
use_locking(*
validate_shape(*
T0
Ä
save_31/Assign_67Assignvf/dense_2/kernel/Adam_1save_31/RestoreV2:67*
_output_shapes
:	*
use_locking(*
validate_shape(*$
_class
loc:@vf/dense_2/kernel*
T0
á

save_31/restore_shardNoOp^save_31/Assign^save_31/Assign_1^save_31/Assign_10^save_31/Assign_11^save_31/Assign_12^save_31/Assign_13^save_31/Assign_14^save_31/Assign_15^save_31/Assign_16^save_31/Assign_17^save_31/Assign_18^save_31/Assign_19^save_31/Assign_2^save_31/Assign_20^save_31/Assign_21^save_31/Assign_22^save_31/Assign_23^save_31/Assign_24^save_31/Assign_25^save_31/Assign_26^save_31/Assign_27^save_31/Assign_28^save_31/Assign_29^save_31/Assign_3^save_31/Assign_30^save_31/Assign_31^save_31/Assign_32^save_31/Assign_33^save_31/Assign_34^save_31/Assign_35^save_31/Assign_36^save_31/Assign_37^save_31/Assign_38^save_31/Assign_39^save_31/Assign_4^save_31/Assign_40^save_31/Assign_41^save_31/Assign_42^save_31/Assign_43^save_31/Assign_44^save_31/Assign_45^save_31/Assign_46^save_31/Assign_47^save_31/Assign_48^save_31/Assign_49^save_31/Assign_5^save_31/Assign_50^save_31/Assign_51^save_31/Assign_52^save_31/Assign_53^save_31/Assign_54^save_31/Assign_55^save_31/Assign_56^save_31/Assign_57^save_31/Assign_58^save_31/Assign_59^save_31/Assign_6^save_31/Assign_60^save_31/Assign_61^save_31/Assign_62^save_31/Assign_63^save_31/Assign_64^save_31/Assign_65^save_31/Assign_66^save_31/Assign_67^save_31/Assign_7^save_31/Assign_8^save_31/Assign_9
3
save_31/restore_allNoOp^save_31/restore_shard
\
save_32/filename/inputConst*
dtype0*
_output_shapes
: *
valueB Bmodel
t
save_32/filenamePlaceholderWithDefaultsave_32/filename/input*
dtype0*
shape: *
_output_shapes
: 
k
save_32/ConstPlaceholderWithDefaultsave_32/filename*
_output_shapes
: *
dtype0*
shape: 

save_32/StringJoin/inputs_1Const*
dtype0*
_output_shapes
: *<
value3B1 B+_temp_ed00b8305b1543de8a3a97b9339bb973/part
~
save_32/StringJoin
StringJoinsave_32/Constsave_32/StringJoin/inputs_1*
	separator *
_output_shapes
: *
N
T
save_32/num_shardsConst*
dtype0*
value	B :*
_output_shapes
: 
_
save_32/ShardedFilename/shardConst*
value	B : *
_output_shapes
: *
dtype0

save_32/ShardedFilenameShardedFilenamesave_32/StringJoinsave_32/ShardedFilename/shardsave_32/num_shards*
_output_shapes
: 
ř
save_32/SaveV2/tensor_namesConst*
dtype0*
_output_shapes
:D*¨
valueBDBbeta1_powerBbeta1_power_1Bbeta1_power_2Bbeta2_powerBbeta2_power_1Bbeta2_power_2Bpenalty/penalty_paramBpenalty/penalty_param/AdamBpenalty/penalty_param/Adam_1Bpenalty/penalty_param/Adam_2Bpenalty/penalty_param/Adam_3Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1B
pi/log_stdBpi/log_std/AdamBpi/log_std/Adam_1Bvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1
ń
save_32/SaveV2/shape_and_slicesConst*
valueBDB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:D
Ö
save_32/SaveV2SaveV2save_32/ShardedFilenamesave_32/SaveV2/tensor_namessave_32/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta1_power_2beta2_powerbeta2_power_1beta2_power_2penalty/penalty_parampenalty/penalty_param/Adampenalty/penalty_param/Adam_1penalty/penalty_param/Adam_2penalty/penalty_param/Adam_3pi/dense/biaspi/dense/bias/Adampi/dense/bias/Adam_1pi/dense/kernelpi/dense/kernel/Adampi/dense/kernel/Adam_1pi/dense_1/biaspi/dense_1/bias/Adampi/dense_1/bias/Adam_1pi/dense_1/kernelpi/dense_1/kernel/Adampi/dense_1/kernel/Adam_1pi/dense_2/biaspi/dense_2/bias/Adampi/dense_2/bias/Adam_1pi/dense_2/kernelpi/dense_2/kernel/Adampi/dense_2/kernel/Adam_1
pi/log_stdpi/log_std/Adampi/log_std/Adam_1vc/dense/biasvc/dense/bias/Adamvc/dense/bias/Adam_1vc/dense/kernelvc/dense/kernel/Adamvc/dense/kernel/Adam_1vc/dense_1/biasvc/dense_1/bias/Adamvc/dense_1/bias/Adam_1vc/dense_1/kernelvc/dense_1/kernel/Adamvc/dense_1/kernel/Adam_1vc/dense_2/biasvc/dense_2/bias/Adamvc/dense_2/bias/Adam_1vc/dense_2/kernelvc/dense_2/kernel/Adamvc/dense_2/kernel/Adam_1vf/dense/biasvf/dense/bias/Adamvf/dense/bias/Adam_1vf/dense/kernelvf/dense/kernel/Adamvf/dense/kernel/Adam_1vf/dense_1/biasvf/dense_1/bias/Adamvf/dense_1/bias/Adam_1vf/dense_1/kernelvf/dense_1/kernel/Adamvf/dense_1/kernel/Adam_1vf/dense_2/biasvf/dense_2/bias/Adamvf/dense_2/bias/Adam_1vf/dense_2/kernelvf/dense_2/kernel/Adamvf/dense_2/kernel/Adam_1*R
dtypesH
F2D

save_32/control_dependencyIdentitysave_32/ShardedFilename^save_32/SaveV2*
_output_shapes
: *
T0**
_class 
loc:@save_32/ShardedFilename
Ś
.save_32/MergeV2Checkpoints/checkpoint_prefixesPacksave_32/ShardedFilename^save_32/control_dependency*
_output_shapes
:*

axis *
N*
T0

save_32/MergeV2CheckpointsMergeV2Checkpoints.save_32/MergeV2Checkpoints/checkpoint_prefixessave_32/Const*
delete_old_dirs(

save_32/IdentityIdentitysave_32/Const^save_32/MergeV2Checkpoints^save_32/control_dependency*
_output_shapes
: *
T0
ű
save_32/RestoreV2/tensor_namesConst*
dtype0*
_output_shapes
:D*¨
valueBDBbeta1_powerBbeta1_power_1Bbeta1_power_2Bbeta2_powerBbeta2_power_1Bbeta2_power_2Bpenalty/penalty_paramBpenalty/penalty_param/AdamBpenalty/penalty_param/Adam_1Bpenalty/penalty_param/Adam_2Bpenalty/penalty_param/Adam_3Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1B
pi/log_stdBpi/log_std/AdamBpi/log_std/Adam_1Bvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1
ô
"save_32/RestoreV2/shape_and_slicesConst*
valueBDB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:D
î
save_32/RestoreV2	RestoreV2save_32/Constsave_32/RestoreV2/tensor_names"save_32/RestoreV2/shape_and_slices*Ś
_output_shapes
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*R
dtypesH
F2D
Ź
save_32/AssignAssignbeta1_powersave_32/RestoreV2*
T0*
validate_shape(*
_output_shapes
: *(
_class
loc:@penalty/penalty_param*
use_locking(
˛
save_32/Assign_1Assignbeta1_power_1save_32/RestoreV2:1*
_output_shapes
: *(
_class
loc:@penalty/penalty_param*
validate_shape(*
use_locking(*
T0
Ş
save_32/Assign_2Assignbeta1_power_2save_32/RestoreV2:2*
use_locking(*
validate_shape(* 
_class
loc:@vc/dense/bias*
T0*
_output_shapes
: 
°
save_32/Assign_3Assignbeta2_powersave_32/RestoreV2:3*
use_locking(*
T0*(
_class
loc:@penalty/penalty_param*
validate_shape(*
_output_shapes
: 
˛
save_32/Assign_4Assignbeta2_power_1save_32/RestoreV2:4*(
_class
loc:@penalty/penalty_param*
_output_shapes
: *
T0*
use_locking(*
validate_shape(
Ş
save_32/Assign_5Assignbeta2_power_2save_32/RestoreV2:5*
validate_shape(*
T0* 
_class
loc:@vc/dense/bias*
_output_shapes
: *
use_locking(
ş
save_32/Assign_6Assignpenalty/penalty_paramsave_32/RestoreV2:6*
T0*
validate_shape(*
_output_shapes
: *(
_class
loc:@penalty/penalty_param*
use_locking(
ż
save_32/Assign_7Assignpenalty/penalty_param/Adamsave_32/RestoreV2:7*
validate_shape(*
T0*
use_locking(*(
_class
loc:@penalty/penalty_param*
_output_shapes
: 
Á
save_32/Assign_8Assignpenalty/penalty_param/Adam_1save_32/RestoreV2:8*
_output_shapes
: *
validate_shape(*
use_locking(*
T0*(
_class
loc:@penalty/penalty_param
Á
save_32/Assign_9Assignpenalty/penalty_param/Adam_2save_32/RestoreV2:9*
T0*
_output_shapes
: *
use_locking(*(
_class
loc:@penalty/penalty_param*
validate_shape(
Ă
save_32/Assign_10Assignpenalty/penalty_param/Adam_3save_32/RestoreV2:10*(
_class
loc:@penalty/penalty_param*
_output_shapes
: *
validate_shape(*
use_locking(*
T0
ą
save_32/Assign_11Assignpi/dense/biassave_32/RestoreV2:11* 
_class
loc:@pi/dense/bias*
use_locking(*
T0*
validate_shape(*
_output_shapes	
:
ś
save_32/Assign_12Assignpi/dense/bias/Adamsave_32/RestoreV2:12*
validate_shape(*
T0* 
_class
loc:@pi/dense/bias*
use_locking(*
_output_shapes	
:
¸
save_32/Assign_13Assignpi/dense/bias/Adam_1save_32/RestoreV2:13*
use_locking(*
_output_shapes	
:* 
_class
loc:@pi/dense/bias*
T0*
validate_shape(
š
save_32/Assign_14Assignpi/dense/kernelsave_32/RestoreV2:14*"
_class
loc:@pi/dense/kernel*
validate_shape(*
_output_shapes
:	L*
use_locking(*
T0
ž
save_32/Assign_15Assignpi/dense/kernel/Adamsave_32/RestoreV2:15*
_output_shapes
:	L*
T0*"
_class
loc:@pi/dense/kernel*
validate_shape(*
use_locking(
Ŕ
save_32/Assign_16Assignpi/dense/kernel/Adam_1save_32/RestoreV2:16*
T0*
validate_shape(*
use_locking(*"
_class
loc:@pi/dense/kernel*
_output_shapes
:	L
ľ
save_32/Assign_17Assignpi/dense_1/biassave_32/RestoreV2:17*
validate_shape(*
T0*
_output_shapes	
:*"
_class
loc:@pi/dense_1/bias*
use_locking(
ş
save_32/Assign_18Assignpi/dense_1/bias/Adamsave_32/RestoreV2:18*
use_locking(*
T0*"
_class
loc:@pi/dense_1/bias*
validate_shape(*
_output_shapes	
:
ź
save_32/Assign_19Assignpi/dense_1/bias/Adam_1save_32/RestoreV2:19*"
_class
loc:@pi/dense_1/bias*
T0*
_output_shapes	
:*
use_locking(*
validate_shape(
ž
save_32/Assign_20Assignpi/dense_1/kernelsave_32/RestoreV2:20*
validate_shape(* 
_output_shapes
:
*$
_class
loc:@pi/dense_1/kernel*
T0*
use_locking(
Ă
save_32/Assign_21Assignpi/dense_1/kernel/Adamsave_32/RestoreV2:21*
validate_shape(*
use_locking(*
T0* 
_output_shapes
:
*$
_class
loc:@pi/dense_1/kernel
Ĺ
save_32/Assign_22Assignpi/dense_1/kernel/Adam_1save_32/RestoreV2:22*$
_class
loc:@pi/dense_1/kernel*
validate_shape(*
T0*
use_locking(* 
_output_shapes
:

´
save_32/Assign_23Assignpi/dense_2/biassave_32/RestoreV2:23*"
_class
loc:@pi/dense_2/bias*
use_locking(*
validate_shape(*
_output_shapes
:*
T0
š
save_32/Assign_24Assignpi/dense_2/bias/Adamsave_32/RestoreV2:24*
_output_shapes
:*
validate_shape(*
use_locking(*"
_class
loc:@pi/dense_2/bias*
T0
ť
save_32/Assign_25Assignpi/dense_2/bias/Adam_1save_32/RestoreV2:25*
validate_shape(*
use_locking(*
T0*
_output_shapes
:*"
_class
loc:@pi/dense_2/bias
˝
save_32/Assign_26Assignpi/dense_2/kernelsave_32/RestoreV2:26*
_output_shapes
:	*
use_locking(*
validate_shape(*$
_class
loc:@pi/dense_2/kernel*
T0
Â
save_32/Assign_27Assignpi/dense_2/kernel/Adamsave_32/RestoreV2:27*
use_locking(*
T0*
_output_shapes
:	*
validate_shape(*$
_class
loc:@pi/dense_2/kernel
Ä
save_32/Assign_28Assignpi/dense_2/kernel/Adam_1save_32/RestoreV2:28*$
_class
loc:@pi/dense_2/kernel*
_output_shapes
:	*
T0*
use_locking(*
validate_shape(
Ş
save_32/Assign_29Assign
pi/log_stdsave_32/RestoreV2:29*
use_locking(*
_output_shapes
:*
validate_shape(*
T0*
_class
loc:@pi/log_std
Ż
save_32/Assign_30Assignpi/log_std/Adamsave_32/RestoreV2:30*
_class
loc:@pi/log_std*
_output_shapes
:*
use_locking(*
validate_shape(*
T0
ą
save_32/Assign_31Assignpi/log_std/Adam_1save_32/RestoreV2:31*
_class
loc:@pi/log_std*
use_locking(*
T0*
_output_shapes
:*
validate_shape(
ą
save_32/Assign_32Assignvc/dense/biassave_32/RestoreV2:32*
use_locking(*
_output_shapes	
:*
validate_shape(*
T0* 
_class
loc:@vc/dense/bias
ś
save_32/Assign_33Assignvc/dense/bias/Adamsave_32/RestoreV2:33*
use_locking(*
T0*
validate_shape(*
_output_shapes	
:* 
_class
loc:@vc/dense/bias
¸
save_32/Assign_34Assignvc/dense/bias/Adam_1save_32/RestoreV2:34*
use_locking(* 
_class
loc:@vc/dense/bias*
validate_shape(*
_output_shapes	
:*
T0
š
save_32/Assign_35Assignvc/dense/kernelsave_32/RestoreV2:35*
T0*
_output_shapes
:	L*"
_class
loc:@vc/dense/kernel*
use_locking(*
validate_shape(
ž
save_32/Assign_36Assignvc/dense/kernel/Adamsave_32/RestoreV2:36*
_output_shapes
:	L*"
_class
loc:@vc/dense/kernel*
validate_shape(*
use_locking(*
T0
Ŕ
save_32/Assign_37Assignvc/dense/kernel/Adam_1save_32/RestoreV2:37*"
_class
loc:@vc/dense/kernel*
use_locking(*
T0*
validate_shape(*
_output_shapes
:	L
ľ
save_32/Assign_38Assignvc/dense_1/biassave_32/RestoreV2:38*
_output_shapes	
:*
use_locking(*"
_class
loc:@vc/dense_1/bias*
T0*
validate_shape(
ş
save_32/Assign_39Assignvc/dense_1/bias/Adamsave_32/RestoreV2:39*"
_class
loc:@vc/dense_1/bias*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0
ź
save_32/Assign_40Assignvc/dense_1/bias/Adam_1save_32/RestoreV2:40*"
_class
loc:@vc/dense_1/bias*
_output_shapes	
:*
T0*
validate_shape(*
use_locking(
ž
save_32/Assign_41Assignvc/dense_1/kernelsave_32/RestoreV2:41*
use_locking(*
validate_shape(*
T0*$
_class
loc:@vc/dense_1/kernel* 
_output_shapes
:

Ă
save_32/Assign_42Assignvc/dense_1/kernel/Adamsave_32/RestoreV2:42*$
_class
loc:@vc/dense_1/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0
Ĺ
save_32/Assign_43Assignvc/dense_1/kernel/Adam_1save_32/RestoreV2:43*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0*$
_class
loc:@vc/dense_1/kernel
´
save_32/Assign_44Assignvc/dense_2/biassave_32/RestoreV2:44*"
_class
loc:@vc/dense_2/bias*
_output_shapes
:*
validate_shape(*
use_locking(*
T0
š
save_32/Assign_45Assignvc/dense_2/bias/Adamsave_32/RestoreV2:45*
validate_shape(*
use_locking(*
_output_shapes
:*
T0*"
_class
loc:@vc/dense_2/bias
ť
save_32/Assign_46Assignvc/dense_2/bias/Adam_1save_32/RestoreV2:46*"
_class
loc:@vc/dense_2/bias*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
˝
save_32/Assign_47Assignvc/dense_2/kernelsave_32/RestoreV2:47*
_output_shapes
:	*
T0*$
_class
loc:@vc/dense_2/kernel*
use_locking(*
validate_shape(
Â
save_32/Assign_48Assignvc/dense_2/kernel/Adamsave_32/RestoreV2:48*
T0*
use_locking(*$
_class
loc:@vc/dense_2/kernel*
validate_shape(*
_output_shapes
:	
Ä
save_32/Assign_49Assignvc/dense_2/kernel/Adam_1save_32/RestoreV2:49*$
_class
loc:@vc/dense_2/kernel*
_output_shapes
:	*
validate_shape(*
T0*
use_locking(
ą
save_32/Assign_50Assignvf/dense/biassave_32/RestoreV2:50*
use_locking(*
T0*
validate_shape(*
_output_shapes	
:* 
_class
loc:@vf/dense/bias
ś
save_32/Assign_51Assignvf/dense/bias/Adamsave_32/RestoreV2:51* 
_class
loc:@vf/dense/bias*
_output_shapes	
:*
validate_shape(*
use_locking(*
T0
¸
save_32/Assign_52Assignvf/dense/bias/Adam_1save_32/RestoreV2:52*
T0*
validate_shape(*
_output_shapes	
:*
use_locking(* 
_class
loc:@vf/dense/bias
š
save_32/Assign_53Assignvf/dense/kernelsave_32/RestoreV2:53*"
_class
loc:@vf/dense/kernel*
T0*
_output_shapes
:	L*
validate_shape(*
use_locking(
ž
save_32/Assign_54Assignvf/dense/kernel/Adamsave_32/RestoreV2:54*
validate_shape(*"
_class
loc:@vf/dense/kernel*
T0*
_output_shapes
:	L*
use_locking(
Ŕ
save_32/Assign_55Assignvf/dense/kernel/Adam_1save_32/RestoreV2:55*
T0*"
_class
loc:@vf/dense/kernel*
use_locking(*
_output_shapes
:	L*
validate_shape(
ľ
save_32/Assign_56Assignvf/dense_1/biassave_32/RestoreV2:56*
T0*
_output_shapes	
:*"
_class
loc:@vf/dense_1/bias*
use_locking(*
validate_shape(
ş
save_32/Assign_57Assignvf/dense_1/bias/Adamsave_32/RestoreV2:57*
use_locking(*
T0*
_output_shapes	
:*
validate_shape(*"
_class
loc:@vf/dense_1/bias
ź
save_32/Assign_58Assignvf/dense_1/bias/Adam_1save_32/RestoreV2:58*
_output_shapes	
:*
validate_shape(*"
_class
loc:@vf/dense_1/bias*
T0*
use_locking(
ž
save_32/Assign_59Assignvf/dense_1/kernelsave_32/RestoreV2:59*$
_class
loc:@vf/dense_1/kernel*
T0*
validate_shape(* 
_output_shapes
:
*
use_locking(
Ă
save_32/Assign_60Assignvf/dense_1/kernel/Adamsave_32/RestoreV2:60*
use_locking(*
T0*
validate_shape(* 
_output_shapes
:
*$
_class
loc:@vf/dense_1/kernel
Ĺ
save_32/Assign_61Assignvf/dense_1/kernel/Adam_1save_32/RestoreV2:61*
T0* 
_output_shapes
:
*$
_class
loc:@vf/dense_1/kernel*
validate_shape(*
use_locking(
´
save_32/Assign_62Assignvf/dense_2/biassave_32/RestoreV2:62*
T0*
use_locking(*"
_class
loc:@vf/dense_2/bias*
validate_shape(*
_output_shapes
:
š
save_32/Assign_63Assignvf/dense_2/bias/Adamsave_32/RestoreV2:63*
use_locking(*
_output_shapes
:*
T0*
validate_shape(*"
_class
loc:@vf/dense_2/bias
ť
save_32/Assign_64Assignvf/dense_2/bias/Adam_1save_32/RestoreV2:64*
validate_shape(*
T0*
_output_shapes
:*"
_class
loc:@vf/dense_2/bias*
use_locking(
˝
save_32/Assign_65Assignvf/dense_2/kernelsave_32/RestoreV2:65*
use_locking(*
validate_shape(*
T0*
_output_shapes
:	*$
_class
loc:@vf/dense_2/kernel
Â
save_32/Assign_66Assignvf/dense_2/kernel/Adamsave_32/RestoreV2:66*
use_locking(*
_output_shapes
:	*
validate_shape(*
T0*$
_class
loc:@vf/dense_2/kernel
Ä
save_32/Assign_67Assignvf/dense_2/kernel/Adam_1save_32/RestoreV2:67*
T0*$
_class
loc:@vf/dense_2/kernel*
_output_shapes
:	*
use_locking(*
validate_shape(
á

save_32/restore_shardNoOp^save_32/Assign^save_32/Assign_1^save_32/Assign_10^save_32/Assign_11^save_32/Assign_12^save_32/Assign_13^save_32/Assign_14^save_32/Assign_15^save_32/Assign_16^save_32/Assign_17^save_32/Assign_18^save_32/Assign_19^save_32/Assign_2^save_32/Assign_20^save_32/Assign_21^save_32/Assign_22^save_32/Assign_23^save_32/Assign_24^save_32/Assign_25^save_32/Assign_26^save_32/Assign_27^save_32/Assign_28^save_32/Assign_29^save_32/Assign_3^save_32/Assign_30^save_32/Assign_31^save_32/Assign_32^save_32/Assign_33^save_32/Assign_34^save_32/Assign_35^save_32/Assign_36^save_32/Assign_37^save_32/Assign_38^save_32/Assign_39^save_32/Assign_4^save_32/Assign_40^save_32/Assign_41^save_32/Assign_42^save_32/Assign_43^save_32/Assign_44^save_32/Assign_45^save_32/Assign_46^save_32/Assign_47^save_32/Assign_48^save_32/Assign_49^save_32/Assign_5^save_32/Assign_50^save_32/Assign_51^save_32/Assign_52^save_32/Assign_53^save_32/Assign_54^save_32/Assign_55^save_32/Assign_56^save_32/Assign_57^save_32/Assign_58^save_32/Assign_59^save_32/Assign_6^save_32/Assign_60^save_32/Assign_61^save_32/Assign_62^save_32/Assign_63^save_32/Assign_64^save_32/Assign_65^save_32/Assign_66^save_32/Assign_67^save_32/Assign_7^save_32/Assign_8^save_32/Assign_9
3
save_32/restore_allNoOp^save_32/restore_shard "E
save_32/Const:0save_32/Identity:0save_32/restore_all (5 @F8"çA
	variablesŮAÖA
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
~
penalty/penalty_param:0penalty/penalty_param/Assignpenalty/penalty_param/read:02%penalty/penalty_param/initial_value:08
T
beta1_power:0beta1_power/Assignbeta1_power/read:02beta1_power/initial_value:0
T
beta2_power:0beta2_power/Assignbeta2_power/read:02beta2_power/initial_value:0

penalty/penalty_param/Adam:0!penalty/penalty_param/Adam/Assign!penalty/penalty_param/Adam/read:02.penalty/penalty_param/Adam/Initializer/zeros:0

penalty/penalty_param/Adam_1:0#penalty/penalty_param/Adam_1/Assign#penalty/penalty_param/Adam_1/read:020penalty/penalty_param/Adam_1/Initializer/zeros:0
\
beta1_power_1:0beta1_power_1/Assignbeta1_power_1/read:02beta1_power_1/initial_value:0
\
beta2_power_1:0beta2_power_1/Assignbeta2_power_1/read:02beta2_power_1/initial_value:0
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

penalty/penalty_param/Adam_2:0#penalty/penalty_param/Adam_2/Assign#penalty/penalty_param/Adam_2/read:020penalty/penalty_param/Adam_2/Initializer/zeros:0

penalty/penalty_param/Adam_3:0#penalty/penalty_param/Adam_3/Assign#penalty/penalty_param/Adam_3/read:020penalty/penalty_param/Adam_3/Initializer/zeros:0
\
beta1_power_2:0beta1_power_2/Assignbeta1_power_2/read:02beta1_power_2/initial_value:0
\
beta2_power_2:0beta2_power_2/Assignbeta2_power_2/read:02beta2_power_2/initial_value:0
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
vc/dense_2/bias/Adam_1:0vc/dense_2/bias/Adam_1/Assignvc/dense_2/bias/Adam_1/read:02*vc/dense_2/bias/Adam_1/Initializer/zeros:0"đ
trainable_variablesŘŐ
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
~
penalty/penalty_param:0penalty/penalty_param/Assignpenalty/penalty_param/read:02%penalty/penalty_param/initial_value:08"$
train_op

Adam
Adam_1
Adam_2*Ď
serving_defaultť
)
x$
Placeholder:0˙˙˙˙˙˙˙˙˙L$
v
vf/Squeeze:0˙˙˙˙˙˙˙˙˙%
pi
pi/add:0˙˙˙˙˙˙˙˙˙%
vc
vc/Squeeze:0˙˙˙˙˙˙˙˙˙tensorflow/serving/predict