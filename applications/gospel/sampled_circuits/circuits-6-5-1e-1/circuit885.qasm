OPENQASM 2.0;
include "qelib1.inc";
qreg q886[6];
cx q886[3],q886[2];
cx q886[3],q886[4];
cx q886[2],q886[3];
cx q886[1],q886[2];
cx q886[0],q886[1];
