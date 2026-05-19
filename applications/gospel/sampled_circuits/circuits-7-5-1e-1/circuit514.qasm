OPENQASM 2.0;
include "qelib1.inc";
qreg q515[7];
cx q515[5],q515[4];
cx q515[3],q515[4];
cx q515[3],q515[2];
cx q515[1],q515[2];
cx q515[1],q515[0];
