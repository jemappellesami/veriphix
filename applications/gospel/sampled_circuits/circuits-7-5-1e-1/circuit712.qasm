OPENQASM 2.0;
include "qelib1.inc";
qreg q713[7];
cx q713[5],q713[4];
cx q713[4],q713[3];
cx q713[2],q713[3];
cx q713[1],q713[2];
cx q713[1],q713[0];
