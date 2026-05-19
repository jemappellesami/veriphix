OPENQASM 2.0;
include "qelib1.inc";
qreg q552[5];
rx(pi/2) q552[4];
cx q552[4],q552[3];
cx q552[3],q552[2];
cx q552[1],q552[2];
cx q552[0],q552[1];
