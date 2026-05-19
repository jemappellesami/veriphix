OPENQASM 2.0;
include "qelib1.inc";
qreg q552[4];
rx(3*pi/4) q552[3];
cx q552[3],q552[2];
cx q552[2],q552[1];
cx q552[1],q552[0];
