OPENQASM 2.0;
include "qelib1.inc";
qreg q376[3];
rx(3*pi/4) q376[2];
cx q376[2],q376[1];
cx q376[1],q376[2];
cx q376[1],q376[0];
rx(pi/4) q376[1];
