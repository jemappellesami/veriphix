OPENQASM 2.0;
include "qelib1.inc";
qreg q676[3];
rx(3*pi/2) q676[0];
cx q676[0],q676[1];
cx q676[2],q676[1];
rx(pi/4) q676[0];
