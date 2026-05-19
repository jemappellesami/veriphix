OPENQASM 2.0;
include "qelib1.inc";
qreg q645[3];
rz(5*pi/4) q645[2];
cx q645[2],q645[1];
cx q645[0],q645[1];
rx(pi/4) q645[1];
