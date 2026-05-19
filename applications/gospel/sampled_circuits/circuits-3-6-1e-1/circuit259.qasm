OPENQASM 2.0;
include "qelib1.inc";
qreg q260[3];
rz(3*pi/4) q260[2];
cx q260[2],q260[1];
cx q260[1],q260[0];
rx(pi/4) q260[1];
