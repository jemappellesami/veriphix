OPENQASM 2.0;
include "qelib1.inc";
qreg q384[3];
cx q384[1],q384[0];
rz(7*pi/4) q384[1];
cx q384[1],q384[2];
cx q384[1],q384[0];
rx(pi/4) q384[1];
