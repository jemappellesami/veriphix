OPENQASM 2.0;
include "qelib1.inc";
qreg q176[3];
cx q176[0],q176[1];
rz(3*pi/4) q176[2];
cx q176[1],q176[2];
cx q176[1],q176[0];
