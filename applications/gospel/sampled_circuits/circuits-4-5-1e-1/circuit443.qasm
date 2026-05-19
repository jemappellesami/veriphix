OPENQASM 2.0;
include "qelib1.inc";
qreg q444[4];
rz(pi/2) q444[3];
cx q444[3],q444[2];
cx q444[1],q444[2];
cx q444[0],q444[1];
