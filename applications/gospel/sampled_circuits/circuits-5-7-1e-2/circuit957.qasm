OPENQASM 2.0;
include "qelib1.inc";
qreg q958[5];
rz(3*pi/4) q958[4];
cx q958[3],q958[4];
cx q958[2],q958[3];
cx q958[2],q958[1];
cx q958[1],q958[0];
