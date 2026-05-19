OPENQASM 2.0;
include "qelib1.inc";
qreg q199[5];
rz(pi/2) q199[3];
cx q199[2],q199[3];
cx q199[1],q199[2];
cx q199[1],q199[0];
