OPENQASM 2.0;
include "qelib1.inc";
qreg q636[6];
cx q636[1],q636[2];
cx q636[3],q636[2];
cx q636[2],q636[1];
cx q636[0],q636[1];
rx(pi/4) q636[1];
