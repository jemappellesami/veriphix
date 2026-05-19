OPENQASM 2.0;
include "qelib1.inc";
qreg q636[7];
cx q636[4],q636[5];
cx q636[3],q636[4];
cx q636[2],q636[3];
cx q636[2],q636[1];
cx q636[1],q636[0];
