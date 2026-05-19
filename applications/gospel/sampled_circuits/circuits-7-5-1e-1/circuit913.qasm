OPENQASM 2.0;
include "qelib1.inc";
qreg q914[7];
cx q914[4],q914[5];
cx q914[3],q914[4];
cx q914[2],q914[3];
cx q914[2],q914[1];
cx q914[1],q914[0];
