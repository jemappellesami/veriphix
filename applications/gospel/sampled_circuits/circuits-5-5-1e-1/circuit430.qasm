OPENQASM 2.0;
include "qelib1.inc";
qreg q431[5];
cx q431[3],q431[2];
cx q431[4],q431[3];
cx q431[2],q431[3];
cx q431[2],q431[1];
cx q431[0],q431[1];
