OPENQASM 2.0;
include "qelib1.inc";
qreg q710[5];
cx q710[3],q710[4];
cx q710[3],q710[2];
cx q710[2],q710[1];
cx q710[0],q710[1];
