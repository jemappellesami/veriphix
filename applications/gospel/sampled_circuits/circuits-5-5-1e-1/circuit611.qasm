OPENQASM 2.0;
include "qelib1.inc";
qreg q612[5];
cx q612[4],q612[3];
cx q612[2],q612[3];
cx q612[2],q612[1];
cx q612[1],q612[0];
