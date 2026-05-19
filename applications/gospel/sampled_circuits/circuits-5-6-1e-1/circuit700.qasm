OPENQASM 2.0;
include "qelib1.inc";
qreg q701[5];
cx q701[4],q701[3];
cx q701[2],q701[3];
cx q701[2],q701[1];
cx q701[0],q701[1];
rx(pi/4) q701[1];
