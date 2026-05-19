OPENQASM 2.0;
include "qelib1.inc";
qreg q602[6];
cx q602[2],q602[3];
cx q602[4],q602[3];
cx q602[3],q602[2];
cx q602[1],q602[2];
cx q602[1],q602[0];
