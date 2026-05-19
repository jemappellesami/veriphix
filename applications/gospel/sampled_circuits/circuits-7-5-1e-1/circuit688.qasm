OPENQASM 2.0;
include "qelib1.inc";
qreg q689[7];
cx q689[4],q689[5];
cx q689[3],q689[4];
cx q689[3],q689[2];
cx q689[2],q689[1];
cx q689[1],q689[0];
