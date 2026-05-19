OPENQASM 2.0;
include "qelib1.inc";
qreg q116[5];
rx(pi) q116[4];
cx q116[4],q116[3];
cx q116[3],q116[2];
cx q116[2],q116[1];
cx q116[1],q116[0];
