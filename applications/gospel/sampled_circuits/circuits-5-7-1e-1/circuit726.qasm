OPENQASM 2.0;
include "qelib1.inc";
qreg q727[5];
rx(pi) q727[4];
cx q727[3],q727[4];
cx q727[3],q727[2];
cx q727[2],q727[1];
cx q727[1],q727[0];
