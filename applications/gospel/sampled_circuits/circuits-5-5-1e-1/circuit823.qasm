OPENQASM 2.0;
include "qelib1.inc";
qreg q824[5];
cx q824[3],q824[2];
cx q824[4],q824[3];
cx q824[2],q824[3];
cx q824[2],q824[1];
cx q824[1],q824[0];
