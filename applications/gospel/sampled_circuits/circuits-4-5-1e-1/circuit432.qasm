OPENQASM 2.0;
include "qelib1.inc";
qreg q433[4];
cx q433[2],q433[3];
cx q433[3],q433[2];
cx q433[2],q433[1];
cx q433[0],q433[1];
