OPENQASM 2.0;
include "qelib1.inc";
qreg q440[5];
cx q440[3],q440[4];
cx q440[3],q440[2];
cx q440[1],q440[2];
cx q440[0],q440[1];
