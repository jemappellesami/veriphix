OPENQASM 2.0;
include "qelib1.inc";
qreg q646[6];
cx q646[5],q646[4];
cx q646[4],q646[3];
cx q646[2],q646[3];
cx q646[2],q646[1];
cx q646[1],q646[0];
