OPENQASM 2.0;
include "qelib1.inc";
qreg q126[6];
cx q126[5],q126[4];
cx q126[4],q126[3];
cx q126[2],q126[3];
cx q126[2],q126[1];
cx q126[1],q126[0];
