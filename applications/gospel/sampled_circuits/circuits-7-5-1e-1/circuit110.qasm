OPENQASM 2.0;
include "qelib1.inc";
qreg q111[7];
cx q111[4],q111[5];
cx q111[3],q111[4];
cx q111[3],q111[2];
cx q111[2],q111[1];
cx q111[0],q111[1];
