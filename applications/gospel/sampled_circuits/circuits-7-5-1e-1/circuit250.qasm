OPENQASM 2.0;
include "qelib1.inc";
qreg q251[7];
cx q251[5],q251[4];
cx q251[4],q251[3];
cx q251[3],q251[2];
cx q251[2],q251[1];
cx q251[0],q251[1];
