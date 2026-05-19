OPENQASM 2.0;
include "qelib1.inc";
qreg q488[5];
cx q488[3],q488[4];
cx q488[3],q488[2];
cx q488[2],q488[1];
cx q488[0],q488[1];
