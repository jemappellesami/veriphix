OPENQASM 2.0;
include "qelib1.inc";
qreg q547[5];
cx q547[2],q547[3];
cx q547[3],q547[4];
cx q547[3],q547[2];
cx q547[2],q547[1];
cx q547[1],q547[0];
