OPENQASM 2.0;
include "qelib1.inc";
qreg q785[5];
cx q785[1],q785[2];
cx q785[2],q785[3];
cx q785[2],q785[1];
cx q785[1],q785[0];
