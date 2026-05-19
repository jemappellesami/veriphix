OPENQASM 2.0;
include "qelib1.inc";
qreg q322[5];
cx q322[4],q322[3];
cx q322[3],q322[2];
cx q322[1],q322[2];
cx q322[0],q322[1];
