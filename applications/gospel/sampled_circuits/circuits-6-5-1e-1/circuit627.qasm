OPENQASM 2.0;
include "qelib1.inc";
qreg q628[6];
cx q628[4],q628[5];
cx q628[3],q628[4];
cx q628[2],q628[3];
cx q628[2],q628[1];
cx q628[0],q628[1];
