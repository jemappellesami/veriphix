OPENQASM 2.0;
include "qelib1.inc";
qreg q919[5];
cx q919[0],q919[1];
cx q919[2],q919[1];
cx q919[3],q919[2];
cx q919[2],q919[1];
cx q919[0],q919[1];
