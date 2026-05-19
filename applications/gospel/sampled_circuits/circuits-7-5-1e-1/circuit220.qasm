OPENQASM 2.0;
include "qelib1.inc";
qreg q221[7];
cx q221[5],q221[4];
cx q221[4],q221[3];
cx q221[2],q221[3];
cx q221[2],q221[1];
cx q221[0],q221[1];
