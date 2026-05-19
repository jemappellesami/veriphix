OPENQASM 2.0;
include "qelib1.inc";
qreg q299[7];
cx q299[4],q299[5];
cx q299[4],q299[3];
cx q299[3],q299[2];
cx q299[2],q299[1];
cx q299[1],q299[0];
