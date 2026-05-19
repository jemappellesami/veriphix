OPENQASM 2.0;
include "qelib1.inc";
qreg q51[7];
cx q51[4],q51[3];
cx q51[2],q51[3];
cx q51[1],q51[2];
cx q51[1],q51[0];
